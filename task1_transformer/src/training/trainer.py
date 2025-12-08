import torch
import torch.nn as nn
import time
import os
import math
from tqdm import tqdm
from torch.optim import Optimizer
from typing import Optional, Dict, Tuple

from src.model.transformer import Transformer
from src.training.optimization import ScheduledOptim, create_optimizer
from src.data_processing.create_dataset import get_data_loaders

class TransformerTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,  # ScheduledOptim
        criterion: nn.Module,
        train_loader,
        val_loader,
        pad_idx: int,
        device: str,
        config: Optional[Dict] = None,
        save_dir: str = "checkpoints",
        log_freq: int = 10
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.pad_idx = pad_idx
        self.device = device
        self.save_dir = save_dir
        self.log_freq = log_freq
        self.config = config or {}
        
        # Gradient clipping value
        self.clip = self.config.get('clip', 1.0)
        
        # History tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_ppl': [],
            'val_ppl': [],
            'learning_rates': []
        }
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"Trainer initialized on {device}")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
    def create_masks(self, src: torch.Tensor, trg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tạo masks cho source và target sequences.
        
        Args:
            src: Source tensor [batch_size, src_len]
            trg: Target tensor [batch_size, trg_len] (đã bỏ EOS token)
            
        Returns:
            src_mask: [batch_size, 1, 1, src_len]
            trg_mask: [batch_size, 1, trg_len, trg_len]
        """
        # Source padding mask
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        # Target padding mask
        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        # Target subsequent (look-ahead) mask
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        trg_sub_mask = trg_sub_mask.unsqueeze(0).unsqueeze(0)
        
        # Combine masks
        trg_mask = trg_pad_mask & trg_sub_mask
        
        return src_mask, trg_mask
    
    def calculate_ppl(self, loss: float) -> float:
        """Tính perplexity từ loss."""
        try:
            return math.exp(loss)
        except OverflowError:
            return float('inf')
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Huấn luyện một epoch.
        
        Returns:
            avg_loss: Loss trung bình
            ppl: Perplexity
        """
        self.model.train()
        total_loss = 0
        total_tokens = 0
        start_time = time.time()
        
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), 
                   desc=f"Epoch {epoch} [Train]")
        
        for batch_idx, (src, trg) in pbar:
            src = src.to(self.device)
            trg = trg.to(self.device)
            
            # Prepare decoder input and labels
            trg_input = trg[:, :-1]   # Bỏ EOS token
            trg_label = trg[:, 1:]    # Bỏ SOS token
            
            # Create masks
            src_mask, trg_mask = self.create_masks(src, trg_input)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(src, trg_input, src_mask, trg_mask)
            
            # Reshape for loss calculation
            output_flat = output.contiguous().view(-1, output.shape[-1])
            label_flat = trg_label.contiguous().view(-1)
            
            # Calculate loss (ignore padding)
            loss = self.criterion(output_flat, label_flat)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            
            # Update weights and learning rate
            if hasattr(self.optimizer, 'step_and_update_lr'):
                self.optimizer.step_and_update_lr()
            else:
                self.optimizer.step()
            
            # Track metrics
            batch_loss = loss.item()
            batch_tokens = (label_flat != self.pad_idx).sum().item()
            
            total_loss += batch_loss * batch_tokens
            total_tokens += batch_tokens
            
            # Update progress bar
            if batch_idx % self.log_freq == 0:
                current_lr = self.optimizer.get_current_lr() if hasattr(self.optimizer, 'get_current_lr') else \
                           self.optimizer.param_groups[0]['lr']
                
                pbar.set_postfix({
                    'loss': f'{batch_loss:.3f}',
                    'lr': f'{current_lr:.6f}',
                    'ppl': f'{self.calculate_ppl(batch_loss):.2f}'
                })
        
        # Calculate epoch metrics
        avg_loss = total_loss / total_tokens if total_tokens > 0 else total_loss / len(self.train_loader)
        ppl = self.calculate_ppl(avg_loss)
        
        # Record learning rate
        current_lr = self.optimizer.get_current_lr() if hasattr(self.optimizer, 'get_current_lr') else \
                   self.optimizer.param_groups[0]['lr']
        self.history['learning_rates'].append(current_lr)
        
        print(f"  Train Loss: {avg_loss:.4f} | Train PPL: {ppl:.2f} | Time: {time.time() - start_time:.2f}s")
        
        return avg_loss, ppl
    
    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        """
        Đánh giá mô hình trên validation set.
        
        Returns:
            avg_loss: Loss trung bình
            ppl: Perplexity
        """
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        pbar = tqdm(self.val_loader, desc="[Validation]", leave=False)
        
        for src, trg in pbar:
            src = src.to(self.device)
            trg = trg.to(self.device)
            
            # Prepare decoder input and labels
            trg_input = trg[:, :-1]
            trg_label = trg[:, 1:]
            
            # Create masks
            src_mask, trg_mask = self.create_masks(src, trg_input)
            
            # Forward pass
            output = self.model(src, trg_input, src_mask, trg_mask)
            
            # Calculate loss
            output_flat = output.contiguous().view(-1, output.shape[-1])
            label_flat = trg_label.contiguous().view(-1)
            
            loss = self.criterion(output_flat, label_flat)
            
            # Track metrics
            batch_loss = loss.item()
            batch_tokens = (label_flat != self.pad_idx).sum().item()
            
            total_loss += batch_loss * batch_tokens
            total_tokens += batch_tokens
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{batch_loss:.3f}',
                'ppl': f'{self.calculate_ppl(batch_loss):.2f}'
            })
        
        # Calculate validation metrics
        avg_loss = total_loss / total_tokens if total_tokens > 0 else total_loss / len(self.val_loader)
        ppl = self.calculate_ppl(avg_loss)
        
        return avg_loss, ppl
    
    def fit(
        self, 
        epochs: int, 
        validate_every: int = 1,
        early_stopping_patience: Optional[int] = None
    ) -> Dict:
        """
        Huấn luyện mô hình trong nhiều epochs.
        
        Args:
            epochs: Số lượng epochs
            validate_every: Validate sau mỗi N epochs
            early_stopping_patience: Số epochs chờ đợi trước khi dừng sớm
            
        Returns:
            history: Lịch sử training
        """
        print(f"\n{'='*50}")
        print(f"Starting training for {epochs} epochs")
        print(f"{'='*50}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            
            # Train
            train_loss, train_ppl = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_ppl'].append(train_ppl)
            
            # Validate
            if epoch % validate_every == 0:
                val_loss, val_ppl = self.evaluate()
                self.history['val_loss'].append(val_loss)
                self.history['val_ppl'].append(val_ppl)
                
                print(f"  Val Loss:   {val_loss:.4f} | Val PPL:   {val_ppl:.2f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_checkpoint(f"best_model.pt", epoch, val_loss, val_ppl)
                    print(f"  ✓ Saved best model (loss: {val_loss:.4f})")
                else:
                    patience_counter += 1
                    
                # Early stopping check
                if (early_stopping_patience and 
                    patience_counter >= early_stopping_patience):
                    print(f"\nEarly stopping after {epoch} epochs!")
                    break
            else:
                self.history['val_loss'].append(None)
                self.history['val_ppl'].append(None)
            
            # Save checkpoint
            if epoch % self.config.get('save_interval', 5) == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt", epoch, 
                                   val_loss if 'val_loss' in locals() else None,
                                   val_ppl if 'val_ppl' in locals() else None)
        
        best_checkpoint_path = os.path.join(self.save_dir, "best_model.pt")
        if os.path.exists(best_checkpoint_path):
            self.load_checkpoint(best_checkpoint_path)
            print(f"\nLoaded best model from epoch {self._current_epoch if hasattr(self, '_current_epoch') else 'unknown'}")
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best validation PPL: {self.calculate_ppl(best_val_loss):.2f}")
        
        return self.history
    
    def save_checkpoint(self, filename: str, epoch: int, val_loss: float, val_ppl: float):
        """Lưu checkpoint của mô hình."""
        checkpoint_path = os.path.join(self.save_dir, filename)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
            'val_loss': val_loss,
            'val_ppl': val_ppl,
            'config': self.config,
            'history': self.history
        }
        
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str):
        """Tải checkpoint của mô hình."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        self._current_epoch = checkpoint.get('epoch', 0)
        
        print(f"Loaded checkpoint from epoch {self._current_epoch}")
        print(f"Validation loss: {checkpoint.get('val_loss', 'N/A')}")
        
        return checkpoint
    
    def translate_sentence(
        self, 
        sentence: str, 
        src_tokenizer, 
        trg_tokenizer, 
        src_vocab, 
        trg_vocab, 
        max_len: int = 100
    ) -> str:
        """
        Dịch một câu sử dụng mô hình đã huấn luyện.
        
        Args:
            sentence: Câu cần dịch
            src_tokenizer: Tokenizer cho ngôn ngữ nguồn
            trg_tokenizer: Tokenizer cho ngôn ngữ đích
            src_vocab: Vocabulary nguồn
            trg_vocab: Vocabulary đích
            max_len: Độ dài tối đa của câu dịch
            
        Returns:
            Câu đã được dịch
        """
        self.model.eval()
        
        tokens = src_tokenizer.tokenize(sentence)
        src_indices = src_vocab.numericalize(tokens)
        src_tensor = torch.tensor(src_indices).unsqueeze(0).to(self.device)
        
        # Khởi tạo target với SOS token
        trg_indices = [trg_vocab.stoi['<sos>']]
        
        with torch.no_grad():
            for _ in range(max_len):
                trg_tensor = torch.tensor(trg_indices).unsqueeze(0).to(self.device)
                
                # Tạo masks
                src_mask, trg_mask = self.create_masks(src_tensor, trg_tensor)
                
                # Dự đoán
                output = self.model(src_tensor, trg_tensor, src_mask, trg_mask)
                output = output[:, -1, :]  # Lấy prediction cho token cuối cùng
                
                # Lấy token có xác suất cao nhất
                pred_token = output.argmax(-1).item()
                
                # Dừng nếu gặp EOS token
                if pred_token == trg_vocab.stoi['<eos>']:
                    break
                
                trg_indices.append(pred_token)
        
        # Chuyển indices sang tokens
        trg_tokens = [trg_vocab.itos[idx] for idx in trg_indices[1:]]  # Bỏ SOS token
        
        return trg_tokenizer.detokenize(trg_tokens)


# Hàm helper để khởi tạo và chạy training
def create_and_train(
    model_config: Dict,
    data_config: Dict,
    training_config: Dict
):
    """
    Hàm helper để khởi tạo và chạy training hoàn chỉnh.
    
    Args:
        model_config: Cấu hình cho mô hình
        data_config: Cấu hình cho dữ liệu
        training_config: Cấu hình cho training
    """
    
    
    # Set seed
    torch.manual_seed(training_config.get('seed', 42))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(training_config.get('seed', 42))
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    print("Loading data...")
    train_loader, src_vocab, trg_vocab = get_data_loaders(
        src_file=data_config['train_src'],
        trg_file=data_config['train_trg'],
        batch_size=training_config['batch_size'],
        src_lang=data_config['src_lang'],
        trg_lang=data_config['trg_lang'],
        max_strlen=data_config.get('max_strlen', 100),
        seed=training_config.get('seed', 42)
    )
    
    val_loader, _, _ = get_data_loaders(
        src_file=data_config['val_src'],
        trg_file=data_config['val_trg'],
        batch_size=training_config['batch_size'],
        src_vocab=src_vocab,
        trg_vocab=trg_vocab,
        src_lang=data_config['src_lang'],
        trg_lang=data_config['trg_lang'],
        max_strlen=data_config.get('max_strlen', 100),
        seed=training_config.get('seed', 42)
    )
    
    # Create model
    print("Creating model...")
    model = Transformer(
        src_vocab_size=len(src_vocab),
        trg_vocab_size=len(trg_vocab),
        **model_config
    )
    
    # Create optimizer
    optimizer = create_optimizer(
        model,
        d_model=model_config['d_model'],
        warmup_steps=training_config.get('warmup_steps', 4000),
        init_lr=training_config.get('init_lr', 0.2)
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 là pad_idx
    
    # Create trainer
    trainer = TransformerTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        pad_idx=0,
        device=device,
        config={**model_config, **data_config, **training_config},
        save_dir=training_config.get('save_dir', 'checkpoints'),
        log_freq=training_config.get('log_freq', 10)
    )
    
    # Train
    history = trainer.fit(
        epochs=training_config.get('num_epochs', 20),
        validate_every=training_config.get('validate_every', 1),
        early_stopping_patience=training_config.get('early_stopping_patience', 5)
    )
    
    return trainer, history, src_vocab, trg_vocab


# Example usage
if __name__ == "__main__":
    # Cấu hình
    config = {
        'model': {
            'd_model': 512,
            'N': 6,
            'heads': 8,
            'dropout': 0.1
        },
        'data': {
            'train_src': 'data/train.en',
            'train_trg': 'data/train.vi',
            'val_src': 'data/val.en',
            'val_trg': 'data/val.vi',
            'src_lang': 'en',
            'trg_lang': 'vi',
            'max_strlen': 100
        },
        'training': {
            'batch_size': 32,
            'num_epochs': 20,
            'init_lr': 0.2,
            'warmup_steps': 4000,
            'clip': 1.0,
            'save_dir': 'checkpoints',
            'log_freq': 10,
            'validate_every': 1,
            'early_stopping_patience': 5,
            'seed': 42
        }
    }
    
    # Chạy training
    trainer, history, src_vocab, trg_vocab = create_and_train(
        config['model'],
        config['data'],
        config['training']
    )