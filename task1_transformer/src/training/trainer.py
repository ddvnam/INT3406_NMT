# src/training/trainer.py
import torch
import torch.optim as optim
from tqdm import tqdm
import os
from typing import Dict, Any, Tuple

from ..data_processing.dataset import create_data_loaders
from ..model.transformer import Transformer
from .loss import LabelSmoothedCrossEntropyLoss
from .scheduler import TransformerScheduler
from ..evaluation.metrics import evaluate

class Trainer:
    """Main trainer class"""
    def __init__(self, config, tokenizer, train_src, train_tgt, valid_src, valid_tgt):
        self.config = config
        self.tokenizer = tokenizer
        self.train_src = train_src
        self.train_tgt = train_tgt
        self.valid_src = valid_src
        self.valid_tgt = valid_tgt
        
        # Initialize model
        self.model = Transformer(config, tokenizer.get_vocab_size()).to(config.device)
        
        # Loss function
        self.criterion = LabelSmoothedCrossEntropyLoss(
            config.label_smoothing, ignore_index=0
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.lr_base,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=config.weight_decay,
        )
        
        # Scheduler
        self.scheduler = TransformerScheduler(
            self.optimizer, config.warmup_steps, config.lr_base
        )
        
        # Tracking
        self.best_val_loss = float("inf")
        self.best_val_bleu = 0.0
        self.best_val_chrf = 0.0
        
        # Create save directory
        os.makedirs(config.save_dir, exist_ok=True)
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def train_one_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        
        # Create data loader for this epoch
        train_loader, valid_loader, vi_slice_len = create_data_loaders(
            self.train_src, self.train_tgt,
            self.valid_src, self.valid_tgt,
            self.tokenizer, self.config, epoch
        )
        
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for src_batch, tgt_batch in pbar:
            src_batch = src_batch.to(self.config.device)
            tgt_batch = tgt_batch.to(self.config.device)
            
            # Forward pass
            logits = self.model(src_batch, tgt_batch)
            targets = tgt_batch[:, 1:]
            
            # Calculate loss
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )
            
            # Update
            self.optimizer.step()
            self.scheduler.step()
            
            # Update progress bar
            total_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.scheduler.get_lr():.6f}"
            })
        
        return total_loss / len(train_loader), vi_slice_len
    
    def train(self):
        """Main training loop"""
        print(f"\nStarting training for {self.config.num_epochs} epochs...")
        
        for epoch in range(1, self.config.num_epochs + 1):
            # Clean up GPU memory
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            # Train
            train_loss, vi_slice_len = self.train_one_epoch(epoch)
            
            # Evaluate
            eval_results = evaluate(
                self.model, self.criterion, self.tokenizer,
                self.valid_src, self.valid_tgt, self.config
            )
            
            val_loss = eval_results['loss']
            val_bleu = eval_results['bleu']
            val_chrf = eval_results['chrf']
            
            # Logging
            print(f"\nEpoch {epoch}/{self.config.num_epochs}")
            print(f"  vi→en coverage: {vi_slice_len}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Valid Loss: {val_loss:.4f}")
            print(f"  Valid BLEU: {val_bleu:.2f}")
            print(f"  Valid CHRF: {val_chrf:.2f}")
            print(f"  Learning Rate: {self.scheduler.get_lr():.8f}")
            
            # Save checkpoint
            if epoch % self.config.save_every == 0:
                self.save_checkpoint(epoch, train_loss, val_loss, val_bleu, val_chrf)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_bleu = val_bleu
                self.best_val_chrf = val_chrf
                self.save_checkpoint(epoch, train_loss, val_loss, val_bleu, val_chrf, "best_model.pt")
                print(f"  ★ New best model saved (Loss: {self.best_val_loss:.4f})")
        
        print("\nTraining finished!")
        print(f"Best valid loss: {self.best_val_loss:.4f}")
        print(f"Best BLEU: {self.best_val_bleu:.2f}")
        print(f"Best CHRF: {self.best_val_chrf:.2f}")
    
    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: float,
                       val_bleu: float, val_chrf: float, filename: str = "checkpoint.pt"):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_bleu": val_bleu,
            "val_chrf": val_chrf,
            "config": self.config,
        }
        
        path = os.path.join(self.config.save_dir, filename)
        torch.save(checkpoint, path)
        print(f"  ✓ Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"]