"""
Training script for Transformer Neural Machine Translation
Usage: python train.py --config config.yaml
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from src.model.transformer import Transformer
from src.training.optimization import create_optimizer
from src.data_processing.create_dataset import get_data_loaders
from src.training.trainer import TransformerTrainer
from src.data_processing.preprocessing_data import Tokenizer


def set_seed(seed=42):
    """Set seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(config):
    """Prepare data loaders and vocabularies."""
    print("=" * 60)
    print("PREPARING DATA")
    print("=" * 60)
    
    # Load training data
    print(f"Loading training data from:")
    print(f"  Source: {config['data']['train_src']}")
    print(f"  Target: {config['data']['train_trg']}")
    
    train_loader, src_vocab, trg_vocab = get_data_loaders(
        src_file=config['data']['train_src'],
        trg_file=config['data']['train_trg'],
        batch_size=config['training']['batch_size'],
        src_lang=config['data']['src_lang'],
        trg_lang=config['data']['trg_lang'],
        max_strlen=config['data'].get('max_strlen', 100),
        seed=config['training']['seed']
    )
    
    # Load validation data (using same vocab)
    print(f"\nLoading validation data from:")
    print(f"  Source: {config['data']['val_src']}")
    print(f"  Target: {config['data']['val_trg']}")
    
    val_loader, _, _ = get_data_loaders(
        src_file=config['data']['val_src'],
        trg_file=config['data']['val_trg'],
        batch_size=config['training']['batch_size'],
        src_vocab=src_vocab,
        trg_vocab=trg_vocab,
        src_lang=config['data']['src_lang'],
        trg_lang=config['data']['trg_lang'],
        max_strlen=config['data'].get('max_strlen', 100),
        seed=config['training']['seed']
    )
    
    print(f"\nVocabulary sizes:")
    print(f"  Source: {len(src_vocab)} tokens")
    print(f"  Target: {len(trg_vocab)} tokens")
    print(f"\nData statistics:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader, src_vocab, trg_vocab


def build_model(config, src_vocab_size, trg_vocab_size):
    """Build Transformer model."""
    print("\n" + "=" * 60)
    print("BUILDING MODEL")
    print("=" * 60)
    
    model = Transformer(
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        d_model=config['model']['d_model'],
        N=config['model']['n_layers'],
        heads=config['model']['n_heads'],
        dropout=config['model']['dropout']
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model architecture: Transformer")
    print(f"  d_model: {config['model']['d_model']}")
    print(f"  n_layers: {config['model']['n_layers']}")
    print(f"  n_heads: {config['model']['n_heads']}")
    print(f"  dropout: {config['model']['dropout']}")
    print(f"\nParameter count:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model


def setup_training(config, model, train_loader, val_loader, device):
    """Setup optimizer, loss function, and trainer."""
    print("\n" + "=" * 60)
    print("SETTING UP TRAINING")
    print("=" * 60)
    
    # Create optimizer with learning rate schedule
    optimizer = create_optimizer(
        model,
        d_model=config['model']['d_model'],
        warmup_steps=config['training']['warmup_steps'],
        init_lr=config['training']['init_lr'],
        betas=tuple(config['training']['betas']),
        eps=config['training']['eps']
    )
    
    # Loss function (ignore padding index)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 is pad_idx
    
    print(f"Optimizer: Adam with Transformer LR schedule")
    print(f"  Initial LR: {config['training']['init_lr']}")
    print(f"  Warmup steps: {config['training']['warmup_steps']}")
    print(f"  Betas: {config['training']['betas']}")
    print(f"  Epsilon: {config['training']['eps']}")
    print(f"\nLoss function: CrossEntropyLoss (ignore padding)")
    print(f"\nTraining config:")
    print(f"  Epochs: {config['training']['num_epochs']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Gradient clip: {config['training']['clip']}")
    print(f"  Device: {device}")
    print(f"  Save directory: {config['training']['save_dir']}")
    
    # Create trainer
    trainer = TransformerTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        pad_idx=0,  # pad_idx is always 0
        device=device,
        config=config,
        save_dir=config['training']['save_dir'],
        log_freq=config['training']['log_freq']
    )
    
    return trainer


def save_config_and_vocabs(config, src_vocab, trg_vocab, save_dir):
    """Save configuration and vocabularies."""
    import pickle
    
    # Save config
    config_path = os.path.join(save_dir, 'config.yaml')
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Save vocabularies
    vocab_path = os.path.join(save_dir, 'vocabs.pkl')
    with open(vocab_path, 'wb') as f:
        pickle.dump({'src_vocab': src_vocab, 'trg_vocab': trg_vocab}, f)
    
    print(f"\nSaved config to: {config_path}")
    print(f"Saved vocabularies to: {vocab_path}")


def test_translation(trainer, src_vocab, trg_vocab, test_sentences=None):
    """Test translation on sample sentences."""
    print("\n" + "=" * 60)
    print("TEST TRANSLATION")
    print("=" * 60)
    
    src_tokenizer = Tokenizer('en')
    trg_tokenizer = Tokenizer('vi')
    
    if test_sentences is None:
        test_sentences = [
            "Hello world",
            "How are you?",
            "I love machine learning",
            "This is a test sentence"
        ]
    
    for i, sentence in enumerate(test_sentences, 1):
        try:
            translation = trainer.translate_sentence(
                sentence,
                src_tokenizer,
                trg_tokenizer,
                src_vocab,
                trg_vocab,
                max_len=50
            )
            print(f"{i}. Source: {sentence}")
            print(f"   Translation: {translation}")
        except Exception as e:
            print(f"{i}. Error translating: {sentence}")
            print(f"   Error: {e}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Transformer for NMT')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test the model without training')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override device if specified
    if args.device:
        config['training']['device'] = args.device
    
    # Set device
    if config['training']['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['training']['device'])
    
    # Set seed for reproducibility
    set_seed(config['training']['seed'])
    
    print("=" * 60)
    print("TRANSFORMER NEURAL MACHINE TRANSLATION")
    print("=" * 60)
    print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Create save directory
    os.makedirs(config['training']['save_dir'], exist_ok=True)
    
    # Prepare data
    train_loader, val_loader, src_vocab, trg_vocab = prepare_data(config)
    
    # Build model
    model = build_model(config, len(src_vocab), len(trg_vocab))
    model = model.to(device)
    
    # Setup training
    trainer = setup_training(config, model, train_loader, val_loader, device)
    
    # Save configuration and vocabularies
    save_config_and_vocabs(config, src_vocab, trg_vocab, config['training']['save_dir'])
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Test-only mode
    if args.test_only:
        print("\nRunning in test-only mode...")
        test_translation(trainer, src_vocab, trg_vocab)
        return
    
    # Train model
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    history = trainer.fit(
        epochs=config['training']['num_epochs'],
        validate_every=config['training']['validate_every'],
        early_stopping_patience=config['training']['early_stopping_patience']
    )
    
    # Save final model
    final_path = os.path.join(config['training']['save_dir'], 'final_model.pt')
    trainer.save_checkpoint(final_path, trainer._current_epoch if hasattr(trainer, '_current_epoch') else config['training']['num_epochs'],
                           history['val_loss'][-1] if history['val_loss'][-1] is not None else float('inf'),
                           history['val_ppl'][-1] if history['val_ppl'][-1] is not None else float('inf'))
    
    # Test translation with trained model
    print("\nTesting translation with trained model...")
    test_translation(trainer, src_vocab, trg_vocab)
    
    # Plot training history (if matplotlib is available)
    try:
        plot_training_history(history, config['training']['save_dir'])
    except ImportError:
        print("\nMatplotlib not available. Skipping plots.")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Models saved in: {config['training']['save_dir']}")
    print(f"Training finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")


def plot_training_history(history, save_dir):
    """Plot training history."""
    try:
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Plot loss
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
        val_epochs = [i+1 for i, val in enumerate(history['val_loss']) if val is not None]
        val_losses = [val for val in history['val_loss'] if val is not None]
        if val_losses:
            plt.plot(val_epochs, val_losses, 'r-', label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot perplexity
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history['train_ppl'], 'b-', label='Train PPL')
        val_ppls = [val for val in history['val_ppl'] if val is not None]
        if val_ppls:
            plt.plot(val_epochs, val_ppls, 'r-', label='Val PPL')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.title('Training and Validation Perplexity')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        print(f"Training plots saved to: {plot_path}")
        
    except ImportError:
        pass


if __name__ == '__main__':
    import time
    main()