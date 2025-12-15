# main.py
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import ModelConfig, setup_seed
from src.data_processing.preprocess import read_file, clean_data, split_data
from src.data_processing.tokenizer import Tokenizer
from src.training.trainer import Trainer

def main():
    # Configuration
    config = ModelConfig(
        train_src_file="/path/to/train.en.txt",
        train_tgt_file="/path/to/train.vi.txt",
        spm_prefix="/path/to/spm_model",
        save_dir="./checkpoints"
    )
    
    # Setup
    setup_seed(config.seed)
    
    # Load data
    print("Loading data...")
    src_sentences = read_file(config.train_src_file)
    tgt_sentences = read_file(config.train_tgt_file)
    
    # Clean data
    src_sentences, tgt_sentences = clean_data(src_sentences, tgt_sentences)
    print(f"Loaded {len(src_sentences)} parallel sentences")
    
    # Split data
    train_src, train_tgt, valid_src, valid_tgt = split_data(
        src_sentences, tgt_sentences, 0.95
    )
    print(f"Train: {len(train_src)}, Valid: {len(valid_src)}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = Tokenizer(f"{config.spm_prefix}.model")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Train
    trainer = Trainer(
        config, tokenizer,
        train_src, train_tgt,
        valid_src, valid_tgt
    )
    trainer.train()

if __name__ == "__main__":
    main()