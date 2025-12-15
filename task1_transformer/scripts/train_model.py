import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ModelConfig, set_seed
from src.data_processing.preprocess import read_file, clean_data
from src.data_processing.tokenizer import Tokenizer
from src.training.trainer import Trainer
import sentencepiece

def main():
    set_seed(ModelConfig.seed)

    # Config
    config = ModelConfig(
        train_src_file="data/processed/train.en.txt",
        train_tgt_file="data/processed/train.vi.txt",
        spm_prefix="models/tokenizer/spm_en_vi_joint",
        save_dir="checkpoints"
    )

    # Load data
    print("Loading data...")
    src_sentences = read_file(config.train_src_file)
    tgt_sentences = read_file(config.train_tgt_file)

    valid_src_sentences = read_file("data/processed/valid.en.txt")
    valid_tgt_sentences = read_file("data/processed/valid.vi.txt")
    
    print("Loading tokenizer...")
    tokenizer = Tokenizer(f"{config.spm_prefix}.model")
    print("Tokenizer loaded.")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

    # Trainer
    trainer = Trainer(
        config, tokenizer,
        src_sentences, tgt_sentences,
        valid_src_sentences, valid_tgt_sentences
    )

    trainer.train()

if __name__ == "__main__":
    main()