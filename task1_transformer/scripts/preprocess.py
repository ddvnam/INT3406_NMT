import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_processing.preprocess import read_file, clean_data, split_data


def main():
    SRC_FILE = "data/raw/train.en.txt"
    TGT_FILE = "data/raw/train.vi.txt"

    # Load data
    print("Loading data...")
    src_sentences = read_file(SRC_FILE)
    tgt_sentences = read_file(TGT_FILE)
    print(f"Loaded {len(src_sentences)} parallel sentences")
    print(f"Loaded {len(tgt_sentences)} parallel sentences")

    # Clean data
    print("Cleaning data...")
    src_clean, tgt_clean = clean_data(src_sentences, tgt_sentences)
    print(f"Cleaned data has {len(src_clean)} parallel sentences")

    # Split data
    print("Splitting data...")
    train_src, train_tgt, valid_src, valid_tgt = split_data(
        src_clean, tgt_clean, train_ratio=0.95
    )

    print(f"Train set: {len(train_src)} sentences")
    print(f"Validation set: {len(valid_src)} sentences")


    # Save the data
    os.makedirs("data/processed", exist_ok=True)

    with open("data/processed/train.en.txt", "w", encoding="utf-8") as f:
        for line in train_src:
            f.write(line + "\n")

    with open("data/processed/train.vi.txt", "w", encoding="utf-8") as f:
        for line in train_tgt:
            f.write(line + "\n")

    with open("data/processed/valid.en.txt", "w", encoding="utf-8") as f:
        for line in valid_src:
            f.write(line + "\n")

    with open("data/processed/valid.vi.txt", "w", encoding="utf-8") as f:
        for line in valid_tgt:
            f.write(line + "\n")
    print("Data saved to data/processed/")


if __name__ == "__main__":
    main()