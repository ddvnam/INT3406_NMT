import sys
import os
os.environ["SENTENCEPIECE_LOGLEVEL"] = "ERROR"

import sentencepiece as spm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.tokenizer import train_tokenizer

def main():
    # Tạo file corpus kết hợp từ train data
    CORPUS_FILE = "data/processed/tokenizer_corpus.txt"
    
    print("Đang tạo corpus cho tokenizer...")
    
    # Đọc dữ liệu train
    with open("data/processed/train.en.txt", "r", encoding="utf-8") as f:
        en_lines = f.readlines()
    
    with open("data/processed/train.vi.txt", "r", encoding="utf-8") as f:
        vi_lines = f.readlines()
    
    # Ghi corpus kết hợp
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        for en, vi in zip(en_lines, vi_lines):
            f.write(en.strip() + "\n")
            f.write(vi.strip() + "\n")
    
    print(f"Đã tạo corpus với {len(en_lines) * 2} dòng")
    
    # Huấn luyện tokenizer
    print("Đang huấn luyện tokenizer...")
    OUTPUT_PREFIX = "models/tokenizer/spm_en_vi_joint"
    os.makedirs("models/tokenizer", exist_ok=True)
    
    train_tokenizer(
        corpus_path=CORPUS_FILE,
        output_prefix=OUTPUT_PREFIX,
        vocab_size=16000,
        model_type="bpe",
        user_defined_symbols=["<2en>", "<2vi>"]
    )
    
    print("Hoàn thành huấn luyện tokenizer!")

if __name__ == "__main__":
    main()