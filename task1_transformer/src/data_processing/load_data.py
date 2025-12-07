import os
from typing import Tuple, List

def read_data(src_file: str, trg_file: str) -> Tuple[List[str], List[str]]:
    """
    Đọc dữ liệu nguồn và đích từ file văn bản.
    
    Args:
        src_file (str): Đường dẫn tới file ngôn ngữ nguồn.
        trg_file (str): Đường dẫn tới file ngôn ngữ đích.
        
    Returns:
        tuple: (list các câu nguồn, list các câu đích)
        
    Raises:
        FileNotFoundError: Nếu file không tồn tại.
    """
    if not os.path.exists(src_file):
        raise FileNotFoundError(f"Source file not found: {src_file}")
    if not os.path.exists(trg_file):
        raise FileNotFoundError(f"Target file not found: {trg_file}")
    
    print(f"Reading data from {src_file} and {trg_file}...")
    
    with open(src_file, encoding='utf-8') as f:
        src_data = f.read().strip().split('\n')

    with open(trg_file, encoding='utf-8') as f:
        trg_data = f.read().strip().split('\n')

    return src_data, trg_data

if __name__ == "__main__":
    with open("test.en", "w", encoding="utf-8") as f: f.write("Hello\nWorld")
    with open("test.vi", "w", encoding="utf-8") as f: f.write("Xin chào\nThế giới")
    
    src, trg = read_data("test.en", "test.vi")
    print(f"Read {len(src)} sentences.")
    print(f"Sample: {src[0]} -> {trg[0]}")
    
    os.remove("test.en")
    os.remove("test.vi")