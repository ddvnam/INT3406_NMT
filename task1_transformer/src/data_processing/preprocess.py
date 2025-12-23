# src/data_processing/preprocess.py
import pandas as pd
import numpy as np
from typing import List, Tuple
import html
import re
import os

def read_file(path: str) -> List[str]:
    """Read text file and return list of lines"""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]

def clean_data(
    src_sentences: List[str], 
    tgt_sentences: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Clean parallel data:
    1. Decode HTML entities (&apos; -> ', &quot; -> ")
    2. Remove HTML tags (<br>, <div>...)
    3. Remove empty rows
    """
    df = pd.DataFrame({"src": src_sentences, "tgt": tgt_sentences})
    
    # --- BƯỚC 1: Xử lý HTML Entities (ví dụ: &apos; -> ') ---
    df["src"] = df["src"].apply(lambda x: html.unescape(str(x)))
    df["tgt"] = df["tgt"].apply(lambda x: html.unescape(str(x)))

    # --- BƯỚC 2: Xóa các thẻ HTML (ví dụ: <div>, <script>) ---
    # Pattern r'<[^>]+>' nghĩa là tìm chuỗi bắt đầu bằng < và kết thúc bằng >
    tag_pattern = r'<[^>]+>'
    df["src"] = df["src"].str.replace(tag_pattern, "", regex=True)
    df["tgt"] = df["tgt"].str.replace(tag_pattern, "", regex=True)

    # --- BƯỚC 3: Chuẩn hóa khoảng trắng ---
    df["src"] = df["src"].str.replace(r'\s+', ' ', regex=True).str.strip()
    df["tgt"] = df["tgt"].str.replace(r'\s+', ' ', regex=True).str.strip()

    # --- BƯỚC 4: Lọc dòng rỗng ---
    # Sau khi clean có thể một số dòng trở thành rỗng
    df_clean = df[ (df["src"] != "") & (df["tgt"] != "") ].copy()
    
    return df_clean["src"].tolist(), df_clean["tgt"].tolist()

def split_data(
    src_sentences: List[str],
    tgt_sentences: List[str],
    train_ratio: float = 0.95
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Split data into train and validation sets"""
    split_idx = int(train_ratio * len(src_sentences))
    if split_idx % 2 != 0:
        split_idx -= 1
    
    train_src = src_sentences[:split_idx]
    train_tgt = tgt_sentences[:split_idx]
    
    # Take every other sample for validation to keep balance
    valid_src = src_sentences[split_idx::2]
    valid_tgt = tgt_sentences[split_idx::2]
    
    return train_src, train_tgt, valid_src, valid_tgt

def analyze_length(
    sentences: List[str],
    tokenizer,
    max_tokens: int = None
):
    """Analyze sentence lengths"""
    lengths = [len(tokenizer.encode(s)) for s in sentences]
    
    stats = {
        "mean": np.mean(lengths),
        "min": np.min(lengths),
        "max": np.max(lengths),
        "std": np.std(lengths),
        "p95": np.percentile(lengths, 95)
    }
    
    if max_tokens:
        filtered = [s for s in sentences if len(tokenizer.encode(s)) <= max_tokens]
        stats["filtered_count"] = len(filtered)
        stats["filtered_ratio"] = len(filtered) / len(sentences)
    
    return stats