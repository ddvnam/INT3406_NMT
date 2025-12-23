import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sentencepiece as spm
import os
import sys
from dataclasses import dataclass
from typing import List, Optional
import string

from src.model.transformer import Transformer
from src.evaluation.decoding import greedy_decode, beam_search_decode

# ==========================================
# 1. CẤU HÌNH (CONFIGURATION)
# ==========================================
@dataclass
class ModelConfig:
    vocab_size: int = 16000
    d_model: int = 512          
    n_heads: int = 8
    n_kv_heads: int = 4
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1        
    rope_base: float = 10000.0
    max_len: int = 128
    
    # --- SPECIAL TOKENS ---
    pad_token: str = "<pad>"
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    unk_token: str = "<unk>"
    en_token: str = "<2en>"
    vi_token: str = "<2vi>"
    
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_inference_components(checkpoint_path, tokenizer_path):
    config = ModelConfig()
    
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(tokenizer_path)
    real_vocab_size = sp_model.GetPieceSize()
    
    print(f"Initializing model with vocab size: {real_vocab_size}")
    model = Transformer(config, real_vocab_size).to(config.device)
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    
    # Xử lý 2 trường hợp save
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model, sp_model, config

def translate(text, model, sp_model, config, beam_size=3, target_lang="vi"):
    text = text.strip() # Xóa khoảng trắng thừa
    
    # Tự động thêm dấu chấm nếu chưa có dấu kết câu
    if text and text[-1] not in ".?!":
        # Nếu câu bắt đầu bằng từ để hỏi thì thêm dấu ?, ngược lại thêm .
        wh_words = ["what", "where", "when", "who", "why", "how", "which"]
        first_word = text.split()[0].lower()
        
        if first_word in wh_words:
            text += "?"
        else:
            text += "."
        
    lang_token = config.vi_token if target_lang == "vi" else config.en_token
    input_text = f"{lang_token} {text}"
    
    src_ids = sp_model.encode(input_text, out_type=int)
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(config.device)
    
    if beam_size > 1:
        result = beam_search_decode(model, src_tensor, sp_model, config, beam_size=beam_size)[0]
    else:
        result = greedy_decode(model, src_tensor, sp_model, config)[0]
    
    if target_lang == "en":
        # Remove html tags if any like &apos;, &quot;, <br />, etc.
        html_tags = {
            "&apos;": "'",
            "&quot;": '"',
            "&amp;": "&",
            "&lt;": "<",
            "&gt;": ">",
        }
        for tag, char in html_tags.items():
            result = result.replace(tag, char)

    return result

if __name__ == "__main__":
    # --- ĐƯỜNG DẪN CẤU HÌNH TỰ ĐỘNG ---
    # Giả sử bạn đang chạy lệnh từ thư mục gốc: task1_transformer
    base_dir = os.getcwd()
    
    # Đường dẫn tương đối dựa trên cấu trúc thư mục của bạn
    CHECKPOINT_FILE = os.path.join(base_dir, "checkpoints", "last.pt")
    
    # Tokenizer có thể ở trong 'models' hoặc 'data', hãy kiểm tra lại
    # Dựa trên ảnh của bạn, có vẻ tokenizer ở trong 'model/tokenizer' hoặc tương tự
    # Đây là đường dẫn dự đoán, bạn hãy sửa lại nếu cần:
    TOKENIZER_FILE = os.path.join(base_dir, "model", "tokenizer", "spm_en_vi_joint.model")
    
    # Nếu không tìm thấy, thử đường dẫn khác (ví dụ trong thư mục models)
    if not os.path.exists(TOKENIZER_FILE):
         TOKENIZER_FILE = os.path.join(base_dir, "models", "tokenizer", "spm_en_vi_joint.model")

    print(f"Working Directory: {base_dir}")
    print(f"Checkpoint Path: {CHECKPOINT_FILE}")
    print(f"Tokenizer Path:  {TOKENIZER_FILE}")
    print("-" * 50)

    if os.path.exists(CHECKPOINT_FILE) and os.path.exists(TOKENIZER_FILE):
        try:
            model, sp_model, config = load_inference_components(CHECKPOINT_FILE, TOKENIZER_FILE)
            print("Model loaded successfully!")
            print("-" * 50)
            
            while True:
                text = input("\nNhập câu tiếng Anh (gõ 'q' để thoát): ")
                if text.lower() in ['q', 'quit', 'exit']:
                    break
                
                translated = translate(text, model, sp_model, config, beam_size=3, target_lang="en")
                print(f"Dịch: {translated}")
                
        except Exception as e:
            print(f"\n❌ Lỗi khi chạy: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n❌ Lỗi: Không tìm thấy file model hoặc tokenizer.")
        print("Vui lòng kiểm tra lại đường dẫn file:")
        print(f" - Model tồn tại?: {os.path.exists(CHECKPOINT_FILE)}")
        print(f" - Tokenizer tồn tại?: {os.path.exists(TOKENIZER_FILE)}")