# src/data_processing/tokenizer.py
import sentencepiece as spm
import os
from typing import List

class Tokenizer:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
    
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode text to token IDs"""
        return self.sp.encode(text, out_type=int)
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text"""
        return self.sp.decode(ids)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return self.sp.GetPieceSize()
    
    def get_token_id(self, token: str) -> int:
        """Get ID for a specific token"""
        return self.sp.piece_to_id(token)
    
    def get_token_from_id(self, id: int) -> str:
        """Get token from ID"""
        return self.sp.id_to_piece(id)
    
    def add_lang_token(self, text: str, lang_token: str) -> str:
        """Add language token to text"""
        text = text.strip()
        if text.startswith(("<2vi>", "<2en>")):
            return text
        return f"{lang_token} {text}"

def train_tokenizer(
    corpus_path: str,
    output_prefix: str,
    vocab_size: int = 16000,
    model_type: str = "bpe",
    user_defined_symbols: List[str] = None
):
    if user_defined_symbols is None:
        user_defined_symbols = ["<2en>", "<2vi>"]

    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    spm.SentencePieceTrainer.Train(
        input=corpus_path,
        model_prefix=output_prefix,
        vocab_size=16000,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=["<2en>", "<2vi>"],
    )
    print("DONE TRAIN TOKENIZER")
    print("Model exists:", os.path.exists(output_prefix + ".model"))
    print("Vocab exists:", os.path.exists(output_prefix + ".vocab"))