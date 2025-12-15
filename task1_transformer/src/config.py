from dataclasses import dataclass
import torch
import os

@dataclass
class ModelConfig:
    # Path
    train_src_file: str  
    train_tgt_file: str
    spm_prefix: str
    save_dir: str = "task1_transformer/src/out"

    # Data
    vocab_size: int = 16000
    max_len: int = 128

    # Model
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 4
    num_layers: int = 3
    num_layers: int = 3
    dropout: float = 0.1
    d_ff: int = 2048
    rope_base: float = 10000.0

    # Training
    batch_size: int = 1
    num_epochs: int = 1
    lr_base: float = 2e-4
    warmup_steps: int = 200 
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    
    # Special Tokens
    pad_token: str = "<pad>"
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    unk_token: str = "<unk>"
    en_token: str = "<2en>"
    vi_token: str = "<2vi>"

    # Bidirectional 
    # --- BIDIRECTIONAL ---
    vi2en_epoch_ratio: float = 0.7
    span_mask_prob: float = 0.0

    # Device
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers: int = 1
    save_every: int = 1
    seed: int = 42

def set_seed(seed: int):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



