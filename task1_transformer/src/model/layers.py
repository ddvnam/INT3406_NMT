import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import GroupedQueryAttentionRoPE
from .norm import RMSNorm

class FFN_SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.dff = d_ff
        self.linear1 = nn.Linear(d_model, 2*d_ff, bias = False)
        self.linear2 = nn.Linear(d_ff, d_model, bias = False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear1(x)
        g, v = h[..., : self.dff], h[..., self.dff :]
        s = g * torch.sigmoid(g)
        out = self.linear2(s * v)
        out = self.dropout(out)
        return out
    
class EncoderLayer(nn.Module):
    """Transformer Encoder Layer"""
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.self_attn = GroupedQueryAttentionRoPE(
            config.d_model, config.n_heads, config.n_kv_heads, 
            config.dropout, config.rope_base
        )
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = FFN_SwiGLU(config.d_model, config.d_ff, config.dropout)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, src_pad_mask=None):
        # Self-attention with residual
        x1 = self.norm1(x)
        attn = self.self_attn(x1, x1, x1, key_padding_mask=src_pad_mask)
        x = x + self.dropout(attn)
        
        # FFN with residual
        x2 = self.norm2(x)
        return x + self.ffn(x2)

class DecoderLayer(nn.Module):
    """Transformer Decoder Layer"""
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.self_attn = GroupedQueryAttentionRoPE(
            config.d_model, config.n_heads, config.n_kv_heads, 
            config.dropout, config.rope_base
        )
        self.norm2 = RMSNorm(config.d_model)
        self.cross_attn = GroupedQueryAttentionRoPE(
            config.d_model, config.n_heads, config.n_kv_heads, 
            config.dropout, config.rope_base
        )
        self.norm3 = RMSNorm(config.d_model)
        self.ffn = FFN_SwiGLU(config.d_model, config.d_ff, config.dropout)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, y, enc_output, tgt_pad_mask=None, tgt_causal_mask=None, src_pad_mask=None):
        # Self-attention with residual
        y1 = self.norm1(y)
        attn1 = self.self_attn(y1, y1, y1, key_padding_mask=tgt_pad_mask)
        y = y + self.dropout(attn1)
        
        # Cross-attention with residual
        y2 = self.norm2(y)
        attn2 = self.cross_attn(y2, enc_output, enc_output, key_padding_mask=src_pad_mask)
        y = y + self.dropout(attn2)
        
        # FFN with residual
        y3 = self.norm3(y)
        return y + self.ffn(y3)