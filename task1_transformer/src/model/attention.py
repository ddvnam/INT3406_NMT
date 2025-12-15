import torch
import torch.nn as nn
import torch.nn.functional as F 
import math 
from typing import Optional

class RoPE(nn.Module):
    def __init__(self, d_model: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)
        self._cos = None
        self._sin = None
        self._seq_len_cached = 0

    def _maybe_update_cache(self, seq_len: int, device, dtype):
        if seq_len > self._seq_len_cached or self._cos is None or self._cos.device != device:
            self._seq_len_cached = seq_len
            positions = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(positions, self.inv_freq.to(device))
            self._cos = freqs.cos()
            self._sin = freqs.sin()

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None):
        seq_len = seq_len or x.size(-2)
        self._maybe_update_cache(seq_len, x.device, x.dtype)
        return self._cos[:seq_len], self._sin[:seq_len]
        
def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    return torch.stack([rot1, rot2], dim=-1).flatten(-2)

class GroupedQueryAttentionRoPE(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float, rope_base: float):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        self.rope = RoPE(self.d_k, base=rope_base)

    def forward(self, q, k, v, key_padding_mask=None, attn_mask=None):
        B, T_q = q.size(0), q.size(1)
        T_k = k.size(1)
        Q = self.W_q(q).view(B, T_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(B, T_k, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(B, T_k, self.n_kv_heads, self.d_k).transpose(1, 2)
        cos_q, sin_q = self.rope(Q, T_q)
        cos_k, sin_k = self.rope(K, T_k)
        Q = apply_rope(Q, cos_q, sin_q)
        K = apply_rope(K, cos_k, sin_k)
        K = K.repeat_interleave(self.n_groups, dim=1)
        V = V.repeat_interleave(self.n_groups, dim=1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T_q, -1)
        return self.W_o(out)
