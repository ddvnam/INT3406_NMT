import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
import unittest

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout= 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, attn_mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)  # [B,1,Lq,Lk] -> broadcast over heads
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        
        value_weights = F.softmax(scores, dim=-1)

        if self.dropout is not None:
            value_weights = self.dropout(value_weights)
        
        output = torch.matmul(value_weights, V)
        return output, value_weights
    
def main():
    attention = ScaledDotProductAttention(dropout=0.0)
    Q = torch.randn(32, 8, 30, 512)
    K = torch.randn(32, 8, 30, 512)
    V = torch.randn(32, 8, 30, 512)
    output, attn_weights = attention(Q, K, V)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()