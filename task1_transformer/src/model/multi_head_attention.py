import torch
import torch.nn as nn
import torch.nn.functional as F 
from .scaled_dot_product import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        assert d_model % heads == 0, "d_model must be divisible by heads"

        self.d_model = d_model
        self.d_k = d_model // heads
        self.heads = heads
        self.attention = None
        self.scaled_dot_product = ScaledDotProductAttention(dropout)

        self.wQ = nn.Linear(d_model, d_model)
        self.wK = nn.Linear(d_model, d_model)
        self.wV = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask= None):
        '''
        q: [batch_size, seq_len, d_model]
        k: [batch_size, seq_len, d_model]
        v: [batch_size, seq_len, d_model]
        output: [batch_size, seq_len, d_model]
        '''

        batch_size = Q.size(0)

        Q = self.wQ(Q).view(batch_size, -1, self.heads, self.d_k) # [batch_size, seq_len, heads, d_k]
        K = self.wK(K).view(batch_size, -1, self.heads, self.d_k) # [batch_size, seq_len, heads, d_k]
        V = self.wV(V).view(batch_size, -1, self.heads, self.d_k) # [batch_size, seq_len, heads, d_k]

        Q = Q.transpose(1, 2)  # [batch_size, heads, seq_len, d_k]
        K = K.transpose(1, 2)  # [batch_size, heads, seq_len, d_k]
        V = V.transpose(1, 2)  # [batch_size, heads, seq_len, d_k]

        output, attn_weights = self.scaled_dot_product(Q, K, V, attn_mask=mask)  # output: [batch_size, heads, seq_len, d_k]

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # [batch_size, seq_len, d_model]

        output = self.out(output)  # [batch_size, seq_len, d_model]

        return output 

def main():
    mha = MultiHeadAttention(heads=8, d_model=512, dropout=0.1)
    Q = torch.randn(32, 30, 512)
    K = torch.randn(32, 30, 512)
    V = torch.randn(32, 30, 512)
    output = mha(Q, K, V)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()