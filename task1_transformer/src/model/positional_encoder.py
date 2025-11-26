import torch
import torch.nn as nn
import math

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        # Calculate the positional encodings once in log space
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices in the array; 2i
        # Apply cosine to odd indices in the array; 2i+1
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.d_model = d_model

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)

        x = x + self.pe[:, :seq_len]

        return self.dropout(x) # Apply dropout for avoiding overfitting and improving generalization