import torch
import torch.nn as nn
import math

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embedder, self).__init__()
        self.vocab_size = vocab_size   
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        '''
        Args:
            x: Input tensor of shape [batch_size, seq_len]
        Returns:
            Embedded tensor of shape [batch_size, seq_len, d_model]
        '''
        return self.embedding(x)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

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
    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)

        x = x + self.pe[:, :seq_len]

        return self.dropout(x) # Apply dropout for avoiding overfitting and improving generalization
    

def main():
    # check the shape of output equals to input shape
    embedder = Embedder(vocab_size=232, d_model=512)
    x = torch.randint(0, 232, (32, 30))  # [batch_size, seq_len]
    embedded_x = embedder(x)
    print("Embedder output shape:", embedded_x.shape)

    # check positional encoder
    pe = PositionalEncoder(d_model=512, max_len=200, dropout=0.1)
    x_pe = pe(embedded_x)
    print("Positional Encoder output shape:", x_pe.shape)

if __name__ == "__main__":
    main()