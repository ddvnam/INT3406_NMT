import os
import sys
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from embeddings import Embedder, PositionalEncoder
from encoder_layer import EncoderLayer
from norm import Norm
from shared.utils.transformer_utils import get_clones

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout= 0.1):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src, src_mask):
        '''
            Args:
            src: Input tensor of shape [batch_size, src_seq_len]
            src_mask: Source mask tensor [batch_size, 1, src_seq_len]

            Returns:
            Output tensor of shape [batch_size, src_seq_len, d_model]
        '''
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, src_mask)
        x = self.norm(x)
        return x

def main():
    # check the shape of output equals to input shape
    encoder = Encoder(vocab_size=10000, d_model=512, N=6, heads=8, dropout=0.1)
    src = torch.randint(0, 10000, (32, 40))  # [batch_size, src_seq_len]
    src_mask = torch.randn(32, 1, 40)  # [batch_size, 1, src_seq_len]
    encoder_output = encoder(src, src_mask)
    print("Encoder output shape:", encoder_output.shape)
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)

    print("Total params:", total_params)
    print("Trainable params:", trainable_params)

if __name__ == "__main__":
    main()