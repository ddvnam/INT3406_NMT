import torch
import torch.nn as nn
import os 
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from shared.utils.transformer_utils import get_clones
from .embeddings import Embedder, PositionalEncoder
from .decoder_layer import DecoderLayer
from .norm import Norm

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout=0.1):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    
    def forward(self, trg, encoder_output, src_mask, trg_mask):
        '''
            Args:
            trg: Input tensor of shape [batch_size, trg_seq_len]
            encoder_output: Encoder output tensor of shape [batch_size, src_seq_len, d_model]
            src_mask: Source mask tensor [batch_size, 1, src_seq_len]
            trg_mask: Target mask tensor [batch_size, 1, trg_seq_len]

            Returns:
            Output tensor of shape [batch_size, trg_seq_len, d_model]
        '''
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, encoder_output, src_mask, trg_mask)
        x = self.norm(x)
        return x

def main():
    # check the shape of output equals to input shape
    decoder = Decoder(vocab_size=232, d_model=512, N=6, heads=8, dropout=0.1)
    trg = torch.randint(0, 232, (32, 30))  # [batch_size, trg_seq_len]
    encoder_output = torch.randn(32, 40, 512)  # [batch_size, src_seq_len, d_model]
    src_mask = torch.randn(32, 1, 40)  # [batch_size, 1, src_seq_len]
    trg_mask = torch.randn(32, 1, 30)  # [batch_size, 1, trg_seq_len]
    decoder_output = decoder(trg, encoder_output, src_mask, trg_mask)
    print("Decoder output shape:", decoder_output.shape)
    total_params = sum(p.numel() for p in decoder.parameters())
    trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)

    print("Total params:", total_params)
    print("Trainable params:", trainable_params)

if __name__ == "__main__":
    main()
