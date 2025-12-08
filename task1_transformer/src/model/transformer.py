import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, N, heads, dropout= 0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab_size, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab_size)
    
    def forward(self, src, trg, src_mask, trg_mask):
        '''
            Args:
            src: Source input tensor of shape [batch_size, src_seq_len]
            trg: Target input tensor of shape [batch_size, trg_seq_len]
            src_mask: Source mask tensor [batch_size, 1, src_seq_len]
            trg_mask: Target mask tensor [batch_size, 1, trg_seq_len]

            Returns:
            Output tensor of shape [batch_size, trg_seq_len, trg_vocab_size]
        '''
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(trg, encoder_output, src_mask, trg_mask)
        output = self.out(decoder_output)
        return output

def main():
    # check the shape of output equals to trg input shape with vocab size
    transformer = Transformer(src_vocab_size=10000, trg_vocab_size=232, d_model=512, N=6, heads=8, dropout=0.1)
    src = torch.randint(0, 10000, (32, 40))  # [batch_size, src_seq_len]
    trg = torch.randint(0, 232, (32, 30))  # [batch_size, trg_seq_len]
    src_mask = torch.randn(32, 1, 40)  # [batch_size, 1, src_seq_len]
    trg_mask = torch.randn(32, 1, 30)  # [batch_size, 1, trg_seq_len]
    output = transformer(src, trg, src_mask, trg_mask)
    print("Transformer output shape:", output.shape)
    total_params = sum(p.numel() for p in transformer.parameters())
    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)

    print("Total params:", total_params)
    print("Trainable params:", trainable_params)
    
if __name__ == "__main__":
    main()