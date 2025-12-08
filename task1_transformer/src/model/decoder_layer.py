import torch
import torch.nn as nn
from .norm import Norm
from .multi_head_attention import MultiHeadAttention
from .feed_forward_network import FeedForwardNetwork

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attention_1 = MultiHeadAttention(heads, d_model, dropout)
        self.attention_2 = MultiHeadAttention(heads, d_model, dropout)

        self.ff = FeedForwardNetwork(d_model, dropout=dropout)
    
    def forward(self, x, encoder_output, src_mask, trg_mask):
        '''
            Args:
            x: Input tensor of shape [batch_size, trg_seq_len, d_model]
            encoder_output: Encoder output tensor of shape [batch_size, src_seq_len, d_model]
            src_mask: Source mask tensor [batch_size, 1, src_seq_len]
            trg_mask: Target mask tensor [batch_size, 1, trg_seq_len]
        '''
        # Masked multi-head attention sub-layer
        x2 = self.norm_1(x)
        # First attention sub-layer (self-attention) for target sequence
        x = x + self.dropout_1(self.attention_1(x2, x2, x2, mask=trg_mask))
        x2 = self.norm_2(x)
        # Second attention sub-layer (encoder-decoder attention)
        x = x + self.dropout_2(self.attention_2(x2, encoder_output, encoder_output, mask=src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x

def main():
    # check the shape of output equals to input shape
    decoder_layer = DecoderLayer(d_model=512, heads=8, dropout=0.1)
    x = torch.randn(32, 30, 512)  # [batch_size, trg_seq_len, d_model]
    encoder_output = torch.randn(32, 40, 512)  # [batch_size, src_seq_len, d_model]
    src_mask = torch.randn(32, 1, 40)  # [batch_size, 1, src_seq_len]
    trg_mask = torch.randn(32, 1, 30)  # [batch_size, 1, trg_seq_len]
    decoder_output = decoder_layer(x, encoder_output, src_mask, trg_mask)
    print("Decoder output shape:", decoder_output.shape)
    total_params = sum(p.numel() for p in decoder_layer.parameters())
    trainable_params = sum(p.numel() for p in decoder_layer.parameters() if p.requires_grad)

    print("Total params:", total_params)
    print("Trainable params:", trainable_params)

if __name__ == "__main__":
    main()