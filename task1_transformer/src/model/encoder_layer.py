import torch
import torch.nn as nn
from .norm import Norm
from .multi_head_attention import MultiHeadAttention
from .feed_forward_network import FeedForwardNetwork

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)
        self.attenion = MultiHeadAttention(heads, d_model, dropout)
        self.ff = FeedForwardNetwork(d_model, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        '''
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            mask: Optional mask tensor
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        '''
        # Multi-head attention sub-layer
        x2 = self.norm1(x)

        x = x + self.dropout1(self.attenion(x2, x2, x2, mask=mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.ff(x2))
        return x
    
def main():
    encoder_layer = EncoderLayer(d_model=512, heads=8, dropout=0.1)
    x = torch.randn(32, 30, 512)  # [batch_size, seq_len, d_model]
    output = encoder_layer(x)
    print("Output shape:", output.shape)
    
if __name__ == "__main__":
    main() # check whether the shape of output equals to input shape