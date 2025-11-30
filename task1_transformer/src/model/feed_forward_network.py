import torch
import torch.nn as nn
import torch.nn.functional as F 

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff = 2048, dropout= 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        '''
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
        '''
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    