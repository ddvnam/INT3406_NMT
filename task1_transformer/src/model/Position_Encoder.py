import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional
import math
import logging

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_length=200, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self._max_seq_length = max_seq_length
        
        pe = torch.zeros(max_seq_length, d_model)
        
        for pos in range(max_seq_length):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos/(10000**(2*i/d_model)))
                pe[pos, i+1] = math.cos(pos/(10000**((2*i+1)/d_model)))
        pe = pe.unsqueeze(0)        
        self.register_buffer('pe', pe)

        @torch.jit.script
        def splice_by_size(source, target):
            length = target.size(1)
            return source[:, :length]

        self.splice_by_size = splice_by_size
    
    def forward(self, x):
        if(x.shape[1] > self._max_seq_length):
            logging.warn("Input longer than maximum supported length for PE detected. Build a model with a larger input_max_length limit if you want to keep the input; or ignore if you want the input trimmed")
            x = x[:, 0:self._max_seq_length]
        
        x = x * math.sqrt(self.d_model)
        
        spliced_pe = self.splice_by_size(self.pe, x) 
        pe = spliced_pe.requires_grad_(False)
        
        x = x + pe
        x = self.dropout(x)
        
        return x
