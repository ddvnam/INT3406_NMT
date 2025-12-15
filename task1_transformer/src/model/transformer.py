import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .layers import EncoderLayer, DecoderLayer  
from .norm import RMSNorm

class Transformer(nn.Module):
    def __init__(self, config, vocab_size: int):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        # Encoder
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.num_layers)]
        )
        self.encoder_final_norm = RMSNorm(config.d_model)

        # Decoder

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_layers)]    
        )

        self.decoder_final_norm = RMSNorm(config.d_model)

        # output
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.emb_scale = math.sqrt(config.d_model)

        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    
    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        # Masks
        src_pad = (src_ids == 0)
        tgt_pad = (tgt_ids == 0)
        tgt_in = tgt_ids[:, :-1]
        tgt_pad_in = tgt_pad[:, :-1]
        
        # Causal mask for decoder
        T = tgt_in.size(1)
        tgt_causal = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=src_ids.device), 
            diagonal=1
        )
        
        # Encoder
        src_emb = self.emb_dropout(self.embedding(src_ids) * self.emb_scale)
        enc_out = src_emb
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_pad)
        enc_out = self.encoder_final_norm(enc_out)
        
        # Decoder
        tgt_emb = self.emb_dropout(self.embedding(tgt_in) * self.emb_scale)
        dec_out = tgt_emb
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, tgt_pad_in, tgt_causal, src_pad)
        dec_out = self.decoder_final_norm(dec_out)
        
        # Output projection
        return F.linear(dec_out, self.embedding.weight, self.output_bias)
    
    