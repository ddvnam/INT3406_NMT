import torch
import torch.nn as nn
from scaled_dot_product import ScaledDotProductAttention
import unittest

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout= 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = d_model // num_heads

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias = False)
        self.W_k = nn.Linear(d_model, d_model, bias = False)
        self.W_v = nn.Linear(d_model, d_model, bias = False)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)
    
    def forward(self, Q, K, V, mask=None):
        '''
        Compute multi-head attention.
        
        Args:
            Q: Queries matrix [batch_size, seq_len_q, d_model]
            K: Keys matrix [batch_size, seq_len_k, d_model]
            V: Values matrix [batch_size, seq_len_v, d_model]
            mask: Optional mask tensor [batch_size, seq_len_q, seq_len_k]
        
        Returns:
            output: Multi-head attention output [batch_size, seq_len_q, d_model]
        '''
        batch_size, seq_len_q = Q.size(0), Q.size(1)


        # Linear projections and head splitting
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.dk).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.dk).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.dk).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1) # add head dimension
        
        output, attention_weights = self.attention(Q, K, V, mask)

        # Concat and apply final linear layer
        output = output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        output = self.W_o(output)

        return output, attention_weights
        
class MultiheadAttentionTest(unittest.TestCase):
    def test_multi_head_attention(self):
        d_model = 512
        num_heads = 8
        batch_size = 2
        seq_len_q = 10

        # create model
        mha = MultiHeadAttention(d_model, num_heads)
        mha.eval()

        # Sample input
        Q = torch.randn(batch_size, seq_len_q, d_model)
        K = torch.randn(batch_size, seq_len_q, d_model)
        V = torch.randn(batch_size, seq_len_q, d_model)

        output, attention_weights = mha(Q, K, V)

        self.assertEqual(output.shape, (batch_size, seq_len_q, d_model))
        self.assertEqual(attention_weights.shape, (batch_size, num_heads, seq_len_q, seq_len_q))

        # Check attention with input which has different seq lengths
        Q_new = torch.randn(batch_size, 15, d_model)
        K_new = torch.randn(batch_size, 20, d_model)
        V_new = torch.randn(batch_size, 20, d_model)

        print("Testing with different sequence lengths...")
        output_new, attention_weights_new = mha(Q_new, K_new, V_new)

        self.assertEqual(output_new.shape, (batch_size, 15, d_model))
        self.assertEqual(attention_weights_new.shape, (batch_size, num_heads, 15, 20))

if __name__ == "__main__":
    unittest.main()