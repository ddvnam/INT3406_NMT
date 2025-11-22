import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
import unittest

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout= 0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask= None):
        '''
        Compute the scaled dot-product attention.
        
        Args:
            Q: Queries matrix [batch_size, seq_len_q, d_k]
            K: Keys matrix [batch_size, seq_len_k, d_k]
            V: Values matrix [batch_size, seq_len_v, d_v]
            mask: Optional mask tensor [batch_size, seq_len_q, seq_len_k]
        
        Returns:
            output: Attention output [batch_size, seq_len_q, d_v]
            attention: Attention weights [batch_size, seq_len_q, seq_len_k]
        '''
        dk = Q.size(-1)

        # Compute the scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dk)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Compute the output
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
class TestScaledDotProductAttention(unittest.TestCase):
    def setUp(self):
        self.attention = ScaledDotProductAttention(dropout=0.0)
        self.attention.eval()

    def test_shapes(self):
        batch_size = 2
        seq_len_q = 4
        seq_len_k = 6
        d_k = 8
        d_v = 10

        Q = torch.randn(batch_size, seq_len_q, d_k)
        K = torch.randn(batch_size, seq_len_k, d_k)
        V = torch.randn(batch_size, seq_len_k, d_v)

        output, attention_weights = self.attention(Q, K, V)

        self.assertEqual(output.shape, (batch_size, seq_len_q, d_v))
        self.assertEqual(attention_weights.shape, (batch_size, seq_len_q, seq_len_k))
        
    def test_mask_logic(self):
        # Test xem mask có che được thông tin đúng không
        Q = torch.randn(1, 1, 8)
        K = torch.randn(1, 2, 8)
        V = torch.randn(1, 2, 8)

        mask = torch.tensor([[[1, 0]]])  # Chỉ cho phép truy cập key đầu tiên

        output, attention_weights = self.attention(Q, K, V, mask=mask)

        # Kiểm tra xem trọng số attention cho key thứ hai có phải là 0 không
        self.assertAlmostEqual(attention_weights[0, 0, 1].item(), 0.0, places=5)

        # Trọng số của các vị trị khác phải là 1
        self.assertAlmostEqual(attention_weights[0, 0, 0].item(), 1.0, places=5)
        
if __name__ == '__main__':
    unittest.main()