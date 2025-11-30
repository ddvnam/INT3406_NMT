import torch
import torch.nn as nn

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
            / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def main():
    # check the shape of output equals to input shape
    norm = Norm(d_model=512)
    x = torch.randn(32, 30, 512)  # [batch_size, seq_len, d_model]
    normed_x = norm(x)
    print("Norm output shape:", normed_x.shape)
    
    print("Input before normalize:", x[0,0,:5])
    print("Input after normalize:", normed_x[0,0,:5])

    total_params = sum(p.numel() for p in norm.parameters())
    trainable_params = sum(p.numel() for p in norm.parameters() if p.requires_grad)

    print("Total params:", total_params)
    print("Trainable params:", trainable_params)

if __name__ == "__main__":
    main()