import torch
import torch.nn as nn

class SimplifiedSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        attn_scores = x @ x.transpose(-2, -1)
        attn_weights = torch.softmax(attn_scores / x.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ x
        return context_vec

if __name__ == "__main__":
    torch.manual_seed(123)
    inputs = torch.rand(8, 4, 256)
    
    attention = SimplifiedSelfAttention()
    output = attention(inputs)
    
    print(output.shape)
