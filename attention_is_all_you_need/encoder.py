import numpy as np
import torch
from einops import rearrange
from torch import nn

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)
  
class AttentionHead(nn.Module):
    def __init__(self, d_model, dim):
        # Assuming we will be passed x of shape (batch, tokens, dim)
        super().__init__()
        self.to_qkv = nn.Linear(d_model, dim * 3)
        self.scale_factor = dim ** -0.5

    def forward(self, x):
        qkv = self.to_qkv(x) # b, tokens, dim * 3 (n = 3)
        q, k, v = tuple(rearrange('b t (d n) -> n b t d', n=3))  # Now each of b, tokens, dim
        k = torch.transpose(k, -1, -2)  # b, dim, tokens
        attention = torch.softmax(torch.bmm(q, k) * self.scale_factor, dim=-1)  # b, tokens, tokens
        return torch.bmm(attention, v)  # b, tokens, dim

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, dim, num_heads=8):
        super().__init__()
        self.heads = nn.ModuleList([
            AttentionHead(d_model, dim) for _ in range(num_heads)
        ])  # Each shape b, tokens, dim
        self.W_0 = nn.Linear(dim * num_heads, d_model)  # b, tokens, d_model -> b, tokens, d_model
    
    def forward(self, x):
        # num_heads, b, tokens, dim
        return self.W_0(torch.cat(
            [head(x) for head in self.heads], dim=-1)
        )

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads=8, hidden_dim=2048):
        super().__init__()
        self.MHA = MultiHeadAttention(dim, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.layer_norm_attention = torch.LayerNorm([d_model])
        self.layer_norm_ff = torch.LayerNorm([d_model])

    def forward(self, x):
        x = x + self.layer_norm_attention(self.MHA(x))
        x = x + self.layer_norm_ff(self.ff(x))
        return x

class EncoderStack(nn.Module):
    def __init__(self, n_layers, d_model, n_heads):
        super().__init__()
        # Use nn.Sequential since there is forward() demand
        modules = [EncoderBlock(d_model, n_heads) for _ in n_layers]
        self.encoders = nn.Sequential(*modules)

    def forward(self, x):
        # x should be of shape b, tokens, d_model
        return self.encoders(x)
