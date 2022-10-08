import torch

from einops import rearrange
from torch import nn

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)

class PositonalEncoding(nn.Module):
    def __init__(self, seq_len, d_model, dropout=0.2, max_len: int = 5000):
        super().__init__()
        positions = torch.arange(max_len)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.view(1, 0, 2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

  
class AttentionHead(nn.Module):
    def __init__(self, d_model):
        # Assuming we will be passed x of shape (batch, tokens, d_model)
        super().__init__()
        self.to_qkv = nn.Linear(d_model, d_model * 3)
        self.scale_factor = (1 / d_model ** 0.5)

    def forward(self, x, attention_mask):
        qkv = self.to_qkv(x) # b, tokens, dim * 3 (n = 3)
        q, k, v = tuple(rearrange(qkv, 'b t (d n) -> n b t d', n=3))  # Now each of b, tokens, dim
        k = torch.transpose(k, 1, 2)  # b, dim, tokens
        attention_scores = torch.bmm(q, k) * self.scale_factor  # Shape of b, tokens, tokens
        masked_scores = attention_scores.masked_fill(attention_mask[:, None, :] == 0, float('-inf'))
        attention = torch.softmax(masked_scores, dim=-1)  # b, tokens, tokens
        scores = torch.bmm(attention, v) # b, tokens, d_modeln
        return scores # b, tokens, d_model

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.heads = nn.ModuleList([
            AttentionHead(d_model) for _ in range(num_heads)
        ])  # Each shape b, tokens, dim
        self.W_0 = nn.Linear(d_model * num_heads, d_model)  # b, tokens, d_model -> b, tokens, d_model
    
    def forward(self, x, attention_mask):
        # num_heads, b, tokens, dim
        return self.W_0(torch.cat(
            [head(x, attention_mask) for head in self.heads], dim=-1)
        )

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads=8, hidden_dim=2048, dropout_prob=0.1):
        super().__init__()
        self.MHA = MultiHeadAttention(d_model, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.layer_norm_attention = nn.LayerNorm([d_model])
        self.layer_norm_ff = nn.LayerNorm([d_model])
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, attention_mask):
        mha = self.MHA(x, attention_mask)
        z = self.layer_norm_attention(mha)
        x = x + self.dropout(z)
        z = self.ff(x)
        x = x + self.dropout(self.layer_norm_ff(x))
        return x