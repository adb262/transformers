import torch

from .modelUtils import EncoderBlock, Embedder


class BOWBERT(nn.Module):
    """Implementation of BERT without positional encoding (BOW BERT)."""

    def __init__(self, vocab_size, n_encoders, d_model, n_heads, out_size, pad_token_idx):
        super().__init__()
        self.pad_token_idx = pad_token_idx
        self.embedder = Embedder(vocab_size, d_model)
        self.inference_ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, out_size),
            nn.Softmax(dim=-1),
        )
        # Use nn.Sequential since there is forward() demand
        modules = [EncoderBlock(d_model, n_heads) for _ in range(n_encoders)]
        #self.encoders = nn.Sequential(*modules)
        self.encoders = EncoderBlock(d_model, n_heads)

    def forward(self, x):
        attention_mask = torch.where(x == self.pad_token_idx, 0, 1)
        # x should be of shape b, tokens, d_model
        x = self.inference_ff(self.encoders(self.embedder(x), attention_mask)[:, 0])
        # x now in shape b, out_size
        return x
      
    def predict(self, x):
        x = self.inference_ff(self.encoders(self.embedder(x)))
        return torch.argmax(x, dim=-1)