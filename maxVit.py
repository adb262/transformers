import torch

from torch import nn
from modelUtils import MultiHeadAttention
from einops.layers.torch import Rearrange, Reduce

class SqueezeExcitationBlock(nn.Module):
    def __init__(self, d_model, ratio=16):
        super().__init__()
        self.se = nn.Sequential(
            Reduce('b h w c -> b c', 'mean'),
            nn.Linear(d_model, d_model/ratio),
            nn.ReLU(),
            nn.Linear(d_model/ratio, d_model),
            nn.Sigmoid(dim=-1),
            Rearrange('b c -> b c 1 1'),
        )

    def forward(self, x):
        return x * self.se(x)  # We multiply as this is ultimately a scaling factor


class MbConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, d_model):
        super().__init__()
        self.mbconv = nn.Sequential(
            nn.Conv2d(in_dim, d_model, 1),
            nn.BatchNorm2d(d_model),
            nn.GeLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3),
            nn.BatchNorm2d(d_model),
            nn.GeLU(),
            SqueezeExcitationBlock(d_model),
            nn.Conv2d(d_model, out_dim, 1)
        )
    
    def forward(self, x):
        return self.mbconv(x)