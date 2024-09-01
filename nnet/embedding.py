import torch as T
from torch import nn
from .config import ModelConfig

class SinusoidalEncoding(nn.Module):
    def __init__(self, config: ModelConfig):
        super(SinusoidalEncoding, self).__init__()
        
        pos_encoding = T.zeros(config.max_seq_len, config.d_model)
        pos = T.arange(0, config.max_seq_len, dtype=T.float).unsqueeze(1)
        i = T.arange(0, config.d_model // 2, dtype=T.float).unsqueeze(0)
        angles = pos / 10000 ** (2*i / config.d_model)

        pos_encoding[:, 0::2] = angles.sin()
        pos_encoding[:, 1::2] = angles.cos()
        pos_encoding = pos_encoding.unsqueeze(0)

        self.register_buffer("pos_encoding", pos_encoding, persistent=False)
    
    def forward(self, x : T.Tensor):
        # Input x is of shape [B, T, D_MODEL]
        T = x.size(1)
        x = x + self.pos_encoding[:, :T]
        return x