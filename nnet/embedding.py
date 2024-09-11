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

class RelativeSinusoialPositionalEncoding(nn.Module):
    def __init__(self, config: ModelConfig):
        super(RelativeSinusoialPositionalEncoding, self).__init__()
        self.max_len = config.max_relative_posenc_len
        pos_encoding = T.zeros(2 * self.max_len - 1, config.d_model)

        # Positions [max_len-1, ...., max_len-1]
        pos_left = T.arange(start=self.max_len-1, end=0, step=-1, dtype=T.float)
        pos_right = T.arange(start=0, end=-self.max_len, step=-1, dtype=T.float)
        
        pos = T.cat([pos_left, pos_right], dim=0).unsqueeze(1)

        # Angles
        angles = pos / 10000**(2 * T.arange(0, config.d_model // 2, dtype=T.float).unsqueeze(0) / config.d_model)

        # Rel Sinusoidal PE
        pos_encoding[:, 0::2] = angles.sin()
        pos_encoding[:, 1::2] = angles.cos()

        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding", pos_encoding, persistent=False)
    
    def forward(self, x:T.Tensor) -> T.Tensor:
        bt_size, seq_len, _ = x.shape
        enc = self.pos_encoding[:, self.max_len - seq_len : self.max_len]
        return enc.repeat(bt_size, 1, 1)
        