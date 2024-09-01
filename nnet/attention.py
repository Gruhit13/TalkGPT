import torch as T
from torch import nn
import torch.nn.functional as F
from .config import ModelConfig
from typing import Optional
import numpy as np
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super(MultiHeadAttention, self).__init__()

        assert config.d_model % config.n_heads == 0, "Model-Dim must be divisible by n_heads"

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = self.d_model // self.n_heads
        self.scale = math.sqrt(self.head_dim)

        self.projection = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.att_dropout = nn.Dropout(config.attn_dropout)

        self.output_proj = nn.Linear(self.d_model, self.d_model)
        self.context_dropout = nn.Dropout(config.attn_dropout)
    
    def forward(self, x: T.Tensor, mask: Optional[T.Tensor] = None) -> T.Tensor:
        """
        Arugments:
            x: Input tensor of shape [B, T, D]
            mask: input mask of shape [B, 1, T, T]
        """

        bt_size, seq_len, _ = x.shape
        
        x = self.projection(x)
        q, k, v = x.chunk(3, dim=-1)
        
        q = q.view(bt_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(bt_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(bt_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(2, 3))/self.scale

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        
        attn = self.att_dropout(F.softmax(attn, dim=-1))

        context = attn @ v
        context = context.transpose(1, 2)
        context = context.contiguous().view(bt_size, seq_len, -1)
        context = self.context_dropout(self.output_proj(context))

        return (context, attn)