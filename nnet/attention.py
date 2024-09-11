import torch as T
from torch import nn
import torch.nn.functional as F
from .config import ModelConfig
from .embedding import RelativeSinusoialPositionalEncoding
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

class RelativePosMultiheadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super(RelativePosMultiheadAttention, self).__init__()

        assert config.d_model % config.n_heads == 0, "Model-Dim must be divisible by n_heads"

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = self.d_model // self.n_heads
        self.scale = math.sqrt(self.head_dim)

        self.pos_enc = RelativeSinusoialPositionalEncoding(config)

        self.projection = nn.Linear(self.d_model, self.d_model * 3, bias=False)
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

        # Get the relative sinusoidal position encoding
        E = self.pos_enc(q)

        # [B, S, D] => [B, S, N_H, H_D] => [B, N_H, S, H_D]
        q = q.view(bt_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(bt_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(bt_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        E = E.view(bt_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention score calculation
        att_score_K = q @ k.transpose(2, 3)
        att_score_E = self.relative_to_absoulute(q @ E.transpose(2, 3))
        score = (att_score_K + att_score_E) / self.scale

        if mask is not None:
            score = score.masked_fill(mask, -np.inf)
        
        score = self.att_dropout(F.softmax(score, dim=-1))

        context = score @ v
        context = context.transpose(1, 2)
        context = context.contiguous().view(bt_size, seq_len, -1)
        
        return (self.output_proj(context), score)
    
    def relative_to_absoulute(self, x: T.Tensor) -> T.Tensor:
        
        bt_size, num_heads, seq_len1, seq_len2 = x.size()

        x = F.pad(x, pad=(1, 0), value=0)

        x = x.reshape(bt_size, num_heads, -1)

        x = F.pad(x, pad=(seq_len2 - seq_len1, 0), value=0)

        x = x.reshape(bt_size, num_heads, 1 + seq_len1, seq_len2)

        return x[:, :, 1:]
