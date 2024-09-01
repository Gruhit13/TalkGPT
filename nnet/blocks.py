import torch as T
from torch import nn
import torch.nn.functional as F
from .config import ModelConfig
from .attention import MultiHeadAttention
from typing import Optional

class FeedForwardNetwork(nn.Module):
    def __init__(self, config: ModelConfig):
        super(FeedForwardNetwork, self).__init__()

        self.linear1 = nn.Linear(config.d_model, config.hidden_dim)
        self.linear2 = nn.Linear(config.hidden_dim, config.d_model)
        self.dropout = nn.Dropout(config.ffn_dropout)
    
    def forward(self, x: T.Tensor) -> T.Tensor:
        x = F.gelu(self.linear1(x))
        x = self.dropout(self.linear2(x))

        return x


class DecoderBlock(nn.Module):
    def __init__(self, config : ModelConfig):
        super(DecoderBlock, self).__init__()
        
        self.mha = MultiHeadAttention(config)
        self.ffn = FeedForwardNetwork(config)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)

    def forward(self, x: T.Tensor, mask: Optional[T.Tensor] = None) -> T.Tensor:
        (context, _ ) = self.mha(x, mask)
        x = self.ln1(x + context)

        ffn_oup = self.ffn(x)
        x = self.ln2(x + ffn_oup)
        return x

class PostMelSpecBlock(nn.Module):
    def __init__(self, config : ModelConfig):
        super(PostMelSpecBlock, self).__init__()

        self.first_layer = nn.Conv1d(
            in_channels=config.n_mels,
            out_channels=config.postmel_hidden_dim,
            kernel_size=config.kernel_size,
            padding=config.padding,
            stride=config.stride
        )

        self.intermediate_layer = nn.ModuleList([
            nn.Conv1d(
                in_channels=config.postmel_hidden_dim,
                out_channels=config.postmel_hidden_dim,
                kernel_size=config.kernel_size,
                padding=config.padding,
                stride=config.stride
            )   for _ in range(config.n_postmelblock-2)
        ])

        self.last_layer = nn.Conv1d(
            in_channels=config.postmel_hidden_dim,
            out_channels=config.n_mels,
            kernel_size=config.kernel_size,
            padding=config.padding,
            stride=config.stride
        )

        self.drop_prob = config.postmel_dropout
    
    def forward(self, x: T.Tensor) -> T.Tensor:
        """
        x = [B, NMEL_DIM, T] 

        Over here instead of using a dropout layer, pytorch's Functional is used to 
        provide diversity in respose at the time of inference. This type of implementation
        is from a research paper, don't ask which one because I do not remember as read a 
        lot in past few days.
        """
        # [B, T, NMEL_DIM] => [B, NMEL_DIM, T]
        x = x.transpose(1, 2)
        x = F.dropout(T.tanh(self.first_layer(x)), p=self.drop_prob)

        for layer in self.intermediate_layer:
            x = F.dropout(T.tanh(layer(x)), p=self.drop_prob)
        
        x = F.dropout(T.tanh(self.last_layer(x)), p=self.drop_prob)

        # [B, NMEL_DIM, T] => [B, T, NMEL_DIM]
        return x.transpose(1, 2)