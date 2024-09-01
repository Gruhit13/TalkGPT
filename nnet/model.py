import torch as T
from torch import nn
from .config import ModelConfig
from .blocks import DecoderBlock, PostMelSpecBlock
from .embedding import SinusoidalEncoding
from typing import Optional

class TalkGPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super(TalkGPT, self).__init__()

        self.mel_embedding = nn.Linear(config.n_mels, config.d_model, bias=False)
        self.pos_encoding = SinusoidalEncoding(config)

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(config) for _ in range(config.n_blocks)
        ])

        self.melspec_linear = nn.Linear(config.d_model, config.n_mels)
        self.melspec_post_process = PostMelSpecBlock(config)

        # Will use the output as 1 to control the stopping frame sensistivity
        self.stop_token_linear = nn.Linear(config.d_model, 1, bias=False)
    
    def forward(self, x: T.Tensor, mask: Optional[T.Tensor] = None) -> T.Tensor:
        x = self.mel_embedding(x)
        x = self.pos_encoding(x)

        for block in self.decoder_blocks:
            x = block(x, mask)
        
        mel_spec = self.melspec_linear(x)
        mel_spec = mel_spec + self.melspec_post_process(mel_spec)

        stop_label = self.stop_token_linear(x)

        return mel_spec, stop_label
    
    def get_mask(self, x, mask: Optional[T.Tensor] = None) -> T.Tensor:
        seq_len = x.size(1)
        causal_mask = T.tril(T.ones(1, seq_len, seq_len)).to(x.device)

        if mask is not None:
            assert len(mask.shape) == 2, "Mask must be of shape [B, Seq_len]"
            # [B, seq_len] => [B, 1, seq_len, 1]
            mask = mask.unsqueeze(1).unsqueeze(3)
            causal_mask = mask * causal_mask

            causal_mask = (1 - causal_mask).bool()
            return causal_mask
        else:
            causal_mask = causal_mask.repeat(x.size(0), 1, 1)
            causal_mask = (1 - causal_mask.unsqueeze(1)).bool()
            return causal_mask