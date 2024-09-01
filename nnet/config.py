class ModelConfig:
    max_melspec_len: int = 512
    n_fft: int = 2048
    hop_length: int = int(n_fft / 8)
    win_length: int = int(n_fft / 2)
    n_mels: int = 80


    d_model: int = 128
    hidden_dim: int = d_model * 4
    max_seq_len: int = 512

    n_heads: int = 4
    attn_dropout: float = 0.1
    ffn_dropout: float = 0.1
    n_blocks: int = 4

    kernel_size: int = 5
    padding: str = "same"
    stride: int = 1
    n_postmelblock: int = 4
    postmel_hidden_dim: int = n_mels * 4
    postmel_dropout: int = 0.1