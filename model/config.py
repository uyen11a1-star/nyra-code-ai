from dataclasses import dataclass

@dataclass
class NyraConfig:
    # Kiến trúc Decoder-only Transformer
    vocab_size: int = 32000       # Vocab size cho code + text
    hidden_size: int = 1024       # D (Embedding dim)
    intermediate_size: int = 2752 # FFN dim (khoảng 2.68 * D)
    num_hidden_layers: int = 24   # L (Số layer)
    num_attention_heads: int = 16 # H (Head dim = 64)
    num_key_value_heads: int = 16 # GQA (bằng số heads tức là MHA thường)
    
    # Context & Training
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = True # Share weight giữa input và output embed
    
    # Tính toán params:
    # Embed: 32000 * 1024 = 32.7M
    # Layer: 24 * (4 * 1024^2 (Attention) + 3 * 1024 * 2752 (MLP)) ≈ 24 * (4.19M + 8.45M) ≈ 303M
    # Tổng: Khoảng 302M parameters.
