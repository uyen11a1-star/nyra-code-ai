from dataclasses import dataclass

@dataclass
class NyraConfig:
    vocab_size: int = 32000
    hidden_size: int = 1024
    intermediate_size: int = 2752
    num_hidden_layers: int = 23   
    num_attention_heads: int = 16 
    # SỬA LỖI: Giảm KV heads để dùng GQA (tiết kiệm VRAM khi dùng KV Cache)
    num_key_value_heads: int = 8  
    
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = True 
