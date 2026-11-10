import torch
from torch import nn
import torch.nn.functional as F
from model.config import NyraConfig
from model.attention import NyraAttention
from model.initialization import initialize_weights

class NyraRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class NyraMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class NyraDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = NyraAttention(config)
        self.mlp = NyraMLP(config)
        self.input_layernorm = NyraRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = NyraRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states, present_key_value = self.self_attn(
            hidden_states, attention_mask, position_ids, past_key_value, use_cache
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, present_key_value

class NyraModel(nn.Module):
    def __init__(self, config: NyraConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([NyraDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = NyraRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        
        initialize_weights(self, config.initializer_range)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False):
        if position_ids is None:
            past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            position_ids = torch.arange(past_length, input_ids.shape[1] + past_length, device=input_ids.device).unsqueeze(0)
        
        hidden_states = self.embed_tokens(input_ids)
        
        presents = () if use_cache else None
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            hidden_states, present_key_value = layer(
                hidden_states, attention_mask, position_ids, past_key_value, use_cache
            )
            if use_cache:
                presents = presents + (present_key_value,)
                
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, presents
