# Copyright © 2025 FunASR (original model implementation)
# Copyright © Anthony DePasquale (MLX port)
# Ported to MLX from https://github.com/modelscope/FunASR
# License: licenses/funasr.txt

"""
Qwen3 LLM implementation for Fun-ASR model.

Based on mlx-lm's Qwen3 implementation, adapted for the Fun-ASR speech
recognition task.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@dataclass
class Qwen3Config:
    """Configuration for Qwen3 model."""

    vocab_size: int = 151936
    hidden_size: int = 1024
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8  # GQA
    intermediate_size: int = 3072
    max_position_embeddings: int = 40960
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    head_dim: int = 64
    rope_scaling: Optional[dict] = None


class Attention(nn.Module):
    """
    Qwen3 attention with Grouped Query Attention (GQA) and RoPE.
    """

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim = config.head_dim

        self.scale = head_dim**-0.5

        # Projections
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        # QK normalization (per-head RMSNorm) - Qwen3 specific
        self.q_norm = nn.RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=config.rms_norm_eps)

        # Rotary embeddings
        self.rope = nn.RoPE(
            head_dim,
            traditional=False,
            base=config.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        B, L, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # Reshape for multi-head attention
        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        # Apply QK normalization
        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        # Apply RoPE
        if cache is not None:
            queries = self.rope(queries, offset=cache[0].shape[2])
            keys = self.rope(keys, offset=cache[0].shape[2])
            keys = mx.concatenate([cache[0], keys], axis=2)
            values = mx.concatenate([cache[1], values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        new_cache = (keys, values)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output), new_cache


class MLP(nn.Module):
    """
    Qwen3 MLP with SwiGLU activation.
    """

    def __init__(self, config: Qwen3Config):
        super().__init__()
        dim = config.hidden_size
        hidden_dim = config.intermediate_size

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """
    Single Qwen3 transformer block.
    """

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        # Self-attention with pre-norm and residual
        r = x
        x = self.input_layernorm(x)
        x, cache = self.self_attn(x, mask, cache)
        x = r + x

        # MLP with pre-norm and residual
        r = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = r + x

        return x, cache


class Qwen3Model(nn.Module):
    """
    Qwen3 transformer model (without LM head).
    """

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        input_embeddings: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
    ) -> Tuple[mx.array, Optional[List[Tuple[mx.array, mx.array]]]]:
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(input_ids)

        # Create causal mask if needed
        if mask is None and h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        new_cache = []
        for layer, c in zip(self.layers, cache):
            h, layer_cache = layer(h, mask, c)
            new_cache.append(layer_cache)

        return self.norm(h), new_cache


class Qwen3ForCausalLM(nn.Module):
    """
    Qwen3 model with language modeling head.
    """

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)

        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        input_embeddings: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
    ) -> Tuple[mx.array, Optional[List[Tuple[mx.array, mx.array]]]]:
        out, cache = self.model(input_ids, input_embeddings, mask, cache)

        if self.config.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(out)
        else:
            logits = self.lm_head(out)

        return logits, cache

    def get_input_embeddings(self) -> nn.Embedding:
        """Get the input embedding layer."""
        return self.model.embed_tokens

    @property
    def layers(self):
        return self.model.layers


def create_qwen3_from_config(config_dict: dict) -> Qwen3ForCausalLM:
    """
    Create a Qwen3 model from a configuration dictionary.
    """
    config = Qwen3Config(
        vocab_size=config_dict.get("vocab_size", 151936),
        hidden_size=config_dict.get("hidden_size", 1024),
        num_hidden_layers=config_dict.get("num_hidden_layers", 28),
        num_attention_heads=config_dict.get("num_attention_heads", 16),
        num_key_value_heads=config_dict.get("num_key_value_heads", 8),
        intermediate_size=config_dict.get("intermediate_size", 3072),
        max_position_embeddings=config_dict.get("max_position_embeddings", 40960),
        rope_theta=config_dict.get("rope_theta", 1000000.0),
        rms_norm_eps=config_dict.get("rms_norm_eps", 1e-6),
        tie_word_embeddings=config_dict.get("tie_word_embeddings", True),
        head_dim=config_dict.get("head_dim", 64),
    )
    return Qwen3ForCausalLM(config)
