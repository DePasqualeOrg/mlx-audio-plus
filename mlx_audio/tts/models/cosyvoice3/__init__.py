# Copyright © Anthony DePasquale
# Ported to MLX from https://github.com/FunAudioLLM/CosyVoice

"""CosyVoice3 TTS model for MLX."""

from .config import (
    CosyVoice3Config,
    DiTConfig,
    FlowConfig,
    HiFiGANConfig,
    LLMConfig,
    ModelConfig,
)
from .convolution import (
    CausalConv1d,
    CausalConv1dDownSample,
    CausalConv1dUpsample,
    PreLookaheadLayer,
)
from .cosyvoice3 import CosyVoice3, Model, load_cosyvoice3
from .dit import DiT, DiTBlock, InputEmbedding, TimestepEmbedding
from .flow import CausalMaskedDiffWithDiT, CosyVoice3ConditionalCFM, build_flow_model
from .hifigan import CausalHiFTGenerator
from .llm import CosyVoice3LM, Qwen2Encoder, top_k_sampling
from .scripts.convert import convert_from_source

__all__ = [
    # Main model (generate API compatible)
    "Model",
    "ModelConfig",
    # Core model
    "CosyVoice3",
    "load_cosyvoice3",
    # Configuration
    "CosyVoice3Config",
    "LLMConfig",
    "DiTConfig",
    "FlowConfig",
    "HiFiGANConfig",
    # LLM components
    "CosyVoice3LM",
    "Qwen2Encoder",
    "top_k_sampling",
    # DiT components
    "DiT",
    "DiTBlock",
    "TimestepEmbedding",
    "InputEmbedding",
    # Flow components
    "CausalMaskedDiffWithDiT",
    "CosyVoice3ConditionalCFM",
    "build_flow_model",
    # Convolution components
    "CausalConv1d",
    "CausalConv1dDownSample",
    "CausalConv1dUpsample",
    "PreLookaheadLayer",
    # Vocoder
    "CausalHiFTGenerator",
    # Conversion
    "convert_from_source",
]
