# Copyright © Anthony DePasquale
# Ported to MLX from https://github.com/FunAudioLLM/CosyVoice

"""Configuration classes for CosyVoice3."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..base import BaseModelArgs


@dataclass
class LLMConfig:
    """Configuration for Qwen2-based LLM in CosyVoice3."""

    llm_input_size: int = 896
    llm_output_size: int = 896
    speech_token_size: int = 6561
    # CosyVoice3 uses +200 extended vocabulary
    extended_vocab_size: int = 200
    mix_ratio: List[int] = field(default_factory=lambda: [5, 15])

    # Qwen2 model config (for reference, actual model loaded from mlx-lm)
    hidden_size: int = 896
    num_hidden_layers: int = 24
    intermediate_size: int = 4864
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936


@dataclass
class DiTConfig:
    """Configuration for Diffusion Transformer (DiT) module."""

    dim: int = 1024
    depth: int = 22
    heads: int = 16
    dim_head: int = 64
    ff_mult: int = 2
    dropout: float = 0.1
    mel_dim: int = 80
    mu_dim: int = 80
    spk_dim: int = 80
    out_channels: int = 80
    static_chunk_size: int = 50  # chunk_size * token_mel_ratio
    num_decoding_left_chunks: int = -1
    long_skip_connection: bool = False


@dataclass
class FlowConfig:
    """Configuration for Flow Matching module (CosyVoice3 with DiT)."""

    input_size: int = 512
    output_size: int = 80
    spk_embed_dim: int = 192
    output_type: str = "mel"
    vocab_size: int = 6561
    input_frame_rate: int = 25
    only_mask_loss: bool = True
    token_mel_ratio: int = 2
    pre_lookahead_len: int = 3
    n_timesteps: int = 10

    # PreLookaheadLayer config
    pre_lookahead_channels: int = 512

    # DiT config (embedded)
    dit: DiTConfig = field(default_factory=DiTConfig)

    # CFM params
    cfm_sigma_min: float = 1e-6
    cfm_t_scheduler: str = "cosine"
    cfm_inference_cfg_rate: float = 0.7


@dataclass
class HiFiGANConfig:
    """Configuration for Causal HiFi-GAN vocoder (CosyVoice3 24kHz)."""

    in_channels: int = 80
    base_channels: int = 512
    nb_harmonics: int = 8
    sampling_rate: int = 24000
    nsf_alpha: float = 0.1
    nsf_sigma: float = 0.003
    nsf_voiced_threshold: float = 10.0
    upsample_rates: List[int] = field(default_factory=lambda: [8, 5, 3])
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [16, 11, 7])
    istft_n_fft: int = 16
    istft_hop_len: int = 4
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    source_resblock_kernel_sizes: List[int] = field(default_factory=lambda: [7, 11])
    source_resblock_dilation_sizes: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5]]
    )
    # Causal-specific config
    conv_pre_look_right: int = 4
    causal: bool = True


@dataclass
class CosyVoice3Config:
    """Full configuration for CosyVoice3 model."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    hifigan: HiFiGANConfig = field(default_factory=HiFiGANConfig)

    # Model paths
    llm_path: Optional[str] = None
    flow_path: Optional[str] = None
    hifigan_path: Optional[str] = None

    # Generation defaults
    default_sampling: int = 25
    max_token_text_ratio: float = 20.0
    min_token_text_ratio: float = 2.0

    @classmethod
    def from_pretrained(cls, model_path: str) -> "CosyVoice3Config":
        """Load configuration from a pretrained model directory."""
        import json
        from pathlib import Path

        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)

            # Parse sub-configs
            llm_config = LLMConfig(**config_dict.get("llm", {}))

            # Parse flow config
            flow_dict = config_dict.get("flow", {})
            dit_dict = flow_dict.pop("dit", flow_dict.pop("estimator", {}))
            dit_config = DiTConfig(**dit_dict) if dit_dict else DiTConfig()
            flow_config = FlowConfig(**flow_dict, dit=dit_config)

            # Parse hifigan config (may be named "hift" in config.json)
            hifigan_dict = config_dict.get("hifigan", config_dict.get("hift", {}))
            hifigan_config = HiFiGANConfig(**hifigan_dict)

            return cls(
                llm=llm_config,
                flow=flow_config,
                hifigan=hifigan_config,
            )
        return cls()


@dataclass
class ModelConfig(BaseModelArgs):
    """Model configuration for CosyVoice3 (compatible with generate API)."""

    model_type: str = "cosyvoice3"
    sample_rate: int = 24000
    model_path: Optional[str] = None

    # Internal config (loaded from model)
    _cosyvoice3_config: Optional[CosyVoice3Config] = field(default=None, repr=False)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ModelConfig":
        """Create config from dictionary."""
        return cls(
            model_type=config.get("model_type", "cosyvoice3"),
            sample_rate=config.get("sample_rate", 24000),
            model_path=config.get("model_path"),
        )
