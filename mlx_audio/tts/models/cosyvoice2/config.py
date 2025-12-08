# Copyright (c) 2024 Alibaba Inc
# MLX port by mlx-audio contributors
#
# Licensed under the Apache License, Version 2.0

"""Configuration classes for CosyVoice2."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..base import BaseModelArgs


@dataclass
class LLMConfig:
    """Configuration for Qwen2-based LLM."""

    llm_input_size: int = 896
    llm_output_size: int = 896
    speech_token_size: int = 6561
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
class FlowConfig:
    """Configuration for Flow Matching module (CosyVoice2)."""

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

    # Encoder config (CosyVoice2 uses simpler encoder without CNN module)
    encoder_input_size: int = 512
    encoder_output_size: int = 512
    encoder_attention_heads: int = 8
    encoder_linear_units: int = 2048
    encoder_num_blocks: int = 6
    encoder_num_up_blocks: int = 4  # CosyVoice2 uses 4 up_blocks
    encoder_dropout_rate: float = 0.1
    encoder_positional_dropout_rate: float = 0.1
    encoder_attention_dropout_rate: float = 0.1
    encoder_normalize_before: bool = True
    encoder_macaron_style: bool = False  # CosyVoice2 uses False
    encoder_use_cnn_module: bool = False  # CosyVoice2 uses False
    encoder_cnn_module_kernel: int = 15
    encoder_causal: bool = True
    encoder_upsample_stride: int = 2
    encoder_static_chunk_size: int = 25  # CosyVoice2 uses 25 (chunk_size)
    encoder_pos_enc_layer_type: str = "rel_pos_espnet"  # CosyVoice2 uses rel_pos_espnet

    # Decoder config (CosyVoice2 CFM)
    # Note: in_channels=320 is for the estimator (decoder) which receives:
    # x (80) + mu (80) + spks (80) + cond (80) = 320 concatenated channels
    decoder_in_channels: int = 320  # CosyVoice2 estimator input channels
    decoder_out_channel: int = 80
    decoder_channels: List[int] = field(default_factory=lambda: [256])  # Single element
    decoder_dropout: float = 0.0
    decoder_attention_head_dim: int = 64
    decoder_n_blocks: int = 4
    decoder_num_mid_blocks: int = 12  # CosyVoice2 uses 12
    decoder_num_heads: int = 8
    decoder_act_fn: str = "gelu"
    decoder_static_chunk_size: int = (
        50  # CosyVoice2: chunk_size * token_mel_ratio = 25 * 2
    )
    decoder_num_decoding_left_chunks: int = -1  # CosyVoice2 uses -1 (full left context)

    # CFM params
    cfm_in_channels: int = 240  # CosyVoice2 CFM in_channels (separate from estimator)
    cfm_sigma_min: float = 1e-6
    cfm_t_scheduler: str = "cosine"
    cfm_inference_cfg_rate: float = 0.7


@dataclass
class HiFiGANConfig:
    """Configuration for HiFi-GAN vocoder (CosyVoice2 24kHz)."""

    in_channels: int = 80
    base_channels: int = 512
    nb_harmonics: int = 8
    sampling_rate: int = 24000
    nsf_alpha: float = 0.1
    nsf_sigma: float = 0.003
    nsf_voiced_threshold: float = 10.0
    # CosyVoice2 uses [8, 5, 3] not [10, 6, 4, 2]
    upsample_rates: List[int] = field(default_factory=lambda: [8, 5, 3])
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [16, 11, 7])
    istft_n_fft: int = 16
    istft_hop_len: int = 4
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    source_resblock_kernel_sizes: List[int] = field(default_factory=lambda: [7, 7, 11])
    source_resblock_dilation_sizes: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    use_interpolation: bool = True  # For 24kHz mode


@dataclass
class CosyVoice2Config:
    """Full configuration for CosyVoice2 model."""

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
    def from_pretrained(cls, model_path: str) -> "CosyVoice2Config":
        """Load configuration from a pretrained model directory."""
        import json
        from pathlib import Path

        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)

            # Parse sub-configs
            llm_config = LLMConfig(**config_dict.get("llm", {}))

            # Parse flow config - flatten nested encoder/decoder
            flow_dict = config_dict.get("flow", {})
            flow_kwargs = {}
            for k, v in flow_dict.items():
                if k == "encoder" and isinstance(v, dict):
                    # Flatten encoder config with encoder_ prefix
                    for ek, ev in v.items():
                        flow_kwargs[f"encoder_{ek}"] = ev
                elif k == "decoder" and isinstance(v, dict):
                    # Flatten decoder config with decoder_ prefix
                    for dk, dv in v.items():
                        if dk == "out_channels":
                            flow_kwargs["decoder_out_channel"] = dv
                        else:
                            flow_kwargs[f"decoder_{dk}"] = dv
                else:
                    flow_kwargs[k] = v
            flow_config = FlowConfig(**flow_kwargs)

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
    """Model configuration for CosyVoice2 (compatible with generate API)."""

    model_type: str = "cosyvoice2"
    sample_rate: int = 24000
    model_path: Optional[str] = None

    # Internal config (loaded from model)
    _cosyvoice2_config: Optional[CosyVoice2Config] = field(default=None, repr=False)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ModelConfig":
        """Create config from dictionary."""
        return cls(
            model_type=config.get("model_type", "cosyvoice2"),
            sample_rate=config.get("sample_rate", 24000),
            model_path=config.get("model_path"),
        )
