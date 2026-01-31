# Qwen3-TTS Model Components

This document lists all components of the Qwen3-TTS model implementation in Python MLX for reference when porting to Swift MLX.

## Table of Contents

1. [Overview](#overview)
2. [External Dependencies](#external-dependencies)
3. [Configuration Classes](#configuration-classes)
4. [Main Model](#main-model)
5. [Talker Components](#talker-components)
6. [Code Predictor Components](#code-predictor-components)
7. [Speech Tokenizer Components](#speech-tokenizer-components)
8. [Speaker Encoder Components](#speaker-encoder-components)
9. [Utility Functions](#utility-functions)
10. [Weight Shapes Reference](#weight-shapes-reference)

---

## Overview

Qwen3-TTS is a text-to-speech model with the following high-level architecture:

```
Text Input → [Text Tokenizer] → [Talker Model] → Codec Tokens → [Speech Tokenizer Decoder] → Audio
                                     ↓
                              [Code Predictor]
                              (multi-codebook)
```

For voice cloning (ICL mode):
```
Reference Audio → [Speech Tokenizer Encoder] → Reference Codes
                                                    ↓
Text + Reference → [Talker Model] → Generated Codes → [Speech Tokenizer Decoder] → Audio
```

**File Locations:**
- `mlx_audio/tts/models/qwen3_tts/qwen3_tts.py` - Main model
- `mlx_audio/tts/models/qwen3_tts/talker.py` - Talker transformer
- `mlx_audio/tts/models/qwen3_tts/speech_tokenizer.py` - Audio encoder/decoder
- `mlx_audio/tts/models/qwen3_tts/speaker_encoder.py` - ECAPA-TDNN speaker encoder
- `mlx_audio/tts/models/qwen3_tts/config.py` - Configuration dataclasses

---

## External Dependencies

These are external modules used by Qwen3-TTS that would also need to be ported to Swift MLX.

### From mlx_lm

| Component | Import Path | Used By |
|-----------|-------------|---------|
| KVCache | `mlx_lm.models.cache.KVCache` | Talker, Code Predictor, Speech Tokenizer |

**KVCache** is a key-value cache for efficient autoregressive generation. It stores past key/value states and provides:
- `update_and_fetch(k, v)` - Update cache with new k/v and return full sequence
- `offset` - Current position in the sequence

### From mlx_audio.dsp

| Component | Import Path | Used By |
|-----------|-------------|---------|
| mel_filters | `mlx_audio.dsp.mel_filters` | Main Model (mel spectrogram) |
| stft | `mlx_audio.dsp.stft` | Main Model (mel spectrogram) |

### From mlx_audio.codec.models.mimi

Used by the Speech Tokenizer Encoder for ICL voice cloning:

| Component | Import Path | Description |
|-----------|-------------|-------------|
| SeanetConfig | `mlx_audio.codec.models.mimi.modules.SeanetConfig` | Config for convolutional encoder |
| SeanetEncoder | `mlx_audio.codec.models.mimi.modules.SeanetEncoder` | Convolutional audio encoder |
| TransformerConfig | `mlx_audio.codec.models.mimi.modules.TransformerConfig` | Config for transformer |
| ProjectedTransformer | `mlx_audio.codec.models.mimi.modules.ProjectedTransformer` | Transformer with projection |
| ConvDownsample1d | `mlx_audio.codec.models.mimi.modules.ConvDownsample1d` | Frame rate downsampler |
| SplitResidualVectorQuantizer | `mlx_audio.codec.models.mimi.modules.SplitResidualVectorQuantizer` | Split RVQ (as MimiSplitRVQ) |
| _reset_kv_cache | `mlx_audio.codec.models.mimi.mimi._reset_kv_cache` | Helper to reset KV cache |

### From mlx_audio.tts.models.base

| Component | Import Path | Description |
|-----------|-------------|-------------|
| GenerationResult | `mlx_audio.tts.models.base.GenerationResult` | Dataclass for generation output |
| BaseModelArgs | `mlx_audio.tts.models.base.BaseModelArgs` | Base config class |

**GenerationResult** contains:
- `audio: mx.array` - Generated audio waveform
- `samples: int` - Number of audio samples
- `sample_rate: int` - Audio sample rate
- `segment_idx: int` - Segment index
- `token_count: int` - Number of tokens generated
- `audio_duration: str` - Formatted duration string
- `real_time_factor: float` - RTF (audio_duration / processing_time)
- `prompt: dict` - Token statistics
- `audio_samples: dict` - Sample statistics
- `processing_time_seconds: float` - Processing time
- `peak_memory_usage: float` - Peak memory in GB

---

## Configuration Classes

**File:** `config.py`

### Helper Function: filter_dict_for_dataclass()
Filters a dictionary to only include keys that are valid dataclass fields.

**Parameters:**
- `cls: Type[T]` - Dataclass type
- `data: Dict[str, Any]` - Dictionary to filter

**Returns:** Filtered dictionary with only valid fields

### 1. ModelConfig (BaseModelArgs)
Main configuration for Qwen3-TTS model.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| model_type | str | "qwen3_tts" | Model identifier |
| talker_config | Qwen3TTSTalkerConfig | None | Talker model config |
| speaker_encoder_config | Qwen3TTSSpeakerEncoderConfig | None | Speaker encoder config |
| tokenizer_config | Qwen3TTSTokenizerConfig | None | Speech tokenizer config |
| tokenizer_type | str | "qwen3_tts_tokenizer_12hz" | Tokenizer type |
| tts_model_size | str | "0b6" | Model size (0b6, 1b5, etc.) |
| tts_model_type | str | "base" | Type: "base", "custom_voice", "voice_design" |
| im_start_token_id | int | 151644 | Chat template start token |
| im_end_token_id | int | 151645 | Chat template end token |
| tts_pad_token_id | int | 151671 | TTS padding token |
| tts_bos_token_id | int | 151672 | TTS beginning of speech token |
| tts_eos_token_id | int | 151673 | TTS end of speech token |
| sample_rate | int | 24000 | Audio sample rate |

### 2. Qwen3TTSTalkerConfig
Configuration for the main talker model.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| code_predictor_config | Qwen3TTSTalkerCodePredictorConfig | None | Code predictor config |
| vocab_size | int | 3072 | Codec vocabulary size |
| hidden_size | int | 1024 | Transformer hidden dimension |
| intermediate_size | int | 3072 | MLP intermediate dimension |
| num_hidden_layers | int | 28 | Number of transformer layers |
| num_attention_heads | int | 16 | Number of attention heads |
| num_key_value_heads | int | 8 | Number of KV heads (GQA) |
| head_dim | int | 128 | Per-head dimension |
| hidden_act | str | "silu" | Activation function |
| max_position_embeddings | int | 32768 | Maximum sequence length |
| rms_norm_eps | float | 1e-6 | RMSNorm epsilon |
| rope_theta | float | 1000000.0 | RoPE base frequency |
| rope_scaling | Dict | {"mrope_section": [24, 20, 20]} | MRoPE configuration |
| attention_bias | bool | False | Use attention bias |
| num_code_groups | int | 16 | Number of codebook groups |
| text_hidden_size | int | 2048 | Text embedding dimension |
| text_vocab_size | int | 151936 | Text vocabulary size |
| codec_eos_token_id | int | 2150 | Codec end token |
| codec_think_id | int | 2154 | Think mode token |
| codec_nothink_id | int | 2155 | No-think mode token |
| codec_think_bos_id | int | 2156 | Think BOS token |
| codec_think_eos_id | int | 2157 | Think EOS token |
| codec_pad_id | int | 2148 | Codec padding token |
| codec_bos_id | int | 2149 | Codec BOS token |
| codec_language_id | Dict[str, int] | None | Language ID mapping |
| spk_id | Dict[str, List[int]] | None | Speaker ID mapping |
| spk_is_dialect | Dict[str, str] | None | Speaker dialect mapping |

### 3. Qwen3TTSTalkerCodePredictorConfig
Configuration for the code predictor sub-model.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| vocab_size | int | 2048 | Codec vocabulary size |
| hidden_size | int | 1024 | Hidden dimension |
| intermediate_size | int | 3072 | MLP intermediate size |
| num_hidden_layers | int | 5 | Number of layers |
| num_attention_heads | int | 16 | Number of attention heads |
| num_key_value_heads | int | 8 | Number of KV heads |
| head_dim | int | 128 | Per-head dimension |
| hidden_act | str | "silu" | Activation function |
| max_position_embeddings | int | 65536 | Maximum sequence length |
| rms_norm_eps | float | 1e-6 | RMSNorm epsilon |
| rope_theta | float | 1000000.0 | RoPE base frequency |
| attention_bias | bool | False | Use attention bias |
| num_code_groups | int | 16 | Number of codebook groups |

### 4. Qwen3TTSTokenizerDecoderConfig
Configuration for the speech tokenizer decoder.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| latent_dim | int | 1024 | Latent dimension |
| codebook_dim | int | 512 | Codebook dimension |
| codebook_size | int | 2048 | Codebook size |
| decoder_dim | int | 1536 | Initial decoder dimension |
| hidden_size | int | 512 | Transformer hidden size |
| intermediate_size | int | 1024 | MLP intermediate size |
| head_dim | int | 64 | Attention head dimension |
| num_attention_heads | int | 16 | Number of attention heads |
| num_key_value_heads | int | 16 | Number of KV heads |
| num_hidden_layers | int | 8 | Number of transformer layers |
| num_quantizers | int | 16 | Number of RVQ levels |
| num_semantic_quantizers | int | 1 | Semantic quantizer levels |
| max_position_embeddings | int | 8000 | Maximum sequence length |
| rms_norm_eps | float | 1e-5 | RMSNorm epsilon |
| rope_theta | float | 10000.0 | RoPE base frequency |
| sliding_window | int | 72 | Sliding window size |
| layer_scale_initial_scale | float | 0.01 | Layer scale init |
| upsample_rates | List[int] | [8, 5, 4, 3] | Decoder upsample rates |
| upsampling_ratios | List[int] | [2, 2] | Pre-decoder upsample rates |

### 5. Qwen3TTSTokenizerEncoderConfig
Configuration for the speech tokenizer encoder.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| frame_rate | float | 12.5 | Output frame rate |
| audio_channels | int | 1 | Audio channels |
| codebook_dim | int | 256 | Codebook dimension |
| codebook_size | int | 2048 | Codebook size |
| hidden_size | int | 512 | Hidden dimension |
| intermediate_size | int | 2048 | MLP intermediate size |
| num_attention_heads | int | 8 | Number of attention heads |
| num_key_value_heads | int | 8 | Number of KV heads |
| num_hidden_layers | int | 8 | Number of transformer layers |
| num_quantizers | int | 32 | Number of RVQ levels |
| num_semantic_quantizers | int | 1 | Semantic quantizer levels |
| num_filters | int | 64 | Base filter count |
| num_residual_layers | int | 1 | Residual layers per block |
| kernel_size | int | 7 | Conv kernel size |
| residual_kernel_size | int | 3 | Residual conv kernel size |
| last_kernel_size | int | 3 | Final conv kernel size |
| compress | int | 2 | Compression factor |
| dilation_growth_rate | int | 2 | Dilation growth rate |
| sampling_rate | int | 24000 | Input sample rate |
| upsampling_ratios | List[int] | [8, 6, 5, 4] | Encoder downsample ratios |
| use_causal_conv | bool | True | Use causal convolutions |
| max_position_embeddings | int | 8000 | Maximum sequence length |
| rope_theta | float | 10000.0 | RoPE base frequency |
| sliding_window | int | 250 | Sliding window size |
| layer_scale_initial_scale | float | 0.01 | Layer scale init |

### 6. Qwen3TTSTokenizerConfig
Configuration for the speech tokenizer (wraps encoder + decoder).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| encoder_config | Qwen3TTSTokenizerEncoderConfig | None | Encoder config |
| decoder_config | Qwen3TTSTokenizerDecoderConfig | None | Decoder config |
| encoder_valid_num_quantizers | int | 16 | Valid quantizer count |
| input_sample_rate | int | 24000 | Input sample rate |
| output_sample_rate | int | 24000 | Output sample rate |
| decode_upsample_rate | int | 1920 | Decode upsample rate |
| encode_downsample_rate | int | 1920 | Encode downsample rate |

### 7. Qwen3TTSSpeakerEncoderConfig
Configuration for ECAPA-TDNN speaker encoder.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| mel_dim | int | 128 | Mel spectrogram bins |
| enc_dim | int | 1024 | Output embedding dimension |
| enc_channels | List[int] | [512, 512, 512, 512, 1536] | Channel sizes per layer |
| enc_kernel_sizes | List[int] | [5, 3, 3, 3, 1] | Kernel sizes per layer |
| enc_dilations | List[int] | [1, 2, 3, 4, 1] | Dilation rates per layer |
| enc_attention_channels | int | 128 | Attention pooling channels |
| enc_res2net_scale | int | 8 | Res2Net scale factor |
| enc_se_channels | int | 128 | Squeeze-excitation channels |
| sample_rate | int | 24000 | Audio sample rate |

---

## Main Model

**File:** `qwen3_tts.py`

### Model (nn.Module)
The main Qwen3-TTS model class.

**Components:**
| Component | Type | Description |
|-----------|------|-------------|
| talker | Qwen3TTSTalkerForConditionalGeneration | Main transformer model |
| speaker_encoder | Qwen3TTSSpeakerEncoder | Optional speaker encoder (for base models) |
| speech_tokenizer | Qwen3TTSSpeechTokenizer | Audio codec (loaded separately) |
| tokenizer | AutoTokenizer | Text tokenizer (HuggingFace) |

**Key Methods:**
| Method | Description |
|--------|-------------|
| `generate()` | Main TTS generation (routes to appropriate method) |
| `generate_custom_voice()` | Generation with predefined speaker + optional instruct |
| `generate_voice_design()` | Generation with natural language voice description |
| `_generate_icl()` | In-context learning voice cloning |
| `extract_speaker_embedding()` | Extract x-vector from reference audio |
| `_prepare_generation_inputs()` | Build input embeddings for talker |
| `_prepare_icl_generation_inputs()` | Build ICL input embeddings |
| `_sample_token()` | Token sampling with top-k, top-p, repetition penalty |
| `load_speech_tokenizer()` | Load speech tokenizer model |
| `from_pretrained()` | Load model from HuggingFace |
| `post_load_hook()` | Initialize tokenizer and speech tokenizer after weight loading |
| `sanitize()` | Convert PyTorch weights to MLX format |
| `model_quant_predicate()` | Determine which layers to quantize |

---

## Talker Components

**File:** `talker.py`

### 1. RMSNorm (nn.Module)
RMS Layer Normalization.

**Parameters:**
- `dims: int` - Normalization dimension
- `eps: float = 1e-6` - Epsilon for numerical stability

**Weights:**
- `weight: [dims]` - Scale parameter

### 2. RotaryEmbedding (nn.Module)
Standard Rotary Position Embedding for code predictor.

**Parameters:**
- `dim: int` - Embedding dimension
- `max_position_embeddings: int = 32768`
- `base: float = 10000.0`

**Internal State:**
- `_inv_freq: [dim/2]` - Inverse frequencies

### 3. TalkerRotaryEmbedding (nn.Module)
Multimodal Rotary Embedding with interleaved MRoPE for 3D positions.

**Parameters:**
- `dim: int` - Embedding dimension
- `max_position_embeddings: int = 32768`
- `base: float = 10000.0`
- `mrope_section: List[int] = [24, 20, 20]` - Dimensions for T/H/W

**Internal State:**
- `_inv_freq: [dim/2]` - Inverse frequencies

**Key Methods:**
- `apply_interleaved_mrope()` - Combines 3D frequencies with interleaved layout

### 4. TalkerAttention (nn.Module)
Multi-head attention with MRoPE and QK normalization.

**Parameters:**
- `config: Qwen3TTSTalkerConfig`
- `layer_idx: int`

**Submodules:**
- `q_proj: nn.Linear(hidden_size, num_heads * head_dim)`
- `k_proj: nn.Linear(hidden_size, num_kv_heads * head_dim)`
- `v_proj: nn.Linear(hidden_size, num_kv_heads * head_dim)`
- `o_proj: nn.Linear(num_heads * head_dim, hidden_size)`
- `q_norm: RMSNorm(head_dim)`
- `k_norm: RMSNorm(head_dim)`

### 5. TalkerMLP (nn.Module)
SwiGLU MLP block.

**Parameters:**
- `config: Qwen3TTSTalkerConfig`

**Submodules:**
- `gate_proj: nn.Linear(hidden_size, intermediate_size, bias=False)`
- `up_proj: nn.Linear(hidden_size, intermediate_size, bias=False)`
- `down_proj: nn.Linear(intermediate_size, hidden_size, bias=False)`

**Forward:** `down_proj(silu(gate_proj(x)) * up_proj(x))`

### 6. ResizeMLP (nn.Module)
MLP for resizing hidden dimensions (text projection).

**Parameters:**
- `input_size: int`
- `intermediate_size: int`
- `output_size: int`
- `hidden_act: str = "silu"`
- `bias: bool = False`

**Submodules:**
- `linear_fc1: nn.Linear(input_size, intermediate_size, bias)`
- `linear_fc2: nn.Linear(intermediate_size, output_size, bias)`

### 7. TalkerDecoderLayer (nn.Module)
Single transformer decoder layer.

**Parameters:**
- `config: Qwen3TTSTalkerConfig`
- `layer_idx: int`

**Submodules:**
- `self_attn: TalkerAttention`
- `mlp: TalkerMLP`
- `input_layernorm: RMSNorm(hidden_size)`
- `post_attention_layernorm: RMSNorm(hidden_size)`

### 8. Qwen3TTSTalkerModel (nn.Module)
Main talker transformer model.

**Parameters:**
- `config: Qwen3TTSTalkerConfig`

**Submodules:**
- `codec_embedding: nn.Embedding(vocab_size, hidden_size)` - Codec token embeddings
- `text_embedding: nn.Embedding(text_vocab_size, text_hidden_size)` - Text token embeddings
- `layers: List[TalkerDecoderLayer]` - num_hidden_layers decoder layers
- `norm: RMSNorm(hidden_size)` - Final layer norm
- `rotary_emb: TalkerRotaryEmbedding` - MRoPE embeddings

**Key Methods:**
- `make_cache()` - Create KV cache for all layers

### 9. Qwen3TTSTalkerForConditionalGeneration (nn.Module)
Full talker model for conditional generation.

**Parameters:**
- `config: Qwen3TTSTalkerConfig`

**Submodules:**
- `model: Qwen3TTSTalkerModel` - Main transformer
- `text_projection: ResizeMLP` - Projects text embeddings to codec space
- `codec_head: nn.Linear(hidden_size, vocab_size)` - First codebook prediction
- `code_predictor: Qwen3TTSTalkerCodePredictor` - Multi-codebook predictor

**Key Methods:**
- `get_input_embeddings()` - Returns codec_embedding
- `get_text_embeddings()` - Returns text_embedding
- `make_cache()` - Create KV cache

---

## Code Predictor Components

**File:** `talker.py`

### 1. CodePredictorAttention (nn.Module)
Attention for code predictor with standard RoPE.

**Parameters:**
- `config: Qwen3TTSTalkerCodePredictorConfig`
- `layer_idx: int`

**Submodules:**
- `q_proj, k_proj, v_proj, o_proj: nn.Linear`
- `q_norm, k_norm: RMSNorm(head_dim)`

### 2. CodePredictorMLP (nn.Module)
SwiGLU MLP for code predictor.

**Submodules:**
- `gate_proj: nn.Linear(hidden_size, intermediate_size, bias=False)`
- `up_proj: nn.Linear(hidden_size, intermediate_size, bias=False)`
- `down_proj: nn.Linear(intermediate_size, hidden_size, bias=False)`

### 3. CodePredictorDecoderLayer (nn.Module)
Decoder layer for code predictor.

**Submodules:**
- `self_attn: CodePredictorAttention`
- `mlp: CodePredictorMLP`
- `input_layernorm: RMSNorm(hidden_size)`
- `post_attention_layernorm: RMSNorm(hidden_size)`

### 4. CodePredictorModel (nn.Module)
Inner model for code predictor.

**Parameters:**
- `config: Qwen3TTSTalkerCodePredictorConfig`
- `talker_hidden_size: int`

**Submodules:**
- `codec_embedding: List[nn.Embedding]` - (num_code_groups - 1) embeddings
- `layers: List[CodePredictorDecoderLayer]` - num_hidden_layers layers
- `norm: RMSNorm(hidden_size)`
- `rotary_emb: RotaryEmbedding`

### 5. Qwen3TTSTalkerCodePredictor (nn.Module)
Code predictor sub-model for multi-codebook prediction.

**Parameters:**
- `config: Qwen3TTSTalkerCodePredictorConfig`
- `talker_hidden_size: int`

**Submodules:**
- `small_to_mtp_projection: nn.Linear` - Optional size projection
- `model: CodePredictorModel` - Inner transformer
- `lm_head: List[nn.Linear]` - (num_code_groups - 1) prediction heads

---

## Speech Tokenizer Components

**File:** `speech_tokenizer.py`

### Decoder Components

#### 1. DepthwiseConvWeight (nn.Module)
Container for depthwise conv weights to match PyTorch key structure.

**Parameters:**
- `out_channels: int`
- `kernel_size: int`
- `in_per_group: int`

**Weights:**
- `weight: [out_channels, kernel_size, in_per_group]`
- `bias: [out_channels]`

#### 2. CausalConv1d (nn.Module)
Causal 1D convolution with proper padding. Supports grouped convolutions.

**Parameters:**
- `in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1`

**Submodules:**
- `conv: nn.Conv1d` (for groups=1) or `DepthwiseConvWeight` (for grouped/depthwise convs)

**Key Methods:**
- `_get_extra_padding()` - Compute extra padding for alignment

#### 3. CausalTransposeConv1d (nn.Module)
Causal transposed convolution for upsampling.

**Parameters:**
- `in_channels, out_channels, kernel_size, stride=1`

**Submodules:**
- `conv: nn.ConvTranspose1d`
- `trim_right: int` - Amount to trim from right for causal behavior

#### 4. SnakeBeta (nn.Module)
Snake activation: `x + (1/beta) * sin²(x * alpha)`

**Parameters:**
- `channels: int`

**Weights:**
- `alpha: [channels]`
- `beta: [channels]`

#### 5. ConvNeXtBlock (nn.Module)
ConvNeXt block for feature processing.

**Parameters:**
- `dim: int`

**Submodules:**
- `dwconv: CausalConv1d(dim, dim, kernel_size=7, groups=dim)`
- `norm: nn.LayerNorm(dim)`
- `pwconv1: nn.Linear(dim, 4*dim)`
- `pwconv2: nn.Linear(4*dim, dim)`
- `gamma: [dim]` - Layer scale

#### 6. DecoderRMSNorm (nn.Module)
RMS normalization for decoder.

**Weights:**
- `weight: [hidden_size]`

#### 7. LayerScale (nn.Module)
Layer scale for residual connections.

**Weights:**
- `scale: [channels]`

#### 8. DecoderRotaryEmbedding (nn.Module)
Rotary embedding for decoder transformer.

**Internal State:**
- `_inv_freq: [dim/2]`

#### 9. DecoderAttention (nn.Module)
Multi-head attention for decoder transformer.

**Submodules:**
- `q_proj, k_proj, v_proj, o_proj: nn.Linear`

#### 10. DecoderMLP (nn.Module)
SwiGLU MLP for decoder.

**Submodules:**
- `gate_proj, up_proj, down_proj: nn.Linear`

#### 11. DecoderTransformerLayer (nn.Module)
Transformer layer for decoder.

**Submodules:**
- `self_attn: DecoderAttention`
- `mlp: DecoderMLP`
- `input_layernorm: DecoderRMSNorm`
- `post_attention_layernorm: DecoderRMSNorm`
- `self_attn_layer_scale: LayerScale`
- `mlp_layer_scale: LayerScale`

#### 12. DecoderTransformer (nn.Module)
Full transformer for decoder.

**Submodules:**
- `layers: List[DecoderTransformerLayer]` - num_hidden_layers layers
- `norm: DecoderRMSNorm`
- `rotary_emb: DecoderRotaryEmbedding`
- `input_proj: nn.Linear(latent_dim, hidden_size)`
- `output_proj: nn.Linear(hidden_size, latent_dim)`

#### 13. Vector Quantization Components

**EuclideanCodebook:**
- `embed: nn.Embedding(codebook_size, dim)`

**VectorQuantization:**
- `project_out: nn.Linear` (optional)
- `codebook: EuclideanCodebook`

**ResidualVectorQuantization:**
- `layers: List[VectorQuantization]` - num_quantizers layers

**ResidualVectorQuantizer:**
- `input_proj: nn.Conv1d` (optional)
- `output_proj: nn.Conv1d` (optional)
- `vq: ResidualVectorQuantization`

**SplitResidualVectorQuantizer:**
- `rvq_first: ResidualVectorQuantizer` - Semantic quantizer
- `rvq_rest: ResidualVectorQuantizer` - Acoustic quantizer

#### 14. DecoderResidualUnit (nn.Module)
Residual unit for decoder.

**Submodules:**
- `act1: SnakeBeta`
- `conv1: CausalConv1d(dim, dim, kernel_size=7, dilation)`
- `act2: SnakeBeta`
- `conv2: CausalConv1d(dim, dim, kernel_size=1)`

#### 15. DecoderBlockUpsample (nn.Module)
Upsample layer wrapper for decoder block (causal transpose conv).

**Parameters:**
- `in_dim: int`
- `out_dim: int`
- `upsample_rate: int`

**Submodules:**
- `conv: nn.ConvTranspose1d(in_dim, out_dim, kernel_size=2*upsample_rate, stride=upsample_rate)`
- `trim_right: int` - `kernel_size - upsample_rate` for causal trimming

#### 16. DecoderBlock (nn.Module)
Decoder block with upsampling.

**Parameters:**
- `config: Qwen3TTSTokenizerDecoderConfig`
- `layer_idx: int`

**Submodules (as block list):**
- `block[0]: SnakeBeta(in_dim)` - Activation
- `block[1]: DecoderBlockUpsample(in_dim, out_dim, upsample_rate)` - Transpose conv
- `block[2]: DecoderResidualUnit(out_dim, dilation=1)`
- `block[3]: DecoderResidualUnit(out_dim, dilation=3)`
- `block[4]: DecoderResidualUnit(out_dim, dilation=9)`

#### 17. DecoderInitialConv (nn.Module)
Initial convolution wrapper for decoder. Matches PyTorch key: `decoder.decoder.0.conv.*`

**Parameters:**
- `latent_dim: int`
- `decoder_dim: int`
- `kernel_size: int = 7`

**Submodules:**
- `conv: nn.Conv1d(latent_dim, decoder_dim, kernel_size)`

#### 18. DecoderOutputSnake (nn.Module)
Output SnakeBeta layer. Matches PyTorch key: `decoder.decoder.5.*`

**Parameters:**
- `channels: int`

**Weights:**
- `alpha: [channels]`
- `beta: [channels]`

#### 19. DecoderOutputConv (nn.Module)
Output convolution layer. Matches PyTorch key: `decoder.decoder.6.conv.*`

**Parameters:**
- `channels: int`
- `kernel_size: int = 7`

**Submodules:**
- `conv: nn.Conv1d(channels, 1, kernel_size)`

#### 20. Qwen3TTSSpeechTokenizerDecoder (nn.Module)
Full decoder for speech tokenizer.

**Submodules:**
- `pre_transformer: DecoderTransformer`
- `quantizer: SplitResidualVectorQuantizer`
- `pre_conv: CausalConv1d(codebook_dim, latent_dim, kernel_size=3)`
- `upsample: List[List[CausalTransposeConv1d, ConvNeXtBlock]]` - Upsampling blocks
- `decoder: List`:
  - `decoder[0]: DecoderInitialConv`
  - `decoder[1-4]: DecoderBlock` (4 blocks)
  - `decoder[5]: DecoderOutputSnake`
  - `decoder[6]: DecoderOutputConv`

**Key Methods:**
- `chunked_decode()` - Decode in chunks for long sequences

### Encoder Components

#### 21. Qwen3TTSSpeechTokenizerEncoder (nn.Module)
Encoder using Mimi components (for ICL voice cloning).

**Parameters:**
- `config: Qwen3TTSTokenizerEncoderConfig`

**Submodules:**
- `encoder: SeanetEncoder` - Convolutional encoder (from Mimi)
- `encoder_transformer: ProjectedTransformer` - Transformer (from Mimi)
- `downsample: ConvDownsample1d` - Frame rate downsample (from Mimi)
- `quantizer: MimiSplitRVQ` - Split RVQ quantizer (from Mimi)
- `encoder_cache: List[KVCache]` - Cached KV states

**Key Methods:**
- `encode(audio) -> codes` - Encode audio waveform to codes

### Full Tokenizer

#### 22. Qwen3TTSSpeechTokenizer (nn.Module)
Full speech tokenizer (encoder + decoder).

**Parameters:**
- `config: Qwen3TTSTokenizerConfig`

**Submodules:**
- `decoder: Qwen3TTSSpeechTokenizerDecoder`
- `encoder_model: Qwen3TTSSpeechTokenizerEncoder` (optional, for ICL)

**Key Methods:**
- `encode(audio) -> codes` - Audio to codes (requires encoder)
- `decode(codes) -> (audio, audio_lengths)` - Codes to audio
- `has_encoder` - Property indicating ICL support

**Static Methods:**
- `sanitize(weights)` - Convert PyTorch weights to MLX format

---

## Speaker Encoder Components

**File:** `speaker_encoder.py`

### 1. TimeDelayNetBlock (nn.Module)
TDNN block with 1D conv, reflect padding, and ReLU.

**Parameters:**
- `in_channels, out_channels, kernel_size, dilation`

**Submodules:**
- `conv: nn.Conv1d`

### 2. Res2NetBlock (nn.Module)
Multi-scale feature extraction block.

**Parameters:**
- `in_channels, out_channels, scale=8, kernel_size=3, dilation=1`

**Submodules:**
- `blocks: List[TimeDelayNetBlock]` - (scale - 1) blocks

### 3. SqueezeExcitationBlock (nn.Module)
Channel attention block.

**Parameters:**
- `in_channels, se_channels, out_channels`

**Submodules:**
- `conv1: nn.Conv1d(in_channels, se_channels, kernel_size=1)`
- `conv2: nn.Conv1d(se_channels, out_channels, kernel_size=1)`

### 4. SqueezeExcitationRes2NetBlock (nn.Module)
TDNN-Res2Net-TDNN-SE block.

**Submodules:**
- `tdnn1: TimeDelayNetBlock(in, out, kernel_size=1)`
- `res2net_block: Res2NetBlock`
- `tdnn2: TimeDelayNetBlock(out, out, kernel_size=1)`
- `se_block: SqueezeExcitationBlock`

### 5. AttentiveStatisticsPooling (nn.Module)
Attentive statistics pooling.

**Submodules:**
- `tdnn: TimeDelayNetBlock(channels*3, attention_channels, 1, 1)`
- `conv: nn.Conv1d(attention_channels, channels, kernel_size=1)`

### 6. Qwen3TTSSpeakerEncoder (nn.Module)
ECAPA-TDNN speaker encoder.

**Submodules:**
- `blocks[0]: TimeDelayNetBlock` - Initial TDNN
- `blocks[1-3]: SqueezeExcitationRes2NetBlock` - SE-Res2Net layers
- `mfa: TimeDelayNetBlock` - Multi-layer feature aggregation
- `asp: AttentiveStatisticsPooling`
- `fc: nn.Conv1d(enc_channels[-1]*2, enc_dim, kernel_size=1)` - Final projection

**Input:** Mel spectrogram `[batch, time, mel_dim]`
**Output:** Speaker embedding `[batch, enc_dim]`

---

## Utility Functions

### From qwen3_tts.py

#### mel_spectrogram()
Compute mel spectrogram from audio waveform.

**Parameters:**
- `audio: mx.array` - Audio waveform
- `n_fft: int = 1024`
- `num_mels: int = 128`
- `sample_rate: int = 24000`
- `hop_size: int = 256`
- `win_size: int = 1024`
- `fmin: float = 0.0`
- `fmax: float = 12000.0`

**Returns:** `[batch, frames, n_mels]`

#### check_array_shape_qwen3()
Check if Conv1d weights are already in MLX format.

**Parameters:**
- `arr: mx.array` - Array to check (must be 3D)

**Returns:** `bool` - True if MLX format, False if PyTorch format

#### format_duration()
Format duration in seconds as HH:MM:SS.mmm string.

**Parameters:**
- `seconds: float` - Duration in seconds

**Returns:** Formatted string (e.g., "00:01:23.456")

### From talker.py

#### rotate_half()
Rotates half the hidden dims of the input for RoPE.

**Parameters:**
- `x: mx.array` - Input tensor `[..., dim]`

**Returns:** Tensor with first and second halves swapped and negated

#### apply_rotary_pos_emb()
Applies standard Rotary Position Embedding to query and key tensors.

**Parameters:**
- `q, k: mx.array` - Query and key tensors `[batch, heads, seq, head_dim]`
- `cos, sin: mx.array` - Position embeddings `[batch, seq, head_dim]`

**Returns:** Tuple of rotated (q, k)

#### apply_multimodal_rotary_pos_emb()
Applies Multimodal RoPE (MRoPE) to query and key tensors.

**Parameters:**
- `q, k: mx.array` - Query and key tensors
- `cos, sin: mx.array` - Combined MRoPE position embeddings
- `unsqueeze_dim: int = 1` - Dimension to unsqueeze for broadcasting

**Returns:** Tuple of rotated (q, k)

### From speech_tokenizer.py

#### rotate_half() (duplicate)
Same as talker.py version, duplicated for decoder transformer.

#### apply_rotary_pos_emb() (duplicate)
Same as talker.py version, duplicated for decoder transformer.

### From speaker_encoder.py

#### reflect_pad_1d()
Apply reflect padding to the time dimension (axis=1) in NLC format.

**Parameters:**
- `x: mx.array` - Input tensor `[batch, time, channels]`
- `pad: int` - Number of samples to pad on each side

**Returns:** Padded tensor `[batch, time + 2*pad, channels]`

---

## Weight Shapes Reference

### Talker Model Weights

```
model.codec_embedding.weight: [vocab_size, hidden_size]  # e.g., [3072, 1024]
model.text_embedding.weight: [text_vocab_size, text_hidden_size]  # e.g., [151936, 2048]

# Per layer (num_hidden_layers times):
model.layers.{i}.self_attn.q_proj.weight: [num_heads * head_dim, hidden_size]
model.layers.{i}.self_attn.k_proj.weight: [num_kv_heads * head_dim, hidden_size]
model.layers.{i}.self_attn.v_proj.weight: [num_kv_heads * head_dim, hidden_size]
model.layers.{i}.self_attn.o_proj.weight: [hidden_size, num_heads * head_dim]
model.layers.{i}.self_attn.q_norm.weight: [head_dim]
model.layers.{i}.self_attn.k_norm.weight: [head_dim]
model.layers.{i}.mlp.gate_proj.weight: [intermediate_size, hidden_size]
model.layers.{i}.mlp.up_proj.weight: [intermediate_size, hidden_size]
model.layers.{i}.mlp.down_proj.weight: [hidden_size, intermediate_size]
model.layers.{i}.input_layernorm.weight: [hidden_size]
model.layers.{i}.post_attention_layernorm.weight: [hidden_size]

model.norm.weight: [hidden_size]

text_projection.linear_fc1.weight: [intermediate_size, text_hidden_size]
text_projection.linear_fc1.bias: [intermediate_size]
text_projection.linear_fc2.weight: [hidden_size, intermediate_size]
text_projection.linear_fc2.bias: [hidden_size]

codec_head.weight: [vocab_size, hidden_size]
```

### Code Predictor Weights

```
code_predictor.model.codec_embedding.{i}.weight: [vocab_size, talker_hidden_size]  # i = 0..14

# Per layer (num_hidden_layers times):
code_predictor.model.layers.{i}.self_attn.q_proj.weight: [num_heads * head_dim, hidden_size]
# ... similar to talker layers

code_predictor.model.norm.weight: [hidden_size]

code_predictor.lm_head.{i}.weight: [vocab_size, hidden_size]  # i = 0..14

# Optional projection (when talker_hidden_size != code_predictor_hidden_size)
code_predictor.small_to_mtp_projection.weight: [code_predictor_hidden_size, talker_hidden_size]
code_predictor.small_to_mtp_projection.bias: [code_predictor_hidden_size]
```

### Speech Tokenizer Decoder Weights

```
# Transformer
decoder.pre_transformer.input_proj.weight: [hidden_size, latent_dim]
decoder.pre_transformer.output_proj.weight: [latent_dim, hidden_size]
decoder.pre_transformer.layers.{i}.*  # Similar to talker layers
decoder.pre_transformer.norm.weight: [hidden_size]

# Quantizer
decoder.quantizer.rvq_first.input_proj.weight: [dim, kernel, input_dim]  # Conv1d MLX format
decoder.quantizer.rvq_first.output_proj.weight: [output_dim, kernel, dim]
decoder.quantizer.rvq_first.vq.layers.{i}.codebook.embed.weight: [codebook_size, codebook_dim]
# Similar for rvq_rest

# Pre-conv
decoder.pre_conv.conv.weight: [latent_dim, kernel, codebook_dim]  # Conv1d MLX format

# Upsample blocks
decoder.upsample.{i}.0.conv.weight: [latent_dim, kernel, latent_dim]  # ConvTranspose1d
decoder.upsample.{i}.1.*  # ConvNeXt block weights

# Decoder blocks
decoder.decoder.0.conv.weight: [decoder_dim, kernel, latent_dim]  # Initial conv
decoder.decoder.{1-4}.block.*  # DecoderBlock weights (snake + upsample + residuals)
decoder.decoder.5.alpha, decoder.decoder.5.beta: [output_dim]  # Output snake
decoder.decoder.6.conv.weight: [1, kernel, output_dim]  # Output conv
```

### Speaker Encoder Weights

```
blocks.0.conv.weight: [enc_channels[0], kernel, mel_dim]  # Initial TDNN

# SE-Res2Net blocks
blocks.{1-3}.tdnn1.conv.weight: [out, kernel, in]
blocks.{1-3}.res2net_block.blocks.{i}.conv.weight: [hidden, kernel, in]
blocks.{1-3}.tdnn2.conv.weight: [out, kernel, out]
blocks.{1-3}.se_block.conv1.weight: [se, kernel, in]
blocks.{1-3}.se_block.conv2.weight: [out, kernel, se]

mfa.conv.weight: [enc_channels[-1], kernel, enc_channels[-1]]
asp.tdnn.conv.weight: [attention_channels, kernel, channels*3]
asp.conv.weight: [channels, kernel, attention_channels]
fc.weight: [enc_dim, kernel, channels*2]  # Final projection
```

---

## Conv Weight Format Notes

**MLX Conv1d format:** `[out_channels, kernel_size, in_channels]`
**PyTorch Conv1d format:** `[out_channels, in_channels, kernel_size]`

**MLX ConvTranspose1d format:** `[out_channels, kernel_size, in_channels]`
**PyTorch ConvTranspose1d format:** `[in_channels, out_channels, kernel_size]`

The `sanitize()` methods handle these conversions automatically.
