# Qwen3-ForcedAligner Component Documentation

This document provides a comprehensive list of all components in the Qwen3-ForcedAligner-0.6B model for porting from Python MLX to Swift MLX.

## Model Overview

**Model:** Qwen3-ForcedAligner-0.6B
**Task:** Forced alignment (aligning text transcriptions with audio at word/character level)
**Architecture:** Encoder-decoder transformer with audio encoder and text decoder
**Supported Languages:** 11 languages (Chinese, English, Japanese, Korean, Arabic, German, French, Spanish, Portuguese, Indonesian, Italian)
**Max Audio Duration:** 5 minutes (300 seconds)

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Qwen3ForcedAligner                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Qwen3ASRForConditionalGeneration            │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │        Qwen3ASRThinkerForConditionalGeneration     │  │   │
│  │  │  ┌────────────────┐  ┌──────────────────────────┐  │  │   │
│  │  │  │ AudioEncoder   │  │  TextModel (Thinker)     │  │  │   │
│  │  │  │ (Qwen3ASR      │  │  (Qwen3ASRThinkerText    │  │  │   │
│  │  │  │  AudioEncoder) │  │   Model)                 │  │  │   │
│  │  │  └────────────────┘  └──────────────────────────┘  │  │   │
│  │  │                           ↓                        │  │   │
│  │  │                    ┌──────────────┐                │  │   │
│  │  │                    │   LM Head    │                │  │   │
│  │  │                    │ (classify_   │                │  │   │
│  │  │                    │  num output) │                │  │   │
│  │  │                    └──────────────┘                │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. Configuration Classes

### 1.1 Qwen3ASRConfig
Top-level configuration that wraps the thinker configuration.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `thinker_config` | dict/Qwen3ASRThinkerConfig | None | Configuration for the thinker model |
| `support_languages` | List[str] | None | List of supported languages |

### 1.2 Qwen3ASRThinkerConfig
Configuration for the thinker component (audio encoder + text model).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio_config` | dict/Qwen3ASRAudioEncoderConfig | None | Audio encoder configuration |
| `text_config` | dict/Qwen3ASRTextConfig | None | Text model configuration |
| `audio_token_id` | int | 151646 | Token ID for audio placeholder |
| `audio_start_token_id` | int | 151647 | Token ID for audio start |
| `user_token_id` | int | 872 | Token ID for user |
| `initializer_range` | float | 0.02 | Weight initialization std |

### 1.3 Qwen3ASRAudioEncoderConfig
Configuration for the audio encoder.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_mel_bins` | int | 128 | Number of mel frequency bins |
| `encoder_layers` | int | 32 | Number of transformer encoder layers |
| `encoder_attention_heads` | int | 20 | Number of attention heads |
| `encoder_ffn_dim` | int | 5120 | FFN intermediate dimension |
| `d_model` | int | 1280 | Model hidden dimension |
| `dropout` | float | 0.0 | Dropout probability |
| `attention_dropout` | float | 0.0 | Attention dropout |
| `activation_function` | str | "gelu" | Activation function |
| `activation_dropout` | float | 0.0 | Activation dropout |
| `scale_embedding` | bool | False | Scale embeddings by sqrt(d_model) |
| `max_source_positions` | int | 1500 | Max sequence length for positional embeddings |
| `n_window` | int | 100 | Chunk size for conv and flash attn |
| `output_dim` | int | 3584 | Output dimension (projects to text hidden size) |
| `n_window_infer` | int | 400 | Inference window size |
| `conv_chunksize` | int | 500 | Convolution chunk size |
| `downsample_hidden_size` | int | 480 | Conv2d hidden size |

### 1.4 Qwen3ASRTextConfig
Configuration for the text decoder model.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | int | 151936 | Vocabulary size |
| `hidden_size` | int | 4096 | Hidden dimension |
| `intermediate_size` | int | 22016 | MLP intermediate dimension |
| `num_hidden_layers` | int | 32 | Number of decoder layers |
| `num_attention_heads` | int | 32 | Number of attention heads |
| `num_key_value_heads` | int | 32 | Number of KV heads (for GQA) |
| `head_dim` | int | 128 | Dimension per attention head |
| `hidden_act` | str | "silu" | Activation function (SwiGLU) |
| `max_position_embeddings` | int | 128000 | Max position embeddings |
| `rms_norm_eps` | float | 1e-6 | RMSNorm epsilon |
| `rope_theta` | float | 5000000.0 | RoPE base frequency |
| `rope_scaling` | dict | None | MRoPE configuration with `mrope_section` |
| `attention_bias` | bool | False | Use bias in attention projections |
| `attention_dropout` | float | 0.0 | Attention dropout |

---

## 2. Audio Encoder Components

### 2.1 Qwen3ASRAudioEncoder
Main audio encoder that processes mel spectrogram input.

**Input:** `(batch, num_mel_bins, time)` mel spectrogram
**Output:** `(total_frames, output_dim)` audio embeddings

#### 2.1.1 Convolutional Downsampling Stack

```python
conv2d1 = nn.Conv2d(1, downsample_hidden_size, kernel_size=3, stride=2, padding=1)
conv2d2 = nn.Conv2d(downsample_hidden_size, downsample_hidden_size, kernel_size=3, stride=2, padding=1)
conv2d3 = nn.Conv2d(downsample_hidden_size, downsample_hidden_size, kernel_size=3, stride=2, padding=1)
```

- **Input shape:** `(batch, 1, mel_bins, time)`
- **Activation:** GELU after each conv
- **Total downsampling:** 8x in time dimension
- **Output projection:** `conv_out = Linear(downsample_hidden_size * flattened_freq, d_model)`

#### 2.1.2 SinusoidsPositionEmbedding
Sinusoidal positional embeddings (non-learnable).

```python
class SinusoidsPositionEmbedding:
    def __init__(self, length, channels, max_timescale=10000):
        # Precomputes sin/cos positional embeddings
        # Shape: (length, channels)
```

**Parameters:**
- `length`: max_source_positions (1500)
- `channels`: d_model (1280)
- `max_timescale`: 10000

#### 2.1.3 Qwen3ASRAudioEncoderLayer
Single transformer encoder layer.

```
┌─────────────────────────────────────┐
│  Input hidden_states                │
│           ↓                         │
│  self_attn_layer_norm (LayerNorm)   │
│           ↓                         │
│  self_attn (Qwen3ASRAudioAttention) │
│           ↓                         │
│      + residual                     │
│           ↓                         │
│  final_layer_norm (LayerNorm)       │
│           ↓                         │
│  fc1 (Linear → encoder_ffn_dim)     │
│           ↓                         │
│  activation_fn (GELU)               │
│           ↓                         │
│  fc2 (Linear → d_model)             │
│           ↓                         │
│      + residual                     │
│           ↓                         │
│  Output hidden_states               │
└─────────────────────────────────────┘
```

**Components:**
| Component | Type | Input Dim | Output Dim |
|-----------|------|-----------|------------|
| `self_attn_layer_norm` | LayerNorm | d_model | d_model |
| `self_attn` | Qwen3ASRAudioAttention | d_model | d_model |
| `final_layer_norm` | LayerNorm | d_model | d_model |
| `fc1` | Linear | d_model | encoder_ffn_dim |
| `fc2` | Linear | encoder_ffn_dim | d_model |

#### 2.1.4 Qwen3ASRAudioAttention
Multi-head self-attention for audio encoder.

```python
class Qwen3ASRAudioAttention:
    embed_dim = d_model  # 1280
    num_heads = encoder_attention_heads  # 20
    head_dim = embed_dim // num_heads  # 64

    q_proj = Linear(embed_dim, embed_dim, bias=True)
    k_proj = Linear(embed_dim, embed_dim, bias=True)
    v_proj = Linear(embed_dim, embed_dim, bias=True)
    out_proj = Linear(embed_dim, embed_dim, bias=True)
```

**Notes:**
- Uses `cu_seqlens` for variable-length sequence handling
- Non-causal attention (bidirectional)
- Scaling: `head_dim ** -0.5`

#### 2.1.5 Output Projection
```python
ln_post = LayerNorm(d_model)
proj1 = Linear(d_model, d_model)
act = GELU
proj2 = Linear(d_model, output_dim)
```

---

## 3. Text Model Components

### 3.1 Qwen3ASRThinkerTextModel
Transformer decoder model for processing text tokens merged with audio features.

#### 3.1.1 Embedding Layer
```python
embed_tokens = Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
```

#### 3.1.2 Qwen3ASRThinkerTextRotaryEmbedding (MRoPE)
Multimodal Rotary Position Embedding supporting 3D positions (temporal, height, width).

```python
class Qwen3ASRThinkerTextRotaryEmbedding:
    mrope_section = [24, 20, 20]  # dims for T, H, W

    def forward(self, x, position_ids):
        # position_ids: (3, batch, seq_len) for T, H, W dimensions
        # Returns: cos, sin for rotary embeddings
```

**Key method: `apply_interleaved_mrope`**
- Interleaves frequency components from 3 dimensions
- Pattern: [T,H,W,T,H,W,...] instead of [T,T,T,...,H,H,H,...,W,W,W,...]

#### 3.1.3 Qwen3ASRThinkerTextDecoderLayer
Single transformer decoder layer.

```
┌─────────────────────────────────────────┐
│  Input hidden_states                    │
│           ↓                             │
│  input_layernorm (RMSNorm)              │
│           ↓                             │
│  self_attn (Qwen3ASRThinkerTextAttn)    │
│           ↓                             │
│      + residual                         │
│           ↓                             │
│  post_attention_layernorm (RMSNorm)     │
│           ↓                             │
│  mlp (Qwen3ASRThinkerTextMLP)           │
│           ↓                             │
│      + residual                         │
│           ↓                             │
│  Output hidden_states                   │
└─────────────────────────────────────────┘
```

#### 3.1.4 Qwen3ASRThinkerTextAttention
Multi-head attention with Grouped Query Attention (GQA) and QK normalization.

```python
class Qwen3ASRThinkerTextAttention:
    head_dim = 128
    num_key_value_groups = num_attention_heads // num_key_value_heads
    scaling = head_dim ** -0.5

    q_proj = Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
    k_proj = Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
    v_proj = Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
    o_proj = Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)

    # QK Normalization (per-head RMSNorm)
    q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
    k_norm = RMSNorm(head_dim, eps=rms_norm_eps)
```

**Forward pass:**
1. Project Q, K, V
2. Apply Q/K normalization (per-head)
3. Reshape to `(batch, heads, seq, head_dim)`
4. Apply RoPE to Q and K
5. Expand K, V for GQA (`repeat_kv`)
6. Compute attention scores with scaling
7. Apply causal mask
8. Softmax + dropout
9. Attend to values
10. Output projection

#### 3.1.5 Qwen3ASRThinkerTextMLP (SwiGLU)
```python
class Qwen3ASRThinkerTextMLP:
    gate_proj = Linear(hidden_size, intermediate_size, bias=False)
    up_proj = Linear(hidden_size, intermediate_size, bias=False)
    down_proj = Linear(intermediate_size, hidden_size, bias=False)
    act_fn = SiLU

    def forward(self, x):
        return down_proj(act_fn(gate_proj(x)) * up_proj(x))
```

#### 3.1.6 Qwen3ASRThinkerTextRMSNorm / Qwen3ASRTextRMSNorm
```python
class RMSNorm:
    weight = Parameter(ones(hidden_size))
    variance_epsilon = eps  # 1e-6

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * rsqrt(variance + variance_epsilon)
        return weight * hidden_states.to(input_dtype)
```

### 3.2 Final Normalization and Output
```python
norm = RMSNorm(hidden_size, eps=rms_norm_eps)  # Final layer norm
```

---

## 4. Model Head

### 4.1 Classification Head (Forced Aligner)
For forced alignment, the LM head outputs timestamp classifications:

```python
# When model_type contains "forced_aligner":
lm_head = Linear(hidden_size, classify_num, bias=False)
```

**Note:** `classify_num` is the number of timestamp buckets (determined by `timestamp_segment_time`).

---

## 5. Utility Functions

### 5.1 rotate_half
```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return concat((-x2, x1), dim=-1)
```

### 5.2 apply_rotary_pos_emb
```python
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

### 5.3 repeat_kv (for GQA)
```python
def repeat_kv(hidden_states, n_rep):
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)
```

### 5.4 _get_feat_extract_output_lengths
Computes output length after audio convolutions:
```python
def _get_feat_extract_output_lengths(input_lengths):
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths
```

---

## 6. Inference Pipeline

### 6.1 Qwen3ForcedAligner Wrapper
High-level inference wrapper.

**Key attributes:**
- `model`: Qwen3ASRForConditionalGeneration
- `processor`: Qwen3ASRProcessor (tokenizer + feature extractor)
- `aligner_processor`: Qwen3ForceAlignProcessor (text preprocessing)
- `timestamp_token_id`: Token ID for timestamp markers
- `timestamp_segment_time`: Time resolution per timestamp bucket (ms)

### 6.2 Qwen3ForceAlignProcessor
Text preprocessing for different languages.

**Methods:**
- `tokenize_chinese_mixed(text)`: Handles CJK + Latin mixed text
- `tokenize_japanese(text)`: Uses `nagisa` for Japanese tokenization
- `tokenize_korean(text)`: Uses `soynlp.tokenizer.LTokenizer`
- `tokenize_space_lang(text)`: Space-separated languages (English, etc.)
- `encode_timestamp(text, language)`: Formats text with timestamp tokens
- `parse_timestamp(word_list, timestamp)`: Extracts timestamps from model output
- `fix_timestamp(data)`: Post-processing to fix non-monotonic timestamps (LIS-based)

### 6.3 Alignment Flow
```
1. Audio preprocessing:
   - Load audio (path/URL/base64/numpy)
   - Convert to mono 16kHz float32
   - Extract mel spectrogram features

2. Text preprocessing:
   - Tokenize text by language
   - Insert <timestamp> tokens between words/characters
   - Format: "<|audio_start|><|audio_pad|><|audio_end|>word1<timestamp><timestamp>word2<timestamp><timestamp>..."

3. Model forward:
   - Process audio through AudioEncoder
   - Merge audio embeddings with text embeddings
   - Run through TextModel decoder
   - Get logits from classification head

4. Post-processing:
   - Extract timestamp predictions (argmax of logits)
   - Map to milliseconds using timestamp_segment_time
   - Fix non-monotonic timestamps
   - Return word-level alignments
```

---

## 7. Data Types Summary

### 7.1 Input Types
| Input | Shape | Type |
|-------|-------|------|
| `input_features` | `(batch, mel_bins, time)` | float32 |
| `feature_attention_mask` | `(batch, time)` | int64 |
| `input_ids` | `(batch, seq_len)` | int64 |
| `attention_mask` | `(batch, seq_len)` | int64 |

### 7.2 Intermediate Shapes (0.6B model)
| Tensor | Shape | Notes |
|--------|-------|-------|
| Audio embeddings | `(total_frames, 3584)` | After audio encoder |
| Text embeddings | `(batch, seq_len, hidden_size)` | 1536 for 0.6B |
| Attention Q | `(batch, num_heads, seq_len, head_dim)` | 12 heads, 128 dim |
| Attention K/V | `(batch, num_kv_heads, seq_len, head_dim)` | 2 KV heads (GQA) |

### 7.3 Output Types
| Output | Type | Description |
|--------|------|-------------|
| `ForcedAlignItem` | dataclass | `text`, `start_time`, `end_time` |
| `ForcedAlignResult` | dataclass | List of `ForcedAlignItem` |

---

## 8. Special Tokens

| Token | ID | Purpose |
|-------|-----|---------|
| `<|audio_start|>` | 151647 | Marks start of audio features |
| `<|audio_pad|>` | 151646 | Audio feature placeholder |
| `<|audio_end|>` | 151648 | Marks end of audio features |
| `<timestamp>` | config.timestamp_token_id | Timestamp prediction position |
| `<|endoftext|>` | 151643 | End of text |
| `<|im_end|>` | 151645 | End of instruction |

---

## 9. Model Size Reference (0.6B)

| Component | Parameters | Notes |
|-----------|------------|-------|
| Audio Encoder | ~100M | 32 layers, d_model=1280 |
| Text Model | ~500M | 28 layers, hidden=1536 |
| Total | ~600M | |

### Text Model 0.6B Configuration
```python
hidden_size = 1536
intermediate_size = 8960
num_hidden_layers = 28
num_attention_heads = 12
num_key_value_heads = 2
head_dim = 128
```

---

## 10. Implementation Notes for Swift MLX

### 10.1 Key Considerations
1. **MRoPE Implementation**: The 3D positional encoding with interleaved frequencies is critical
2. **GQA**: Support for different Q and KV head counts
3. **QK Normalization**: Per-head RMSNorm on Q and K before attention
4. **Variable-length Audio**: Handle `cu_seqlens` for efficient batch processing
5. **Conv2d Stack**: Three conv layers with stride 2 for 8x downsampling
6. **SwiGLU**: MLP with gate/up projections multiplied before down projection

### 10.2 Activation Functions Required
- GELU (audio encoder)
- SiLU/Swish (text model MLP)

### 10.3 Normalization Layers Required
- LayerNorm (audio encoder)
- RMSNorm (text model)

### 10.4 Attention Variants
- Standard MHA (audio encoder, bias=True)
- GQA with QK norm (text model, bias=False)

---

## References

- **Model:** [Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)
- **GitHub:** [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
- **Blog:** [Qwen3-ASR & Qwen3-ForcedAligner](https://qwen.ai/blog?id=qwen3asr)
