# Qwen3-ASR Model Components

This document details all components of the Qwen3-ASR (FunASR) model for porting from Python MLX to Swift MLX.

## Architecture Overview

Qwen3-ASR is an end-to-end speech recognition model with three main stages:

```
Audio Input (16kHz)
       │
       ▼
┌─────────────────────────────────┐
│    Audio Preprocessing          │
│    (Mel Spectrogram + LFR)      │
│    Output: (T, 560)             │
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│    SenseVoice Encoder           │
│    (SANM + FSMN)                │
│    Output: (T, 512)             │
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│    Audio Adaptor                │
│    (Downsample + Transform)     │
│    Output: (T/2, 1024)          │
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│    Qwen3 LLM Decoder            │
│    (28-layer Transformer)       │
│    Output: Text tokens          │
└─────────────────────────────────┘
```

---

## 1. Audio Preprocessing

**File:** `mlx_audio/stt/models/funasr/audio.py`

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `SAMPLE_RATE` | 16000 | Audio sample rate in Hz |
| `N_FFT` | 400 | FFT window size (25ms at 16kHz) |
| `HOP_LENGTH` | 160 | STFT hop length (10ms) |
| `N_MELS` | 80 | Number of mel filterbank channels |
| `LFR_M` | 7 | LFR frame stacking count |
| `LFR_N` | 6 | LFR subsampling factor |

### Functions

#### `log_mel_spectrogram`
- **Input:** Audio waveform (1D array)
- **Output:** Log mel spectrogram `(n_frames, 80)`
- **Operations:**
  1. Apply Hamming window (not Hann)
  2. Compute STFT
  3. Compute power spectrum (magnitude squared)
  4. Apply mel filterbank (80 filters, HTK scale, Slaney normalization)
  5. Apply log: `log(max(mel_spec, 1e-10))`

#### `apply_lfr` (Low Frame Rate)
- **Input:** Mel features `(T, 80)`
- **Output:** LFR features `(ceil(T/6), 560)`
- **Operations:**
  1. Pad left with `(7-1)//2 = 3` copies of first frame
  2. Pad right as needed for alignment
  3. Stack 7 consecutive frames
  4. Subsample by factor of 6
  5. Reshape to `(T_out, 80*7) = (T_out, 560)`

#### `apply_cmvn` (Cepstral Mean and Variance Normalization)
- **Input:** Features, optional precomputed mean/istd
- **Output:** Normalized features
- **Formula:** `(features + cmvn_mean) * cmvn_istd`
- **Note:** `cmvn_mean` is the negative mean (additive shift)

---

## 2. SenseVoice Encoder

**File:** `mlx_audio/stt/models/funasr/encoder.py`

### SenseVoiceEncoderConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_dim` | 560 | Input dimension (80 * 7) |
| `encoder_dim` | 512 | Encoder hidden dimension |
| `num_heads` | 4 | Attention heads |
| `ffn_dim` | 2048 | Feed-forward dimension |
| `kernel_size` | 11 | FSMN kernel size |
| `sanm_shift` | 0 | Asymmetric context shift |
| `num_encoders0` | 1 | Initial encoder layers |
| `num_encoders` | 49 | Main encoder layers |
| `num_tp_encoders` | 20 | Time-pooling encoder layers |
| `dropout` | 0.0 | Dropout rate |

### SenseVoiceEncoder

**Architecture:**
```
Input (T, 560)
    │
    ▼ scale by sqrt(512)
    │
    ▼ encoders0 (1 layer)    : 560 → 512
    │
    ▼ encoders (49 layers)   : 512 → 512
    │
    ▼ after_norm (LayerNorm)
    │
    ▼ tp_encoders (20 layers): 512 → 512
    │
    ▼ tp_norm (LayerNorm)
    │
Output (T, 512)
```

### EncoderLayerSANM

Pre-norm transformer block with SANM attention:

```
x ──► LayerNorm ──► SANM Attention ──► Dropout ──► (+) ──►
       │                                            ▲
       └────────────────────────────────────────────┘ (residual, only if in_size == size)

     ──► LayerNorm ──► FFN ──► (+) ──► output
           │                    ▲
           └────────────────────┘ (residual)
```

**Components:**
- `norm1`: `nn.LayerNorm(in_size)`
- `self_attn`: `MultiHeadedAttentionSANM`
- `norm2`: `nn.LayerNorm(size)`
- `feed_forward`: `PositionwiseFeedForward`
- `dropout`: `nn.Dropout`

### MultiHeadedAttentionSANM

Self-Attention with Memory (SANM) combining multi-head attention with FSMN.

**Parameters:**
- `n_head`: 4
- `in_feat`: Input dimension
- `n_feat`: Output dimension (512)
- `d_k`: 512 / 4 = 128 (head dimension)

**Layers:**
- `linear_q_k_v`: `nn.Linear(in_feat, n_feat * 3, bias=True)` - Combined Q/K/V projection
- `linear_out`: `nn.Linear(n_feat, n_feat, bias=True)` - Output projection
- `fsmn_block`: `nn.Conv1d(n_feat, n_feat, kernel_size=11, groups=n_feat, bias=False)` - Depthwise conv
- `dropout`: `nn.Dropout`

**Forward Pass:**
1. Project Q, K, V with combined linear
2. Apply FSMN to V:
   - Transpose: `(B, T, D) → (B, D, T)`
   - Pad: left=5, right=5 (for kernel_size=11, sanm_shift=0)
   - Depthwise Conv1d
   - Add residual
   - Apply dropout and mask
3. Reshape for multi-head: `(B, T, H, D_k) → (B, H, T, D_k)`
4. Scaled dot-product attention: `scale = d_k^(-0.5) = 128^(-0.5)`
5. Reshape back: `(B, H, T, D_k) → (B, T, H*D_k)`
6. Output projection
7. **Add FSMN memory to attention output**

**FSMN Padding Calculation:**
```python
left_padding = (kernel_size - 1) // 2 + sanm_shift  # = 5 for default
right_padding = kernel_size - 1 - left_padding      # = 5 for default
```

### PositionwiseFeedForward

**Layers:**
- `w_1`: `nn.Linear(d_model, d_ff, bias=True)` - 512 → 2048
- `w_2`: `nn.Linear(d_ff, d_model, bias=True)` - 2048 → 512
- `dropout`: `nn.Dropout`

**Forward:** `w_2(dropout(relu(w_1(x))))`

---

## 3. Audio Adaptor

**File:** `mlx_audio/stt/models/funasr/adaptor.py`

### AudioAdaptorConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `downsample_rate` | 2 | Frame grouping factor |
| `encoder_dim` | 512 | Input dimension from encoder |
| `llm_dim` | 1024 | Output dimension for LLM |
| `ffn_dim` | 2048 | Intermediate projection dimension |
| `n_layer` | 2 | Number of transformer blocks |
| `attention_heads` | 8 | Attention heads in transformer blocks |
| `dropout` | 0.0 | Dropout rate |

### AudioAdaptor

**Architecture:**
```
Input (T, 512)
    │
    ▼ Pad to divisible by 2
    │
    ▼ Reshape: (T, 512) → (T/2, 1024)  [group 2 frames]
    │
    ▼ linear1: 1024 → 2048, ReLU
    │
    ▼ linear2: 2048 → 1024
    │
    ▼ 2x EncoderLayer (transformer blocks)
    │
Output (T/2, 1024)
```

**Layers:**
- `linear1`: `nn.Linear(encoder_dim * k, ffn_dim, bias=True)` - 1024 → 2048
- `linear2`: `nn.Linear(ffn_dim, llm_dim, bias=True)` - 2048 → 1024
- `blocks`: List of 2 `EncoderLayer`

### MultiHeadedAttention (Standard)

**Parameters:**
- `n_head`: 8
- `n_feat`: 1024
- `d_k`: 1024 / 8 = 128

**Layers:**
- `linear_q`: `nn.Linear(n_feat, n_feat, bias=True)`
- `linear_k`: `nn.Linear(n_feat, n_feat, bias=True)`
- `linear_v`: `nn.Linear(n_feat, n_feat, bias=True)`
- `linear_out`: `nn.Linear(n_feat, n_feat, bias=True)`
- `dropout`: `nn.Dropout`

**Forward:** Standard multi-head attention with scale = `d_k^(-0.5)`

### EncoderLayer (Adaptor)

Pre-norm transformer block:

**Layers:**
- `norm1`: `nn.LayerNorm(size)` - size=1024
- `self_attn`: `MultiHeadedAttention`
- `norm2`: `nn.LayerNorm(size)`
- `feed_forward`: `PositionwiseFeedForward(d_model=1024, d_ff=256)` - Note: d_ff = llm_dim // 4
- `dropout`: `nn.Dropout`

---

## 4. Qwen3 LLM Decoder

**File:** `mlx_audio/stt/models/funasr/qwen3.py`

### Qwen3Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vocab_size` | 151936 | Vocabulary size |
| `hidden_size` | 1024 | Model hidden dimension |
| `num_hidden_layers` | 28 | Number of transformer layers |
| `num_attention_heads` | 16 | Query attention heads |
| `num_key_value_heads` | 8 | Key/Value heads (GQA) |
| `intermediate_size` | 3072 | MLP intermediate dimension |
| `max_position_embeddings` | 40960 | Maximum sequence length |
| `rope_theta` | 1000000.0 | RoPE base frequency |
| `rms_norm_eps` | 1e-6 | RMSNorm epsilon |
| `tie_word_embeddings` | True | Share input/output embeddings |
| `head_dim` | 64 | Dimension per attention head |

### Qwen3ForCausalLM

**Components:**
- `model`: `Qwen3Model`
- `lm_head`: `nn.Linear(hidden_size, vocab_size, bias=False)` (only if not tied)

**Forward:**
- If `tie_word_embeddings=True`: Use `embed_tokens.as_linear(out)` for output projection

### Qwen3Model

**Components:**
- `embed_tokens`: `nn.Embedding(vocab_size, hidden_size)` - 151936 × 1024
- `layers`: List of 28 `TransformerBlock`
- `norm`: `nn.RMSNorm(hidden_size, eps=1e-6)` - Final layer norm

**Forward:**
1. Get embeddings (from input_ids or input_embeddings)
2. Create causal mask if needed
3. Process through 28 transformer blocks
4. Apply final RMSNorm

### TransformerBlock

Pre-norm transformer block:

```
x ──► input_layernorm ──► self_attn ──► (+) ──►
       (RMSNorm)                          ▲
           └──────────────────────────────┘ (residual)

     ──► post_attention_layernorm ──► mlp ──► (+) ──► output
           (RMSNorm)                           ▲
               └───────────────────────────────┘ (residual)
```

**Components:**
- `input_layernorm`: `nn.RMSNorm(hidden_size, eps=1e-6)`
- `self_attn`: `Attention`
- `post_attention_layernorm`: `nn.RMSNorm(hidden_size, eps=1e-6)`
- `mlp`: `MLP`

### Attention (Qwen3)

**Grouped Query Attention (GQA) with QK-Normalization and RoPE**

**Parameters:**
- `dim`: 1024
- `n_heads`: 16 (query heads)
- `n_kv_heads`: 8 (key/value heads)
- `head_dim`: 64
- `scale`: 64^(-0.5) = 0.125

**Layers:**
- `q_proj`: `nn.Linear(dim, n_heads * head_dim, bias=False)` - 1024 → 1024
- `k_proj`: `nn.Linear(dim, n_kv_heads * head_dim, bias=False)` - 1024 → 512
- `v_proj`: `nn.Linear(dim, n_kv_heads * head_dim, bias=False)` - 1024 → 512
- `o_proj`: `nn.Linear(n_heads * head_dim, dim, bias=False)` - 1024 → 1024
- `q_norm`: `nn.RMSNorm(head_dim, eps=1e-6)` - Per-head Q normalization
- `k_norm`: `nn.RMSNorm(head_dim, eps=1e-6)` - Per-head K normalization
- `rope`: `nn.RoPE(head_dim, traditional=False, base=1000000.0)`

**Forward:**
1. Project Q, K, V
2. Reshape: `(B, L, dim) → (B, n_heads, L, head_dim)`
3. Apply per-head RMSNorm to Q and K (Qwen3-specific)
4. Apply RoPE (with offset for cached positions)
5. Concatenate with KV cache if present
6. Scaled dot-product attention
7. Reshape and output projection

**KV Cache Format:** Tuple of (keys, values), each `(B, n_kv_heads, L, head_dim)`

### MLP (SwiGLU)

**Layers:**
- `gate_proj`: `nn.Linear(dim, hidden_dim, bias=False)` - 1024 → 3072
- `up_proj`: `nn.Linear(dim, hidden_dim, bias=False)` - 1024 → 3072
- `down_proj`: `nn.Linear(hidden_dim, dim, bias=False)` - 3072 → 1024

**Forward:** `down_proj(silu(gate_proj(x)) * up_proj(x))`

---

## 5. Main Model Integration

**File:** `mlx_audio/stt/models/funasr/funasr.py`

### FunASRConfig

Combines all component configs:
- `encoder`: `SenseVoiceEncoderConfig`
- `adaptor`: `AudioAdaptorConfig`
- `llm`: `Qwen3Config`

**Additional parameters:**
- `sample_rate`: 16000
- `n_mels`: 80
- `lfr_m`: 7
- `lfr_n`: 6
- `sos_token`: `"<|startofspeech|>"`
- `eos_token`: `"<|endofspeech|>"`
- `im_start_token`: `"<|im_start|>"`
- `im_end_token`: `"<|im_end|>"`
- `max_tokens`: 512
- `temperature`: 0.0

### Model Class

**Components:**
- `audio_encoder`: `SenseVoiceEncoder`
- `audio_adaptor`: `AudioAdaptor`
- `llm`: `Qwen3ForCausalLM`
- `_tokenizer`: HuggingFace tokenizer

**Inference Pipeline:**
1. `encode_audio()`: Audio → preprocessed features → encoder → adaptor → embeddings
2. `_prepare_prompt()`: Build chat template with audio embeddings
3. `_merge_embeddings()`: Insert audio embeddings at speech token positions
4. `stream_generate()`: Autoregressive token generation with KV caching

---

## 6. Layer Summary for Swift Implementation

### Normalization Layers

| Layer | Location | Parameters |
|-------|----------|------------|
| `nn.LayerNorm` | Encoder | `(dim,)` |
| `nn.RMSNorm` | Qwen3 LLM | `(dim, eps=1e-6)` |

### Linear Layers

| Layer | In | Out | Bias |
|-------|-----|-----|------|
| Encoder Q/K/V | in_feat | n_feat*3 | Yes |
| Encoder out | n_feat | n_feat | Yes |
| Encoder FFN w1 | d_model | d_ff | Yes |
| Encoder FFN w2 | d_ff | d_model | Yes |
| Adaptor Q/K/V/out | n_feat | n_feat | Yes |
| Adaptor linear1 | 1024 | 2048 | Yes |
| Adaptor linear2 | 2048 | 1024 | Yes |
| LLM Q proj | 1024 | 1024 | No |
| LLM K proj | 1024 | 512 | No |
| LLM V proj | 1024 | 512 | No |
| LLM O proj | 1024 | 1024 | No |
| LLM gate_proj | 1024 | 3072 | No |
| LLM up_proj | 1024 | 3072 | No |
| LLM down_proj | 3072 | 1024 | No |

### Convolutional Layers

| Layer | In Ch | Out Ch | Kernel | Groups | Bias |
|-------|-------|--------|--------|--------|------|
| FSMN block | 512 | 512 | 11 | 512 (depthwise) | No |

### Embeddings

| Layer | Vocab | Dim |
|-------|-------|-----|
| embed_tokens | 151936 | 1024 |

### Activation Functions

- **ReLU**: Encoder FFN, Adaptor linear1
- **SiLU (Swish)**: Qwen3 MLP gate

---

## 7. Weight Key Mapping

Weight keys in saved models follow this pattern:

```
audio_encoder.encoders0.0.norm1.weight
audio_encoder.encoders0.0.self_attn.linear_q_k_v.weight
audio_encoder.encoders0.0.self_attn.fsmn_block.conv.weight
audio_encoder.encoders.{0-48}.{...}
audio_encoder.tp_encoders.{0-19}.{...}
audio_encoder.after_norm.{weight,bias}
audio_encoder.tp_norm.{weight,bias}

audio_adaptor.linear1.{weight,bias}
audio_adaptor.linear2.{weight,bias}
audio_adaptor.blocks.{0,1}.norm1.{weight,bias}
audio_adaptor.blocks.{0,1}.self_attn.linear_{q,k,v,out}.{weight,bias}
audio_adaptor.blocks.{0,1}.feed_forward.w_{1,2}.{weight,bias}

llm.model.embed_tokens.weight
llm.model.layers.{0-27}.input_layernorm.weight
llm.model.layers.{0-27}.self_attn.{q,k,v,o}_proj.weight
llm.model.layers.{0-27}.self_attn.{q,k}_norm.weight
llm.model.layers.{0-27}.post_attention_layernorm.weight
llm.model.layers.{0-27}.mlp.{gate,up,down}_proj.weight
llm.model.norm.weight
```

---

## 8. Special Considerations for Swift Port

### Conv1d Weight Format
- PyTorch: `(out_channels, in_channels/groups, kernel_size)`
- MLX Python: Expects `(batch, seq, channels)` input format
- FSMN weights need special handling during conversion

### Attention Mask Format
- Use additive masks: `0.0` for valid, `-inf` for masked positions

### RoPE Implementation
- `traditional=False` (default)
- `base=1000000.0` for Qwen3

### KV Cache
- Format: List of 28 tuples, each `(keys, values)`
- Keys/Values shape: `(B, n_kv_heads, L, head_dim)`

### Quantization
- Supported: 4-bit and 8-bit
- Non-quantized components: normalization layers, embeddings, FSMN convolutions, audio encoder

---

## 9. Model Statistics

| Component | Parameters (approx) |
|-----------|---------------------|
| Audio Encoder | ~90M |
| Audio Adaptor | ~10M |
| Qwen3 LLM (28 layers) | ~500M |
| **Total** | ~600M |

**Layer Counts:**
- Encoder SANM layers: 1 + 49 + 20 = 70
- Adaptor transformer layers: 2
- LLM transformer layers: 28
