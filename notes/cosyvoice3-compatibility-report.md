# CosyVoice 3 MLX Compatibility Report

**Date:** December 2024
**Status:** Not Compatible - Significant Porting Work Required

## Executive Summary

CosyVoice 3 (Fun-CosyVoice3-0.5B-2512) introduces major architectural changes compared to CosyVoice 2, making it **incompatible** with the existing MLX port. The most significant change is replacing the U-Net style flow decoder with a **Diffusion Transformer (DiT)** architecture. Porting CosyVoice 3 to MLX will require approximately 1000-1150 lines of new code.

## Model Overview

### CosyVoice 3 Key Features
- 9 languages (Chinese, English, Japanese, Korean, German, Spanish, French, Italian, Russian)
- 18+ Chinese dialects/accents
- Bi-streaming support (text-in streaming + audio-out streaming)
- 150ms latency for streaming
- Pronunciation inpainting (Chinese Pinyin, English CMU phonemes)
- Instruct support (languages, dialects, emotions, speed, volume)

### Model Files (Fun-CosyVoice3-0.5B-2512)
```
├── llm.pt                    # Language model weights
├── llm.rl.pt                 # RL-tuned LLM weights (optional)
├── flow.pt                   # Flow/diffusion model weights
├── hift.pt                   # HiFi-GAN vocoder weights
├── speech_tokenizer_v3.onnx  # Speech tokenizer (v3, different from v2)
├── campplus.onnx             # Speaker encoder
├── cosyvoice3.yaml           # Model configuration
└── CosyVoice-BlankEN/        # Qwen2 text encoder
```

## Architecture Comparison

### 1. Language Model (LLM)

| Aspect | CosyVoice 2 | CosyVoice 3 |
|--------|-------------|-------------|
| Class | `Qwen2LM` | `CosyVoice3LM` (extends Qwen2LM) |
| Base Model | Qwen2 (0.5B) | Qwen2 (0.5B) - same |
| Speech Token Size | 6561 | 6561 - same |
| Special Tokens | +3 (eos, fill, extra) | +200 (extended vocabulary) |
| Special Token Embedding | Separate `llm_embedding` (2 tokens) | Unified `speech_embedding` |
| SOS Token | Index 0 in separate embedding | `speech_token_size + 0` |
| EOS Token | `speech_token_size` | `speech_token_size + 1` |
| Task ID Token | Index 1 in separate embedding | `speech_token_size + 2` |
| Fill Token | `speech_token_size + 2` | `speech_token_size + 3` |
| LLM Decoder | `Linear(896, 6564)` with bias | `Linear(896, 6761, bias=False)` |

**Code Diff (key changes):**
```python
# CosyVoice2
self.llm_embedding = nn.Embedding(2, llm_input_size)  # separate
self.speech_embedding = nn.Embedding(speech_token_size + 3, llm_input_size)
self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 3)
self.sos = 0
self.task_id = 1
self.eos_token = speech_token_size
self.fill_token = speech_token_size + 2

# CosyVoice3
self.speech_embedding = nn.Embedding(speech_token_size + 200, llm_input_size)  # unified
self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 200, bias=False)
self.sos = speech_token_size + 0
self.eos_token = speech_token_size + 1
self.task_id = speech_token_size + 2
self.fill_token = speech_token_size + 3
```

### 2. Flow/Diffusion Model (MAJOR CHANGE)

| Aspect | CosyVoice 2 | CosyVoice 3 |
|--------|-------------|-------------|
| Main Class | `CausalMaskedDiffWithXvec` | `CausalMaskedDiffWithDiT` |
| Encoder | `UpsampleConformerEncoder` (6 blocks + 4 up blocks) | **None** - just `PreLookaheadLayer` |
| Decoder/Estimator | U-Net style `ConditionalCFM` | **DiT** (Diffusion Transformer) |
| Token Processing | Conformer encoder with upsampling | Simple `repeat_interleave` |
| CFM Class | `CausalConditionalCFM` | `CausalConditionalCFM` (same algorithm) |

#### DiT Architecture (NEW)
```yaml
# From cosyvoice3.yaml
estimator: DiT
  dim: 1024
  depth: 22
  heads: 16
  dim_head: 64
  ff_mult: 2
  mel_dim: 80
  mu_dim: 80
  spk_dim: 80
  out_channels: 80
  static_chunk_size: 50  # chunk_size * token_mel_ratio
  num_decoding_left_chunks: -1
```

#### DiT Components to Implement
1. **`DiT`** - Main transformer backbone
   - `TimestepEmbedding` - Sinusoidal + MLP for time conditioning
   - `InputEmbedding` - Combines x, cond, mu, spks
   - `RotaryEmbedding` - From x_transformers
   - 22x `DiTBlock` layers
   - `AdaLayerNormZero_Final` - Final output normalization

2. **`DiTBlock`** - Transformer block with adaptive layer norm
   - `AdaLayerNormZero` - Produces shift/scale/gate for attention and FFN
   - `Attention` with `AttnProcessor`
   - `FeedForward` with GELU activation

3. **`InputEmbedding`** - Input projection
   - Projects concatenated [x, cond, mu, spks] to model dim
   - `CausalConvPositionEmbedding` for positional encoding

4. **Supporting Modules**
   - `SinusPositionEmbedding`
   - `CausalConvPositionEmbedding`
   - `GRN` (Global Response Normalization)
   - `ConvNeXtV2Block` (optional, for text embedding)

#### PreLookaheadLayer (Simple)
```python
class PreLookaheadLayer(nn.Module):
    def __init__(self, in_channels: int, channels: int, pre_lookahead_len: int = 3):
        self.conv1 = nn.Conv1d(in_channels, channels, kernel_size=pre_lookahead_len + 1)
        self.conv2 = nn.Conv1d(channels, in_channels, kernel_size=3)

    def forward(self, inputs, context=None):
        # Causal convolution with optional lookahead context
        # Returns: inputs + conv_output (residual)
```

### 3. Vocoder (HiFi-GAN)

| Aspect | CosyVoice 2 | CosyVoice 3 |
|--------|-------------|-------------|
| Class | `HiFTGenerator` | `CausalHiFTGenerator` |
| Convolutions | Standard Conv1d | `CausalConv1d` throughout |
| Upsampling | `ConvTranspose1d` | `CausalConv1dUpsample` |
| Downsampling | `Conv1d` | `CausalConv1dDownSample` |
| F0 Predictor | `ConvRNNF0Predictor` | `CausalConvRNNF0Predictor` |
| Sample Rate | 24kHz | 24kHz - same |
| Upsample Rates | [8, 5, 3] | [8, 5, 3] - same |

#### New Causal Convolution Classes
```python
# From cosyvoice/transformer/convolution.py
class CausalConv1d(nn.Module):
    # Supports 'left' and 'right' causal types
    # Left: standard causal (past only)
    # Right: lookahead (future context)

class CausalConv1dDownSample(nn.Module):
    # Causal downsampling

class CausalConv1dUpsample(nn.Module):
    # Causal upsampling (no ConvTranspose, uses interpolation + conv)
```

#### CausalConvRNNF0Predictor
```python
class CausalConvRNNF0Predictor(nn.Module):
    def __init__(self, num_class=1, in_channels=80, cond_channels=512):
        self.condnet = nn.Sequential(
            CausalConv1d(in_channels, cond_channels, kernel_size=4, causal_type='right'),
            nn.ELU(),
            CausalConv1d(cond_channels, cond_channels, kernel_size=3, causal_type='left'),
            nn.ELU(),
            # ... 3 more CausalConv1d layers
        )
        self.classifier = nn.Linear(cond_channels, num_class)
```

### 4. Speech Tokenizer

- **CosyVoice 2:** `speech_tokenizer_v2.onnx` (or similar)
- **CosyVoice 3:** `speech_tokenizer_v3.onnx` (new version)

Both use ONNX format and can likely use the same loading infrastructure, but the v3 model may have different input/output specifications.

## Porting Effort Estimate

### New Components Required

| Component | Lines of Code | Difficulty | Priority |
|-----------|---------------|------------|----------|
| `DiT` (main class) | ~100 | High | Critical |
| `DiTBlock` | ~50 | High | Critical |
| `Attention` + `AttnProcessor` | ~80 | Medium | Critical |
| `AdaLayerNormZero` / `AdaLayerNormZero_Final` | ~40 | Medium | Critical |
| `TimestepEmbedding` | ~30 | Low | Critical |
| `InputEmbedding` | ~40 | Medium | Critical |
| `CausalConvPositionEmbedding` | ~30 | Low | Critical |
| `FeedForward` | ~20 | Low | Critical |
| `GRN` | ~15 | Low | Critical |
| `PreLookaheadLayer` | ~30 | Low | Critical |
| `CosyVoice3LM` modifications | ~100 | Medium | Critical |
| `CausalConv1d` | ~40 | Medium | Critical |
| `CausalConv1dUpsample` | ~30 | Medium | Critical |
| `CausalConv1dDownSample` | ~30 | Medium | Critical |
| `CausalHiFTGenerator` | ~150 | Medium | Critical |
| `CausalConvRNNF0Predictor` | ~40 | Low | Critical |
| Weight conversion script | ~150 | Medium | Critical |
| Config/model loading | ~100 | Low | Critical |
| **Total** | **~1050-1150** | **Medium-High** | - |

### Reusable Components (No Changes Needed)

1. **Qwen2 LLM Backend** - Identical underlying model
2. **CFM Solver Algorithm** - Same `solve_euler` logic
3. **ONNX Loading** - Same infrastructure for tokenizer/speaker encoder
4. **Mel Spectrogram Processing** - Same parameters
5. **Sampling Functions** - `ras_sampling`, `nucleus_sampling`, `top_k_sampling`
6. **Basic Flow Matching** - `BASECFM` class

### Reusable with Modifications

1. **`Qwen2LM`** -> `CosyVoice3LM` (~30% new code)
2. **`HiFTGenerator`** -> `CausalHiFTGenerator` (~50% new code)
3. **`CosyVoice2ConditionalCFM`** - Same algorithm, new estimator

## Implementation Roadmap

### Phase 1: DiT Core (Highest Priority)
1. Implement `SinusPositionEmbedding`
2. Implement `TimestepEmbedding`
3. Implement `CausalConvPositionEmbedding`
4. Implement `InputEmbedding`
5. Implement `AdaLayerNormZero` and `AdaLayerNormZero_Final`
6. Implement `FeedForward` and `GRN`
7. Implement `Attention` with `AttnProcessor`
8. Implement `DiTBlock`
9. Implement `DiT` main class

### Phase 2: Flow Module
1. Implement `PreLookaheadLayer`
2. Create `CausalMaskedDiffWithDiT` wrapper
3. Update CFM to use DiT estimator

### Phase 3: Vocoder
1. Implement `CausalConv1d`, `CausalConv1dUpsample`, `CausalConv1dDownSample`
2. Implement `CausalConvRNNF0Predictor`
3. Implement `CausalHiFTGenerator`

### Phase 4: LLM Updates
1. Modify `Qwen2LM` -> `CosyVoice3LM`
2. Update special token handling
3. Update embedding dimensions

### Phase 5: Integration
1. Weight conversion script
2. Config loading updates
3. End-to-end testing

## Technical Notes

### RotaryEmbedding
CosyVoice 3 uses `RotaryEmbedding` from `x_transformers`. MLX has built-in RoPE support that can be adapted:
```python
# x_transformers usage in DiT:
self.rotary_embed = RotaryEmbedding(dim_head)
rope = self.rotary_embed.forward_from_seq_len(seq_len)
# Then apply_rotary_pos_emb(query, freqs, scale)
```

### Attention Masking
DiT uses chunk-based attention masking for streaming:
```python
if streaming:
    attn_mask = add_optional_chunk_mask(x, mask, False, False, 0, static_chunk_size, -1)
else:
    attn_mask = add_optional_chunk_mask(x, mask, False, False, 0, 0, -1).repeat(...)
```

### Configuration
Key config values from `cosyvoice3.yaml`:
```yaml
sample_rate: 24000
llm_input_size: 896
llm_output_size: 896
spk_embed_dim: 192
token_frame_rate: 25
token_mel_ratio: 2
chunk_size: 25
speech_token_size: 6561
```

## Conclusion

Porting CosyVoice 3 to MLX is a substantial undertaking, primarily due to the DiT architecture replacement. The core CFM algorithm remains the same, but the neural network estimator is completely different.

**Recommendation:** If CosyVoice 3's improved quality is desired, allocate dedicated development time for the DiT implementation. Consider implementing DiT as a reusable module since it's becoming a popular architecture for diffusion models.

## References

- [CosyVoice 3 Paper](https://arxiv.org/abs/2505.17589)
- [CosyVoice GitHub](https://github.com/FunAudioLLM/CosyVoice)
- [Fun-CosyVoice3-0.5B-2512 Model](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- [DiT Paper (Scalable Diffusion Models with Transformers)](https://arxiv.org/abs/2212.09748)
