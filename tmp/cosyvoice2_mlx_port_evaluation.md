# CosyVoice2 MLX Port Evaluation

## What's Already Ported (from Chatterbox)

The mlx-audio codebase already has **most components** needed for CosyVoice2:

| Component | Location | Reusable? |
|-----------|----------|-----------|
| **S3 Tokenizer (FSQ)** | `mlx_audio/codec/models/s3/` | ✅ 100% |
| **UpsampleConformerEncoder** | `mlx_audio/tts/models/chatterbox/s3gen/transformer/` | ✅ 100% (parameterized) |
| **CausalMaskedDiffWithXvec** | `mlx_audio/tts/models/chatterbox/s3gen/flow.py` | ✅ 100% (parameterized) |
| **ConditionalCFM + Decoder** | `mlx_audio/tts/models/chatterbox/s3gen/flow_matching.py` | ✅ 100% |
| **HiFTGenerator (HiFi-GAN)** | `mlx_audio/tts/models/chatterbox/s3gen/hifigan.py` | ✅ 100% (parameterized) |
| **CAMPPlus (Speaker Encoder)** | `mlx_audio/tts/models/chatterbox/s3gen/xvector.py` | ✅ 100% |
| **Matcha CFM Components** | `mlx_audio/tts/models/chatterbox/s3gen/matcha/` | ✅ 100% |

**What's NOT ported yet:**
- Qwen2LM wrapper with speech token embeddings (~600-800 lines)
- Weight conversion script (~200-300 lines)
- Frontend/tokenizer integration (~200-300 lines)

**Recently Fixed (now fully parameterized):**
- `UpsampleConformerEncoder`: Added `num_up_blocks`, `pre_lookahead_len`, `upsample_stride` params
- `CausalMaskedDiffWithXvec`: Added `n_timesteps` parameter to `__init__` and `inference()`
- `HiFTGenerator`/`SineGen`/`SourceModuleHnNSF`: Added `use_interpolation` param for 24kHz support

---

## Executive Summary

CosyVoice2 is a streaming-capable TTS model from Alibaba with these main components:

```
┌──────────────────┐    ┌─────────────────────────┐    ┌───────────────────────┐    ┌─────────────────┐
│ Text Processing  │ → │ Qwen2 LLM (Speech LM)   │ → │ Flow Matching (CFM)   │ → │ HiFi-GAN        │
│ (Tokenizer)      │    │ Generates speech tokens │    │ Tokens → Mel          │    │ Mel → Waveform  │
└──────────────────┘    └─────────────────────────┘    └───────────────────────┘    └─────────────────┘
                               ↑                              ↑
                         S3 Speech Tokenizer           Speaker Embedding
                         (FSQ, already ported)         (CAMPPlus)
```

### Key Features
- **Streaming support**: Bidirectional streaming with interleaved text/speech tokens
- **Zero-shot voice cloning**: Clone voices from short audio prompts
- **Multilingual**: Chinese, English, Japanese, Korean support
- **High quality**: Human-parity naturalness with FSQ tokenization

---

## Component-by-Component Analysis

### 1. S3 Speech Tokenizer (FSQ) ✅ ALREADY PORTED

| Aspect | Status |
|--------|--------|
| Location | `mlx_audio/codec/models/s3/model_v2.py` |
| MLX Implementation | Complete with FSQCodebook, AudioEncoderV2, FSMN attention |
| Vocabulary | 6561 tokens (3^8) |
| Token Rate | 25 Hz |
| Sample Rate | 16 kHz |

The existing `S3TokenizerV2` matches CosyVoice2's `speech_tokenizer_v2.onnx`. Key components:
- `FSQCodebook`: Finite Scalar Quantization with 8-dimensional projection
- `AudioEncoderV2`: Conv1d + FSMN attention blocks with rotary embeddings
- `FSQVectorQuantization`: Encodes audio features to discrete tokens

**No additional work needed.**

---

### 2. Qwen2-based LLM ❌ NEEDS PORTING (Major Work)

**Source Files:**
- `cosyvoice/llm/llm.py` (612 lines)

**Key Classes to Port:**

```python
class Qwen2Encoder(torch.nn.Module):
    """Wraps Qwen2ForCausalLM with custom forward methods"""
    def __init__(self, pretrain_path):
        self.model = Qwen2ForCausalLM.from_pretrained(pretrain_path)

    def forward(self, xs, xs_lens):
        # Full sequence forward with attention mask

    def forward_one_step(self, xs, masks, cache=None):
        # Single step with KV cache for autoregressive generation

class Qwen2LM(TransformerLM):
    """Main LLM for CosyVoice2 speech token generation"""

    # Embeddings
    speech_embedding: nn.Embedding(speech_token_size + 3, llm_input_size)
    llm_embedding: nn.Embedding(2, llm_input_size)  # sos/eos, task_id

    # Output projection
    llm_decoder: nn.Linear(llm_output_size, speech_token_size + 3)

    # Special tokens
    sos_eos = 0      # Start/end of sequence
    task_id = 1      # Task identifier
    fill_token = 2   # Fill token for streaming

    # Inference methods
    def inference():           # Standard token-by-token generation
    def inference_bistream():  # Streaming with interleaved text/speech
```

**Key Parameters (CosyVoice2-0.5B):**
| Parameter | Value |
|-----------|-------|
| `llm_input_size` | 896 (Qwen2-0.5B hidden size) |
| `llm_output_size` | 896 |
| `speech_token_size` | 6561 |
| `mix_ratio` | [5, 15] (text:speech for streaming) |
| `stop_token_ids` | [6561, 6562, 6563] |

**Architecture Flow:**
```
Text tokens → Qwen2 embed_tokens → [sos_emb, text_emb, task_id_emb, speech_emb]
           → Qwen2 forward → llm_decoder → speech token logits → sampling
```

**Dependencies:**
- `mlx-lm` has Qwen2 support (can use `mlx_lm.models.qwen2`)
- Need custom wrapper for embedding injection and caching

**Effort Estimate:** ~800-1000 lines of MLX code

---

### 3. Flow Matching (CFM + Conformer) ✅ MOSTLY PORTED (Minor Work)

**Existing MLX Components** (in `mlx_audio/tts/models/chatterbox/s3gen/`):

| Component | File | Status |
|-----------|------|--------|
| `BASECFM` | `matcha/flow_matching.py` | ✅ Complete |
| `CFMParams` | `matcha/flow_matching.py` | ✅ Complete |
| `ConditionalCFM` | `flow_matching.py` | ✅ Complete |
| `CausalConditionalCFM` | `flow_matching.py` | ✅ Complete |
| `SinusoidalPosEmb` | `matcha/decoder.py` | ✅ Complete |
| `TimestepEmbedding` | `matcha/decoder.py` | ✅ Complete |
| `Block1D` | `matcha/decoder.py` | ✅ Complete |
| `ResnetBlock1D` | `matcha/decoder.py` | ✅ Complete |
| `Downsample1D` | `matcha/decoder.py` | ✅ Complete |
| `Upsample1D` | `matcha/decoder.py` | ✅ Complete |
| `CausalConv1d` | `decoder.py` | ✅ Complete |
| `CausalBlock1D` | `decoder.py` | ✅ Complete |
| `CausalResnetBlock1D` | `decoder.py` | ✅ Complete |
| `ConditionalDecoder` | `decoder.py` | ✅ Complete |
| `DiffusersAttention` | `matcha/transformer.py` | ✅ Complete |
| `BasicTransformerBlock` | `matcha/transformer.py` | ✅ Complete |
| `FeedForward` | `matcha/transformer.py` | ✅ Complete |

**What's Still Needed:**

1. **`LengthRegulator`** (~50 lines)
   - Simple duration-based upsampling (if needed)

**Already Available (now parameterized):**
- `CausalMaskedDiffWithXvec` - fully parameterized with `n_timesteps`, `vocab_size`, `input_size`, etc.
- `CausalConditionalCFM` - flow matching decoder
- `ConditionalDecoder` - U-Net estimator

**CosyVoice2-specific parameters** (all configurable):
```python
token_mel_ratio: int = 2    # 50Hz tokens → 100Hz mel
pre_lookahead_len: int = 3  # Streaming lookahead
vocab_size: int = 6561      # FSQ vocabulary
input_size: int = 512       # Encoder hidden size
output_size: int = 80       # Mel channels
n_timesteps: int = 10       # Diffusion steps (now parameterized)
```

**Effort Estimate:** ~50-100 lines (thin wrapper + length regulator if needed)

---

### 4. UpsampleConformerEncoder ✅ ALREADY PORTED

**Existing MLX Components** (in `mlx_audio/tts/models/chatterbox/s3gen/transformer/`):

| Component | File | Status |
|-----------|------|--------|
| `UpsampleConformerEncoder` | `upsample_encoder.py` | ✅ Complete (531 lines) |
| `PreLookaheadLayer` | `upsample_encoder.py` | ✅ Complete |
| `Upsample1D` | `upsample_encoder.py` | ✅ Complete |
| `ConformerEncoderLayer` | `encoder_layer.py` | ✅ Complete |
| `RelPositionMultiHeadedAttention` | `attention.py` | ✅ Complete |
| `MultiHeadedAttention` | `attention.py` | ✅ Complete |
| `ConvolutionModule` | `convolution.py` | ✅ Complete |
| `PositionwiseFeedForward` | `positionwise_feed_forward.py` | ✅ Complete |
| `RelPositionalEncoding` | `embedding.py` | ✅ Complete |
| `LinearNoSubsampling` | `subsampling.py` | ✅ Complete |
| `make_pad_mask` | `upsample_encoder.py` | ✅ Complete |
| `subsequent_chunk_mask` | `upsample_encoder.py` | ✅ Complete |
| `add_optional_chunk_mask` | `upsample_encoder.py` | ✅ Complete |

**Architecture matches CosyVoice2:**
- 6 encoder layers + 2x upsample + 4 encoder layers
- PreLookaheadLayer with pre_lookahead_len=3
- Relative position attention
- Static chunk masking for streaming

**Now fully parameterized** (Dec 2024):
- `num_up_blocks` - configurable post-upsample layers (default: 4)
- `pre_lookahead_len` - configurable lookahead window (default: 3)
- `upsample_stride` - configurable upsample factor (default: 2)

**Effort Estimate:** 0 lines - complete and parameterized

---

### 5. HiFi-GAN Vocoder ✅ FULLY PORTED (Parameterized)

**Existing MLX Components** (in `mlx_audio/tts/models/chatterbox/s3gen/hifigan.py`):

| Component | Status |
|-----------|--------|
| `HiFTGenerator` | ✅ Complete (800+ lines) |
| `Snake` activation | ✅ Complete |
| `ResBlock` | ✅ Complete |
| `SineGen` | ✅ Complete (with `use_interpolation` param) |
| `SourceModuleHnNSF` | ✅ Complete (with `use_interpolation` param) |
| `stft` / `istft` | ✅ Complete (pure MLX) |
| `hann_window_periodic` | ✅ Complete |
| `_linear_interpolate_1d_to_size` | ✅ Complete (for interpolation mode) |

**Configuration Comparison:**

| Parameter | CosyVoice2 | Chatterbox (existing) |
|-----------|------------|----------------------|
| Sample Rate | **24000 Hz** | 22050 Hz |
| Mel Bins | 80 | 80 |
| Upsample Rates | [10, 6, 4, 2] | [8, 8] |
| ISTFT n_fft | 16 | 16 |
| ISTFT hop_len | 4 | 4 |
| Base Channels | 512 | 512 |
| Harmonics | 8 | 8 |
| **use_interpolation** | **True** | False |

**The `HiFTGenerator` class accepts all these as constructor parameters!**

**Now fully parameterized** (Dec 2024):
- `use_interpolation` parameter added to `SineGen`, `SourceModuleHnNSF`, and `HiFTGenerator`
- When `use_interpolation=True`: Uses interpolation-based phase computation (for 24kHz)
- When `use_interpolation=False`: Uses direct cumsum phase computation (for 22050Hz)
- Both modes work through the same unified class - no code duplication

**Usage:**
```python
# Chatterbox (22050Hz) - default
hift = HiFTGenerator(sampling_rate=22050, use_interpolation=False, ...)

# CosyVoice2 (24000Hz)
hift = HiFTGenerator(sampling_rate=24000, use_interpolation=True, ...)
```

**Effort Estimate:** 0 lines - complete and parameterized

---

### 6. Speaker Encoder (CAMPPlus) 🟡 ALREADY AVAILABLE

**Status:** Chatterbox has CAMPPlus at `mlx_audio/tts/models/chatterbox/s3gen/xvector.py`

**CosyVoice2 Usage:**
- Input: 80-dim Fbank features at 16kHz
- Output: 192-dim speaker embedding
- Original uses ONNX (`campplus.onnx`)

**Integration:**
```python
# Extract speaker embedding
feat = kaldi.fbank(speech, num_mel_bins=80, sample_frequency=16000)
feat = feat - feat.mean(dim=0, keepdim=True)  # CMVN
embedding = campplus(feat)  # [1, 192]
```

**Effort Estimate:** ~50 lines for integration wrapper

---

### 7. Text Processing / Frontend ❌ NEEDS PORTING (Medium Work)

**Source Files:**
- `cosyvoice/cli/frontend.py` (216 lines)
- `cosyvoice/tokenizer/tokenizer.py` (280 lines)

**Components:**

```python
class CosyVoiceFrontEnd:
    """Text and audio preprocessing pipeline"""

    # Text tokenization
    tokenizer: QwenTokenizer  # Based on transformers AutoTokenizer

    # Speech processing (we have MLX versions)
    speech_tokenizer_session  # S3TokenizerV2 (ONNX → MLX)
    campplus_session          # CAMPPlus (ONNX → MLX)

    # Text normalization
    zh_tn_model: ZhNormalizer  # Chinese text normalization (wetext)
    en_tn_model: EnNormalizer  # English text normalization
    inflect_parser             # Number spelling

    def text_normalize(text, split=True):
        # Normalize and optionally split into sentences

    def frontend_zero_shot(tts_text, prompt_text, prompt_speech):
        # Prepare inputs for zero-shot synthesis

class QwenTokenizer:
    """Wrapper around HuggingFace tokenizer"""

    special_tokens = {
        'eos_token': '<|endoftext|>',
        'pad_token': '<|endoftext|>',
        'additional_special_tokens': [
            '<|im_start|>', '<|im_end|>', '<|endofprompt|>',
            '[breath]', '<strong>', '</strong>', '[noise]',
            '[laughter]', '[cough]', ...
        ]
    }
```

**Text Normalization Options:**
1. **ttsfrd** (Alibaba's TTS frontend) - Requires separate installation
2. **wetext** (Open source) - Pure Python, easier to integrate

**Dependencies:**
- `tiktoken`: BPE tokenization
- `transformers`: AutoTokenizer for Qwen
- `wetext` or `ttsfrd`: Text normalization
- `inflect`: Number to words conversion

**Simplification Option:**
For initial port, can skip text normalization and require pre-normalized input.

**Effort Estimate:** ~300-400 lines, mainly integration work

---

## Proposed Directory Structure

```
mlx_audio/tts/models/cosyvoice2/
├── __init__.py
├── config.py                    # ModelConfig, LLMConfig, FlowConfig
├── cosyvoice2.py               # Main CosyVoice2 class
│
├── llm/
│   ├── __init__.py
│   └── llm.py                  # Qwen2LM, Qwen2Encoder
│
├── flow/
│   ├── __init__.py
│   ├── flow.py                 # CausalMaskedDiffWithXvec
│   ├── flow_matching.py        # CausalConditionalCFM
│   ├── decoder.py              # ConditionalDecoder (U-Net)
│   └── length_regulator.py     # LengthRegulator
│
├── transformer/
│   ├── __init__.py
│   ├── upsample_encoder.py     # UpsampleConformerEncoder
│   ├── attention.py            # RelativeMultiHeadAttention
│   ├── convolution.py          # ConvolutionModule
│   ├── encoder_layer.py        # ConformerEncoderLayer
│   ├── embedding.py            # RelPositionalEncoding
│   └── subsampling.py          # LinearSubsampling
│
├── hifigan/
│   ├── __init__.py
│   └── generator.py            # HiFTGenerator (24kHz variant)
│
├── frontend/
│   ├── __init__.py
│   ├── frontend.py             # CosyVoiceFrontEnd
│   └── tokenizer.py            # QwenTokenizer wrapper
│
└── scripts/
    └── convert_cosyvoice2.py   # Weight conversion script
```

---

## Total Effort Estimate (REVISED)

| Component | Lines of Code | Status | Priority |
|-----------|--------------|--------|----------|
| LLM (Qwen2LM) | 600-800 | ❌ New | P0 |
| Flow Matching Wrapper | 50-100 | ✅ Mostly reuse (parameterized) | P0 |
| Transformer Encoder | 0 | ✅ Complete (parameterized) | - |
| HiFi-GAN 24kHz | 0 | ✅ Complete (parameterized) | - |
| Frontend | 200-300 | ❌ New (simplified) | P2 |
| Integration & Config | 300-400 | 🟡 New | P1 |
| Weight Conversion | 200-300 | ❌ New | P1 |
| **Total** | **~1350-2000** | - | - |

**Significant reduction from original 3000-4000 estimate due to existing components!**

**Verified working:** Chatterbox TTS generates audio correctly with parameterized components (tested Dec 2024).

---

## Critical Dependencies

| Dependency | Purpose | Status |
|------------|---------|--------|
| `mlx-lm` | Qwen2 model support | ✅ Available |
| `tiktoken` | BPE tokenization | ✅ Available |
| `transformers` | AutoTokenizer | ✅ Available |
| `wetext` | Text normalization | Optional |
| `einops` | Tensor operations | ✅ Available |

---

## Recommended Implementation Phases (REVISED)

### Phase 1: LLM Integration (Main Work)
1. Create `Qwen2Encoder` wrapper using `mlx-lm`'s Qwen2
2. Implement speech token embeddings and projection layers
3. Implement `inference()` with KV caching
4. Test standalone speech token generation

### Phase 2: Flow Matching Wrapper (Light Work)
1. Create `CausalMaskedDiffWithXvec` class (~150 lines)
   - Reuse existing `CausalConditionalCFM` decoder
   - Reuse existing `UpsampleConformerEncoder`
2. Add token embedding and speaker projection layers
3. Test mel spectrogram generation

### Phase 3: Integration
1. Create weight conversion script
2. Configure `HiFTGenerator` for 24kHz with `use_interpolation=True`
3. Implement `CosyVoice2` main class
4. End-to-end testing

### Phase 4: Frontend (Optional)
1. Add QwenTokenizer wrapper
2. Basic text normalization (or skip, require pre-normalized input)

### Phase 5: Optimization
1. Streaming inference support
2. Quantization support
3. Performance benchmarking

---

## Model Variants to Support

| Model | Parameters | Sample Rate | Notes |
|-------|------------|-------------|-------|
| CosyVoice2-0.5B | ~500M | 24kHz | Primary target |
| CosyVoice-300M | ~300M | 22kHz | Legacy v1 |
| CosyVoice-300M-SFT | ~300M | 22kHz | Speaker fine-tuned |
| CosyVoice-300M-Instruct | ~300M | 22kHz | Instruction-following |

**Recommendation:** Start with CosyVoice2-0.5B as it's the recommended model with best quality.

---

## Weight Conversion Notes

**PyTorch → MLX Mapping:**

```python
# LLM weights
"llm.model.model.embed_tokens" → "llm.llm.model.embed_tokens"
"llm.speech_embedding" → "llm.speech_embedding"
"llm.llm_embedding" → "llm.llm_embedding"
"llm.llm_decoder" → "llm.llm_decoder"

# Flow weights
"flow.input_embedding" → "flow.input_embedding"
"flow.encoder.*" → "flow.encoder.*"
"flow.decoder.*" → "flow.decoder.*"

# HiFi-GAN weights
"hift.conv_pre" → "hift.conv_pre"
"hift.ups.*" → "hift.ups.*"
"hift.resblocks.*" → "hift.resblocks.*"
```

**Conv1d Transposition:**
MLX Conv1d uses `(out_channels, kernel_size, in_channels)` vs PyTorch's `(out_channels, in_channels, kernel_size)`.

---

## References

- [CosyVoice GitHub](https://github.com/FunAudioLLM/CosyVoice)
- [CosyVoice 2 Paper (arXiv:2412.10117)](https://arxiv.org/abs/2412.10117)
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) - Flow matching reference
- [mlx-lm Qwen2](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm)
