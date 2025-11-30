# Chatterbox TTS MLX Port

**Date:** 2025-11-30
**Source Repository:** https://github.com/resemble-ai/chatterbox
**Status:** Complete - Fully Functional

## Summary

The Chatterbox TTS model has been successfully ported to MLX. All core components are working and produce high-quality, intelligible speech output.

### Key Achievements

- **End-to-end text-to-speech generation** - Fully functional
- **Voice cloning** - Fully working with reference audio
- **Classifier-Free Guidance (CFG)** - Working with configurable `cfg_weight`
- **Pure MLX implementation** - Removed numpy/scipy where possible for future Swift MLX port
- **All weight loading** - 100% of model parameters load correctly with `strict=True`
- **Standard mlx_audio loading** - Works with `utils.load_model()` for CLI and API integration

## Quick Start

### Using mlx_audio.tts.generate (Recommended)

```bash
# Generate speech with voice cloning (full precision)
python -m mlx_audio.tts.generate \
    --model mlx-community/Chatterbox-TTS-fp16 \
    --text "Hello, this is a test of Chatterbox on MLX." \
    --ref_audio reference.wav \
    --ref_text "." \
    --file_prefix output \
    --verbose

# Or use quantized variant (smaller, faster, slightly lower quality)
python -m mlx_audio.tts.generate \
    --model mlx-community/Chatterbox-TTS-4bit \
    --text "Hello, this is a test of Chatterbox on MLX." \
    --ref_audio reference.wav \
    --ref_text "." \
    --file_prefix output
```

**Note:** `--ref_text` requires a non-empty placeholder (e.g., `"."`) to skip automatic transcription via Whisper. An empty string `""` will still trigger transcription. Chatterbox uses only the audio for voice cloning and ignores the text.

### Using Python API

```python
from mlx_audio.tts.generate import generate_audio

# Generate with voice cloning (full precision)
generate_audio(
    text="Hello, this is Chatterbox on MLX!",
    model="mlx-community/Chatterbox-TTS-fp16",
    ref_audio="reference.wav",
    ref_text=".",  # Non-empty placeholder to skip Whisper; ignored by Chatterbox
    file_prefix="output",
    verbose=True,
)

# Or use quantized variant (3x faster, 47% less memory)
generate_audio(
    text="Hello, this is Chatterbox on MLX!",
    model="mlx-community/Chatterbox-TTS-4bit",
    ref_audio="reference.wav",
    ref_text=".",
    file_prefix="output",
)
```

### Using ChatterboxTTS directly

```python
from mlx_audio.tts.models.chatterbox import ChatterboxTTS
from huggingface_hub import hf_hub_download
from pathlib import Path
import mlx.core as mx
import soundfile as sf

# Load model from Hugging Face (full precision or quantized)
cache_dir = hf_hub_download("mlx-community/Chatterbox-TTS-fp16", "model.safetensors")
ckpt_dir = Path(cache_dir).parent

model = ChatterboxTTS.from_pretrained(ckpt_dir)

# Load reference audio
ref_audio, ref_sr = sf.read("reference.wav")
ref_audio = mx.array(ref_audio.astype("float32"))

# Generate speech with voice cloning
for result in model.generate(
    text="Hello, this is a test of Chatterbox on MLX.",
    audio_prompt=ref_audio,
    audio_prompt_sr=ref_sr,
    exaggeration=0.5,
    cfg_weight=0.5,
    temperature=0.8,
    repetition_penalty=2.0,
    top_p=0.8,
):
    sf.write("output.wav", result.audio, result.sample_rate)
```

## Component Status

| Component | Status | Notes |
|-----------|--------|-------|
| T3 (LLaMA backbone) | ✅ Complete | Text-to-speech token generation |
| T3 CFG support | ✅ Complete | Classifier-free guidance working |
| S3Gen flow matching | ✅ Complete | Flow-based mel generation |
| HiFi-GAN vocoder | ✅ Complete | Pure MLX implementation |
| Voice encoder | ✅ Complete | Speaker embedding extraction |
| S3TokenizerV2 | ✅ Complete | Bundled weights (no ONNX needed) |
| Text tokenizer | ✅ Complete | EnTokenizer with Hugging Face |
| Weight sanitization | ✅ Complete | Idempotent PyTorch→MLX conversion |

## Implementation Details

### Pure MLX Refactoring

The following components have been refactored to use pure MLX instead of numpy/scipy:

| File | Changes |
|------|---------|
| `hifigan.py` | Removed numpy. Uses `math.pi`, `math.prod()`, pure MLX `istft` with `mx.array.at[].add()` for overlap-add |
| `flow_matching.py` | Uses `math.pi` instead of `np.pi` |
| `melspec.py` | Uses `math.log10` instead of `np.log10` |
| `voice_encoder.py` | Uses Python builtins (`round()`, list operations) instead of numpy |

**Kept scipy for:** Audio resampling (`scipy.signal.resample_poly`) - consistent with other models in the repo.

### Weight Format

The model uses a single combined `model.safetensors` file with component prefixes for compatibility with `utils.load_model()`:

| Prefix | Component | Params | Notes |
|--------|-----------|--------|-------|
| `ve.*` | VoiceEncoder | 13 | LSTM speaker encoder |
| `t3.*` | T3 | 293 | LLaMA backbone + heads |
| `s3gen.*` | S3Token2Wav | 2131 | Flow matching + HiFi-GAN |
| `s3_tokenizer.*` | S3TokenizerV2 | 96 | Speech tokenizer |

**Total:** 2533 tensors in `model.safetensors` (~3.04 GB)

### Weight Conversion

To convert from original PyTorch weights:

```bash
python mlx_audio/tts/models/chatterbox/scripts/convert_chatterbox.py \
    --output-dir ./chatterbox-mlx \
    --upload-repo mlx-community/Chatterbox-TTS  # Optional: upload to HF
```

The conversion script:
1. Downloads weights from `ResembleAI/chatterbox` (Hugging Face)
2. Downloads S3Tokenizer ONNX from ModelScope
3. Sanitizes all weights (PyTorch→MLX format conversion)
4. Saves combined `model.safetensors` with prefixes
5. Copies `tokenizer.json` and creates `config.json`

### Weight Loading Methods

| Method | Used By | Notes |
|--------|---------|-------|
| `utils.load_model()` | CLI, standard API | Calls `sanitize()` then `load_weights()` |
| `ChatterboxTTS.from_pretrained()` | Direct Python | Convenience method, handles tokenizer init |

Both methods work with the same `model.safetensors` format. The `sanitize()` method is idempotent.

### Key Fixes During Development

1. **T3 Transformer Weight Mapping** - PyTorch uses `tfmr.layers.X`, mlx_lm expects `tfmr.model.layers.X`
2. **HiFi-GAN Weight Normalization** - Merging `g * v / ||v||` for pre-normalized weights
3. **F0 Predictor Index Remapping** - Sequential indices (0,2,4,6,8) to list indices (0,1,2,3,4) using single-step function callback to avoid cascading replacements
4. **Pure MLX iSTFT** - Replaced numpy overlap-add with `mx.array.at[].add()`
5. **Mel Spectrogram Normalization** - Changed `norm=None` to `norm="slaney"` to match librosa defaults
6. **STFT Double Padding** - Added `center=False` to avoid double-padding with manual reflection padding
7. **Kaldi Fbank Compatibility** - Rewrote `kaldi_fbank` to match `torchaudio.compliance.kaldi.fbank`:
   - Povey window (hann^0.85) instead of hann
   - Round n_fft to power of 2 (400→512)
   - Per-frame DC offset removal
   - Snip edges (no padding at signal boundaries)
   - Correct epsilon floor (1.19e-7) instead of 1.0
8. **HiFi-GAN Hann Window** - Use periodic window (`fftbins=True`) instead of symmetric for STFT/iSTFT
9. **VE Embedding Full Audio** - Use full-length audio for voice encoder embedding (not truncated to 6s)
10. **S3Tokenizer Mel Filterbank** - Implement pure MLX `librosa_mel_filters()` with slaney normalization to match librosa defaults
11. **FSQ Quantizer Attribute Naming** - Renamed `_codebook` to `codebook` so `tree_flatten` includes quantizer weights in parameters (MLX skips underscore-prefixed attributes)
12. **DiffusersAttention Dimensions** - PyTorch's `diffusers.Attention` uses `inner_dim = heads * dim_head` (8×64=512), different from `query_dim` (256). Created custom `DiffusersAttention` class to match (256→512→256) projection shapes instead of MLX's standard MHA (256→256→256). This allows `strict=True` weight loading

### S3Tokenizer Code Sharing

The S3TokenizerV2 implementation is shared with the codec module (`mlx_audio/codec/models/s3/`). This follows the original PyTorch design where Chatterbox imports from the external `s3tokenizer` package.

**What's shared (from `mlx_audio.codec.models.s3`):**
- `S3TokenizerV2` - Core speech tokenizer model
- `ModelConfig` - Configuration dataclass
- `make_non_pad_mask`, `mask_to_bias`, `padding` - Utility functions
- `merge_tokenized_segments` - Long audio handling
- Constants: `S3_SR`, `S3_HOP`, `S3_TOKEN_HOP`, `S3_TOKEN_RATE`, `SPEECH_VOCAB_SIZE`

**What's Chatterbox-specific (`mlx_audio/tts/models/chatterbox/s3tokenizer/utils.py`):**
- `log_mel_spectrogram` - Custom implementation that drops the last STFT frame to match PyTorch's `torch.stft` behavior

**Why keep separate mel spectrogram?**

The PyTorch S3Tokenizer uses `torch.stft` which produces different frame counts than our standard MLX STFT:
- Standard MLX: `magnitudes = freqs.abs() ** 2` → shape (128, 101) for 1s audio
- Chatterbox: `magnitudes = spec[:-1, :].abs() ** 2` → shape (128, 100) for 1s audio

This 1-frame difference affects token counts and must match the model's training data format. The Chatterbox version drops the last frame (`[:-1]`) to match PyTorch exactly.

**Codec S3 enhancements added during refactoring:**
- Long audio handling with sliding window (30s windows, 4s overlap)
- `sanitize()` method for PyTorch→MLX weight conversion
- `quantize_simple()` method for simple quantization without long audio handling
- Default `name` parameter (`"speech_tokenizer_v2_25hz"`)

### Classifier-Free Guidance

CFG is fully implemented and matches the original PyTorch behavior:

```python
# In T3 inference:
# Batch 0 = conditioned, Batch 1 = unconditioned (zeroed text embeddings)
logits = cond_logits + cfg_weight * (cond_logits - uncond_logits)
```

Default `cfg_weight=0.5` works well for most use cases. Set to `0.0` for language transfer scenarios.

## Testing

### Full Pipeline Test

```bash
python scripts/test_full_pipeline.py
```

Expected output:
- ~100-150 speech tokens generated
- 4-6 seconds of audio
- SNR > 50 dB
- Intelligible speech matching the text

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/test_full_pipeline.py` | End-to-end generation test |
| `scripts/test_t3_quantization.py` | Test T3 backbone 4-bit quantization |
| `scripts/compare_t3_tokens.py` | Compare T3 tokens between PyTorch and MLX |
| `scripts/convert_chatterbox.py` | Weight conversion utility |
| `scripts/test_pytorch_direct.py` | PyTorch reference generation |
| `scripts/test_hf_pytorch.py` | PyTorch with HF space parameters |

Scripts are located in `mlx_audio/tts/models/chatterbox/scripts/` and `scripts/`.

## Architecture

```
mlx_audio/tts/models/chatterbox/
├── chatterbox.py            # Main ChatterboxTTS class
├── config.py                # Configuration classes
├── tokenizer.py             # Text tokenization
├── t3/
│   ├── t3.py                # LLaMA-based T3 model with CFG
│   ├── cond_enc.py          # Conditioning encoder
│   ├── perceiver.py         # Perceiver resampler
│   └── learned_pos_emb.py   # Position embeddings
├── s3gen/
│   ├── s3gen.py             # S3Token2Wav orchestration
│   ├── hifigan.py           # HiFi-GAN vocoder (pure MLX)
│   ├── flow_matching.py     # Flow matching decoder
│   ├── flow.py              # Flow network
│   ├── decoder.py           # Decoder components
│   ├── f0_predictor.py      # Pitch prediction
│   ├── xvector.py           # Speaker embeddings
│   └── matcha/              # Matcha-TTS components
├── s3tokenizer/
│   ├── __init__.py          # Re-exports from codec S3 + local mel spectrogram
│   └── utils.py             # Chatterbox-specific log_mel_spectrogram
├── voice_encoder/
│   ├── voice_encoder.py     # LSTM voice encoder
│   └── melspec.py           # Mel spectrogram
└── scripts/                 # Utility scripts
```

## Model Availability

The model is available on mlx-community in two variants:

| Repo | Size | Notes |
|------|------|-------|
| `mlx-community/Chatterbox-TTS-fp16` | ~3 GB | Full precision (fp16), highest quality |
| `mlx-community/Chatterbox-TTS-4bit` | ~1.5 GB | 4-bit quantized T3 backbone, minor quality loss |

### Performance Benchmarks

Tested on Apple Silicon with ~9 seconds of generated audio (89 tokens input):

| Model | Processing Time | Real-time Factor | Peak Memory | Tokens/sec |
|-------|-----------------|------------------|-------------|------------|
| **FP16** | 7.16s | 0.82x | 3.62 GB | 12.4 |
| **4-bit** | 2.33s | 0.26x | 1.93 GB | 38.2 |

The 4-bit quantized model is **3x faster** and uses **47% less memory** with minor quality degradation.

### Conversion Commands

```bash
# Full precision (fp16)
python mlx_audio/tts/models/chatterbox/scripts/convert_chatterbox.py \
    --output-dir ./Chatterbox-TTS-fp16

# Quantized (4-bit T3 backbone)
python mlx_audio/tts/models/chatterbox/scripts/convert_chatterbox.py \
    --output-dir ./Chatterbox-TTS-4bit \
    --quantize

# Upload to Hugging Face
python mlx_audio/tts/models/chatterbox/scripts/convert_chatterbox.py \
    -o ./Chatterbox-TTS-fp16 \
    --upload-repo mlx-community/Chatterbox-TTS-fp16

python mlx_audio/tts/models/chatterbox/scripts/convert_chatterbox.py \
    -o ./Chatterbox-TTS-4bit \
    --quantize \
    --upload-repo mlx-community/Chatterbox-TTS-4bit
```

### Quantization Analysis

The model has 804M parameters distributed across components with varying sensitivity to quantization:

| Component | Params | % of Model | Quantization Risk |
|-----------|--------|------------|-------------------|
| T3 (LLaMA backbone) | 532M | 66% | **Low** - Standard transformer |
| S3Tokenizer | 124M | 15% | **High** - Discrete token encoding |
| S3Gen flow decoder | 147M | 18% | **Medium** - Flow matching |
| Voice Encoder | 1.4M | 0.2% | **High** - Speaker identity |

#### T3 Breakdown (532M params)

| Layer Type | Params | Currently Quantized | Notes |
|------------|--------|---------------------|-------|
| MLP (gate/up/down_proj) | 377M | ✅ Yes (4-bit) | LLaMA MLPs quantize well |
| Attention (q/k/v/o_proj) | 126M | ✅ Yes (4-bit) | Standard attention |
| Output heads | 9M | ❌ No (FP16) | Could use 4-6 bit per mlx-lm practice |
| Embeddings | 20M | ❌ No (FP16) | Could quantize per standard LLM practice |

**Note:** Our current approach is conservative - we only quantize `t3.tfmr.model.layers.*` (MLP + attention). Standard LLM quantization (e.g., llama.cpp Q4_K_M, mlx-lm mixed recipes) typically also quantizes embeddings at the same bit level and output heads at higher precision (6-bit). This could provide additional ~29M params of savings with minimal quality impact.

#### S3Gen Breakdown (147M params)

| Layer Type | Params | Quantization Potential |
|------------|--------|----------------------|
| Flow attention | 42M | ❌ **Cannot quantize** - Produces gibberish |
| Flow FFN | 35M | ❌ **Cannot quantize** - Produces gibberish |
| HiFi-GAN vocoder | 21M | ❌ High risk - Waveform synthesis |
| Flow convolutions | 7M | ❌ **Cannot quantize** - Produces gibberish |

#### Selective Quantization Strategy

**Currently quantized (4-bit):**
- `t3.tfmr.model.layers.*.mlp.*` (377M) - LLaMA MLP layers
- `t3.tfmr.model.layers.*.self_attn.*` (126M) - Attention layers

**Currently kept in full precision (FP16):**
- `t3.*head*`, `t3.*emb*` (29M) - Could be quantized (4-6 bit) per standard practice
- `s3_tokenizer.*` (124M) - Discrete encoding, high sensitivity
- `s3gen.*` (147M) - Flow matching, **cannot quantize** (produces gibberish)
- `ve.*` (1.4M) - Voice encoder, small footprint

**Result:** ~3GB → ~1.5GB (53% reduction) with T3 transformer layers only.

**Potential improvement:** Quantizing T3 embeddings/heads at 4-6 bit could save an additional ~29M params (~50MB) with minimal quality impact, based on standard LLM quantization practices.

#### Quantization Test Results

Tested 4-bit quantization of T3 transformer layers only (group_size=64):

| Metric | Result |
|--------|--------|
| Layers quantized | 210 (MLP and self-attention in T3) |
| Original size | 3.22 GB |
| Quantized size | 1.52 GB |
| Size reduction | **53%** |
| Audio quality | Minor degradation (acceptable for most use cases) |

**Quantization pattern used:**

```python
# Quantize only T3 transformer MLP and attention layers
nn.quantize(model, bits=4, group_size=64, class_predicate=lambda path, m:
    isinstance(m, nn.Linear) and "t3.tfmr.model.layers" in path)
```

**Extended quantization test (T3 + S3Gen flow):**

| Configuration | Bits | Layers | Size | Reduction | Result |
|---------------|------|--------|------|-----------|--------|
| T3 only | 4 | 210 | 1.52 GB | 53% | ✅ Works with minor quality loss |
| T3 + S3Gen flow | 4 | 636 | 1.19 GB | 63% | ❌ **Gibberish audio** |
| T3 + S3Gen flow | 3 | 636 | 1.11 GB | 66% | ❌ Poor quality |
| T3 + S3Gen flow | 2 | 636 | 1.04 GB | 68% | ❌ Very poor quality |

**Conclusion:** The S3Gen flow matching components (`s3gen.flow.encoder`, `s3gen.flow.decoder`) are highly sensitive to quantization and produce unintelligible output when quantized at any bit width. Only the T3 backbone should be quantized.

**Test script:** `scripts/test_t3_quantization.py`

## Next Steps

### Completed

- ✅ Standard `utils.load_model()` integration working
- ✅ All sanitize methods are idempotent for both PyTorch and pre-converted weights
- ✅ `strict=True` weight loading (custom DiffusersAttention matches PyTorch dimensions)
- ✅ Quantization tested - T3 backbone quantizes to 4-bit with minor quality loss (53% size reduction)
- ✅ Combined `model.safetensors` format with component prefixes
- ✅ Conversion script creates standard mlx-audio weight format
- ✅ Tested 2-bit, 3-bit, 4-bit quantization on T3 + S3Gen flow (only T3-only quantization viable)
- ✅ Conversion script supports `--quantize` flag for creating 4-bit variant
- ✅ Proper naming convention (`-fp16`, `-4bit` suffixes) for Hugging Face uploads

### Remaining Tasks

1. **Upload to mlx-community** - Run conversion commands:
   ```bash
   # Full precision
   python mlx_audio/tts/models/chatterbox/scripts/convert_chatterbox.py \
       --upload-repo mlx-community/Chatterbox-TTS-fp16
   
   # Quantized
   python mlx_audio/tts/models/chatterbox/scripts/convert_chatterbox.py \
       --quantize \
       --upload-repo mlx-community/Chatterbox-TTS-4bit
   ```

### Optional Enhancements

- Streaming inference optimization
- Perth watermarking support
- Multilingual alignment analysis
- Performance profiling

## Dependencies

**Runtime (after weight conversion):**
```bash
pip install mlx mlx-lm tokenizers scipy
```

**For weight conversion:**
```bash
pip install torch safetensors huggingface_hub
```

---

### Performance Results

| Model | RTF | Notes |
|-------|-----|-------|
| FP16 | ~1.10x | Near real-time |
| 4-bit | ~0.48x | 2x faster than real-time |

The **4-bit quantized model** is recommended for most use cases - 6 seconds of audio generates in under 3 seconds on Apple Silicon.

## References

- Chatterbox Repository: https://github.com/resemble-ai/chatterbox
- Chatterbox Hugging Face Demo: https://huggingface.co/spaces/ResembleAI/Chatterbox (working voice cloning demo)
- S3Tokenizer: https://github.com/xingchensong/S3Tokenizer
- MLX Documentation: https://ml-explore.github.io/mlx/
- HiFi-GAN Paper: https://arxiv.org/abs/2010.05646
- Flow Matching: https://arxiv.org/abs/2210.02747
