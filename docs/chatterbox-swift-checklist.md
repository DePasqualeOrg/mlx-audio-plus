# Chatterbox Swift Port Checklist

**Date:** 2025-12-01
**Source:** MLX Python implementation at `mlx_audio/tts/models/chatterbox/`
**Reference:** [Chatterbox MLX Port Documentation](chatterbox-mlx-port.md)

This document provides a comprehensive checklist of all components that need to be ported from Python MLX to Swift MLX.

---

## Important Notes

### Weight Loading Simplification

The Python MLX port includes `sanitize()` methods that handle PyTorch → MLX weight conversion (transposing Conv1d weights, remapping LSTM keys, etc.). These were used during the initial conversion process.

**For the Swift port:**
- Pre-converted weights are already available on Hugging Face (`mlx-community/Chatterbox-TTS-fp16`, `mlx-community/Chatterbox-TTS-4bit`)
- The Swift implementation will load these pre-converted weights directly
- **The complex `sanitize()` logic is NOT needed in Swift** — weights are already in MLX format
- Only basic weight loading (from safetensors) is required

This simplifies the Swift port significantly. Items marked with ~~strikethrough~~ in the sanitize rows indicate they can be skipped.

### Weight File Structure

The Hugging Face models (`mlx-community/Chatterbox-TTS-fp16`, `mlx-community/Chatterbox-TTS-4bit`) contain a **single `model.safetensors` file** with all component weights using prefixes:

| Prefix | Component | Tensor Count |
|--------|-----------|--------------|
| `ve.*` | VoiceEncoder | 13 |
| `t3.*` | T3 | 293 |
| `s3gen.*` | S3Token2Wav | 2131 |
| `s3_tokenizer.*` | S3TokenizerV2 | 96 |
| **Total** | | **2533** |

**Implication for Swift:**
- The S3Tokenizer weights are bundled in the same safetensors file as Chatterbox
- When loading, filter weights by prefix to distribute to each component
- The S3Tokenizer Swift module should accept weights passed to it (not load its own file)
- Alternative: S3Tokenizer could also support standalone weight files for use with other models (CosyVoice, etc.)

### Scipy Dependencies

The Python implementation uses `scipy.signal.resample_poly()` for polyphase audio resampling in three places:

| Location | Function | Purpose |
|----------|----------|---------|
| [chatterbox.py:54](../mlx_audio/tts/models/chatterbox/chatterbox.py#L54) | `resample_audio()` | Resample input audio to S3 tokenizer rate (16kHz) and vocoder rate (24kHz) |
| [s3gen/s3gen.py:29](../mlx_audio/tts/models/chatterbox/s3gen/s3gen.py#L29) | `resample_audio()` | Resample reference audio for speaker conditioning |
| [voice_encoder/voice_encoder.py:405](../mlx_audio/tts/models/chatterbox/voice_encoder/voice_encoder.py#L405) | `embeds_from_wavs()` | Resample audio to voice encoder rate (16kHz) |

**What `resample_poly` does:**
- Polyphase resampling using an FIR anti-aliasing filter
- Resamples by rational factor `up/down` (e.g., 16000/44100 → 160/441 after GCD)
- Uses "edge" padding to handle boundary conditions

**Swift Options:**

1. **Accelerate/vDSP** (Recommended)
   - `vDSP_desamp` for downsampling with anti-aliasing
   - `vDSP_vgenp` for polynomial interpolation
   - May need to implement polyphase filtering manually using `vDSP_conv`

2. **AVFoundation**
   - `AVAudioConverter` can resample between formats
   - Higher-level API, less control over filter characteristics
   - May have different edge handling behavior

3. **Custom Implementation**
   - Implement polyphase FIR resampling directly in MLX Swift
   - Most control, but more work
   - Could use MLX's `conv1d` for the FIR filtering

4. **Require Pre-resampled Audio**
   - Simplest option: require input audio at correct sample rates
   - Document required rates: 16kHz for S3 tokenizer, 24kHz for S3Gen conditioning
   - Defer resampling to the caller (Swift app can use AVFoundation)

**Recommendation:** Start with option 4 (require pre-resampled audio) for initial port, then add AVFoundation-based resampling as a convenience feature. This avoids blocking the port on matching scipy's exact resampling behavior.

### Audio File Loading

The Python implementation uses `soundfile` to load audio files. In Swift, we can use Apple's native frameworks to load common audio formats and resample in one step.

**Supported Formats via AVFoundation:**
- WAV, AIFF, CAF (uncompressed)
- MP3, AAC, M4A (compressed)
- FLAC, ALAC (lossless compressed)
- Any format supported by Core Audio

**Swift Implementation Approach:**

```swift
import AVFoundation

func loadAudio(from url: URL, targetSampleRate: Double = 16000) throws -> MLXArray {
    let audioFile = try AVAudioFile(forReading: url)

    // Create output format at target sample rate, mono, float32
    let outputFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: targetSampleRate,
        channels: 1,
        interleaved: false
    )!

    // Create converter
    guard let converter = AVAudioConverter(from: audioFile.processingFormat, to: outputFormat) else {
        throw AudioError.converterCreationFailed
    }

    // Calculate output buffer size
    let inputLength = AVAudioFrameCount(audioFile.length)
    let ratio = targetSampleRate / audioFile.processingFormat.sampleRate
    let outputLength = AVAudioFrameCount(Double(inputLength) * ratio)

    // Allocate output buffer
    guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: outputLength) else {
        throw AudioError.bufferAllocationFailed
    }

    // Convert with resampling
    var error: NSError?
    let inputBlock: AVAudioConverterInputBlock = { inNumPackets, outStatus in
        // Read from file...
    }
    converter.convert(to: outputBuffer, error: &error, withInputFrom: inputBlock)

    // Convert to MLXArray
    let floatData = outputBuffer.floatChannelData![0]
    return MLXArray(Array(UnsafeBufferPointer(start: floatData, count: Int(outputBuffer.frameLength))))
}
```

**Benefits:**
- Single API handles both file loading AND resampling
- Supports all common audio formats out of the box
- Hardware-accelerated on Apple Silicon
- Handles stereo→mono conversion automatically
- No need for separate scipy-equivalent resampling

**This means:** The Swift port can accept any common audio format as reference audio, with automatic resampling to the required sample rates (16kHz for tokenizer/voice encoder, 24kHz for S3Gen conditioning).

### mlx_lm Dependency (LLaMA Model)

The T3 model uses `mlx_lm` for its LLaMA-520M backbone. The Python implementation imports:

```python
from mlx_lm.models.llama import Model as LlamaModel
from mlx_lm.models.llama import ModelArgs as LlamaModelConfig
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler, make_logits_processors
```

**Swift Options:**

1. **Use mlx-swift-lm** (Recommended)
   - [ml-explore/mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) - Official MLX Swift LLM library
   - Direct Swift equivalent of `mlx_lm`
   - Includes LLaMA model, KV caching, sampling utilities
   - Well-tested and maintained

2. **Port LLaMA directly**
   - Port only the specific LLaMA components needed for T3
   - More control, but more work
   - May be necessary if mlx-swift-lm doesn't expose needed APIs (e.g., input embeddings instead of token IDs)

**Key Components Needed from LLaMA:**
- `LlamaModel` - Forward pass with input embeddings (not token IDs)
- `ModelArgs` / `LlamaModelConfig` - Configuration (hidden_size=1024, num_layers=16, etc.)
- KV cache for autoregressive generation
- Sampler with temperature, top_p, repetition_penalty

**Note:** T3 doesn't use LLaMA's embedding layer or output head directly—it has its own text/speech embeddings and heads. The LLaMA backbone is used only for the transformer layers.

### Text Tokenizer (tokenizer.json)

The Python implementation uses Hugging Face's `tokenizers` library to load `tokenizer.json`:

```python
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("tokenizer.json")
```

**Swift Options:**

1. **swift-transformers Tokenizers module** (Recommended)
   - [huggingface/swift-transformers](https://github.com/huggingface/swift-transformers) includes a `Tokenizers` module
   - Native Swift implementation of HF tokenizers
   - Supports `tokenizer.json` format directly
   - Can import just the `Tokenizers` module without the full package

2. **Manual JSON parsing**
   - Parse `tokenizer.json` vocabulary manually
   - Chatterbox uses a simple character-level tokenizer
   - Special tokens: `[START]`, `[STOP]`, `[SPACE]`, `[UNK]`

**tokenizer.json Structure:**
```json
{
  "model": {
    "type": "BPE" or "WordLevel",
    "vocab": { "[START]": 0, "[STOP]": 1, ... }
  },
  "added_tokens": [...],
  ...
}
```

**Key Operations:**
- `encode(text)` → token IDs (replace spaces with `[SPACE]` first)
- `decode(ids)` → text (replace `[SPACE]` back to spaces)
- `token_to_id("[START]")` → SOT token ID
- `token_to_id("[STOP]")` → EOT token ID

### Testing Strategy

**Verification Approach:**

1. **Component-level testing**
   - Test each component in isolation against Python outputs
   - Save intermediate tensors from Python for comparison
   - Use small test inputs for quick iteration

2. **Key checkpoints to verify:**
   - Mel spectrogram output (compare against Python)
   - S3Tokenizer output tokens (should match exactly)
   - VoiceEncoder embeddings (should be close, may have small numerical differences)
   - T3 generated tokens (should match given same random seed)
   - Final audio waveform (listen test + compare spectrograms)

3. **Test data:**
   - Use the same reference audio files for Python and Swift
   - Save Python intermediate outputs as `.npy` or `.safetensors` for comparison
   - Consider creating a test script that exports all intermediate tensors

4. **Numerical tolerance:**
   - Expect exact match for integer outputs (tokens)
   - Allow small tolerance (1e-5 to 1e-3) for floating point due to:
     - Different FFT implementations
     - Float32 vs Float16 precision differences
     - Different operation ordering

---

## Overview

The Chatterbox TTS model consists of 4 major components:
1. **T3** - Text-to-speech token generator (LLaMA-520M backbone)
2. **S3Gen** - Speech token to waveform decoder (flow matching + HiFi-GAN)
3. **VoiceEncoder** - Speaker embedding extraction (LSTM-based)
4. **S3TokenizerV2** - Speech tokenization for reference audio

Total: ~2533 tensors, ~804M parameters

---

## 1. Main Model Orchestration

### 1.1 ChatterboxTTS (Model)
- **Python file:** [chatterbox.py](../mlx_audio/tts/models/chatterbox/chatterbox.py)
- **Status:** [ ] Not started

| Item | Description | Status |
|------|-------------|--------|
| `Model` class | Main TTS orchestrator | [ ] |
| `Conditionals` dataclass | T3 and S3Gen conditioning container | [ ] |
| `from_pretrained()` | Load model from safetensors | [ ] |
| ~~`sanitize()`~~ | ~~Route weight sanitization to components~~ | N/A |
| `load_weights()` | Load weights into components | [ ] |
| `prepare_conditionals()` | Extract speaker/emotion embeddings | [ ] |
| `generate()` | Main text-to-audio pipeline | [ ] |
| `resample_audio()` | Polyphase audio resampling | [ ] |
| `punc_norm()` | Punctuation normalization | [ ] |
| `drop_invalid_tokens()` | Filter SOS/EOS tokens | [ ] |

**Constants:**
- `S3_SR = 16000` (tokenizer sample rate)
- `S3GEN_SR = 24000` (output sample rate)
- `SPEECH_VOCAB_SIZE = 6561`
- `ENC_COND_LEN = 96000` (6 seconds @ 16kHz)
- `DEC_COND_LEN = 240000` (10 seconds @ 24kHz)

---

## 2. T3 (Text-to-Speech Token Generator)

### 2.1 T3 Model
- **Python file:** [t3/t3.py](../mlx_audio/tts/models/chatterbox/t3/t3.py)
- **Status:** [ ] Not started

| Item | Description | Status |
|------|-------------|--------|
| `T3` class | Main T3 model | [ ] |
| ~~`sanitize()`~~ | ~~Weight conversion (tfmr.* → tfmr.model.*)~~ | N/A |
| `prepare_conditioning()` | Process T3Cond into embeddings | [ ] |
| `prepare_input_embeds()` | Build [cond \| text \| speech] embeddings | [ ] |
| `__call__()` | Forward pass (training) | [ ] |
| `inference()` | Generation with KV caching + CFG | [ ] |

**Sub-components:**
- `self.tfmr` - LLaMA model (need to port or use mlx-swift-lm)
- `self.cond_enc` - T3CondEnc
- `self.text_emb` - Text token embeddings
- `self.speech_emb` - Speech token embeddings
- `self.text_pos_emb` - Learned position embeddings (text)
- `self.speech_pos_emb` - Learned position embeddings (speech)
- `self.text_head` - Text output projection
- `self.speech_head` - Speech output projection

### 2.2 T3 Conditioning Encoder
- **Python file:** [t3/cond_enc.py](../mlx_audio/tts/models/chatterbox/t3/cond_enc.py)
- **Status:** [ ] Not started

| Item | Description | Status |
|------|-------------|--------|
| `T3CondEnc` class | Conditioning encoder | [ ] |
| `T3Cond` dataclass | Conditioning inputs container | [ ] |
| `spkr_enc` | Speaker embedding projection | [ ] |
| `emotion_adv_fc` | Emotion exaggeration layer | [ ] |
| `perceiver` | Perceiver resampler for prompt | [ ] |

### 2.3 Perceiver Resampler
- **Python file:** [t3/perceiver.py](../mlx_audio/tts/models/chatterbox/t3/perceiver.py)
- **Status:** [ ] Not started

| Item | Description | Status |
|------|-------------|--------|
| `Perceiver` class | Variable→fixed length resampler | [ ] |
| `AttentionBlock` class | Cross-attention block | [ ] |
| `AttentionQKV` class | Multi-head attention primitive | [ ] |
| Learnable query tokens | 32 tokens × 1024 dim | [ ] |

### 2.4 Learned Position Embeddings
- **Python file:** [t3/learned_pos_emb.py](../mlx_audio/tts/models/chatterbox/t3/learned_pos_emb.py)
- **Status:** [ ] Not started

| Item | Description | Status |
|------|-------------|--------|
| `LearnedPositionEmbeddings` class | Learned positional encoding | [ ] |
| `get_fixed_embedding()` | Get embeddings for specific indices | [ ] |

---

## 3. S3Gen (Speech Token to Waveform)

### 3.1 S3Token2Wav
- **Python file:** [s3gen/s3gen.py](../mlx_audio/tts/models/chatterbox/s3gen/s3gen.py)
- **Status:** [ ] Not started

| Item | Description | Status |
|------|-------------|--------|
| `resample_audio()` | Scipy polyphase resampling wrapper | [ ] |
| `S3Token2Mel` class | Flow matching decoder (speaker_encoder + flow) | [ ] |
| `S3Token2Wav` class | Complete pipeline (S3Token2Mel + HiFi-GAN) | [ ] |
| `embed_ref()` | Speaker conditioning from reference | [ ] |
| `__call__()` | Generate mel + waveform | [ ] |
| `flow_inference()` | Token-to-mel only inference | [ ] |
| `hift_inference()` | Mel-to-wav only inference | [ ] |
| `inference()` | Full pipeline with source caching | [ ] |
| `trim_fade` | Fade-in window to reduce artifacts | [ ] |

### 3.2 Flow Network
- **Python file:** [s3gen/flow.py](../mlx_audio/tts/models/chatterbox/s3gen/flow.py)
- **Status:** [ ] Not started

| Item | Description | Status |
|------|-------------|--------|
| `CausalMaskedDiffWithXvec` class | Causal masked diffusion model | [ ] |
| `input_embedding` | Token embedding layer | [ ] |
| `spk_embed_affine_layer` | Speaker embedding projection | [ ] |
| `encoder` | UpsampleConformerEncoder | [ ] |
| `encoder_proj` | Encoder output projection | [ ] |
| `decoder` | CausalConditionalCFM | [ ] |
| `inference()` | Streaming inference | [ ] |

### 3.3 Flow Matching
- **Python file:** [s3gen/flow_matching.py](../mlx_audio/tts/models/chatterbox/s3gen/flow_matching.py)
- **Status:** [ ] Not started

| Item | Description | Status |
|------|-------------|--------|
| `CFM_PARAMS` | Default CFMParams instance | [ ] |
| `ConditionalCFM` class | Base conditional flow matching with CFG | [ ] |
| `CausalConditionalCFM` class | Causal CFM with fixed noise | [ ] |
| `rand_noise` | Pre-generated noise for determinism | [ ] |
| `__call__()` | Forward diffusion with caching | [ ] |
| `solve_euler()` | Euler solver with CFG | [ ] |

### 3.4 HiFi-GAN Vocoder
- **Python file:** [s3gen/hifigan.py](../mlx_audio/tts/models/chatterbox/s3gen/hifigan.py)
- **Status:** [ ] Not started

| Item | Description | Status |
|------|-------------|--------|
| `HiFTGenerator` class | HiFi-GAN with NSF vocoder | [ ] |
| `decode()` | Mel + source to waveform | [ ] |
| `inference()` | Inference-mode forward | [ ] |
| `_f0_upsample()` | F0 nearest-neighbor upsampling | [ ] |
| `_stft()` / `_istft()` | STFT wrapper methods | [ ] |
| `ResBlock` class | Residual block with Snake | [ ] |
| `Snake` class | Snake activation: x + (1/α) * sin²(αx) | [ ] |
| `SineGen` class | Sine wave generator for harmonics | [ ] |
| `_f02uv()` | F0 to voiced/unvoiced | [ ] |
| `SourceModuleHnNSF` class | Neural Source Filter module | [ ] |
| `stft()` | STFT function (standalone) | [ ] |
| `istft()` | iSTFT with overlap-add using `at[].add()` | [ ] |
| `hann_window_periodic()` | Periodic Hann window | [ ] |
| `get_padding()` | Padding for 'same' convolution | [ ] |

### 3.5 Conditional Decoder
- **Python file:** [s3gen/decoder.py](../mlx_audio/tts/models/chatterbox/s3gen/decoder.py)
- **Status:** [ ] Not started

| Item | Description | Status |
|------|-------------|--------|
| `CausalConv1d` class | Causal 1D convolution with left padding | [ ] |
| `CausalBlock1D` class | Causal block with LayerNorm | [ ] |
| `CausalResnetBlock1D` class | Causal ResNet block | [ ] |
| `DownBlock` class | Container for down block components | [ ] |
| `MidBlock` class | Container for mid block components | [ ] |
| `UpBlock` class | Container for up block components | [ ] |
| `ConditionalDecoder` class | Causal U-Net decoder for flow matching | [ ] |

### 3.6 Conformer Encoder
- **Python file:** [s3gen/transformer/upsample_encoder.py](../mlx_audio/tts/models/chatterbox/s3gen/transformer/upsample_encoder.py)
- **Status:** [ ] Not started

| Item | Description | Status |
|------|-------------|--------|
| `UpsampleConformerEncoder` class | Main encoder | [ ] |

### 3.7 Transformer Components
- **Python files:** [s3gen/transformer/](../mlx_audio/tts/models/chatterbox/s3gen/transformer/)
- **Status:** [ ] Not started

| Item | File | Status |
|------|------|--------|
| `MultiHeadedAttention` | attention.py | [ ] |
| `ConformerEncoderLayer` | encoder_layer.py | [ ] |
| `RelPositionalEncoding` | embedding.py | [ ] |
| `Swish`, `GLU` | activation.py | [ ] |
| `ConvolutionModule` | convolution.py | [ ] |
| `PositionwiseFeedForward` | positionwise_feed_forward.py | [ ] |
| `Conv2dSubsampling` | subsampling.py | [ ] |

### 3.8 F0 Predictor
- **Python file:** [s3gen/f0_predictor.py](../mlx_audio/tts/models/chatterbox/s3gen/f0_predictor.py)
- **Status:** [ ] Not started

| Item | Description | Status |
|------|-------------|--------|
| `ConvRNNF0Predictor` class | Pitch prediction network | [ ] |
| 5× Conv1d layers | Indices 0,2,4,6,8 with ELU | [ ] |
| Linear classifier | Final output layer | [ ] |

### 3.9 CAMPPlus Speaker Encoder
- **Python file:** [s3gen/xvector.py](../mlx_audio/tts/models/chatterbox/s3gen/xvector.py)
- **Status:** [ ] Not started

| Item | Description | Status |
|------|-------------|--------|
| `CAMPPlus` class | X-vector speaker encoder | [ ] |
| `inference()` | Extract speaker embedding | [ ] |
| `kaldi_fbank()` | Kaldi-compatible filterbank features | [ ] |
| `_povey_window()` | Povey window (hann^0.85) | [ ] |
| `_next_power_of_2()` | Round to next power of 2 | [ ] |
| `BasicResBlock` class | 2D residual block | [ ] |
| `FCM` class | Feature Channel Module | [ ] |
| `TDNNLayer` class | Time-Delay Neural Network layer | [ ] |
| `CAMLayer` class | Context Attentive Module | [ ] |
| `CAMDenseTDNNLayer` class | CAM Dense TDNN layer | [ ] |
| `CAMDenseTDNNBlock` class | Dense block with multiple layers | [ ] |
| `TransitLayer` class | Transition layer | [ ] |
| `DenseLayer` class | Final embedding layer | [ ] |
| `StatsPool` class | Statistics pooling | [ ] |
| `statistics_pooling()` | Mean + std pooling | [ ] |
| `get_nonlinear()` | Create activation layers from config string | [ ] |
| `conv1d_pytorch_format()` | Apply MLX Conv1d to PyTorch-format input | [ ] |

**Note:** xvector.py is a large file (~757 lines) with many subcomponents for the CAM++ speaker encoder architecture.

### 3.10 Mel Spectrogram (S3Gen)
- **Python file:** [s3gen/mel.py](../mlx_audio/tts/models/chatterbox/s3gen/mel.py)
- **Status:** [ ] Not started

| Item | Description | Status |
|------|-------------|--------|
| `mel_spectrogram()` | Compute mel-spectrogram | [ ] |
| `_reflect_pad_2d()` | Reflect padding for batch | [ ] |

**Parameters:** n_fft=1920, num_mels=80, hop_size=480, sample_rate=24000

### 3.11 Matcha-TTS Components
- **Python files:** [s3gen/matcha/](../mlx_audio/tts/models/chatterbox/s3gen/matcha/)
- **Status:** [ ] Not started

| Item | File | Description | Status |
|------|------|-------------|--------|
| `SinusoidalPosEmb` | decoder.py | Sinusoidal timestep embeddings | [ ] |
| `TimestepEmbedding` | decoder.py | MLP for timestep embedding | [ ] |
| `Block1D` | decoder.py | 1D conv block with GroupNorm | [ ] |
| `ResnetBlock1D` | decoder.py | 1D ResNet block with time embedding | [ ] |
| `Downsample1D` | decoder.py | Stride-2 downsampling | [ ] |
| `Upsample1D` | decoder.py | Transposed conv upsampling | [ ] |
| `CFMParams` | flow_matching.py | CFM configuration dataclass | [ ] |
| `BASECFM` | flow_matching.py | Base conditional flow matching | [ ] |
| `BasicTransformerBlock` | transformer.py | Self-attention + FFN block | [ ] |
| `DiffusersAttention` | transformer.py | Diffusers-style attention | [ ] |
| `FeedForward` | transformer.py | GELU feed-forward network | [ ] |

---

## 4. Voice Encoder

### 4.1 Voice Encoder Model
- **Python file:** [voice_encoder/voice_encoder.py](../mlx_audio/tts/models/chatterbox/voice_encoder/voice_encoder.py)
- **Status:** [ ] Not started

| Item | Description | Status |
|------|-------------|--------|
| `VoiceEncoder` class | Speaker embedding extractor | [ ] |
| `StackedLSTM` class | Multi-layer LSTM wrapper | [ ] |
| `__call__()` | Forward pass with L2 normalization | [ ] |
| `inference()` | Full utterance embedding with partials | [ ] |
| `embeds_from_mels()` | Embeddings from mel spectrograms | [ ] |
| `embeds_from_wavs()` | Extract embeddings from waveforms | [ ] |
| `utt_to_spk_embed()` | Aggregate utterance embeddings | [ ] |
| `voice_similarity()` | Cosine similarity for embeddings | [ ] |
| ~~`sanitize()`~~ | ~~PyTorch LSTM weight conversion~~ | N/A |
| ~~`sanitize_lstm_weights()`~~ | ~~Convert LSTM keys~~ | N/A |
| `get_num_wins()` | Calculate windowing | [ ] |
| `get_frame_step()` | Compute frame step | [ ] |

**Config (VoiceEncConfig):**
- `num_mels: 40`
- `sample_rate: 16000`
- `speaker_embed_size: 256`
- `ve_hidden_size: 256`
- `ve_partial_frames: 160`

### 4.2 Mel Spectrogram (Voice Encoder)
- **Python file:** [voice_encoder/melspec.py](../mlx_audio/tts/models/chatterbox/voice_encoder/melspec.py)
- **Status:** [ ] Not started

| Item | Description | Status |
|------|-------------|--------|
| `melspectrogram()` | Compute mel-spectrogram | [ ] |
| `kaldi_fbank()` | Kaldi-compatible filterbank | [ ] |
| `_povey_window()` | Povey window (hann^0.85) | [ ] |
| `_next_power_of_2()` | Round to next power of 2 | [ ] |

**Parameters:** n_fft=400→512, hop_size=160, num_mels=40, fmin=0, fmax=8000

---

## 5. S3 Tokenizer (Separate Module)

> **Note:** The S3 tokenizer will be ported as a **separate Swift module** since it is shared across multiple TTS models (Chatterbox, CozySpeech, etc.). This mirrors the Python architecture where `mlx_audio/codec/models/s3/` is shared.

**Python files:**
- [codec/models/s3/__init__.py](../mlx_audio/codec/models/s3/__init__.py) - Package exports and constants
- [codec/models/s3/model.py](../mlx_audio/codec/models/s3/model.py) - S3Tokenizer V1 (287 lines)
- [codec/models/s3/model_v2.py](../mlx_audio/codec/models/s3/model_v2.py) - S3TokenizerV2 (595 lines)
- [codec/models/s3/utils.py](../mlx_audio/codec/models/s3/utils.py) - Utility functions (150 lines)

**Swift target:** Separate S3Tokenizer module (dependency of Chatterbox)

### 5.1 Constants and Configuration
- **Status:** [ ] Not started

| Item | Description | Status |
|------|-------------|--------|
| `S3_SR` | Sample rate: 16000 | [ ] |
| `S3_HOP` | Hop length: 160 (100 frames/sec) | [ ] |
| `S3_TOKEN_HOP` | Token hop: 640 (25 tokens/sec) | [ ] |
| `S3_TOKEN_RATE` | Token rate: 25 Hz | [ ] |
| `SPEECH_VOCAB_SIZE` | Vocabulary: 6561 (3^8) | [ ] |
| `ModelConfig` dataclass | n_mels, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer | [ ] |

### 5.2 S3TokenizerV2 (Main Class)
- **Python file:** [model_v2.py](../mlx_audio/codec/models/s3/model_v2.py)
- **Status:** [ ] Not started

| Item | Description | Status |
|------|-------------|--------|
| `S3TokenizerV2` class | Main tokenizer class | [ ] |
| `__call__()` | Forward pass | [ ] |
| `quantize()` | Main quantization with long audio handling | [ ] |
| `quantize_simple()` | Simple quantization (no long audio) | [ ] |
| `_quantize_mixed_batch()` | Mixed short/long audio handling | [ ] |
| ~~`sanitize()`~~ | ~~PyTorch → MLX weight conversion~~ | N/A |
| `from_pretrained()` | Model loading from HF Hub | [ ] |

### 5.3 AudioEncoderV2
- **Python file:** [model_v2.py](../mlx_audio/codec/models/s3/model_v2.py)
- **Status:** [ ] Not started

| Item | Description | Status |
|------|-------------|--------|
| `AudioEncoderV2` class | Main encoder | [ ] |
| `conv1` | Conv1d(n_mels→n_state, k=3, stride=variable) | [ ] |
| `conv2` | Conv1d(n_state→n_state, k=3, stride=2) | [ ] |
| `_freqs_cis` | Precomputed rotary frequencies | [ ] |
| `blocks` | List of ResidualAttentionBlock | [ ] |

### 5.4 Attention Components (V2)
- **Python file:** [model_v2.py](../mlx_audio/codec/models/s3/model_v2.py)
- **Status:** [ ] Not started

| Item | Description | Status |
|------|-------------|--------|
| `FSMNMultiHeadAttention` class | FSMN-enhanced multi-head attention | [ ] |
| `query`, `key`, `value`, `out` | Linear projections | [ ] |
| `fsmn_block` | Conv1d for sequential memory | [ ] |
| `forward_fsmn()` | FSMN forward pass | [ ] |
| `qkv_attention()` | QKV attention with RoPE | [ ] |
| `ResidualAttentionBlock` class | Attention + MLP block | [ ] |
| `attn_ln`, `mlp_ln` | LayerNorm layers | [ ] |
| `mlp` | FFN (4x expansion) | [ ] |

### 5.5 Rotary Position Embeddings
- **Python file:** [model_v2.py](../mlx_audio/codec/models/s3/model_v2.py)
- **Status:** [ ] Not started

| Item | Description | Status |
|------|-------------|--------|
| `precompute_freqs_cis()` | Precompute RoPE frequencies | [ ] |
| `apply_rotary_emb()` | Apply RoPE to Q and K | [ ] |

### 5.6 FSQ Quantization (V2)
- **Python file:** [model_v2.py](../mlx_audio/codec/models/s3/model_v2.py)
- **Status:** [ ] Not started

| Item | Description | Status |
|------|-------------|--------|
| `FSQCodebook` class | Finite Scalar Quantization codebook | [ ] |
| `project_down` | Linear(dim→8) | [ ] |
| `preprocess()` | Reshape for quantization | [ ] |
| `encode()` | FSQ encoding (tanh + round + base-3) | [ ] |
| `FSQVectorQuantization` class | FSQ wrapper | [ ] |
| `encode()` | Full encoding pipeline | [ ] |
| `decode()` | Decode and rearrange | [ ] |

### 5.7 S3Tokenizer V1 (Legacy)
- **Python file:** [model.py](../mlx_audio/codec/models/s3/model.py)
- **Status:** [ ] Not started
- **Note:** V1 may not be needed for Chatterbox, but included for completeness

| Item | Description | Status |
|------|-------------|--------|
| `S3Tokenizer` class | V1 tokenizer | [ ] |
| `sinusoids()` | Sinusoidal positional embeddings | [ ] |
| `MultiHeadAttention` class | Basic multi-head attention | [ ] |
| `ResidualAttentionBlock` class | V1 attention block | [ ] |
| `AudioEncoder` class | V1 encoder | [ ] |
| `EuclideanCodebook` class | Euclidean distance codebook | [ ] |
| `VectorQuantization` class | V1 VQ wrapper | [ ] |

### 5.8 Utility Functions
- **Python file:** [utils.py](../mlx_audio/codec/models/s3/utils.py)
- **Status:** [ ] Not started

| Item | Description | Status |
|------|-------------|--------|
| `log_mel_spectrogram()` | Audio → log mel spectrogram | [ ] |
| `make_non_pad_mask()` | Create binary mask for padding | [ ] |
| `mask_to_bias()` | Convert mask to attention bias | [ ] |
| `padding()` | Batch-pad variable-length sequences | [ ] |
| `merge_tokenized_segments()` | Merge overlapping tokenized segments | [ ] |
| `fetch_from_hub()` | Download models from HF Hub | [ ] |

### 5.9 Chatterbox-specific Mel Spectrogram
- **Python file:** [s3tokenizer/utils.py](../mlx_audio/tts/models/chatterbox/s3tokenizer/utils.py)
- **Status:** [ ] Not started

| Item | Description | Status |
|------|-------------|--------|
| `log_mel_spectrogram()` | Log-mel with last frame dropped | [ ] |

**Note:** Drops last STFT frame to match PyTorch `torch.stft` behavior. This is Chatterbox-specific and may live in the Chatterbox module rather than the shared S3Tokenizer module.

### 5.10 Key Algorithms Summary

| Algorithm | Description | Used In |
|-----------|-------------|---------|
| Rotary Position Embeddings (RoPE) | Rotation-based positional encoding | V2 attention |
| FSMN | Feedforward Sequential Memory Network | V2 attention |
| Finite Scalar Quantization (FSQ) | Base-3 quantization to 6561 codes | V2 quantizer |
| Euclidean Codebook | L2 distance-based VQ | V1 quantizer |
| Sliding Window | 30s windows with 4s overlap | Long audio handling |

---

## 6. Text Tokenizer

### 6.1 EnTokenizer
- **Python file:** [tokenizer.py](../mlx_audio/tts/models/chatterbox/tokenizer.py)
- **Status:** [ ] Not started

| Item | Description | Status |
|------|-------------|--------|
| `EnTokenizer` class | English text tokenizer | [ ] |
| `text_to_tokens()` | Text → token IDs | [ ] |
| `encode()` | Encode with space handling | [ ] |
| `decode()` | Decode token IDs → text | [ ] |

**Special Tokens:**
- `[START]` - Start of text
- `[STOP]` - End of text
- `[UNK]` - Unknown token
- `[SPACE]` - Space token

**Note:** Uses Hugging Face `tokenizers` library in Python. Swift will need equivalent tokenizer loading.

---

## 7. Configuration Classes

### 7.1 T3Config
- **Python file:** [config.py](../mlx_audio/tts/models/chatterbox/config.py)
- **Status:** [ ] Not started

| Item | Description | Status |
|------|-------------|--------|
| `T3Config` class | T3 model configuration | [ ] |
| Text vocab size | 704 (English) | [ ] |
| Speech vocab size | 8194 | [ ] |
| LLaMA config | 520M model | [ ] |
| Speaker embed size | 256 | [ ] |

### 7.2 ModelConfig
- **Python file:** [config.py](../mlx_audio/tts/models/chatterbox/config.py)
- **Status:** [ ] Not started

| Item | Description | Status |
|------|-------------|--------|
| `ModelConfig` class | Main configuration | [ ] |
| Sample rates | S3=16000, S3Gen=24000 | [ ] |
| Conditioning lengths | 6s, 10s | [ ] |

---

## 8. Critical Implementation Notes for Swift

### 8.1 Audio Processing
| Item | Python Approach | Swift Considerations |
|------|-----------------|---------------------|
| Audio resampling | `scipy.signal.resample_poly` | Need polyphase resampling implementation |
| Audio file loading | `soundfile` | Use AVFoundation or similar |

### 8.2 Weight Loading
| Item | Python Approach | Swift Considerations |
|------|-----------------|---------------------|
| ~~LSTM weights~~ | ~~Custom `sanitize_lstm_weights()`~~ | **Not needed** - weights pre-converted |
| ~~Conv1d weights~~ | ~~Transpose (out, in, kernel) → (out, kernel, in)~~ | **Not needed** - weights pre-converted |
| Safetensors loading | Python safetensors | Built into mlx-swift |
| Weight distribution | Filter by prefix (`ve.*`, `t3.*`, etc.) | Same approach in Swift |

### 8.3 Kaldi Compatibility
| Item | Description |
|------|-------------|
| Povey window | `hann^0.85` instead of standard hann |
| Power-of-2 rounding | n_fft: 400 → 512 |
| Snip edges | No padding at signal boundaries |
| DC offset removal | Per-frame DC offset removal |

### 8.4 Special Algorithms
| Item | Description |
|------|-------------|
| Flow matching | Euler solver with cosine time scheduling |
| CFG formula | `logits = cond + cfg_weight * (cond - uncond)` |
| iSTFT | Pure MLX with overlap-add using `array.at[].add()` |
| Perceiver | Learnable query tokens as parameters |

---

## 9. Weight File Structure

All weights stored in single `model.safetensors` with prefixes:

| Prefix | Component | Tensor Count |
|--------|-----------|--------------|
| `ve.*` | VoiceEncoder | 13 |
| `t3.*` | T3 | 293 |
| `s3gen.*` | S3Token2Wav | 2131 |
| `s3_tokenizer.*` | S3TokenizerV2 | 96 |
| **Total** | | **2533** |

---

## 10. Dependencies to Address

### Python Dependencies Used
- `mlx` - Core ML framework
- `mlx_lm` - LLaMA model (for T3 backbone)
- `tokenizers` - Hugging Face tokenizers
- `scipy.signal` - Audio resampling
- `safetensors` - Weight loading

### Swift Equivalents Needed

| Dependency | Python | Swift Option | Status |
|------------|--------|--------------|--------|
| Core ML framework | `mlx` | [ml-explore/mlx-swift](https://github.com/ml-explore/mlx-swift) | Available |
| LLaMA model | `mlx_lm` | [ml-explore/mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) | Available |
| Text tokenizer | `tokenizers` | [huggingface/swift-transformers](https://github.com/huggingface/swift-transformers) `Tokenizers` module | Available |
| Safetensors loading | `safetensors` | Built into mlx-swift | Available |
| Audio file loading | `soundfile` | AVFoundation (built-in) | Available |
| Audio resampling | `scipy.signal` | AVFoundation `AVAudioConverter` | Available |
| S3Tokenizer | `mlx_audio.codec.models.s3` | Port as separate Swift module | To be ported |
| Hugging Face Hub | `huggingface_hub` | [huggingface/swift-transformers](https://github.com/huggingface/swift-transformers) Hub utilities | Available |

---

## Progress Summary

| Section | Components | Completed | Percentage |
|---------|------------|-----------|------------|
| 1. Main Model | 10 | 0 | 0% |
| 2. T3 | 15 | 0 | 0% |
| 3. S3Gen | 84 | 0 | 0% |
| 4. Voice Encoder | 14 | 0 | 0% |
| 5. S3 Tokenizer (Separate Module) | 49 | 0 | 0% |
| 6. Text Tokenizer | 4 | 0 | 0% |
| 7. Configuration | 6 | 0 | 0% |
| **Total** | **182** | **0** | **0%** |

### S3Gen Breakdown

| Subsection | Components |
|------------|------------|
| 3.1 S3Token2Wav | 9 |
| 3.2 Flow Network | 7 |
| 3.3 Flow Matching | 6 |
| 3.4 HiFi-GAN | 14 |
| 3.5 Conditional Decoder | 7 |
| 3.6 Conformer Encoder | 1 |
| 3.7 Transformer Components | 7 |
| 3.8 F0 Predictor | 3 |
| 3.9 CAMPPlus | 17 |
| 3.10 Mel Spectrogram | 2 |
| 3.11 Matcha-TTS | 11 |
| **S3Gen Total** | **84** |

### S3 Tokenizer Breakdown (Separate Module)

| Subsection | Components |
|------------|------------|
| 5.1 Constants/Config | 6 |
| 5.2 S3TokenizerV2 | 7 |
| 5.3 AudioEncoderV2 | 5 |
| 5.4 Attention (V2) | 8 |
| 5.5 RoPE | 2 |
| 5.6 FSQ Quantization | 7 |
| 5.7 V1 Legacy | 7 |
| 5.8 Utilities | 6 |
| 5.9 Chatterbox Mel | 1 |
| **S3 Total** | **49** |

---

## References

### Porting Guides
- **[MLX Swift Porting Guide](../../../mlx-swift-lm/Libraries/MLXLMCommon/Documentation.docc/porting.md)** - Comprehensive guide for porting Python MLX to Swift MLX (covers configuration, property wrappers, layers, debugging)
- [MLX Python to Swift API Reference](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/converting-python) - Method/function name mapping

### Reference Implementations
- **`../mlx-lm`** - Python MLX language models (source reference for LLaMA)
- **`../mlx-swift-lm`** - Swift MLX language models (target reference for T3/LLaMA)
- **`mlx_audio_swift/tts/`** - Existing Swift TTS models (Kokoro, Orpheus, Marvis) with reusable components

### Documentation
- [Chatterbox MLX Port Documentation](chatterbox-mlx-port.md)
- [Original Chatterbox Repository](https://github.com/resemble-ai/chatterbox)
- [MLX Swift Documentation](https://ml-explore.github.io/mlx-swift/)
- [S3Tokenizer](https://github.com/xingchensong/S3Tokenizer)
