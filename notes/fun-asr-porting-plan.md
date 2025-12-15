# Fun-ASR-Nano MLX Porting Plan

This document outlines a comprehensive plan for porting Fun-ASR-Nano to MLX for Apple Silicon.

## Source Repositories

- **GitHub**: https://github.com/FunAudioLLM/Fun-ASR
- **HuggingFace**: https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512

## Model Overview

Fun-ASR-Nano is an end-to-end speech recognition model from Tongyi Lab (Alibaba). It combines:
- **Audio Encoder**: SenseVoiceEncoderSmall (SANM-based transformer)
- **Audio Adaptor**: Lightweight transformer to project encoder output to LLM embedding space
- **LLM Decoder**: Qwen3-0.6B for text generation

### Key Features
- 31 language support (focus on East/Southeast Asian languages)
- 7 Chinese dialects, 26 regional accents
- 93% accuracy in high-noise/far-field conditions
- ~2GB total model size (bfloat16)

## Architecture Details

### 1. Audio Frontend (WavFrontend)
```yaml
sample_rate: 16000
n_mels: 80
frame_length: 25ms (400 samples)
frame_shift: 10ms (160 samples)
window: hamming
lfr_m: 7  # Low Frame Rate stacking
lfr_n: 6  # LFR subsampling factor
```

The frontend produces mel-filterbank features with LFR (Low Frame Rate) processing:
- Stack every 7 frames
- Subsample by factor of 6
- Output: `(batch, time/6, 80*7)` = `(batch, time/6, 560)`

### 2. Audio Encoder (SenseVoiceEncoderSmall)

The encoder uses SANM (Self-Attention with Memory) blocks with FSMN (Feedforward Sequential Memory Network):

```
Architecture:
├── encoders0 (1 layer)      # Initial processing, dim 560→512
│   ├── self_attn
│   │   ├── linear_q_k_v: (1536, 560) → Q,K,V packed
│   │   ├── fsmn_block: Conv1d kernel_size=11
│   │   └── linear_out: (512, 512)
│   ├── feed_forward
│   │   ├── w_1: (2048, 512)
│   │   └── w_2: (512, 2048)
│   ├── norm1: LayerNorm(560)
│   └── norm2: LayerNorm(512)
│
├── encoders (49 layers)      # Main encoder, dim 512
│   └── [Same structure as above, but dim 512]
│
├── tp_encoders (20 layers)   # Time-pooling encoder
│   └── [Same structure, dim 512]
│
├── tp_norm: LayerNorm(512)
└── after_norm: LayerNorm(512)
```

**SANM Attention Details**:
- Combined Q/K/V projection with packed weights
- FSMN block: depthwise Conv1d with kernel_size=11 for local context
- 4 attention heads, head_dim=128

### 3. Audio Adaptor (Transformer)

Projects encoder output (512) to LLM dimension (1024):

```
├── linear1: (2048, 512)   # Upproject
├── linear2: (1024, 2048)  # Project to LLM dim
└── blocks (2 layers)
    ├── self_attn
    │   ├── linear_q/k/v: (1024, 1024)
    │   └── linear_out: (1024, 1024)
    ├── feed_forward
    │   ├── w_1: (256, 1024)  # Inverted bottleneck
    │   └── w_2: (1024, 256)
    ├── norm1: LayerNorm(1024)
    └── norm2: LayerNorm(1024)
```

### 4. LLM (Qwen3-0.6B)

Standard Qwen3 architecture with GQA:
```yaml
hidden_size: 1024
num_hidden_layers: 28
num_attention_heads: 16
num_key_value_heads: 8  # GQA
intermediate_size: 3072
vocab_size: 151936
rope_theta: 1000000
```

**Weight Structure**:
- `llm.model.embed_tokens.weight`: (151936, 1024)
- `llm.lm_head.weight`: (151936, 1024) - tied with embed_tokens
- Layers: input_layernorm, self_attn (q/k/v/o_proj, q/k_norm), mlp (gate/up/down_proj)

## Implementation Plan

### Phase 1: Core Infrastructure

#### 1.1 Audio Frontend (`mlx_audio/stt/models/funasr/audio.py`)
```python
# Components to implement:
- log_mel_spectrogram() with hamming window
- LFR (Low Frame Rate) processing
  - Frame stacking (lfr_m=7)
  - Subsampling (lfr_n=6)
```

**Reference**: Similar to Whisper's `audio.py` but with LFR processing.

#### 1.2 SANM Encoder Block (`mlx_audio/stt/models/funasr/encoder.py`)
```python
class FSMNBlock(nn.Module):
    """Feedforward Sequential Memory Network block"""
    # Depthwise Conv1d with kernel_size=11
    # Weight shape: (512, 1, 11) → needs transpose for MLX

class SANMAttention(nn.Module):
    """Self-Attention with Memory (SANM)"""
    # Combined Q/K/V projection
    # FSMN block for local context
    # Multi-head attention with 4 heads

class SANMEncoderLayer(nn.Module):
    """Single encoder layer"""
    # norm1 → self_attn → norm2 → feed_forward

class SenseVoiceEncoder(nn.Module):
    """Full encoder with three encoder stacks"""
    # encoders0 (1 layer, input processing)
    # encoders (49 layers, main encoder)
    # tp_encoders (20 layers, time-pooling)
```

#### 1.3 Audio Adaptor (`mlx_audio/stt/models/funasr/adaptor.py`)
```python
class AudioAdaptor(nn.Module):
    """Projects encoder output to LLM dimension"""
    # linear1/linear2: 512 → 2048 → 1024
    # 2 transformer blocks with self-attention
```

### Phase 2: LLM Integration

#### 2.1 Qwen3 Model (`mlx_audio/stt/models/funasr/qwen3.py`)

**Options**:
1. **Port from scratch** - Full control, matches weight structure exactly
2. **Use mlx-lm Qwen** - Reuse existing implementation, need weight mapping

**Recommendation**: Port from scratch to ensure exact compatibility with the combined model weights.

```python
class Qwen3Attention(nn.Module):
    """GQA attention with RoPE and QK normalization"""
    # q_proj, k_proj, v_proj, o_proj
    # q_norm, k_norm (RMSNorm per head)
    # RoPE with theta=1e6

class Qwen3MLP(nn.Module):
    """SwiGLU MLP"""
    # gate_proj, up_proj, down_proj

class Qwen3Block(nn.Module):
    """Transformer block with pre-norm"""

class Qwen3Model(nn.Module):
    """Full Qwen3 model"""
```

### Phase 3: Integration & Inference

#### 3.1 Main Model Class (`mlx_audio/stt/models/funasr/funasr.py`)
```python
class FunASRNano(nn.Module):
    """Main Fun-ASR model combining all components"""

    def __init__(self, config):
        self.audio_encoder = SenseVoiceEncoder(...)
        self.audio_adaptor = AudioAdaptor(...)
        self.llm = Qwen3Model(...)

    def encode_audio(self, mel, lengths):
        """Process audio through encoder and adaptor"""
        enc_out, enc_lens = self.audio_encoder(mel, lengths)
        adapted = self.audio_adaptor(enc_out, enc_lens)
        return adapted, enc_lens

    def generate(self, audio_path, **kwargs):
        """Full inference pipeline"""
        # 1. Load and preprocess audio
        # 2. Compute mel spectrogram with LFR
        # 3. Encode audio
        # 4. Prepare prompt with audio embeddings
        # 5. Generate with LLM
        # 6. Decode and return text
```

#### 3.2 Tokenization
- Use HuggingFace tokenizers for Qwen3 (tiktoken-based)
- Handle special tokens: `<|im_start|>`, `<|im_end|>`, `<think>`, etc.
- Chat template for system/user/assistant format

### Phase 4: Weight Conversion

#### 4.1 Conversion Script (`mlx_audio/stt/models/funasr/convert.py`)
```python
def convert_funasr_weights(pytorch_path, output_path):
    """Convert PyTorch weights to MLX format"""

    # Load PyTorch checkpoint
    state_dict = torch.load(pytorch_path)['state_dict']

    # Key transformations needed:
    transformations = {
        # Conv1d weights: (out, 1, kernel) → (out, kernel, 1)
        'fsmn_block.weight': lambda w: w.swapaxes(1, 2),

        # Attention Q/K/V renaming (if needed)
        # LLM weight mapping to match our Qwen3 implementation
    }

    # Save as safetensors
```

**Weight Mapping Summary**:
| PyTorch Key | MLX Key | Transform |
|-------------|---------|-----------|
| `audio_encoder.encoders.N.*` | Same | None |
| `*.fsmn_block.weight` | Same | swapaxes(1,2) |
| `audio_adaptor.*` | Same | None |
| `llm.model.*` | Same | None |

### Phase 5: Testing & Validation

#### 5.1 Unit Tests
- Audio preprocessing (compare mel outputs with PyTorch)
- Encoder forward pass
- Adaptor forward pass
- Full model inference

#### 5.2 Integration Tests
- End-to-end transcription on example files
- Multi-language support verification
- Performance benchmarks (latency, throughput)

## File Structure

```
mlx_audio/stt/models/funasr/
├── __init__.py
├── audio.py           # Audio preprocessing, mel spectrogram, LFR
├── encoder.py         # SenseVoiceEncoder, SANM blocks, FSMN
├── adaptor.py         # AudioAdaptor transformer
├── qwen3.py           # Qwen3 LLM implementation
├── funasr.py          # Main FunASRNano model
├── tokenizer.py       # Tokenization utilities
└── convert.py         # Weight conversion script

scripts/
└── convert_funasr.py  # CLI for weight conversion

tests/
└── test_funasr.py     # Test suite
```

## Key Technical Considerations

### 1. FSMN Block Implementation
The FSMN block uses a depthwise convolution for local context:
```python
# PyTorch weight shape: (512, 1, 11) - depthwise conv
# MLX expects: (512, 11, 1) after transpose
# Implementation: nn.Conv1d with groups=input_dim
```

### 2. LFR (Low Frame Rate) Processing
```python
def apply_lfr(features, lfr_m=7, lfr_n=6):
    """Apply Low Frame Rate processing"""
    # Stack lfr_m consecutive frames
    # Subsample by factor of lfr_n
    # Handles padding for incomplete final frames
```

### 3. Audio-LLM Embedding Integration
The model inserts audio embeddings at specific positions in the text sequence:
```python
# 1. Tokenize prompt with placeholder: "<|startofspeech|>...<|endofspeech|>"
# 2. Get text embeddings from LLM
# 3. Replace placeholder region with audio embeddings
# 4. Run LLM generation
```

### 4. Quantization Considerations
- Audio encoder: Likely safe to quantize (transformer layers)
- Audio adaptor: May be sensitive, test carefully
- LLM: Standard 4-bit quantization should work well

## Dependencies

Required:
- mlx >= 0.20.0
- mlx-lm (for reference, not direct use)
- safetensors
- huggingface_hub
- tiktoken (for Qwen tokenizer)
- scipy (for audio processing)
- soundfile (for audio loading)

## Estimated Complexity

| Component | Complexity | LOC Estimate |
|-----------|------------|--------------|
| Audio Frontend | Medium | ~150 |
| SANM Encoder | High | ~400 |
| Audio Adaptor | Low | ~100 |
| Qwen3 LLM | Medium | ~300 |
| Integration | Medium | ~250 |
| Weight Conversion | Medium | ~150 |
| Tests | Medium | ~200 |
| **Total** | | **~1550** |

## Reference Implementations

- **Fun-ASR Source**: https://github.com/FunAudioLLM/Fun-ASR
- **Fun-ASR Weights**: https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512
- **SenseVoice Encoder**: FunASR repo (PyTorch)
- **Qwen3**: mlx-lm repository (https://github.com/ml-explore/mlx-lm)
- **Audio Processing**: Whisper MLX implementation in this repo (`mlx_audio/stt/models/whisper/`)
- **Conformer patterns**: Parakeet implementation in this repo (`mlx_audio/stt/models/parakeet/`)

## Potential Challenges

1. **SANM Attention**: Custom attention mechanism with FSMN - needs careful porting
2. **LFR Processing**: Frame stacking/subsampling at audio frontend level
3. **Multi-turn Dialog**: The model supports multi-turn conversations - need to handle context properly
4. **Prompt Engineering**: Model uses specific prompt format with `<|startofspeech|>` tokens
5. **Thinking Mode**: Model can output `<think>...</think>` blocks - may want to strip or expose

## Next Steps

1. Clone source repositories:
   - `git clone https://github.com/FunAudioLLM/Fun-ASR`
   - `git clone https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512`
2. Set up the directory structure
3. Implement audio frontend with LFR
4. Port SANM encoder block by block, validating against PyTorch
5. Implement audio adaptor
6. Port Qwen3 LLM
7. Create weight conversion script
8. Build inference pipeline
9. Test and validate
10. Optimize performance
11. Create mlx-community model repo

## Appendix: Weight Analysis

### Total Parameters
- Audio Encoder: 914 keys (~200M params)
- Audio Adaptor: 36 keys (~20M params)
- LLM: 311 keys (~600M params)
- **Total**: ~820M parameters

### Model Size
- bfloat16: ~1.6GB
- 4-bit quantized (LLM only): ~400MB + encoder/adaptor

### Inference Memory
- Base model: ~2GB
- With KV cache (max context): ~3-4GB
