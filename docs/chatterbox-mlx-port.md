# Chatterbox TTS MLX Port Feasibility Report

**Date:** 2025-11-28
**Source Repository:** https://github.com/resemble-ai/chatterbox

## Overview

Chatterbox is a 0.5B parameter text-to-speech model from Resemble AI featuring zero-shot voice cloning, multilingual support (23 languages), emotion control, and classifier-free guidance. This document evaluates the feasibility of porting Chatterbox to MLX for Apple Silicon optimization.

## Chatterbox Architecture

### Core Components

| Component | Source Files | Size | Description |
|-----------|--------------|------|-------------|
| **T3** | `models/t3/t3.py` | ~16KB | LLaMA-based transformer backbone for text→speech token generation |
| **S3Gen** | `models/s3gen/` | ~77KB | Speech synthesis module with flow matching and vocoder |
| **S3Tokenizer** | `models/s3tokenizer/` | ~5KB | Speech tokenization |
| **Voice Encoder** | `models/voice_encoder/` | ~10KB | Speaker embedding extraction |

### S3Gen Subcomponents

| File | Size | Purpose |
|------|------|---------|
| `hifigan.py` | ~17KB | HiFi-GAN vocoder with Snake activation and neural source filter |
| `xvector.py` | ~14KB | X-vector speaker embeddings |
| `decoder.py` | ~12KB | Decoder network |
| `flow.py` | ~10KB | Flow-based processing |
| `flow_matching.py` | ~9KB | Flow matching for ODE-based generation |
| `s3gen.py` | ~10KB | Main S3Gen orchestration |
| `f0_predictor.py` | ~2KB | Fundamental frequency (pitch) prediction |

### Key Features

- **Multilingual:** 23 languages including Arabic, Chinese, French, German, Hindi, Japanese, Korean
- **Zero-shot voice cloning:** Clone voices from short audio prompts
- **Emotion control:** Adjustable via `exaggeration` parameter (0.3-0.7 typical)
- **Classifier-free guidance:** `cfg_weight` for speaker adherence vs text fidelity
- **Perth watermarking:** Imperceptible neural watermarks for AI audio detection

### Training Data

- Trained on 0.5M hours of cleaned audio data
- Competitive with closed-source systems (ElevenLabs benchmarks)

## Existing mlx-audio Components

### Already Available

| Component | Location | Reusability |
|-----------|----------|-------------|
| LLaMA backbone | `mlx-lm` integration | Direct reuse via `mlx_lm.models.llama` |
| SNAC codec | `mlx_audio/codec/models/snac.py` | Pattern reference |
| iSTFTNet vocoder | `mlx_audio/tts/models/kokoro/istftnet.py` | Architecture reference |
| Transformer blocks | Multiple models | Pattern reference |
| Voice cloning patterns | Orpheus, OuteTTS | Implementation reference |
| Audio processing | `mlx_audio/utils.py` | Direct reuse |

### Relevant Existing Models

| Model | Similarity to Chatterbox |
|-------|-------------------------|
| **Orpheus** | Uses LLaMA + SNAC, similar architecture pattern |
| **Kokoro** | Has iSTFTNet vocoder, voice system |
| **OuteTTS** | LLaMA-based with DAC codec |
| **Spark** | Complex multi-component architecture |

## Feasibility Assessment

### Overall Rating: **HIGH** ✓

The port is highly feasible due to:
1. Core LLaMA infrastructure already exists
2. Similar architectural patterns are already implemented
3. Model size (0.5B) matches existing Orpheus implementation
4. Well-structured source code with clear component separation

### Component-by-Component Analysis

| Component | Complexity | Effort | Notes |
|-----------|------------|--------|-------|
| T3 (LLaMA backbone) | Low | 1-2 days | Leverage existing `mlx-lm` integration |
| T3 conditioning | Low | 1 day | Emotion/CLAP conditioning adapter |
| HiFi-GAN vocoder | Medium | 3-4 days | Snake activation, NSF modules need implementation |
| Flow matching | Medium-High | 3-4 days | ODE solver translation to MLX |
| X-vector embeddings | Medium | 2 days | Speaker embedding network |
| F0 predictor | Low | 0.5 days | Small pitch prediction module |
| Voice encoder | Medium | 2 days | Mel-spectrogram based encoder |
| S3Tokenizer | Low-Medium | 1-2 days | Speech tokenization |
| Weight conversion | Low | 1 day | PyTorch → MLX checkpoint conversion |

### Estimated Total Effort

- **New code:** ~80-100KB Python
- **Time estimate:** 2-3 weeks for experienced MLX developer
- **Complexity:** Similar to Spark or Dia ports

## Technical Challenges

### 1. Flow Matching Implementation

The flow matching module uses ODE-based generation which requires:
- Translating `torchdiffeq` or similar ODE solvers to MLX
- Careful handling of continuous normalizing flows
- Potential need for custom MLX operations

### 2. HiFi-GAN Specifics

Chatterbox's HiFi-GAN differs from standard implementations:
- **Snake activation:** Periodic activation function `x + 1/α * sin²(αx)`
- **Neural Source Filter (NSF):** Combines sine generation with harmonic merging
- **SineGen:** Harmonic overtone generation with voiced/unvoiced classification

### 3. Weight Conversion

- PyTorch checkpoints need conversion to MLX format
- Attention to weight naming conventions between frameworks
- Verification of numerical equivalence post-conversion

### 4. Watermarking (Optional)

Perth watermarking is a separate concern:
- Can be omitted for initial port
- Requires separate Perth library integration if needed

## Implementation Plan

### Phase 1: Foundation (Week 1)

1. **Set up model structure**
   - Create `mlx_audio/tts/models/chatterbox/` directory
   - Implement base configuration classes
   - Set up weight loading infrastructure

2. **Port T3 model**
   - Adapt LLaMA integration from Orpheus pattern
   - Implement T3-specific conditioning module
   - Add text/speech embedding layers

### Phase 2: Speech Synthesis (Week 2)

3. **Port S3Gen components**
   - Implement HiFi-GAN vocoder with Snake activation
   - Port flow matching module
   - Implement F0 predictor

4. **Port voice components**
   - Implement X-vector speaker embeddings
   - Port voice encoder
   - Implement S3Tokenizer

### Phase 3: Integration (Week 3)

5. **End-to-end pipeline**
   - Connect all components
   - Implement inference pipeline
   - Add voice cloning support

6. **Testing and optimization**
   - Weight conversion and verification
   - Performance benchmarking
   - Memory optimization

## File Structure

```
mlx_audio/tts/models/chatterbox/
├── __init__.py
├── chatterbox.py          # Main model and pipeline
├── config.py              # Model configurations
├── t3/
│   ├── __init__.py
│   ├── t3.py              # T3 transformer model
│   └── conditioning.py    # Emotion/CLAP conditioning
├── s3gen/
│   ├── __init__.py
│   ├── s3gen.py           # Main S3Gen module
│   ├── hifigan.py         # HiFi-GAN vocoder
│   ├── flow_matching.py   # Flow matching
│   ├── f0_predictor.py    # Pitch prediction
│   └── xvector.py         # Speaker embeddings
├── tokenizer/
│   ├── __init__.py
│   └── s3tokenizer.py     # Speech tokenizer
└── voice_encoder/
    ├── __init__.py
    └── voice_encoder.py   # Voice encoding
```

## Recommendations

1. **Proceed with port** - Architecture aligns well with existing mlx-audio patterns
2. **Start with T3** - Leverage existing LLaMA infrastructure for quick wins
3. **Prioritize HiFi-GAN** - Core vocoder is critical path
4. **Defer watermarking** - Optional feature, add post-MVP
5. **Test incrementally** - Verify each component against PyTorch reference

## References

- Chatterbox Repository: https://github.com/resemble-ai/chatterbox
- MLX Documentation: https://ml-explore.github.io/mlx/
- mlx-lm: https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm
- HiFi-GAN Paper: https://arxiv.org/abs/2010.05646
- Flow Matching: https://arxiv.org/abs/2210.02747
