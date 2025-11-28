# TTS Model Comparison: Kokoro, Orpheus, and Marvis

A comparison of the three TTS models available in MLX Audio Swift.

## Table of Contents

1. [Overview: Three Different Paradigms](#overview-three-different-paradigms)
2. [Kokoro: Classical TTS Pipeline](#kokoro-classical-tts-pipeline)
3. [Orpheus: Generative LLM Approach](#orpheus-generative-llm-approach)
4. [Marvis: Multimodal LLM with Voice Cloning](#marvis-multimodal-llm-with-voice-cloning)
5. [Side-by-Side Comparison](#side-by-side-comparison)
6. [Pipeline Visualizations](#pipeline-visualizations)
7. [Audio Codecs Explained](#audio-codecs-explained)
   - [What is a Codebook?](#what-is-a-codebook)
   - [HiFi-GAN, SNAC, and Mimi](#hifi-gan-kokoro)
8. [When to Use Each Model](#when-to-use-each-model)

---

## Overview: Three Different Paradigms

The three models represent fundamentally different approaches to text-to-speech:

| Model | Paradigm | Core Idea |
|-------|----------|-----------|
| **Kokoro** | Classical TTS | Phonemes → Duration/Prosody prediction → Vocoder |
| **Orpheus** | Generative LLM | Text → LLM predicts audio tokens autoregressively → Decode |
| **Marvis** | Multimodal LLM | Text + Reference audio → Dual LLM predicts codebooks → Decode |

### The Fundamental Question Each Model Answers Differently

**"How do we convert text into audio?"**

- **Kokoro**: Break it into explicit stages—figure out the sounds (phonemes), how long each should be (duration), how they should sound (prosody), then synthesize.

- **Orpheus**: Train a big language model where audio is just another kind of token. Let the model figure out everything implicitly by predicting the next token.

- **Marvis**: Use reference audio to capture voice characteristics, then have LLMs predict audio codes frame-by-frame, with quality controlled by how many codebooks we use.

---

## Kokoro: Classical TTS Pipeline

### Philosophy

Kokoro follows the traditional TTS approach: explicitly model each aspect of speech synthesis as a separate, interpretable component.

### Architecture

```
Text
  ↓
[eSpeak-NG Phonemizer] ← Converts spelling to sounds
  ↓
Phoneme sequence: /h ə ˈl oʊ/
  ↓
[CustomAlbert (BERT)] ← Understands linguistic context
  ↓
[Duration Predictor] ← How long is each phoneme?
  ↓
[Prosody Predictor] ← What pitch (F0) and energy?
  ↓
[Text Encoder] ← Acoustic features
  ↓
[Decoder + HiFi-GAN Vocoder] ← Generate waveform
  ↓
Audio (24kHz)
```

### Key Characteristics

| Aspect | Details |
|--------|---------|
| **Text processing** | Explicit phonemization via eSpeak-NG |
| **Core model** | BERT-style encoder (CustomAlbert) |
| **Duration** | Explicitly predicted per phoneme |
| **Prosody** | Explicitly predicted (F0, energy) |
| **Audio generation** | HiFi-GAN neural vocoder |
| **Voice control** | Pre-computed style embeddings (256-dim vectors) |
| **Generation type** | Deterministic (same input → same output) |

### Strengths

- **Fast**: Single forward pass, no iterative sampling
- **Small**: ~200MB model size
- **Interpretable**: Each component has a clear purpose
- **Controllable**: Speed control via duration scaling
- **Mobile-friendly**: Low memory, fast inference

### Limitations

- **Fixed voices**: Limited to pre-computed voice embeddings
- **No expressions**: Can't generate laughs, sighs, etc.
- **Language-dependent**: Phonemizer must support the language

### Key Files

- `KokoroEngine.swift` - Main orchestrator
- `KokoroTokenizer.swift` - Text → phonemes
- `CustomAlbert.swift` - BERT encoder
- `DurationEncoder.swift` - Duration prediction
- `ProsodyPredictor.swift` - F0/energy prediction
- `Generator.swift` - HiFi-GAN vocoder

---

## Orpheus: Generative LLM Approach

### Philosophy

Orpheus treats TTS as a language modeling problem: if we can train LLMs to predict the next text token, why not train them to predict audio tokens too?

### Architecture

```
"tara: Hello world"
  ↓
[BPE Tokenizer] ← Same as text LLMs
  ↓
Token sequence: [voice_prefix_tokens, text_tokens]
  ↓
[Llama-3B Transformer] ← 28 layers, 3072 hidden dim
  │
  │  (autoregressive loop, ~1200 iterations)
  │  Sample token → Append → Sample next → ...
  ↓
Audio token sequence: [start, code, code, ..., end]
  ↓
[Parse into 7-layer SNAC codes]
  ↓
[SNAC Decoder] ← Neural audio codec
  ↓
Audio (24kHz)
```

### Key Characteristics

| Aspect | Details |
|--------|---------|
| **Text processing** | BPE subword tokenization (same as LLMs) |
| **Core model** | Llama-3B transformer (28 layers) |
| **Audio representation** | SNAC codes (7 hierarchical layers) |
| **Voice control** | Text prefix (e.g., "tara: ") |
| **Generation type** | Stochastic (temperature, top-p sampling) |
| **Expressions** | Special tokens: `<laugh>`, `<sigh>`, `<yawn>`, etc. |

### How Audio Tokens Work

Orpheus extends the LLM vocabulary to include audio tokens:

```
Standard vocab:  tokens 0 - 128256 (text)
Audio tokens:    tokens 128257 - 128262 (special markers)
                 + offset-based audio codes

The model just predicts "next token" - whether it's text or audio.
```

The 7-layer SNAC structure captures audio at different scales:
- Layer 0: Coarse structure (rhythm, phonemes)
- Layers 1-6: Progressively finer details (harmonics, texture)

### Strengths

- **High quality**: Large model captures nuances
- **Expressions**: Can generate non-speech sounds (laughs, sighs)
- **Unified architecture**: One model does everything
- **Natural prosody**: Learned implicitly from data

### Limitations

- **Slow**: ~1200 autoregressive iterations per utterance
- **Large**: 3-4GB model size
- **Memory hungry**: Needs quantization for mobile
- **No speed control**: Fixed generation pace

### Key Files

- `OrpheusEngine.swift` - Main orchestrator
- `OrpheusModel.swift` - Llama transformer
- `OrpheusTokenizer.swift` - BPE tokenizer
- `SNACDecoder.swift` - Audio codec decoder

---

## Marvis: Multimodal LLM with Voice Cloning

> **Note**: Marvis (Swift) and Sesame (Python) are the same model architecture. The Python implementation uses the `sesame/csm-1b` model from HuggingFace, while the Swift implementation calls it "Marvis". Both use a dual-LLaMA backbone with the Mimi audio codec.

### Philosophy

Marvis combines the LLM approach with reference audio conditioning. Instead of choosing from fixed voices, you can clone any voice by providing a sample.

### Architecture

```
Reference Audio + "Hello world"
  ↓
[Mimi Encoder] ← Encode reference to audio codes
  ↓
Reference codes: [K codebooks × T frames]
  ↓
[Text Tokenizer] ← Standard LLM tokenization
  ↓
[Frame Construction] ← Combine audio codes + text token per frame
  ↓
[Backbone Llama] ← Predict codebook 0
  ↓
[Decoder Llama] ← Iteratively predict codebooks 1, 2, ..., K
  │
  │  For each codebook:
  │    Sample code → Embed → Predict next codebook
  ↓
Generated codes: [K codebooks × F frames]
  ↓
[Mimi Decoder] ← Neural audio codec
  ↓
Audio (24kHz)
```

### Key Characteristics

| Aspect | Details |
|--------|---------|
| **Text processing** | Standard LLM tokenization |
| **Core model** | Dual Llama (backbone + decoder) |
| **Audio representation** | Mimi codec (8-32 codebooks) |
| **Voice control** | Reference audio embedding |
| **Generation type** | Iterative codebook prediction |
| **Quality levels** | Adjustable: 8 (fast) → 32 (best quality) |

### The Dual-LLM Design

Why two LLMs?

1. **Backbone**: Processes the combined text + reference audio context, predicts the first codebook
2. **Decoder**: Specializes in predicting each subsequent codebook given the previous ones

This separation allows:
- The backbone to focus on high-level content (what to say)
- The decoder to focus on audio details (how it sounds)

### Quality-Speed Tradeoff

Marvis uniquely offers configurable quality:

| Quality Level | Codebooks | Speed | Use Case |
|---------------|-----------|-------|----------|
| Low | 8 | Fastest | Real-time, previews |
| Medium | 16 | Balanced | General use |
| High | 24 | Slower | High-quality output |
| Maximum | 32 | Slowest | Best quality |

### Strengths

- **Voice cloning**: Use any reference audio
- **Streaming**: Native frame-by-frame generation
- **Quality control**: Trade speed for quality
- **Balanced size**: 500MB-1.5GB (quantized)

### Limitations

- **Requires reference audio**: Need a sample to clone
- **No expressions**: Less support for non-speech sounds
- **More complex**: Two models to manage

### Key Files

- `MarvisEngine.swift` - Main orchestrator
- `MarvisModel.swift` - Backbone Llama
- `MarvisDecoder.swift` - Decoder Llama
- `MimiCodec.swift` - Audio encoder/decoder

---

## Side-by-Side Comparison

### Architecture Comparison

| Component | Kokoro | Orpheus | Marvis |
|-----------|--------|---------|--------|
| **Text encoder** | BERT (CustomAlbert) | Llama embedding | Llama embedding |
| **Main model** | Separate modules | Single Llama-3B | Backbone + Decoder Llamas |
| **Audio codec** | HiFi-GAN vocoder | SNAC (7 layers) | Mimi (8-32 codebooks) |
| **Parameters** | ~80M | ~3B | ~100M-250M |

### Generation Comparison

| Aspect | Kokoro | Orpheus | Marvis |
|--------|--------|---------|--------|
| **Method** | Deterministic | Autoregressive sampling | Iterative codebook |
| **Iterations** | 1 (forward pass) | ~1200 tokens | K codebooks × F frames |
| **Sampling** | None | Temperature + Top-P | Top-P per codebook |
| **Streaming** | Chunk callbacks | Full generation | Native streaming |

### Practical Comparison

| Metric | Kokoro | Orpheus | Marvis |
|--------|--------|---------|--------|
| **Model size** | ~200MB | ~3-4GB | ~500MB-1.5GB |
| **Speed (5s audio)** | 100-200ms | 5-10s | 2-5s |
| **Real-time factor** | 0.02-0.05 | 1-2 | 0.4-1.0 |
| **Memory usage** | Low | High | Medium |
| **Mobile suitable** | Yes | With quantization | Yes (quantized) |

### Feature Comparison

| Feature | Kokoro | Orpheus | Marvis |
|---------|--------|---------|--------|
| **Speed control** | Yes | No | Quality tradeoff |
| **Voice cloning** | No | No | Yes |
| **Expressions** | No | Yes (`<laugh>`, etc.) | Limited |
| **Streaming** | Callbacks | No | Native |
| **Languages** | Multi (via eSpeak) | English-focused | Multi |

---

## Pipeline Visualizations

### Kokoro: Staged Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                         KOKORO                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Hello"                                                         │
│     │                                                            │
│     ▼                                                            │
│  ┌──────────────┐                                                │
│  │ Phonemizer   │  Text → Phonemes                               │
│  │ (eSpeak-NG)  │                                                │
│  └──────┬───────┘                                                │
│         │ /həˈloʊ/                                               │
│         ▼                                                        │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     │
│  │    BERT      │────▶│   Duration   │────▶│   Prosody    │     │
│  │  (Context)   │     │  Predictor   │     │  Predictor   │     │
│  └──────────────┘     └──────────────┘     └──────────────┘     │
│         │                    │                    │              │
│         └────────────────────┼────────────────────┘              │
│                              ▼                                   │
│                       ┌──────────────┐                           │
│                       │   Decoder    │                           │
│                       │  (HiFi-GAN)  │                           │
│                       └──────┬───────┘                           │
│                              │                                   │
│                              ▼                                   │
│                         Audio 🔊                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Orpheus: Autoregressive LLM

```
┌─────────────────────────────────────────────────────────────────┐
│                         ORPHEUS                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "tara: Hello"                                                   │
│     │                                                            │
│     ▼                                                            │
│  ┌──────────────┐                                                │
│  │ BPE Tokenizer│  Text → Token IDs                              │
│  └──────┬───────┘                                                │
│         │ [128260, 50256, ...]                                   │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────┐                │
│  │              Llama-3B                        │                │
│  │  ┌─────────────────────────────────────┐    │                │
│  │  │  Embed → [Layer 1] → ... → [Layer 28]│   │                │
│  │  └─────────────────────────────────────┘    │                │
│  │         │                                    │                │
│  │         ▼ (repeat ~1200 times)              │                │
│  │  ┌─────────────────┐                        │                │
│  │  │ Sample next     │◀──┐                    │                │
│  │  │ token           │   │                    │                │
│  │  └────────┬────────┘   │                    │                │
│  │           │            │                    │                │
│  │           └────────────┘                    │                │
│  └─────────────────────────────────────────────┘                │
│         │ [audio_start, code, code, ..., end]                   │
│         ▼                                                        │
│  ┌──────────────┐                                                │
│  │ SNAC Decoder │  7-layer codes → Audio                         │
│  └──────┬───────┘                                                │
│         │                                                        │
│         ▼                                                        │
│     Audio 🔊                                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Marvis: Dual-LLM with Reference

```
┌─────────────────────────────────────────────────────────────────┐
│                         MARVIS                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Reference Audio          "Hello"                                │
│     │                        │                                   │
│     ▼                        ▼                                   │
│  ┌──────────────┐     ┌──────────────┐                          │
│  │ Mimi Encoder │     │  Tokenizer   │                          │
│  └──────┬───────┘     └──────┬───────┘                          │
│         │                    │                                   │
│         └────────┬───────────┘                                   │
│                  ▼                                               │
│         [Audio codes | Text token] per frame                     │
│                  │                                               │
│                  ▼                                               │
│  ┌──────────────────────────┐                                   │
│  │    Backbone Llama        │  Predicts codebook 0               │
│  └────────────┬─────────────┘                                   │
│               │                                                  │
│               ▼                                                  │
│  ┌──────────────────────────┐                                   │
│  │    Decoder Llama         │  Predicts codebooks 1..K           │
│  │  ┌────────────────────┐  │                                   │
│  │  │ For i in 1..K:     │  │                                   │
│  │  │   Sample code[i]   │  │                                   │
│  │  │   Embed & predict  │  │                                   │
│  │  └────────────────────┘  │                                   │
│  └────────────┬─────────────┘                                   │
│               │ [K codebooks × F frames]                         │
│               ▼                                                  │
│  ┌──────────────┐                                                │
│  │ Mimi Decoder │  Codes → Audio                                 │
│  └──────┬───────┘                                                │
│         │                                                        │
│         ▼                                                        │
│     Audio 🔊                                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Audio Codecs Explained

### Why Audio Codecs?

Raw audio is just a sequence of amplitude values (24,000 per second). This is:
- Too long for neural networks to process efficiently
- Doesn't capture the structure of sound

Audio codecs compress audio into a more compact, structured representation.

### What is a Codebook?

A **codebook** is a learned dictionary of audio patterns. Think of it like a palette of sounds.

#### The Analogy: Color Palettes

Imagine compressing an image:

```
Full color image: Each pixel is (R, G, B) = 3 × 8 bits = 24 bits per pixel

Palette-based image:
  1. Create a palette of 256 colors that best represent the image
  2. Each pixel is now just an index (0-255) = 8 bits per pixel
  3. To reconstruct: look up the index in the palette
```

Codebooks work the same way, but for audio:

```
Raw audio: 24,000 float values per second

Codebook-based:
  1. Learn a "palette" of 2048 common audio patterns (the codebook)
  2. Each audio frame becomes an index (0-2047) into the codebook
  3. To reconstruct: look up the index to get the audio pattern
```

#### Vector Quantization

The process of mapping continuous audio to discrete codebook indices is called **vector quantization (VQ)**:

```
Audio frame (continuous)     Codebook (learned)           Output
    [0.23, -0.15, ...]   →   Find closest match    →    Index: 847
                              in 2048 entries
```

During training, the codebook entries are learned to minimize reconstruction error.

#### Why Multiple Codebooks?

One codebook can't capture everything. Using multiple codebooks lets us capture different aspects:

**Single codebook (limited):**
```
Audio → [Codebook] → Index 847 → Reconstructed audio (lossy)
```

**Multiple codebooks (better quality):**
```
Audio → [Codebook 0] → Index 847 → Partial reconstruction
                ↓
        Residual (what's left)
                ↓
      → [Codebook 1] → Index 234 → Better reconstruction
                ↓
        Residual (what's still left)
                ↓
      → [Codebook 2] → Index 1052 → Even better
                ↓
              ...
```

Each codebook captures what the previous ones missed. This is called **residual vector quantization (RVQ)**.

#### Codebooks in Orpheus vs Marvis

| Aspect | Orpheus (SNAC) | Marvis (Mimi) |
|--------|----------------|---------------|
| **Structure** | 7 hierarchical layers | 8-32 residual codebooks |
| **Codebook size** | Variable per layer | 2048 entries each |
| **Organization** | Coarse-to-fine (different time scales) | Residual (same time scale) |
| **Quality control** | Fixed (all 7 layers) | Variable (use 8-32 codebooks) |

**SNAC's approach**: Different layers capture different time scales. Layer 0 is coarse (rhythm), layer 6 is fine (texture).

**Mimi's approach**: All codebooks operate at the same time scale, but each captures finer residual details.

#### Why This Matters for TTS

Codebooks let neural networks work with audio as **discrete tokens** (like words in text):

```
Text LLM:      "Hello" → [15496, 995, ...] → Transformer → Next word
Audio LLM:     Audio   → [847, 234, 1052, ...] → Transformer → Next audio token
```

This is why Orpheus can treat TTS as "just another language modeling problem"—audio becomes tokens that the LLM predicts, just like text tokens.

### HiFi-GAN (Kokoro)

**Type**: Neural vocoder (spectrogram → waveform)

```
Mel-spectrogram [80 bins × T frames]
        │
        ▼
┌───────────────────┐
│ Upsampling blocks │  Increase temporal resolution
│ (transpose conv)  │
├───────────────────┤
│ Residual blocks   │  Refine audio quality
│ (multi-kernel)    │
├───────────────────┤
│ Output conv       │  Final projection
└─────────┬─────────┘
          │
          ▼
  Audio [24000 × T samples]
```

**Key insight**: Doesn't use discrete codes—directly generates continuous waveform from spectrogram.

### SNAC (Orpheus)

**Type**: Hierarchical neural audio codec

```
Audio → Encoder → 7-layer discrete codes → Decoder → Audio

Layer structure (7 layers with different rates):
┌─────────────────────────────────────────┐
│ Layer 0: Coarse (rhythm, phonemes)      │  ←─ Most important
│ Layer 1: ...                            │
│ Layer 2: ...                            │
│ Layer 3: Medium detail                  │
│ Layer 4: ...                            │
│ Layer 5: ...                            │
│ Layer 6: Fine (harmonics, texture)      │  ←─ Finest detail
└─────────────────────────────────────────┘

Codes are interleaved: [L0, L1, L2, L3, L4, L5, L6, L0, L1, ...]
                       |<----- one frame ----->|
```

**Key insight**: Hierarchical structure lets the LLM generate coarse-to-fine. Predicting layer 0 first gives rhythm, then refinements add detail.

### Mimi (Marvis)

**Type**: Multi-codebook residual vector quantizer

```
Audio
  │
  ▼
┌─────────────────┐
│ SeaNet Encoder  │  Convolutional encoder
│ (downsample 32×)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Transformer     │  8 layers, contextual
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ Split Residual Vector Quantizer     │
│                                     │
│  Input → [Codebook 0] → Residual    │
│           [Codebook 1] → Residual   │
│           [Codebook 2] → Residual   │
│           ...                       │
│           [Codebook K] → Final      │
│                                     │
│  Each codebook: 2048 entries        │
└────────┬────────────────────────────┘
         │
         ▼
Codes [K codebooks × T frames]
```

**Key insight**: Residual quantization means each codebook captures what the previous ones missed. More codebooks = better quality but more computation.

### Codec Comparison

| Aspect | HiFi-GAN | SNAC | Mimi |
|--------|----------|------|------|
| **Type** | Vocoder | Hierarchical codec | Multi-codebook codec |
| **Input** | Mel-spectrogram | Audio | Audio |
| **Output** | Waveform | 7-layer codes | K-codebook codes |
| **Discrete tokens** | No | Yes | Yes |
| **Quality control** | Fixed | Fixed | Variable (8-32 books) |
| **Bitrate** | N/A | ~6 kbps | ~1.5-6 kbps |

---

## When to Use Each Model

### Use Kokoro When:

- **Speed matters**: Real-time applications, mobile apps
- **Resources are limited**: Low memory, CPU-only
- **Determinism is needed**: Same input should give same output
- **You need speed control**: Adjust playback speed
- **Simple deployment**: Smallest model, fewest dependencies

**Best for**: Mobile apps, real-time assistants, embedded systems

### Use Orpheus When:

- **Quality is paramount**: Highest naturalness
- **You need expressions**: Laughs, sighs, emotional speech
- **You have GPU resources**: Can handle 3B parameter model
- **Latency is acceptable**: Not real-time, batch processing OK
- **English is the target**: Best optimized for English

**Best for**: Audiobook generation, high-quality content creation, expressive voices

### Use Marvis When:

- **You need voice cloning**: Custom voice from reference audio
- **Streaming is important**: Real-time chunk-by-chunk generation
- **Quality-speed tradeoff**: Different quality levels for different uses
- **Balanced resources**: Medium model size, reasonable speed
- **Multiple languages**: Good multilingual support

**Best for**: Voice cloning applications, streaming TTS, production systems with quality tiers

### Decision Flowchart

```
                    ┌─────────────────┐
                    │ Need voice      │
                    │ cloning?        │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │ Yes                         │ No
              ▼                             ▼
         ┌─────────┐              ┌─────────────────┐
         │ MARVIS  │              │ Need expressions│
         └─────────┘              │ (<laugh>, etc)? │
                                  └────────┬────────┘
                                           │
                            ┌──────────────┴──────────────┐
                            │ Yes                         │ No
                            ▼                             ▼
                       ┌─────────┐              ┌─────────────────┐
                       │ ORPHEUS │              │ Speed/size      │
                       └─────────┘              │ priority?       │
                                                └────────┬────────┘
                                                         │
                                          ┌──────────────┴──────────────┐
                                          │ Yes                         │ No
                                          ▼                             ▼
                                     ┌─────────┐                  ┌─────────┐
                                     │ KOKORO  │                  │ ORPHEUS │
                                     └─────────┘                  └─────────┘
```

---

## Summary

| Model | One-Line Description |
|-------|---------------------|
| **Kokoro** | Fast, small, classical TTS with explicit phoneme/duration/prosody stages |
| **Orpheus** | Large LLM that treats audio as tokens, highest quality with expressions |
| **Marvis** | Dual-LLM with voice cloning, streaming, and configurable quality |

Each model represents a different point in the design space:
- **Kokoro**: Optimize for speed and interpretability
- **Orpheus**: Optimize for quality via scale
- **Marvis**: Optimize for flexibility and features
