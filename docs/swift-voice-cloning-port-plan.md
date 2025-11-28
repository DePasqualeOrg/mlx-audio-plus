# Swift Voice Cloning Port Plan

Voice cloning TTS models for iOS/macOS using MLX Swift.

## Status

| Model | Voice Cloning | Status |
|-------|---------------|--------|
| **OuteTTS** | Yes | ✅ Complete (generation) |
| **Spark** | Yes | 📋 Future |

## OuteTTS

**Model:** `OuteAI/Llama-OuteTTS-1.0-1B` (1B params, DAC 24kHz)
**Performance:** RTF 4.4x (~9s for 2s audio)

### Components

| Component | Status | Location |
|-----------|--------|----------|
| OuteTTS Engine | ✅ | `TTS/OuteTTS/` |
| DAC Codec | ✅ | `Codec/DAC/` |
| Speaker Profile Creation | 🔲 Next | - |

### Next: Speaker Profile Creation

Two modes for creating speaker profiles:

| Mode | Use Case | Word Alignment |
|------|----------|----------------|
| **Preset text** | Guided recording | `SFSpeechRecognizer` (built-in) |
| **Any audio** | Import existing audio | WhisperKit (add dependency) |

**Flow:**
```
Audio input (recorded or imported)
                 ↓
    ┌───────────────────────────┐
    │  Word Alignment           │
    │  - SFSpeechRecognizer     │
    │    (preset text mode)     │
    │  - WhisperKit             │
    │    (any audio mode)       │
    └───────────────────────────┘
                 ↓
         DAC encode full audio
                 ↓
      Segment codes by word boundaries (tps=75)
                 ↓
   Extract features per word (pitch/energy/spectral)
                 ↓
           OuteTTSSpeakerProfile
```

**Components to Build:**

| Component | Description | Approach |
|-----------|-------------|----------|
| Word alignment | Get word timestamps | `SFSpeechRecognizer` or WhisperKit |
| Audio features | Pitch, energy, spectral centroid | Port from Python (~150 lines) |
| Profile builder | Assemble speaker profile | New (~100 lines) |

**Files to Create:**
- `OuteTTSSpeakerCreator.swift` - Main orchestrator
- `OuteTTSAudioFeatures.swift` - Feature extraction (pitch, energy, spectral centroid)

**Dependencies:**
- [WhisperKit](https://github.com/argmaxinc/WhisperKit) - For "any audio" mode (optional)

**Reference Python:** `mlx_audio/tts/models/outetts/audio_processor.py`

---

## Spark (Future)

Higher quality voice cloning with ECAPA-TDNN speaker embeddings. Qwen2-based, ~4,500 lines to port.

**Key components:** BiCodec, Vocos, Wav2Vec2, ECAPA-TDNN, Perceiver resampler

---

## Performance Notes

Key optimizations discovered during OuteTTS port:
1. **Custom Module subclass** - Generic MLXLLM protocols cause 60x slower prefill
2. **Direct token ID building** - Avoid BPE tokenization for special tokens
3. **Strategic eval() calls** - Prevent lazy computation graph buildup

---

## References

- [OuteTTS HuggingFace](https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B)
- [DAC (Descript Audio Codec)](https://github.com/descriptinc/descript-audio-codec)
