# TTS Models Comparison for MLX Porting

This document compares text-to-speech models for potential porting from PyTorch to MLX.

## Memory Calculation Method

Model weight memory (fp16) is calculated as: **parameters × 2 bytes**

Note: Actual runtime memory will be higher due to activations, KV cache, and intermediate computations.

## Comparison Table

| Model | License | Parameters | Weights (fp16) | Priority | Notes |
|-------|---------|------------|----------------|----------|-------|
| **[MeloTTS](https://github.com/myshell-ai/MeloTTS)** | MIT | ~100M + BERT | ~0.4GB | ⭐ High | Lightweight, multilingual, real-time on CPU |
| **[Parler TTS](https://github.com/huggingface/parler-tts)** | Apache 2.0 | 880M / 2.3B | 1.8GB / 4.6GB | ⭐ High | Fully open, HuggingFace backed |
| **[CosyVoice2](https://github.com/FunAudioLLM/CosyVoice)** | Apache 2.0 | 500M | 1.0GB | ⭐ High | Alibaba, excellent multilingual, 150ms streaming |
| **[MegaTTS 3](https://github.com/bytedance/MegaTTS3)** | Apache 2.0 | 450M | 0.9GB | Medium | ByteDance, voice cloning, CPU fallback |
| **[StyleTTS2](https://github.com/yl4579/StyleTTS2)** | GPL (inference dep.) | ~150M | ~0.3GB | Medium | Excellent quality, complex training |
| **[Zonos](https://github.com/Zyphra/Zonos)** | Apache 2.0 | 1.6B | 3.2GB | Medium | Voice cloning, 5s reference audio |
| **[VibeVoice](https://github.com/microsoft/VibeVoice)** | MIT (research only) | 1.5B / 7B | 3.0GB / 14GB | Medium | Microsoft, 90-min synthesis, 4 speakers |
| **[LLMVoX](https://github.com/mbzuai-oryx/LLMVoX)** | CC-BY-NC-SA 4.0 | 30M | 60MB | Lower | Non-commercial, LLM-agnostic streaming |
| **[VyvoTTS](https://github.com/Vyvo-Labs/VyvoTTS)** | MIT | Unknown | Unknown | Lower | Less established, limited documentation |
| **[Granite Speech 3.3](https://huggingface.co/ibm-granite/granite-speech-3.3-8b)** | Apache 2.0 | 8B | 16GB | Lower | IBM, primarily STT not TTS |
| **Chatterbox** (already ported) | MIT | ~1.5B | ~3.0GB | ✅ Done | Resemble AI, 23 languages, emotion control |

## Recommendations

### Top 3 to Port Next

1. **MeloTTS** - MIT license, smallest footprint, runs on CPU, well-documented
2. **Parler TTS** - Apache 2.0, excellent HuggingFace ecosystem, reasonable size
3. **CosyVoice2** - Apache 2.0, strong multilingual, active development

### Models to Avoid or Deprioritize

- **LLMVoX** - Non-commercial license (CC-BY-NC-SA)
- **VibeVoice** - MIT but restricted to research use only
- **Granite Speech** - It's STT (speech-to-text), not TTS
- **StyleTTS2** - GPL dependency complicates licensing

## Detailed Model Notes

### MeloTTS
- Developed by MIT and MyShell.ai
- Supports English, Chinese (mixed English), and other languages
- Uses BERT models for text processing (~400MB download per language)
- Fast enough for real-time CPU inference
- Supports CUDA, CPU, and MPS devices

### Parler TTS
- Fully open-source with all datasets, training code, and weights public
- Mini (938M params) trained on 45K hours of audio
- Large (2.2B params) for higher quality
- Memory: 3.6GB fp32, 1.8GB fp16 inference

### CosyVoice2
- Developed by Alibaba's FunAudioLLM team
- Ultra-low latency streaming (150ms)
- 30-50% reduction in pronunciation errors vs v1
- Supports TensorRT-LLM for 4x acceleration
- **Porting advantage**: Uses S3 tokenizer (same as Chatterbox) - already ported to MLX

### MegaTTS 3
- Developed by ByteDance and Zhejiang University
- Uses Sparse Alignment Enhanced Latent Diffusion Transformer
- Bilingual (Chinese/English) with code-switching support
- Docker images available for GPU and CPU inference
- **Note**: Official release withholds WavVAE encoder for security reasons (only pre-extracted latents available). Community weights available at [drbaph/MegaTTS3-WaveVAE](https://huggingface.co/drbaph/MegaTTS3-WaveVAE) for full voice cloning.

### StyleTTS2
- Excellent quality through style diffusion and adversarial training
- Inference is lightweight (~2GB VRAM)
- Training requires significant resources (24GB+ for full training)
- GPL-licensed inference dependency

### Zonos
- Developed by Zyphra
- Two model variants: transformer and SSM hybrid (both 1.6B)
- Trained on 200K+ hours of multilingual speech
- Voice cloning with just 5 seconds of reference audio
- Hybrid model is ~20% faster than transformer

### VibeVoice
- Developed by Microsoft
- Can synthesize up to 90 minutes with 4 distinct speakers
- Includes watermarking and safety mechanisms
- **Important**: Research use only despite MIT license

### LLMVoX
- Developed by MBZUAI
- Extremely lightweight (30M parameters)
- Designed as add-on for any LLM
- **Important**: Non-commercial license

### Granite Speech 3.3
- Developed by IBM
- **Note**: This is primarily a speech-to-text (STT) model, not TTS
- 8B parameters with 128K context length
- Supports translation to multiple languages

---

*Last updated: December 2024*
