# CosyVoice2 MLX Port Status

## Overview
CosyVoice2 is a streaming TTS model from Alibaba. This document tracks the MLX port status.

## Components

| Component | Status | Notes |
|-----------|--------|-------|
| LLM (Qwen2) | ✅ Working | Token generation works |
| S3TokenizerV2 | ✅ Working | Speech token extraction works |
| Flow Encoder | ✅ Working | Correct 2x upsampling (tokens → mel frames) |
| Flow Decoder (CFM) | ⚠️ Issues | Produces compressed mel output |
| HiFiGAN Vocoder | ⚠️ Partial | GT mel reconstruction has minor distortion |
| Speaker Encoder | ❌ Not ported | Uses ONNX campplus (placeholder zeros used) |

## Current Issue

**Flow model produces gibberish audio** even when given correct speech tokens from reference audio.

### Symptoms
- Mel output statistics are close but not matching ground truth:
  - Flow output: mean=-3.75, std=1.6
  - Ground truth: mean=-4.91, std=2.8
- Output has compressed dynamic range
- Audio is unintelligible

### Verified Working
- Weight loading (all weights match safetensors exactly)
- Random noise (now using pre-generated PyTorch noise for compatibility)
- Encoder output shapes (correct 2x upsampling)

## Likely Causes

1. **Missing causal attention masking**: Original uses `CausalConditionalDecoder` with chunk-based attention masks when `streaming=True`. Our decoder uses `attention_mask=None` (full bidirectional attention).

2. **Missing streaming parameter**: Original passes `streaming` parameter to encoder and decoder which affects attention masking behavior.

3. **CFG rate**: Currently using 0.7, may need adjustment.

## Files Structure

```
mlx_audio/tts/models/cosyvoice2/
├── __init__.py
├── config.py           # Model configuration
├── cosyvoice2.py       # Main model and loading
├── flow_matching.py    # CosyVoice2-specific CFM (NEW)
├── hifigan.py          # Vocoder with F0 predictor
└── llm/
    ├── __init__.py
    └── llm.py          # Qwen2-based LLM
```

## Key Differences from Chatterbox

CosyVoice2 shares flow components with Chatterbox but requires:
- Pre-generated PyTorch random noise (MLX RNG differs from PyTorch)
- 24kHz sample rate (vs 22kHz for Chatterbox)
- Different HiFiGAN source module (SourceModuleHnNSF2 with interpolation)
- Causal attention masking in decoder (not yet implemented)

## Next Steps

1. Implement `CausalConditionalDecoder` with proper attention masking
2. Add `streaming` parameter support to encoder/decoder
3. Test with different CFG rates
4. Consider implementing campplus speaker encoder

## Test Commands

```bash
# Run flow model debug
python debug_flow_simple.py

# Check weight loading
python debug_weights.py

# Verify noise compatibility
python debug_verify_noise.py
```

## References

- Original: https://github.com/FunAudioLLM/CosyVoice
- Chatterbox (similar architecture): https://github.com/resemble-ai/chatterbox
