# Whisper MLX Conversion Plan

This document outlines the plan for converting OpenAI Whisper models to MLX format and uploading them to Hugging Face.

## Source Models

The following OpenAI Whisper models will be converted (excluding deprecated versions):

| Model | Parameters | English-only | Multilingual |
|-------|------------|--------------|--------------|
| tiny | 39M | tiny.en | tiny |
| base | 74M | base.en | base |
| small | 244M | small.en | small |
| medium | 769M | medium.en | medium |
| large-v3 | 1550M | - | large-v3 |
| large-v3-turbo | 809M | - | large-v3-turbo |

**Notes:**
- `large` is an alias for `large-v3`
- `turbo` is an alias for `large-v3-turbo`
- `large-v1` and `large-v2` are excluded as they are superseded by `large-v3`
- English-only models (`.en`) use `gpt2.tiktoken` vocabulary
- Multilingual models use `multilingual.tiktoken` vocabulary

## Quantization Variants

Each model will be converted to the following precisions:

| Precision | Description | Use Case |
|-----------|-------------|----------|
| fp16 | 16-bit floating point | Best quality, larger size |
| 8-bit | 8-bit quantization | Good balance of quality and size |
| 4-bit | 4-bit quantization | Smallest size, some quality tradeoff |

**Note:** MLX supports 2, 3, 4, 5, 6, and 8-bit quantization. We use fp16, 8-bit, and 4-bit as a practical set matching mlx-community conventions.

## Target Repositories

Models will be uploaded to `mlx-community` on Hugging Face with the following naming convention:

```
mlx-community/whisper-{model}-{precision}
```

Examples:
- `mlx-community/whisper-large-v3` (fp16, no suffix)
- `mlx-community/whisper-large-v3-8bit`
- `mlx-community/whisper-large-v3-4bit`

## Conversion Matrix

### English-only Models

| Model | fp16 | 8-bit | 4-bit |
|-------|------|-------|-------|
| whisper-tiny.en | [ ] | [ ] | [ ] |
| whisper-base.en | [ ] | [ ] | [ ] |
| whisper-small.en | [ ] | [ ] | [ ] |
| whisper-medium.en | [ ] | [ ] | [ ] |

### Multilingual Models

| Model | fp16 | 8-bit | 4-bit |
|-------|------|-------|-------|
| whisper-tiny | [ ] | [ ] | [ ] |
| whisper-base | [ ] | [ ] | [ ] |
| whisper-small | [ ] | [ ] | [ ] |
| whisper-medium | [ ] | [ ] | [ ] |
| whisper-large-v3 | [ ] | [ ] | [ ] |
| whisper-large-v3-turbo | [ ] | [ ] | [ ] |

**Total: 30 model variants** (10 models x 3 precisions)

## Conversion Commands

```bash
# fp16 (default)
python -m mlx_audio.stt.models.whisper.scripts.convert \
    --model-id openai/whisper-{model} \
    --output-dir whisper-{model} \
    --upload-repo mlx-community/whisper-{model}

# 8-bit quantization
python -m mlx_audio.stt.models.whisper.scripts.convert \
    --model-id openai/whisper-{model} \
    --output-dir whisper-{model}-8bit \
    --quantize --q-bits 8 \
    --upload-repo mlx-community/whisper-{model}-8bit

# 4-bit quantization
python -m mlx_audio.stt.models.whisper.scripts.convert \
    --model-id openai/whisper-{model} \
    --output-dir whisper-{model}-4bit \
    --quantize --q-bits 4 \
    --upload-repo mlx-community/whisper-{model}-4bit
```

## Dry Run (Local Testing)

To test conversion locally without uploading:

```bash
python -m mlx_audio.stt.models.whisper.scripts.convert \
    --model-id openai/whisper-tiny \
    --output-dir /tmp/whisper-tiny \
    --dry-run
```

## Verification

After conversion, verify each model with:

```bash
mlx_audio.stt --model mlx-community/whisper-{model} --audio test.mp3
```

## Notes

- Conversion requires PyTorch and whisper dependencies (for loading source weights)
- Quantized models may have reduced accuracy, especially for smaller models
- The `large-v3-turbo` model offers a good balance of speed and accuracy
