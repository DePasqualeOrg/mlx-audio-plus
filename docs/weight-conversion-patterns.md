# Weight Conversion Patterns: PyTorch to MLX

This document describes how weights are converted from PyTorch to MLX format in mlx-audio, and compares the approach to mlx-lm.

## Overview

MLX-audio uses a **mixed approach** for weight conversion, unlike mlx-lm which has a more consistent two-step process.

## MLX-LM Approach (for comparison)

MLX-LM uses a clean two-step process:

### Step 1: Pre-Upload Conversion (`mlx_lm/convert.py`)
- **Dtype conversion** (float32 → float16/bfloat16)
- **Quantization** (optional 4-bit/8-bit)
- **Shape transformations** for Conv layers
- Saves to safetensors in **MLX format**

### Step 2: Load-Time Sanitize
- **Lightweight cleanup only** - removes unused weights
- **Config-dependent filtering** (e.g., skip `lm_head.weight` when embeddings are tied)
- **No shape transformations** - weights are already in MLX format

## MLX-Audio Approach

MLX-audio has **two different patterns** depending on the model:

### Pattern 1: Pre-converted Weights (e.g., Whisper)

Models like Whisper have weights that are **already in MLX format** in the mlx-community repos:

```
mlx-community/whisper-tiny
├── config.json
└── weights.npz  ← Conv weights in MLX format (out, kernel, in)
```

- **No sanitize method** - weights load directly
- Uses `model.update(weights)` without transformation
- Conv weight shape: `(384, 3, 384)` = (out_channels, kernel_size, in_channels)

### Pattern 2: PyTorch-Format Weights (e.g., Kokoro)

Models like Kokoro have weights that are **still in PyTorch format** even in mlx-community repos:

```
mlx-community/Kokoro-82M-bf16
├── config.json
└── kokoro-v1_0.safetensors  ← Conv weights in PyTorch format (out, in, kernel)
```

- **sanitize() transforms weights at load time**
- Conv weight shape: `(1, 256, 1)` = (out_channels, in_channels, kernel_size)
- Despite the "bf16" suffix, weights are often still float32

## The Sanitize Method

Located in each model class, `sanitize()` transforms weights from PyTorch to MLX format.

### When Sanitize is Called

From `mlx_audio/tts/utils.py:240-241`:
```python
if quantization is None:
    weights = model.sanitize(weights)
```

Sanitize is **only called for non-quantized models**. Quantized models skip sanitization.

### Common Transformations

#### 1. Conv1d Weight Transposition
PyTorch: `(out_channels, in_channels, kernel_size)`
MLX: `(out_channels, kernel_size, in_channels)`

```python
value = value.transpose(0, 2, 1)  # or .swapaxes(1, 2)
```

#### 2. Conv2d Weight Transposition
PyTorch: `(O, I, H, W)`
MLX: `(O, H, W, I)`

```python
value = value.transpose(0, 2, 3, 1)
```

#### 3. LSTM Weight Renaming
```python
# PyTorch → MLX
"weight_ih_l0" → "Wx"
"weight_hh_l0" → "Wh"
"weight_ih_l0_reverse" → "Wx_backward"  # bidirectional
```

#### 4. Key Renaming
```python
# Attention layers
"attn.to_q" → "attn.query_proj"
"attn.output_proj" → "attn.o_proj"

# LayerNorm
"gamma" → "weight"
"beta" → "bias"

# MLP
"mlp.w1" → "gate_proj"
"mlp.w2" → "down_proj"
"mlp.w3" → "up_proj"
```

#### 5. Skipped Weights
```python
# Remove dynamically computed values
"position_ids"
"num_batches_tracked"
"freqs_cis"
```

## Idempotency Analysis

A sanitize method is **idempotent** if it can be safely called multiple times on the same weights without corrupting them.

### Idempotent Methods

| Model | Location | Approach |
|-------|----------|----------|
| BigVGAN | `indextts/bigvgan.py:82-124` | Compares to model's expected shapes |
| Spark feat_encoder | `spark/modules/encoder_decoder/feat_encoder.py:97-114` | Shape heuristics |
| Voxtral | `stt/models/voxtral/voxtral.py:276-286` | Shape heuristics |
| Sesame | `sesame/sesame.py:518-545` | Key renaming only |
| Bark | `bark/bark.py:429-452` | Key renaming only |
| Dia | `dia/dia.py:108-109` | Pass-through |

Example of shape-aware idempotent sanitize (BigVGAN):
```python
curr_weights = dict(tree_flatten(self.parameters()))

if value.ndim == 3:
    if value.shape != curr_weights[key].shape:  # Only transpose if needed
        value = value.transpose(0, 2, 1)
```

### Non-Idempotent Methods

| Model | Location | Issue |
|-------|----------|-------|
| Kokoro | `kokoro/kokoro.py:221-225` | Unconditional transpose for F0_proj, N_proj |
| Kokoro istftnet | `kokoro/istftnet.py:967-968` | Unconditional transpose for noise_convs |
| Wav2vec | `stt/models/wav2vec/wav2vec.py:699-702` | Unconditional swapaxes |

Example of non-idempotent sanitize (Kokoro):
```python
if "F0_proj.weight" in key:
    sanitized_weights[key] = state_dict.transpose(0, 2, 1)  # Always transposes!
```

**Why this works**: Non-idempotent sanitize methods only work because their mlx-community repos contain PyTorch-format weights. If weights were pre-converted to MLX format, loading would fail.

## Helper Functions

### check_array_shape (`base.py:21-34`)

Heuristic to detect if a 3D tensor is in MLX format:
```python
def check_array_shape(arr):
    if len(arr.shape) != 3:
        return False
    out_channels, kH, KW = arr.shape
    # MLX format: out_channels is largest, kH == KW
    return (out_channels >= kH) and (out_channels >= KW) and (kH == KW)
```

## Conversion Script

MLX-audio has a conversion script at `mlx_audio/tts/convert.py` that:

1. Loads model via `fetch_from_hub()` (which calls `sanitize()`)
2. Optionally converts dtype
3. Optionally quantizes
4. Saves to safetensors

However, since sanitize happens during loading, the saved weights **should** be in MLX format. The inconsistency with Kokoro suggests either:
- The mlx-community repos were created differently (manual upload, not via convert script)
- Or the convert script wasn't used for all models

## Recommendations for New Models

When porting a new model to MLX:

1. **Make sanitize idempotent** - Check shapes before transposing:
   ```python
   def sanitize(self, weights):
       curr_weights = dict(tree_flatten(self.parameters()))
       for key, value in weights.items():
           if "conv" in key and value.ndim == 3:
               if value.shape != curr_weights[key].shape:
                   value = value.transpose(0, 2, 1)
           # ...
   ```

2. **Or use conversion script properly** - Ensure mlx-community repos have pre-converted weights

3. **Document the expected weight format** - Specify whether the model expects PyTorch or MLX format weights

## File References

- Conversion script: `mlx_audio/tts/convert.py`
- Main loading logic: `mlx_audio/tts/utils.py:152-270`
- Shape helper: `mlx_audio/tts/models/base.py:21-34`
- Example idempotent sanitize: `mlx_audio/tts/models/indextts/bigvgan.py:82-124`
- Example non-idempotent sanitize: `mlx_audio/tts/models/kokoro/kokoro.py:172-252`
