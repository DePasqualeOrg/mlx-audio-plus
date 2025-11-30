# PyTorch to MLX Porting Guide

A practical guide for porting PyTorch models to MLX in Python for inference. This document covers best practices, common patterns, and gotchas learned from porting multiple TTS models in this repository.

## Table of Contents

1. [Core Principles](#core-principles)
2. [Module Structure](#module-structure)
3. [Weight Conversion](#weight-conversion)
4. [Performance Optimization](#performance-optimization)
5. [Common Layer Mappings](#common-layer-mappings)
6. [Gotchas and Solutions](#gotchas-and-solutions)
7. [Testing and Validation](#testing-and-validation)
8. [Reference Files](#reference-files)

---

## Core Principles

### 1. Use MLX Built-ins

Always prefer MLX built-in functions over reimplementing functionality or importing external dependencies.

```python
# Good - use MLX built-ins
import mlx.core as mx
import mlx.nn as nn

attention = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
output = nn.gelu(x)

# Avoid - reimplementing or using numpy/scipy unnecessarily
import numpy as np
attention = custom_attention_implementation(q, k, v)
output = 0.5 * x * (1 + np.tanh(...))  # Manual GELU
```

### 2. Replace Loops with Parallel Operations

Vectorize operations using MLX's array operations instead of Python loops.

```python
# Avoid - sequential loop
result = []
for i in range(n):
    result.append(process(x[i]))
result = mx.stack(result)

# Good - parallel operation
result = mx.vmap(process)(x)

# Or use broadcasting/indexing
result = process(x)  # If process supports batch dimension
```

### 3. Lazy Evaluation Awareness

MLX uses lazy evaluation. Use `mx.eval()` strategically to force computation when needed (e.g., before timing measurements or when you need actual values).

```python
result = model(x)
mx.eval(result)  # Force computation to complete
```

---

## Module Structure

### Basic Module Pattern

### Sanitize Method

Every model should implement a `sanitize()` method for weight conversion:

```python
def sanitize(self, weights: dict) -> dict:
    """Convert PyTorch weights to MLX format.

    This method should be idempotent - safe to call multiple times.
    """
    # Get current model's expected shapes
    curr_weights = dict(tree_flatten(self.parameters()))
    sanitized = {}

    for key, value in weights.items():
        new_key = key

        # 1. Key renaming
        new_key = new_key.replace(".gamma", ".weight")
        new_key = new_key.replace(".beta", ".bias")

        # 2. Shape transformations (idempotent - check before transposing)
        if "conv" in new_key and value.ndim == 3:
            if new_key in curr_weights and value.shape != curr_weights[new_key].shape:
                value = mx.swapaxes(value, 1, 2)  # Conv1d transpose

        # 3. Skip unused weights
        if "position_ids" in new_key or "num_batches_tracked" in new_key:
            continue

        sanitized[new_key] = value

    return sanitized
```

---

## Weight Conversion

### Convolution Weight Transposition

| Layer Type | PyTorch Shape | MLX Shape | Transposition |
|------------|---------------|-----------|---------------|
| Conv1d | `(out, in, kernel)` | `(out, kernel, in)` | `swapaxes(1, 2)` |
| Conv2d | `(out, in, H, W)` | `(out, H, W, in)` | `transpose(0, 2, 3, 1)` |
| ConvTranspose1d | `(in, out, kernel)` | `(out, kernel, in)` | `transpose(1, 2, 0)` |

```python
# Conv1d
if value.ndim == 3 and value.shape != expected_shape:
    value = mx.swapaxes(value, 1, 2)

# Conv2d
if value.ndim == 4 and value.shape != expected_shape:
    value = mx.transpose(value, (0, 2, 3, 1))
```

### LSTM Weight Mapping

PyTorch LSTM uses different weight names than MLX:

```python
# PyTorch → MLX: weight_ih_l0 → Wx, weight_hh_l0 → Wh
# Bidirectional: weight_ih_l0_reverse → Wx_backward, etc.
# Biases: bias_ih_l0 → bias_ih (often combined: bias = bias_ih + bias_hh)
```

### Transformer/LLaMA Weight Mapping

```python
# Common attention renames
attention_map = {
    ".attn.to_q.": ".attn.query_proj.",
    ".attn.to_k.": ".attn.key_proj.",
    ".attn.to_v.": ".attn.value_proj.",
    ".attn.to_out.0.": ".attn.out_proj.",
}

# Layer norm renames
norm_map = {
    ".gamma": ".weight",
    ".beta": ".bias",
}

# MLP renames (LLaMA style)
mlp_map = {
    ".mlp.w1.": ".gate_proj.",
    ".mlp.w2.": ".down_proj.",
    ".mlp.w3.": ".up_proj.",
}
```

### Weight Normalization

PyTorch weight normalization stores `g` (magnitude) and `v` (direction) separately:

```python
# Merge parametrized weights: w = g * v / ||v||
if "parametrizations.weight.original0" in key:  # g
    base_key = key.replace(".parametrizations.weight.original0", ".weight")
    g = value
    v = weights[key.replace("original0", "original1")]
    v_norm = mx.sqrt(mx.sum(v * v, axis=tuple(range(1, v.ndim)), keepdims=True))
    merged_weight = g * v / (v_norm + 1e-12)
    sanitized[base_key] = merged_weight
```

---

## Performance Optimization

### 1. Use `mx.fast` Operations

MLX provides optimized implementations for common operations:

```python
# Attention - use mx.fast.scaled_dot_product_attention
output = mx.fast.scaled_dot_product_attention(
    queries, keys, values,
    scale=1.0 / math.sqrt(head_dim),
    mask=mask  # Can be None, "causal", or an mx.array
)

# Rotary Position Embeddings - use mx.fast.rope
queries = mx.fast.rope(queries, dims, traditional=False, base=10000, offset=cache_offset)
keys = mx.fast.rope(keys, dims, traditional=False, base=10000, offset=cache_offset)

# RMS Normalization - use mx.fast.rms_norm
output = mx.fast.rms_norm(x, weight, eps=1e-6)

# Layer Normalization - use mx.fast.layer_norm
output = mx.fast.layer_norm(x, weight, bias, eps=1e-5)
```

Avoid manual implementations of these operations.

### 2. KV Caching for Autoregressive Generation

```python
def __call__(self, x, cache=None):
    q, k, v = self.compute_qkv(x)

    if cache is not None:
        k_cache, v_cache = cache
        k = mx.concatenate([k_cache, k], axis=2)
        v = mx.concatenate([v_cache, v], axis=2)

    new_cache = (k, v)
    output = self.attention(q, k, v)

    return output, new_cache
```

### 3. Precompute Static Values

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Precompute positional encodings once
        pe = mx.zeros((max_len, d_model))
        position = mx.arange(max_len)[:, None]
        div_term = mx.exp(mx.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = mx.sin(position * div_term)
        pe[:, 1::2] = mx.cos(position * div_term)
        self._pe = mx.expand_dims(pe, 0)  # (1, max_len, d_model)

    def __call__(self, x):
        return x + self._pe[:, :x.shape[1]]
```

### 4. Vectorize Repetitive Operations

```python
# Avoid - loop over heads
outputs = []
for i in range(num_heads):
    outputs.append(self.heads[i](x[:, :, i*head_dim:(i+1)*head_dim]))
output = mx.concatenate(outputs, axis=-1)

# Good - reshape and use single operation
x = x.reshape(batch, seq_len, num_heads, head_dim)
x = x.transpose(0, 2, 1, 3)  # (batch, heads, seq, dim)
output = self.attention(x)  # Process all heads at once
```

### 5. In-Place Updates for Overlap-Add

```python
# For operations like iSTFT overlap-add, use at[].add()
output = mx.zeros(output_length)
for i, frame in enumerate(frames):
    start = i * hop_length
    output = output.at[start:start+frame_size].add(frame * window)
```

---

## Common Layer Mappings

| PyTorch | MLX | Notes |
|---------|-----|-------|
| `nn.Linear` | `nn.Linear` | Direct mapping |
| `nn.Conv1d` | `nn.Conv1d` | Transpose weights |
| `nn.Conv2d` | `nn.Conv2d` | Transpose weights |
| `nn.LayerNorm` | `nn.LayerNorm` | Direct mapping |
| `nn.BatchNorm1d/2d` | `nn.BatchNorm` | Check inference mode |
| `nn.LSTM` | `nn.LSTM` | Rename weights |
| `nn.GRU` | `nn.GRU` | Rename weights |
| `nn.Embedding` | `nn.Embedding` | Direct mapping |
| `nn.Dropout` | `nn.Dropout` | Direct mapping |
| `nn.GELU` | `nn.GELU` | Direct mapping |
| `nn.SiLU` | `nn.SiLU` | Also known as Swish |
| `nn.ReLU` | `nn.ReLU` | Direct mapping |
| `nn.Softmax` | `mx.softmax` | Use functional version |
| `nn.GroupNorm` | `nn.GroupNorm` | Direct mapping |
| `nn.RMSNorm` | `nn.RMSNorm` | Or use `mx.fast.rms_norm` |

---

## Gotchas and Solutions

### 1. Underscore-Prefixed Attributes Are Skipped

MLX's `tree_flatten` skips attributes starting with `_`:

```python
# Wrong - will not be included in parameters
self._codebook = mx.array(...)

# Correct - will be included
self.codebook = mx.array(...)
```

### 2. Dimension Ordering for Convolutions

MLX convolutions expect different input ordering. Be explicit about transposes:

```python
# Input for Conv1d: (batch, channels, length) in PyTorch
# For LayerNorm after Conv1d, you may need to transpose
x = mx.swapaxes(x, 1, 2)  # (B, C, T) -> (B, T, C)
x = self.norm(x)
x = mx.swapaxes(x, 1, 2)  # (B, T, C) -> (B, C, T)
```

### 3. STFT/iSTFT Frame Count Differences

PyTorch's `torch.stft` may produce different frame counts:

```python
# If PyTorch produces N frames and MLX produces N+1, drop the last frame
magnitudes = spec[:-1, :].abs() ** 2  # Match PyTorch behavior
```

### 4. Window Functions: Periodic vs Symmetric

For FFT operations, use periodic windows:

```python
# Periodic (for FFT) - use fftbins=True equivalent
window = mx.array(scipy.signal.get_window('hann', n_fft, fftbins=True))

# Symmetric (for filtering) - default scipy behavior
window = mx.array(scipy.signal.get_window('hann', n_fft))
```

### 5. Attention Mask Handling

`mx.fast.scaled_dot_product_attention` accepts three mask types:

```python
# Option 1: None (no masking)
# Option 2: "causal" string (efficient causal masking)
# Option 3: mx.array boolean or additive mask

def create_attention_mask(seq_len, cache_offset=0):
    if seq_len == 1:
        return None  # Single token doesn't need mask
    return "causal"  # Let MLX handle it efficiently
```

### 6. Attention Inner Dimensions

Some models (e.g., diffusers) use different projection dimensions. Check weight shapes carefully - `query_dim -> inner_dim (heads * dim_head) -> query_dim`.

### 7. Idempotent Sanitization

Always check shapes before transforming to make `sanitize()` safe to call multiple times:

```python
# Good - idempotent
if value.shape != expected_shape:
    value = mx.swapaxes(value, 1, 2)

# Bad - will corrupt already-converted weights
value = mx.swapaxes(value, 1, 2)  # Always transposes!
```

### 8. Dtype Handling

Convert float64 to float32 during weight loading:

```python
if value.dtype == mx.float64:
    value = value.astype(mx.float32)
```

### 9. Causal Convolution Padding

For causal convolutions, pad on the left only:

```python
# Left padding for causality
x = mx.pad(x, [(0, 0), (0, 0), (kernel_size - 1, 0)])
```

### 10. No Negative Step Slicing

MLX doesn't support negative step slicing. Use `mx.flip` instead:

```python
# Wrong - will raise an error
reversed_x = x[::-1]

# Correct
reversed_x = mx.flip(x, axis=0)
```

### 11. Arrays Are Immutable

MLX arrays cannot be modified in-place. Avoid augmented assignment on arrays:

```python
# Wrong - will fail or create unexpected behavior
x += 1
x[0] = 5

# Correct
x = x + 1
x = x.at[0].add(5 - x[0])  # For single element updates, or rebuild the array
```

---

## Testing and Validation

### 1. Compare PyTorch vs MLX Outputs

```python
# Generate same input, run both models, compare
np_input = np.random.randn(1, 10, 256).astype(np.float32)
torch_output = torch_model(torch.from_numpy(np_input)).detach().numpy()
mlx_output = np.array(mlx_model(mx.array(np_input)))
np.testing.assert_allclose(mlx_output, torch_output, rtol=1e-4, atol=1e-4)
```

### 2. Verify Weight Loading

```python
model.load_weights(weights_path, strict=True)  # Catch missing/unexpected weights
```

### 3. Test Quantization Sensitivity

Transformers/MLPs typically quantize well at 4-bit. Flow matching, vocoders, and discrete encoders often cannot be quantized:

```python
nn.quantize(model, bits=4, group_size=64, class_predicate=lambda path, m:
    isinstance(m, nn.Linear) and "transformer.layers" in path)
```

---

## Reference Files

### Model Examples in This Repository

| Model | Location | Notable Patterns |
|-------|----------|------------------|
| Chatterbox | `mlx_audio/tts/models/chatterbox/` | Complex multi-component, flow matching, CFG |
| Kokoro | `mlx_audio/tts/models/kokoro/` | LSTM, ISTFTNet vocoder |
| Bark | `mlx_audio/tts/models/bark/` | GPT-style, multiple codebooks |
| IndextTS | `mlx_audio/tts/models/indextts/` | BigVGAN, conformer |

### Key Files to Study

- **Attention**: `mlx_audio/tts/models/chatterbox/s3gen/transformer/attention.py`
- **Convolution**: `mlx_audio/tts/models/chatterbox/s3gen/transformer/convolution.py`
- **LSTM**: `mlx_audio/tts/models/kokoro/modules.py`
- **Weight conversion**: `mlx_audio/tts/models/chatterbox/scripts/convert_chatterbox.py`
- **Base patterns**: `mlx_audio/tts/models/base.py`

### External References

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [mlx-lm Repository](https://github.com/ml-explore/mlx-lm) - LLM porting patterns
- [MLX Examples](https://github.com/ml-explore/mlx-examples) - Official examples

