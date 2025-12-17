# Chatterbox Turbo Conversion Script Design

This document outlines the findings from analyzing the `ResembleAI/chatterbox-turbo` Hugging Face repository and provides guidance for creating a new MLX conversion script.

## Overview

Chatterbox Turbo is a streamlined 350M parameter TTS model that uses a different architecture than the original Chatterbox. While they share some components (VoiceEncoder, S3Tokenizer), the core T3 language model backbone is completely different, requiring a separate conversion script.

## Repository Comparison

### Source Files

| Component | Original Chatterbox | Chatterbox Turbo |
|-----------|---------------------|------------------|
| **HF Repo** | `ResembleAI/chatterbox` | `ResembleAI/chatterbox-turbo` |
| **T3 Weights** | `t3_cfg.safetensors` (292 keys) | `t3_turbo_v1.safetensors` (299 keys) |
| **S3Gen Weights** | `s3gen.safetensors` (2489 keys) | `s3gen_meanflow.safetensors` (2491 keys) |
| **VE Weights** | `ve.safetensors` (16 keys) | `ve.safetensors` (16 keys) |
| **Conditionals** | `conds.pt` (105 KB) | `conds.pt` (166 KB) |
| **Text Tokenizer** | `tokenizer.json` | `vocab.json`, `merges.txt`, `tokenizer_config.json` |
| **Config** | N/A | `t3_turbo_v1.yaml` |

### File Sizes

From the cloned `ResembleAI/chatterbox-turbo` repository:
- `t3_turbo_v1.safetensors`: 1.8 GB
- `s3gen_meanflow.safetensors`: 1016 MB
- `s3gen.safetensors`: 1008 MB (not used by Turbo)
- `ve.safetensors`: 5.5 MB
- `conds.pt`: 166 KB

## Architecture Differences

### T3 Model (Language Model Backbone)

This is the **most significant difference** between the two models.

#### Original Chatterbox T3
- **Backbone**: LLaMA architecture via `mlx_lm.models.llama.Model`
- **Weight structure**: `tfmr.layers.*.self_attn.{q_proj,k_proj,v_proj,o_proj}`, `tfmr.layers.*.mlp.{gate_proj,up_proj,down_proj}`
- **Conditioning**: Perceiver resampler (`cond_enc.perceiver.*`), emotion adversarial FC
- **Position embeddings**: Learned (`text_pos_emb.emb.weight`, `speech_pos_emb.emb.weight`)
- **Config**: `T3Config.english_only()` with `llama_config_name="Llama_520M"`

#### Chatterbox Turbo T3
- **Backbone**: GPT-2 architecture (custom implementation)
- **Weight structure**: `tfmr.h.*.attn.c_attn`, `tfmr.h.*.attn.c_proj`, `tfmr.h.*.mlp.c_fc`, `tfmr.h.*.mlp.c_proj`
- **Conditioning**: Simple speaker encoder (`cond_enc.spkr_enc.*`) - no perceiver
- **Position embeddings**: GPT-2 style (`tfmr.wpe.weight`, `tfmr.wte.weight`)
- **Config**: `T3Config.turbo()` with `llama_config_name="GPT2_medium"`

#### T3 Weight Key Comparison

**Original T3 keys (sample)**:
```
cond_enc.emotion_adv_fc.weight
cond_enc.perceiver.attn.to_q.weight
cond_enc.perceiver.attn.to_k.weight
tfmr.layers.0.self_attn.q_proj.weight
tfmr.layers.0.mlp.gate_proj.weight
text_pos_emb.emb.weight
speech_pos_emb.emb.weight
```

**Turbo T3 keys (sample)**:
```
cond_enc.spkr_enc.weight
cond_enc.spkr_enc.bias
tfmr.h.0.attn.c_attn.weight
tfmr.h.0.attn.c_attn.bias
tfmr.h.0.mlp.c_fc.weight
tfmr.wpe.weight
tfmr.wte.weight
speech_head.bias  # Note: has bias, original doesn't
```

### S3Gen (Speech Token to Waveform)

- **Original**: Uses `s3gen.safetensors`, initialized with `S3Gen(meanflow=False)`
- **Turbo**: Uses `s3gen_meanflow.safetensors`, initialized with `S3Gen(meanflow=True)`
- The meanflow variant is a distilled decoder using only 2 CFM steps instead of 10
- Weight structure is nearly identical (2489 vs 2491 keys)

### VoiceEncoder

- **Identical** between both models
- Same 16 LSTM weights
- Same architecture and sanitize method
- Can reuse conversion code directly

### S3Tokenizer

- **Identical** between both models
- Both use `S3TokenizerV2("speech_tokenizer_v2_25hz")`
- Already converted and available at `mlx-community/S3TokenizerV2`
- **No conversion needed** - shared component

## Reusable Components

The following can be reused from the original conversion script with minimal changes:

1. **VoiceEncoder conversion** - Identical
2. **S3Tokenizer** - Already converted, no work needed
3. **General script structure** - Download, convert, sanitize, save, upload pattern
4. **Utility functions** - `load_pytorch_safetensors`, `numpy_to_mlx`, `save_mlx_safetensors`, etc.
5. **Upload logic** - `upload_to_hub` function

## Components Requiring Changes

### 1. Download Function

```python
def download_chatterbox_turbo_weights(cache_dir: Path) -> Path:
    """Download Chatterbox Turbo weights from Hugging Face.

    Note: conds.pt is intentionally excluded (see personality rights note).
    Tokenizer files are downloaded for conversion to modern format.
    """
    from huggingface_hub import snapshot_download

    ckpt_dir = Path(
        snapshot_download(
            repo_id="ResembleAI/chatterbox-turbo",
            allow_patterns=[
                # Model weights
                "ve.safetensors",
                "t3_turbo_v1.safetensors",
                "s3gen_meanflow.safetensors",
                # Tokenizer (old format - will be converted)
                "vocab.json",
                "merges.txt",
                "tokenizer_config.json",
                "added_tokens.json",
                "special_tokens_map.json",
            ],
            cache_dir=cache_dir,
        )
    )
    return ckpt_dir
```

### 2. Model Imports

```python
# Use Turbo-specific models
from mlx_audio.tts.models.chatterbox_turbo.models.t3 import T3, T3Config
from mlx_audio.tts.models.chatterbox_turbo.models.s3gen import S3Gen
from mlx_audio.tts.models.chatterbox_turbo.models.voice_encoder import VoiceEncoder
```

### 3. Model Instantiation

```python
# T3 - use Turbo config
hp = T3Config.turbo()
t3 = T3(hp)

# S3Gen - use meanflow variant
s3gen = S3Gen(meanflow=True)

# VE - same as original
ve = VoiceEncoder()
```

### 4. Weight File Names

```python
# Load weights from Turbo-specific files
ve_weights = load_pytorch_safetensors(ckpt_dir / "ve.safetensors")
t3_weights = load_pytorch_safetensors(ckpt_dir / "t3_turbo_v1.safetensors")
s3gen_weights = load_pytorch_safetensors(ckpt_dir / "s3gen_meanflow.safetensors")
```

## Quantization Strategy

Following the same strategy as the original Chatterbox conversion, we should:

1. **Quantize**: T3 transformer layers only (the bulk of parameters)
2. **Keep full precision**:
   - VoiceEncoder (small, LSTM sensitive to quantization)
   - S3Gen (audio quality sensitive, flow matching compounds errors)
   - T3 embeddings and output heads (small, quality-critical)
   - T3 conditioning encoder (small, affects speaker similarity)

### Rationale for Selective Quantization

**Why `speech_head` must stay FP16 (CRITICAL)**:
- This is the output projection: `hidden_state (1024-dim) → logits (6563 tokens) → softmax → sample`
- Quantization errors directly corrupt token probabilities
- Small shifts in logits (0.1-0.5) can flip which token wins after softmax
- Wrong speech tokens = wrong phonemes, prosody errors, or gibberish
- Only 13 MB - minimal memory savings, maximum quality risk

**Why `speech_emb` must stay FP16**:
- Input embeddings are the "foundation" - errors propagate through all 24 transformer layers
- Small vocabulary (6,563 tokens) means each embedding row is used frequently
- Only 13 MB saved - not worth the quality degradation

**Why S3Gen must stay FP16**:
- Uses Conditional Flow Matching (CFM) - an ODE-based generative process
- Small errors in velocity prediction compound over integration steps
- Even Turbo's 2-step meanflow distillation is sensitive to quantization noise
- The HiFi-GAN vocoder is especially sensitive for final audio fidelity

### Expected Model Sizes

| Variant | Approximate Size |
|---------|------------------|
| FP16 (no quantization) | ~2.8 GB |
| 4-bit (T3 backbone only) | ~550 MB |
| 8-bit (T3 backbone only) | ~900 MB |

**When in doubt, err on the side of audio quality** - TTS is more sensitive to quantization than text LLMs because errors manifest as audible artifacts (metallic timbre, wrong phonemes, prosody issues).

### Quantization Predicate for Turbo

The original script quantizes `tfmr.model.layers.*`. For Turbo's GPT-2 architecture, we need to target `tfmr.h.*`:

```python
def quantize_t3_turbo_backbone(model, bits: int = 4, group_size: int = 64):
    """
    Selectively quantize the T3 Turbo GPT-2 backbone.

    Only quantizes tfmr.h.* (attention and MLP layers).
    Other components are kept in full precision.

    Args:
        model: T3 model instance
        bits: Quantization bits (default: 4)
        group_size: Quantization group size (default: 64)

    Returns:
        Number of layers quantized
    """
    import mlx.nn as nn

    quantized_count = [0]

    def should_quantize(path, module):
        """Only quantize T3 GPT-2 transformer layers."""
        if isinstance(module, nn.Linear):
            # Target GPT-2 transformer layers: tfmr.h.*
            # For full model with prefix: t3.tfmr.h.*
            if "tfmr.h." in path:
                quantized_count[0] += 1
                return True
        return False

    nn.quantize(
        model, bits=bits, group_size=group_size, class_predicate=should_quantize
    )
    return quantized_count[0]
```

### What Gets Quantized (Turbo)

When quantizing `tfmr.h.*`, these layers will be quantized:
- `tfmr.h.*.attn.c_attn` - Combined Q/K/V projection
- `tfmr.h.*.attn.c_proj` - Output projection
- `tfmr.h.*.mlp.c_fc` - MLP first layer
- `tfmr.h.*.mlp.c_proj` - MLP second layer

### What Stays Full Precision

**Critical for quality** (never quantize):
- `t3.speech_head.*` - Speech output head (token selection)
- `t3.speech_emb.*` - Speech embeddings (error propagation)
- `s3gen.*` - S3Gen decoder (audio fidelity)

**Important for quality** (keep FP16):
- `ve.*` - VoiceEncoder (LSTM layers, speaker similarity)
- `t3.cond_enc.*` - T3 conditioning encoder (speaker identity)
- `t3.tfmr.wpe.*` - Position embeddings (sequence modeling)
- `t3.tfmr.wte.*` - Token embeddings (if used)
- `t3.tfmr.ln_f.*` - Final layer norm (output stability)

**Lower priority** (FP16 preferred but less critical):
- `t3.text_emb.*` - Text embeddings (input only)
- `t3.text_head.*` - Text output head (unused during speech generation)

### If Experimenting with S3Gen Quantization

If memory constraints require quantizing S3Gen (not recommended), these components should **always stay FP16**:
- All convolution weights (`*.conv.conv.weight/bias`) - critical for HiFi-GAN audio quality
- All normalization layers (`*norm*.weight/bias`) - numerical stability
- Position embeddings (`*.pos_enc.pe`, `*.pos_bias_*`) - timing accuracy
- The `trim_fade` parameters - audio processing constants

Only the attention Q/K/V/O projections and MLP linear layers in the transformer blocks could potentially be quantized, but this is not recommended without thorough audio quality testing.

## Config File

Create a `config.json` for the converted model:

```json
{
  "model_type": "chatterbox_turbo",
  "version": "1.0",
  "quantization": {
    "bits": 4,
    "group_size": 64,
    "quantized_components": ["t3.tfmr.h"]
  }
}
```

## Tokenizer Conversion

The Turbo HF repo uses the older GPT-2 tokenizer format (separate `vocab.json` and `merges.txt` files). We should convert to the modern `tokenizer.json` format for consistency with the original Chatterbox conversion.

```python
def convert_tokenizer(ckpt_dir: Path, output_dir: Path):
    """Convert tokenizer to modern format."""
    from transformers import AutoTokenizer

    # Load from old format
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))

    # Save in modern format (produces tokenizer.json)
    tokenizer.save_pretrained(str(output_dir))
```

This produces:
- `tokenizer.json` (3.5 MB) - Modern unified format
- `tokenizer_config.json` - Configuration
- Plus legacy files for backwards compatibility

## Pre-computed Voice Conditionals

### What conds.pt Contains

The `conds.pt` file contains pre-extracted features from a reference audio clip, used as a "default voice" when no reference audio is provided:

```
t3:
  speaker_emb: (1, 256)                  # VoiceEncoder speaker embedding
  cond_prompt_speech_tokens: (1, 375)    # S3Tokenizer speech tokens
  emotion_adv: (1, 1, 1)                 # Emotion exaggeration factor

gen:
  prompt_token: (1, 250)                 # S3Gen reference speech tokens
  prompt_token_len: (1,)                 # Token length
  prompt_feat: (1, 500, 80)              # Mel spectrogram features
  embedding: (1, 192)                    # S3Gen speaker embedding
```

### How Conditionals Are Generated

These features are created by running `prepare_conditionals()` on a reference audio file (must be > 5 seconds):

1. **VoiceEncoder** → `speaker_emb` (256-dim speaker embedding)
2. **S3Tokenizer** → `cond_prompt_speech_tokens` (speech tokens for T3 prompt)
3. **S3Gen.embed_ref()** → `prompt_token`, `prompt_feat`, `embedding` (decoder conditioning)

### Unknown Origin - Personality Rights Concern

**ResembleAI does not publicly document the source of the default voice in `conds.pt`.** We do not know whether it is:
- A synthetic/artificial voice
- A recording from a team member with consent
- Licensed audio from a voice actor
- A sample from training data

**Due to potential personality rights concerns, we may want to EXCLUDE the pre-computed conditionals from our converted model.** Voice cloning technology raises ethical and legal issues around consent, and distributing voice embeddings of unknown origin could potentially violate someone's personality rights.

### Recommendation

**The existing Chatterbox conversion script already excludes `conds.pt`** - it only downloads `ve.safetensors`, `t3_cfg.safetensors`, `s3gen.safetensors`, and `tokenizer.json`. We should follow this same precedent for Turbo.

Options if conditionals are ever needed:

1. **Exclude conditionals entirely** (recommended, matches existing behavior) - Require users to always provide their own reference audio
2. **Generate new conditionals** - Use a known, properly licensed reference audio
3. **Include with disclaimer** - Convert as-is but document the unknown provenance

If excluding conditionals, the model will require `ref_audio` parameter for all generations, which is the typical use case for voice cloning anyway.

### Conditionals Conversion (If Included)

If we decide to include the conditionals, convert `conds.pt` to safetensors format:

```python
def convert_conditionals(ckpt_dir: Path, output_dir: Path):
    """Convert conds.pt to conds.safetensors."""
    import torch
    import numpy as np
    from safetensors.numpy import save_file

    conds_data = torch.load(ckpt_dir / "conds.pt", map_location="cpu", weights_only=True)

    conds_dict = {}

    # T3 conditionals
    if "t3" in conds_data:
        t3_cond = conds_data["t3"]
        if "speaker_emb" in t3_cond:
            conds_dict["t3.speaker_emb"] = t3_cond["speaker_emb"].numpy()
        if "cond_prompt_speech_tokens" in t3_cond:
            conds_dict["t3.cond_prompt_speech_tokens"] = t3_cond["cond_prompt_speech_tokens"].numpy().astype(np.int32)

    # Gen conditionals (S3Gen reference embeddings)
    if "gen" in conds_data:
        for k, v in conds_data["gen"].items():
            if hasattr(v, "numpy"):
                conds_dict[f"gen.{k}"] = v.numpy()

    save_file(conds_dict, output_dir / "conds.safetensors")
```

## README Template

```markdown
---
library_name: mlx-audio-plus
base_model:
- ResembleAI/chatterbox-turbo
tags:
- mlx
pipeline_tag: text-to-speech
---

# {upload_repo}

This model was converted to MLX format from [ResembleAI/chatterbox-turbo](https://huggingface.co/ResembleAI/chatterbox-turbo).

**Note:** This model requires the S3Tokenizer weights from [mlx-community/S3TokenizerV2](https://huggingface.co/mlx-community/S3TokenizerV2), which will be downloaded automatically.

## Use with mlx-audio-plus

```bash
pip install -U mlx-audio-plus
```

### Command line

```bash
mlx_audio.tts --model {upload_repo} --text "Hello, this is Chatterbox Turbo!" --ref_audio reference.wav
```

### Python

```python
from mlx_audio.tts.generate import generate_audio

generate_audio(
    text="Hello, this is Chatterbox Turbo!",
    model="{upload_repo}",
    ref_audio="reference.wav",
    file_prefix="output",
)
```
```

## Implementation Checklist

- [ ] Create `mlx_audio/tts/models/chatterbox_turbo/scripts/convert.py`
- [ ] Implement `download_chatterbox_turbo_weights()` function (excludes conds.pt)
- [ ] Implement `convert_tokenizer()` function (old GPT-2 format → modern tokenizer.json)
- [ ] Implement `quantize_t3_turbo_backbone()` function
- [ ] Implement `convert_all()` main conversion function
- [ ] Implement `generate_readme()` for model card
- [ ] Add CLI argument parsing
- [ ] Test fp16 conversion
- [ ] Test 4-bit quantized conversion
- [ ] Test 8-bit quantized conversion
- [ ] Verify converted model produces audio (with ref_audio)
- [ ] Upload to mlx-community

## Expected Output Structure

```
Chatterbox-Turbo-TTS-fp16/
├── model.safetensors       # Combined weights (ve.*, t3.*, s3gen.*)
├── config.json             # Model configuration
├── tokenizer.json          # Modern unified tokenizer format
├── tokenizer_config.json   # Tokenizer configuration
├── special_tokens_map.json # Special tokens
├── added_tokens.json       # Additional tokens
├── vocab.json              # Legacy format (for compatibility)
├── merges.txt              # Legacy format (for compatibility)
└── README.md               # Model card
```

Note: `conds.safetensors` is intentionally excluded (matching existing Chatterbox conversion). Users must always provide `ref_audio` for voice cloning.

## Notes

1. The Turbo model is optimized for low-latency voice agents with 2-step CFM decoding
2. It does not support CFG (classifier-free guidance) or exaggeration controls
3. Paralinguistic tags (`[laugh]`, `[cough]`, etc.) are natively supported
4. Sample rate is 24kHz (S3GEN_SR) for output, 16kHz for S3Tokenizer input
