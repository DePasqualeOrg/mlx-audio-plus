# Chatterbox Turbo Model Analysis

**Date:** December 15, 2025
**Source:** https://huggingface.co/ResembleAI/chatterbox-turbo
**PyTorch Reference:** https://github.com/resemble-ai/chatterbox

## Executive Summary

Chatterbox Turbo is a new, more efficient variant of the Chatterbox TTS model from Resemble AI. After thorough analysis, **the current MLX port cannot support Chatterbox Turbo without significant modifications**. The key blockers are:

1. Different transformer backbone (GPT-2 vs LLaMA)
2. Different tokenizer (HuggingFace GPT-2 vs custom BPE)
3. MeanFlow decoder (distilled 1-step vs 10-step CFM)

This document provides a comprehensive analysis of the architectural differences and the work required to add Turbo support.

---

## 1. Model Overview

### Chatterbox Turbo Key Features

- **350M parameters** (vs 500M in original)
- **English only** with paralinguistic tags (`[laugh]`, `[cough]`, etc.)
- **1-2 step decoder** (distilled from 10-step CFM)
- **No CFG/emotion control** (simpler inference)
- **GPT-2 backbone** instead of LLaMA

### Model Files (from HuggingFace)

```
chatterbox-turbo/
├── t3_turbo_v1.safetensors    # 1.9 GB - T3 model (GPT-2 based)
├── t3_turbo_v1.yaml           # T3 config
├── s3gen_meanflow.safetensors # 1.0 GB - Distilled decoder
├── s3gen.safetensors          # 1.0 GB - Original decoder (unused in Turbo)
├── ve.safetensors             # 5.7 MB - Voice encoder (same as original)
├── conds.pt                   # Pre-computed conditionals
├── vocab.json                 # GPT-2 vocabulary
├── merges.txt                 # GPT-2 BPE merges
├── tokenizer_config.json      # HuggingFace tokenizer config
├── added_tokens.json          # Special tokens
└── special_tokens_map.json    # Token mappings
```

---

## 2. Architectural Comparison

### 2.1 T3 (Text-to-Token) Model

| Attribute | Original Chatterbox | Chatterbox Turbo |
|-----------|---------------------|------------------|
| **Backbone** | LLaMA 520M | GPT-2 Medium |
| **Layers** | 30 | 24 |
| **Hidden Size** | 1024 | 1024 |
| **Attention Heads** | 16 | 16 |
| **Position Embedding** | Learned (custom) | Internal (GPT-2 native) |
| **Attention Type** | RoPE | Absolute |
| **Text Vocab Size** | 704 (custom BPE) | 50,276 (GPT-2) |
| **Speech Vocab Size** | 8,194 | 6,563 |
| **Perceiver Resampler** | Yes (32 queries) | No |
| **Emotion Control** | Yes (`emotion_adv`) | No |
| **CFG Support** | Yes | No |
| **Speech Prompt Length** | 150 tokens | 375 tokens |
| **Max Speech Tokens** | 4,096 | 604 |
| **Max Text Tokens** | 2,048 | 402 |

#### Weight Structure Differences

**Original (LLaMA-based):**
```
tfmr.model.layers.0.self_attn.q_proj.weight
tfmr.model.layers.0.self_attn.k_proj.weight
tfmr.model.layers.0.self_attn.v_proj.weight
tfmr.model.layers.0.self_attn.o_proj.weight
tfmr.model.layers.0.mlp.gate_proj.weight
tfmr.model.layers.0.mlp.up_proj.weight
tfmr.model.layers.0.mlp.down_proj.weight
tfmr.model.norm.weight
```

**Turbo (GPT-2-based):**
```
tfmr.h.0.attn.c_attn.weight      # Combined QKV projection
tfmr.h.0.attn.c_attn.bias
tfmr.h.0.attn.c_proj.weight      # Output projection
tfmr.h.0.attn.c_proj.bias
tfmr.h.0.mlp.c_fc.weight         # First MLP layer
tfmr.h.0.mlp.c_fc.bias
tfmr.h.0.mlp.c_proj.weight       # Second MLP layer
tfmr.h.0.mlp.c_proj.bias
tfmr.h.0.ln_1.weight/bias        # LayerNorm (not RMSNorm)
tfmr.h.0.ln_2.weight/bias
```

### 2.2 S3Gen (Token-to-Waveform) Decoder

| Attribute | Original | Turbo (MeanFlow) |
|-----------|----------|------------------|
| **CFM Timesteps** | 10 | 2 |
| **Decoder Type** | ConditionalDecoder | ConditionalDecoder + MeanFlow |
| **Time Embedding** | Standard | Mixed (t + r) |
| **Noise Init** | Standard | Random noise input |
| **Weight File** | s3gen.safetensors | s3gen_meanflow.safetensors |

#### MeanFlow Differences

The MeanFlow decoder is a distilled version that requires:

1. **Time Embed Mixer**: Additional MLP that mixes start time `t` and end time `r` embeddings
2. **Different Noise Handling**: Requires explicit noise input during inference
3. **Fewer Timesteps**: Uses 2 steps instead of 10

```python
# MeanFlow specific code in decoder.py
if self.meanflow:
    r = self.time_embeddings(r).to(t.dtype)
    r = self.time_mlp(r)
    concat_embed = torch.cat([t, r], dim=1)
    t = self.time_embed_mixer(concat_embed)
```

### 2.3 Voice Encoder

The voice encoder is **identical** between both models:
- Same architecture (3-layer stacked LSTM)
- Same weight file (`ve.safetensors`)
- Same embedding dimension (256)

### 2.4 Tokenizer

| Attribute | Original | Turbo |
|-----------|----------|-------|
| **Type** | Custom BPE | HuggingFace GPT-2 |
| **Vocab Size** | 704 | 50,276 |
| **Start Text Token** | 255 | 50256 (BOS) |
| **Stop Text Token** | 0 | 50256 (EOS) |
| **Loading** | tokenizer.json | AutoTokenizer.from_pretrained() |

---

## 3. Inference Flow Comparison

### Original Chatterbox

```python
# 1. Prepare conditionals with perceiver resampler
conds = model.prepare_conditionals(ref_wav, exaggeration=0.5)

# 2. Tokenize with custom BPE
text_tokens = tokenizer.text_to_tokens(text)  # Adds BOT/EOT

# 3. T3 inference with CFG
speech_tokens = t3.inference(
    t3_cond=conds.t3,
    text_tokens=text_tokens,
    cfg_weight=0.5,           # Classifier-free guidance
    temperature=0.8,
    repetition_penalty=1.2,
)

# 4. S3Gen with 10 CFM steps
wav = s3gen.inference(speech_tokens, ref_dict=conds.gen, n_cfm_timesteps=10)
```

### Chatterbox Turbo

```python
# 1. Prepare conditionals (no perceiver)
conds = model.prepare_conditionals(ref_wav)  # No exaggeration param

# 2. Tokenize with HuggingFace GPT-2
text_tokens = tokenizer(text, return_tensors="pt").input_ids

# 3. T3 turbo inference (no CFG)
speech_tokens = t3.inference_turbo(
    t3_cond=conds.t3,
    text_tokens=text_tokens,
    temperature=0.8,
    top_k=1000,
    top_p=0.95,
    repetition_penalty=1.2,
)

# 4. S3Gen MeanFlow with 2 CFM steps
wav = s3gen.inference(speech_tokens, ref_dict=conds.gen, n_cfm_timesteps=2)
```

---

## 4. Required Changes for MLX Port

### 4.1 T3 Model Changes (High Priority)

**File: `mlx_audio/tts/models/chatterbox/t3/t3.py`**

1. **Add GPT-2 Backend Support**
   - Current implementation uses `mlx_lm.models.llama.Model`
   - Need to add conditional loading of GPT-2 model
   - Option A: Use `mlx_lm.models.gpt2` if available
   - Option B: Implement custom GPT-2 module

2. **Conditional Architecture**
   ```python
   def __init__(self, hp: T3Config):
       if hp.backbone_type == "gpt2":
           self.tfmr = GPT2Model(self.cfg)
           self.is_gpt = True
       else:
           self.tfmr = LlamaModel(self.cfg)
           self.is_gpt = False
   ```

3. **Remove Position Embeddings for GPT-2**
   ```python
   # Only create position embeddings for non-GPT models
   if hp.input_pos_emb == "learned" and not self.is_gpt:
       self.text_pos_emb = LearnedPositionEmbeddings(...)
       self.speech_pos_emb = LearnedPositionEmbeddings(...)
   ```

4. **Add `inference_turbo()` Method**
   - Simpler than original inference (no CFG)
   - Uses top_k sampling
   - No position embedding additions during generation

5. **Disable Perceiver for Turbo**
   ```python
   # In cond_enc.py
   if hp.use_perceiver_resampler:
       self.perceiver = PerceiverResampler(...)
   else:
       self.perceiver = None
   ```

### 4.2 Config Updates (Medium Priority)

**File: `mlx_audio/tts/models/chatterbox/config.py`**

```python
GPT2_MEDIUM_CONFIG = {
    "model_type": "gpt2",
    "hidden_size": 1024,
    "n_layer": 24,
    "n_head": 16,
    "n_positions": 8196,
    "vocab_size": 50276,
    "activation_function": "gelu_new",
    "layer_norm_epsilon": 1e-05,
}

LLAMA_CONFIGS = {
    "Llama_520M": LLAMA_520M_CONFIG,
    "GPT2_medium": GPT2_MEDIUM_CONFIG,
}

@dataclass
class T3TurboConfig:
    """Configuration for T3 Turbo model."""
    text_tokens_dict_size: int = 50276
    speech_tokens_dict_size: int = 6563
    speech_cond_prompt_len: int = 375
    max_speech_tokens: int = 604
    max_text_tokens: int = 402

    llama_config_name: str = "GPT2_medium"
    input_pos_emb: str = None  # Handled by GPT-2

    use_perceiver_resampler: bool = False
    emotion_adv: bool = False

    start_speech_token: int = 6561
    stop_speech_token: int = 6562
    start_text_token: int = 255
    stop_text_token: int = 0
```

### 4.3 S3Gen MeanFlow Support (Medium Priority)

**File: `mlx_audio/tts/models/chatterbox/s3gen/decoder.py`**

1. **Add MeanFlow Flag**
   ```python
   class ConditionalDecoder(nn.Module):
       def __init__(self, ..., meanflow=False):
           self.meanflow = meanflow
           if self.meanflow:
               self.time_embed_mixer = get_intmeanflow_time_mixer(time_embed_dim)
   ```

2. **Create IntMeanFlow Mixer**

**New File: `mlx_audio/tts/models/chatterbox/s3gen/utils/intmeanflow.py`**

```python
import mlx.core as mx
import mlx.nn as nn

def get_intmeanflow_time_mixer(time_embed_dim: int) -> nn.Module:
    """Create time embedding mixer for meanflow mode."""
    return nn.Sequential(
        nn.Linear(time_embed_dim * 2, time_embed_dim * 4),
        nn.GELU(),
        nn.Linear(time_embed_dim * 4, time_embed_dim),
    )
```

3. **Update Flow Inference**

**File: `mlx_audio/tts/models/chatterbox/s3gen/flow_matching.py`**

```python
def inference(self, ..., meanflow=False, n_timesteps=None):
    n_timesteps = n_timesteps or (2 if meanflow else 10)

    if meanflow:
        # Initialize with random noise
        noise = mx.random.normal(shape=(1, 80, token_len * 2))
        # Use 2-step inference
        ...
```

### 4.4 Tokenizer Updates (Low Priority)

**File: `mlx_audio/tts/models/chatterbox/tokenizer.py`**

```python
from transformers import AutoTokenizer

class TurboTokenizer:
    """GPT-2 based tokenizer for Chatterbox Turbo."""

    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(self, text: str) -> mx.array:
        tokens = self.tokenizer(text, return_tensors="np").input_ids
        return mx.array(tokens)
```

### 4.5 Weight Conversion Script (Medium Priority)

**File: `mlx_audio/tts/models/chatterbox/scripts/convert_turbo.py`**

Key conversion tasks:
1. Map GPT-2 weight names to MLX format
2. Handle combined QKV weights (`c_attn` → separate Q, K, V if needed)
3. Transpose Conv1d weights
4. Load MeanFlow decoder separately

```python
def convert_t3_turbo(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Convert T3 Turbo GPT-2 weights to MLX format."""
    new_weights = {}

    for key, value in weights.items():
        new_key = key

        # GPT-2 uses tfmr.h.X, keep as-is or map to mlx_lm format
        if key.startswith("tfmr.h."):
            # Handle GPT-2 specific weight transformations
            pass

        # Handle combined c_attn weights if needed
        if "c_attn.weight" in key:
            # Shape: (1024, 3072) -> split into q, k, v
            pass

        new_weights[new_key] = value

    return new_weights
```

### 4.6 Main Model Class (Medium Priority)

**New File: `mlx_audio/tts/models/chatterbox/chatterbox_turbo.py`**

```python
class ChatterboxTurbo(nn.Module):
    """Chatterbox Turbo TTS model for MLX."""

    ENC_COND_LEN = 15 * 16000  # 15 seconds at 16kHz
    DEC_COND_LEN = 10 * 24000  # 10 seconds at 24kHz

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.t3 = T3(config.t3_config)
        self.s3gen = S3Gen(meanflow=True)
        self.ve = VoiceEncoder()
        self.tokenizer = TurboTokenizer(config.model_path)

    def generate(
        self,
        text: str,
        audio_prompt: mx.array,
        temperature: float = 0.8,
        top_k: int = 1000,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
    ) -> mx.array:
        # Turbo-specific generation pipeline
        ...
```

---

## 5. Implementation Roadmap

### Phase 1: Core Architecture

1. [ ] Add GPT-2 config to `config.py`
2. [ ] Implement GPT-2 model support in T3 (or integrate `mlx_lm.models.gpt2`)
3. [ ] Add `inference_turbo()` method
4. [ ] Update `T3CondEnc` to handle no perceiver case

### Phase 2: S3Gen MeanFlow

1. [ ] Add `intmeanflow.py` time mixer
2. [ ] Update `ConditionalDecoder` with meanflow support
3. [ ] Update flow matching inference for 2-step generation

### Phase 3: Integration

1. [ ] Create `ChatterboxTurbo` model class
2. [ ] Add GPT-2 tokenizer support
3. [ ] Create conversion script for Turbo weights
4. [ ] Test end-to-end generation

### Phase 4: Testing & Polish

1. [ ] Verify output quality matches PyTorch reference
2. [ ] Performance benchmarking
3. [ ] Documentation updates

---

## 6. Alternative Approaches

### Option A: Separate Model Implementation (Recommended)

Create a completely separate `ChatterboxTurbo` class that doesn't share code with the original. This:
- Avoids breaking existing functionality
- Allows independent optimization
- Cleaner codebase

### Option B: Unified Model with Config Switches

Add config flags to switch between original and turbo modes. This:
- More code sharing
- Higher complexity
- Risk of regressions

### Option C: Partial Support

Implement only the T3 Turbo (most impactful change) and use the original S3Gen with more timesteps. This:
- Faster to implement
- Won't achieve full Turbo speedup
- Easier maintenance

---

## 7. References

- [Chatterbox Turbo HuggingFace](https://huggingface.co/ResembleAI/chatterbox-turbo)
- [Chatterbox GitHub](https://github.com/resemble-ai/chatterbox)
- [Original T3 PyTorch](https://github.com/resemble-ai/chatterbox/blob/main/src/chatterbox/models/t3/t3.py)
- [TTS Turbo PyTorch](https://github.com/resemble-ai/chatterbox/blob/main/src/chatterbox/tts_turbo.py)
- [S3Gen PyTorch](https://github.com/resemble-ai/chatterbox/blob/main/src/chatterbox/models/s3gen/s3gen.py)

---

## 8. Appendix: T3 Turbo Config (from YAML)

Key parameters from `t3_turbo_v1.yaml`:

```yaml
llama_config_name: GPT2_medium
text_tokens_dict_size: 50276
speech_tokens_dict_size: 6563
speech_cond_prompt_len: 250
max_speech_tokens: 604
max_text_tokens: 402
input_pos_emb: handled_internally_by_backbone
use_perceiver_resampler: false
emotion_adv: false
supports_cfg: false
sample_rate: 32000
num_mels: 256
encoder_type: voice_encoder
speaker_embed_size: 256
start_speech_token: 6561
stop_speech_token: 6562
start_text_token: 255
stop_text_token: 0
```
