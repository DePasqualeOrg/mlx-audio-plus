# Whisper Implementation Comparison: mlx-audio-plus vs mlx-whisper

This document provides a line-by-line comparison between the Whisper implementation in this repository (`mlx-audio-plus`) and the official `mlx-whisper` from Apple's ml-explore/mlx-examples.

## Summary

| File | Status | Key Differences |
|------|--------|-----------------|
| `whisper.py` | **Significantly Extended** | +550 lines, adds `generate()` method, `from_pretrained()`, `STTOutput` class |
| `decoding.py` | **Minor Changes** | 3 small diffs (logsumexp, reshape, whitespace) |
| `tokenizer.py` | **Modified** | Loads tiktoken from HuggingFace instead of local assets |
| `audio.py` | **Refactored** | Moves utilities to shared `mlx_audio.utils`, changes function signatures |
| `timing.py` | **Identical** | No differences |
| `writers.py` | **New File** | Does not exist in mlx-whisper; adds subtitle/output writers |

---

## Detailed Comparison

### 1. `whisper.py`

**mlx-whisper:** 266 lines
**mlx-audio-plus:** 867 lines (+601 lines)

#### Major Additions in mlx-audio-plus:

1. **Additional imports** (lines 5-34):
   ```python
   import json
   import sys
   import warnings
   from pathlib import Path
   import tqdm
   from huggingface_hub import snapshot_download
   from mlx.utils import tree_unflatten
   # Plus imports from local modules
   ```

2. **Helper functions** (lines 36-57):
   ```python
   def _format_timestamp(seconds: float)  # Format time as HH:MM:SS.mmm
   def _get_end(segments: List[dict])     # Get last word end time
   ```

3. **New dataclass** (lines 60-63):
   ```python
   @dataclass
   class STTOutput:
       text: str
       segments: List[dict] = None
       language: str = None
   ```

4. **Class renamed** (line 248):
   ```diff
   - class Whisper(nn.Module):
   + class Model(nn.Module):
   ```

5. **dtype stored** (line 252):
   ```diff
   + self.dtype = dtype
   ```

6. **`from_pretrained()` classmethod** (lines 318-354):
   - Loads model from HuggingFace Hub
   - Handles quantization
   - Loads weights from safetensors or npz

7. **`generate()` method** (lines 356-867):
   - Full transcription pipeline (~510 lines)
   - Temperature fallback with quality thresholds
   - Word-level timestamps
   - Hallucination detection
   - Clip timestamps support
   - Progress bar with tqdm
   - Returns `STTOutput` dataclass

---

### 2. `decoding.py`

**Status:** Nearly identical with 3 minor differences

#### Difference 1: logsumexp keepdims (line 268)
```diff
- logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
+ logprobs = logits - mx.logsumexp(logits, axis=-1)
```

#### Difference 2: logsumexp keepdims (line 383)
```diff
- logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
+ logprobs = logits - mx.logsumexp(logits, axis=-1)
```

#### Difference 3: reshape call (line 646)
```diff
- tokens = tokens.reshape((n_audio * self.n_group, len(self.initial_tokens)))
+ tokens = tokens.reshape(
+     tokens, (n_audio * self.n_group, len(self.initial_tokens))
+ )
```
**Note:** This appears to be a bug in mlx-audio-plus - passing `tokens` as first arg to `reshape()` is incorrect.

#### Difference 4: Whitespace (line 606)
```diff
-
  next_tokens, next_completed, next_sum_logprobs, _ = _step(
```

---

### 3. `tokenizer.py`

**Status:** Modified to load from HuggingFace

#### Changes:

1. **Removed import** (line 4):
   ```diff
   - import os
   ```

2. **Added imports** (lines 10-13):
   ```diff
   + from huggingface_hub import hf_hub_download
   +
   + # Repository containing the tiktoken vocabulary files
   + TOKENIZER_REPO = "mlx-community/whisper-tokenizer"
   ```

3. **`get_encoding()` function** (lines 334-338):
   ```diff
   - vocab_path = os.path.join(os.path.dirname(__file__), "assets", f"{name}.tiktoken")
   + filename = f"{name}.tiktoken"
   + vocab_path = hf_hub_download(repo_id=TOKENIZER_REPO, filename=filename)
   ```

4. **Encoding name** (line 366):
   ```diff
   - name=os.path.basename(vocab_path),
   + name=filename,
   ```

---

### 4. `audio.py`

**Status:** Refactored to use shared utilities

#### Removed Functions (moved to `mlx_audio.utils`):
- `load_audio()` - 42 lines removed
- `mel_filters()` - 15 lines removed
- `hanning()` - 3 lines removed
- `stft()` - 25 lines removed

#### Added Imports:
```python
from mlx_audio.stt.utils import load_audio
from mlx_audio.utils import hanning, mel_filters, stft
```

#### Changed Function Signatures:

**`stft()` call** (line 73):
```diff
- freqs = stft(audio, window, nperseg=N_FFT, noverlap=HOP_LENGTH)
+ freqs = stft(audio, window=window, n_fft=N_FFT, hop_length=HOP_LENGTH)
```

**`mel_filters()` call** (line 76):
```diff
- filters = mel_filters(n_mels)
+ filters = mel_filters(SAMPLE_RATE, N_FFT, n_mels, norm="slaney", mel_scale=None)
```

---

### 5. `timing.py`

**Status:** Identical

No differences found between the two implementations.

---

### 6. `writers.py`

**Status:** New file (does not exist in mlx-whisper)

This file provides output writers for various formats:
- `ResultWriter` - Base class
- `WriteTXT` - Plain text output
- `WriteVTT` - WebVTT subtitles
- `WriteSRT` - SRT subtitles
- `WriteJSON` - JSON output
- `WriteTSV` - Tab-separated values

Features word-level timestamp support and subtitle highlighting.

---

## Conclusions

### What mlx-audio-plus adds:
1. **Complete transcription pipeline** via `generate()` method
2. **HuggingFace integration** for model and tokenizer loading
3. **Quality control** (compression ratio, logprob thresholds, hallucination detection)
4. **Output writers** for various subtitle/text formats
5. **Shared utilities** architecture (audio processing in common module)

### What remains identical:
1. Core model architecture (encoder, decoder, attention)
2. Decoding logic (with minor exceptions)
3. Tokenizer logic
4. Timing/alignment algorithms

### Potential Issues Found:
1. `decoding.py` line 646: `tokens.reshape(tokens, ...)` appears incorrect - should be `tokens.reshape(...)`
2. `decoding.py`: Removed `keepdims=True` from logsumexp may affect broadcasting

---

*Generated: 2024-12-12*
