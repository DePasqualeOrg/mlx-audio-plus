# Whisper Decoding Bug: Incorrect `reshape` Call

## Summary

A bug was found in `mlx_audio/stt/models/whisper/decoding.py` at line 646 where an extra argument is passed to the `reshape` method.

## Location

**File:** `mlx_audio/stt/models/whisper/decoding.py`
**Line:** 646-648

## The Bug

```python
# Current code (buggy):
tokens = tokens.reshape(
    tokens, (n_audio * self.n_group, len(self.initial_tokens))
)
```

The `tokens` array is incorrectly passed as the first argument to `reshape()`. The MLX `reshape` method expects only the shape, not the array itself.

## The Fix

```python
# Fixed code (matches official mlx-whisper):
tokens = tokens.reshape((n_audio * self.n_group, len(self.initial_tokens)))
```

## Why It Hasn't Been Noticed

This code is inside a conditional block that only executes when `n_group > 1`:

```python
if self.n_group > 1:  # Line 641
    # ... buggy reshape is here
```

`n_group` is set by:
```python
self.n_group: int = options.beam_size or options.best_of or 1
```

Since:
1. **Beam search is not implemented** (raises `NotImplementedError`)
2. **Default decoding uses `n_group = 1`** (greedy decoding with `temperature=0`)

The bug only triggers when using `best_of > 1` with `temperature > 0`, which is a non-default configuration.

## Evidence

1. **Official mlx-whisper** uses `tokens.reshape((shape))`:
   ```
   /tmp/mlx-examples/whisper/mlx_whisper/decoding.py:
   tokens = tokens.reshape((n_audio * self.n_group, len(self.initial_tokens)))
   ```

2. **All other reshape calls** in this codebase (30+ occurrences) follow the pattern `x.reshape(shape)` without passing the array.

3. **Line 658** in the same file uses correct syntax:
   ```python
   tokens = tokens.reshape(n_audio, self.n_group, -1)
   ```

## Test Case

To reproduce/verify the fix, run transcription with `best_of > 1`:

```python
from mlx_audio.stt import transcribe

# This uses the buggy code path
result = transcribe(
    "audio.mp3",
    model="mlx-community/whisper-tiny",
    temperature=0.5,  # Must be > 0
    best_of=3,        # Must be > 1
)
```

With the current code, this may fail or produce incorrect results.
After the fix, it should work correctly.

## Diff

```diff
--- a/mlx_audio/stt/models/whisper/decoding.py
+++ b/mlx_audio/stt/models/whisper/decoding.py
@@ -643,9 +643,7 @@ class DecodingTask:
             tokens = mx.broadcast_to(
                 tokens, [n_audio, self.n_group, len(self.initial_tokens)]
             )
-            tokens = tokens.reshape(
-                tokens, (n_audio * self.n_group, len(self.initial_tokens))
-            )
+            tokens = tokens.reshape((n_audio * self.n_group, len(self.initial_tokens)))

         # call the main sampling loop
         tokens, sum_logprobs, no_speech_probs = self._main_loop(audio_features, tokens)
```
