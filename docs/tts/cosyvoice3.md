# CosyVoice3

CosyVoice3 is Alibaba's multilingual voice cloning model with four inference modes:
- Cross-lingual: `ref_audio` only
- Zero-shot: `ref_audio` + exact `ref_text`
- Instruct: `ref_audio` + `instruct_text`
- Voice conversion: `source_audio` + `ref_audio`

## Model Selection

CosyVoice3 picks the mode from the inputs you pass:

| Mode | source_audio | ref_audio | ref_text | instruct_text | Notes |
|------|--------------|-----------|----------|---------------|-------|
| Cross-lingual | - | ✓ | - | - | Default when `ref_text` is omitted |
| Zero-shot | - | ✓ | ✓ | - | Best semantic alignment, but only when `ref_text` exactly matches the clipped reference audio |
| Instruct | - | ✓ | - | ✓ | Style control with a reference voice |
| Voice conversion | ✓ | ✓ | - | - | Convert source speech into the target speaker's voice |

## Reference Audio Rules

CosyVoice3 in `mlx-audio-plus` now matches the original PyTorch implementation more closely:
- Reference audio longer than 30 seconds is rejected instead of silently clipped
- Zero-shot only activates when you explicitly provide `ref_text`
- If `ref_text` is omitted, CosyVoice3 uses cross-lingual mode instead of auto-transcribing into zero-shot
- Source audio for voice conversion is also limited to 30 seconds

These rules follow the original PyTorch frontend in:
- `cosyvoice/cli/frontend.py:_extract_speech_token`
- `cosyvoice/cli/frontend.py:frontend_zero_shot`
- `cosyvoice/cli/frontend.py:frontend_cross_lingual`
- `cosyvoice/cli/frontend.py:frontend_vc`

## Best Practice

For zero-shot, prepare the reference clip yourself:
1. Keep it at 30 seconds or less
2. Trim it cleanly so it does not start or end mid-word
3. Provide the exact transcript for that clipped segment

If you do not have an exact transcript, use cross-lingual mode instead.

## CLI Examples

```bash
# Cross-lingual (default)
mlx_audio.tts.generate \
  --model mlx-community/Fun-CosyVoice3-0.5B-2512 \
  --text "Hello, this is a test." \
  --ref_audio reference.wav

# Zero-shot
mlx_audio.tts.generate \
  --model mlx-community/Fun-CosyVoice3-0.5B-2512 \
  --text "Hello, this is a test." \
  --ref_audio reference.wav \
  --ref_text "This is the exact transcription of the clipped reference audio."

# Instruct
mlx_audio.tts.generate \
  --model mlx-community/Fun-CosyVoice3-0.5B-2512 \
  --text "Hello, this is a test." \
  --ref_audio reference.wav \
  --instruct "Speak slowly and calmly."
```

## Python Examples

```python
from mlx_audio.tts.generate import generate_audio

# Cross-lingual
generate_audio(
    text="Hello, this is CosyVoice3 on MLX.",
    model="mlx-community/Fun-CosyVoice3-0.5B-2512",
    ref_audio="reference.wav",
    file_prefix="output",
)

# Zero-shot
generate_audio(
    text="Hello, this is CosyVoice3 on MLX.",
    model="mlx-community/Fun-CosyVoice3-0.5B-2512",
    ref_audio="reference.wav",
    ref_text="This is the exact transcription of the clipped reference audio.",
    file_prefix="output",
)
```

## Streaming Behavior

The streaming flow path also matches the original PyTorch CosyVoice3 CLI more closely now:
- base chunk size is 25 tokens
- chunk hops grow as `25 -> 50 -> 100`
- the 24 kHz prompt mel uses `fmax=None` / Nyquist, matching the original PyTorch runtime mel configuration
