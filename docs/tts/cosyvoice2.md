# CosyVoice2

CosyVoice2 is a speech synthesis model from Alibaba that supports multiple inference modes:
- **Cross-lingual** (default): Zero-shot TTS for any language (reference audio only)
- **Zero-shot**: TTS with reference audio + transcription (better semantic alignment)
- **Instruct**: Control speech style with natural language instructions
- **Voice Conversion (VC)**: Convert source speech to target speaker voice

## Models

| Model | Size | Hugging Face |
|-------|------|-------------|
| fp16 | 1.5 GB | [mlx-community/CosyVoice2-0.5B-fp16](https://huggingface.co/mlx-community/CosyVoice2-0.5B-fp16) |
| 8-bit | 913 MB | [mlx-community/CosyVoice2-0.5B-8bit](https://huggingface.co/mlx-community/CosyVoice2-0.5B-8bit) |
| 4-bit | 742 MB | [mlx-community/CosyVoice2-0.5B-4bit](https://huggingface.co/mlx-community/CosyVoice2-0.5B-4bit) |

## Inference Mode Selection

The mode is selected based on which inputs are provided:

| Mode | source_audio | ref_audio | ref_text | instruct_text | Notes |
|------|--------------|-----------|----------|---------------|-------|
| Cross-lingual | - | ✓ | - | - | **Default** - zero-shot TTS without transcription |
| Zero-shot | - | ✓ | ✓ | - | Better semantic alignment with accurate transcription |
| Instruct | - | ✓ | - | ✓ | Style control with instructions |
| Voice Conversion | ✓ | ✓ | - | - | Convert source speech to target voice |

**Note**: `ref_audio` is required for all modes. CosyVoice2 needs speaker conditioning to produce quality output.

### Voice Conversion (VC) Mode

**Purpose**: Convert the content of one audio (source) to sound like another speaker (target/reference).

**How it works**:
1. Extract speech tokens from source audio (content to convert)
2. Extract mel spectrogram, speech tokens, and speaker embedding from reference audio (target voice)
3. Skip LLM generation - use source speech tokens directly
4. Flow model converts source tokens to target speaker's voice characteristics

**Key difference from other modes**: No text is involved - purely audio-to-audio transformation.

**Current limitation**: Source audio is truncated to 30 seconds (S3 tokenizer constraint).

**Future enhancement - Long audio VC**: For source audio longer than 30 seconds, implement chunked processing:
1. Scan source audio for natural silence points (energy below threshold for >200ms)
2. Split at silence points, keeping chunks under 30 seconds
3. Process each chunk through VC with the same reference audio
4. Concatenate outputs with short crossfade (~50-100ms) at split points

This approach is preferred over fixed-duration splitting because it avoids cutting mid-word/mid-sentence. The silence detection can reuse logic similar to `detect_speech_boundaries()` in generate.py.

## Model Conversion

Convert the original CosyVoice2 model to MLX format:

```bash
# fp16 (default)
python -m mlx_audio.tts.models.cosyvoice2.scripts.convert

# 8-bit quantization
python -m mlx_audio.tts.models.cosyvoice2.scripts.convert --quantize 8

# 4-bit quantization
python -m mlx_audio.tts.models.cosyvoice2.scripts.convert --quantize 4

# Upload to Hugging Face
python -m mlx_audio.tts.models.cosyvoice2.scripts.convert --upload-repo username/repo-name
```

## Fine-Grained Speech Control Tokens

Users can embed special tokens in their input text to control speech output:

### Vocal Effects
| Token | Description | Example |
|-------|-------------|---------|
| `[breath]` | Insert a breath | `"Hello [breath] how are you?"` |
| `[laughter]` | Insert laughter | `"That's funny [laughter]"` |
| `[cough]` | Insert a cough | `"Let me [cough] continue"` |
| `[sigh]` | Insert a sigh | `"[sigh] I'm tired"` |
| `[hissing]` | Hissing sound | |
| `[vocalized-noise]` | Non-speech vocalization | |
| `[lipsmack]` | Lip smacking sound | |
| `[quick_breath]` | Quick breath | |
| `[clucking]` | Clucking sound | |
| `[accent]` | Accent marker | |

### Emphasis Tags
| Token | Description | Example |
|-------|-------------|---------|
| `<strong>text</strong>` | Emphasize/stress text | `"This is <strong>important</strong>"` |
| `<laughter>text</laughter>` | Speak while laughing | `"<laughter>That's hilarious</laughter>"` |
| `[noise]` | Background noise | |

### Example with Fine-Grained Control
```python
# Add laughter to speech
text = "He suddenly [laughter] stopped because he found it funny [laughter]."

# Emphasize key words
text = "She showed extraordinary <strong>courage</strong> and <strong>wisdom</strong>."
```

## CLI Usage Examples

```bash
# Cross-lingual mode (default)
mlx_audio.tts --model mlx-community/CosyVoice2-0.5B-fp16 \
  --text "Hello, this is a test." \
  --ref_audio reference.wav

# Zero-shot mode (with transcription - better semantic alignment)
mlx_audio.tts --model mlx-community/CosyVoice2-0.5B-fp16 \
  --text "Hello, this is a test." \
  --ref_audio reference.wav \
  --ref_text "This is the exact transcription of the reference audio."

# Instruct mode (style control with natural language)
mlx_audio.tts --model mlx-community/CosyVoice2-0.5B-fp16 \
  --text "Hello, this is a test." \
  --ref_audio reference.wav \
  --instruct_text "Speak slowly and calmly"

# Voice Conversion mode (convert source audio to target voice)
mlx_audio.tts --model mlx-community/CosyVoice2-0.5B-fp16 \
  --source_audio source_speech.wav \
  --ref_audio target_voice.wav

# Fine-grained control (with vocal effects)
mlx_audio.tts --model mlx-community/CosyVoice2-0.5B-fp16 \
  --text "Hello [breath] how are you? That's <strong>amazing</strong>! [laughter]" \
  --ref_audio reference.wav
```

## Python Usage

```python
from mlx_audio.tts.generate import generate_audio

# Cross-lingual mode (default)
generate_audio(
    text="Hello, this is CosyVoice2 on MLX!",
    model="mlx-community/CosyVoice2-0.5B-fp16",
    ref_audio="reference.wav",
    file_prefix="output",
)

# Zero-shot mode (with transcription)
generate_audio(
    text="Hello, this is CosyVoice2 on MLX!",
    model="mlx-community/CosyVoice2-0.5B-fp16",
    ref_audio="reference.wav",
    ref_text="Transcription of the reference audio.",
    file_prefix="output",
)

# Instruct mode (style control)
generate_audio(
    text="Hello, this is CosyVoice2 on MLX!",
    model="mlx-community/CosyVoice2-0.5B-fp16",
    ref_audio="reference.wav",
    instruct_text="Speak slowly and calmly",
    file_prefix="output",
)

# Voice Conversion mode
generate_audio(
    text="",
    model="mlx-community/CosyVoice2-0.5B-fp16",
    source_audio="source_speech.wav",
    ref_audio="target_voice.wav",
    file_prefix="output",
)
```
