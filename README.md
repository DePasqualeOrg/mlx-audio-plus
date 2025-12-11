# MLX Audio Plus

In addition to the models from [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio), this package includes the following new models ported to MLX in Python:

- TTS
  - [Chatterbox](https://github.com/resemble-ai/chatterbox)
  - [CosyVoice2](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B)

## Installation

```bash
pip install mlx-audio-plus
```

## Usage

### CLI

```bash
# CosyVoice2: zero-shot mode (reference audio + transcription)
mlx_audio.tts.generate --model mlx-community/CosyVoice2-0.5B-4bit \
    --text "Hello, this is a test of text to speech." \
    --ref_audio reference.wav \
    --ref_text "This is what I said in the reference audio."

# CosyVoice2: cross-lingual mode (no transcription)
mlx_audio.tts.generate --model mlx-community/CosyVoice2-0.5B-4bit \
    --text "Bonjour, comment allez-vous?" \
    --ref_audio reference.wav

# CosyVoice2: instruct mode with style control
mlx_audio.tts.generate --model mlx-community/CosyVoice2-0.5B-4bit \
    --text "I have exciting news!" \
    --ref_audio reference.wav \
    --instruct_text "Speak with excitement and enthusiasm"

# CosyVoice2: voice conversion
mlx_audio.tts.generate --model mlx-community/CosyVoice2-0.5B-4bit \
    --ref_audio target_speaker.wav \
    --source_audio source_speech.wav

# Play audio directly instead of saving
mlx_audio.tts.generate --model mlx-community/CosyVoice2-0.5B-4bit \
    --text "Hello world" \
    --ref_audio reference.wav \
    --play

# Chatterbox: generate speech from reference audio
mlx_audio.tts.generate --model mlx-community/Chatterbox-TTS-4bit \
    --text "The quick brown fox jumped over the lazy dog." \
    --ref_audio reference.wav
```

### Python

```python
from mlx_audio.tts.generate import generate_audio

# CosyVoice2: zero-shot mode (reference audio + transcription)
generate_audio(
    text="Hello, this is a test of text to speech.",
    model="mlx-community/CosyVoice2-0.5B-4bit",
    ref_audio="reference.wav",
    ref_text="This is what I said in the reference audio.",  # Optional
    file_prefix="output",  # Optional
    audio_format="wav",  # Optional
)

# CosyVoice2: cross-lingual mode (no transcription needed)
generate_audio(
    text="Bonjour, comment allez-vous aujourd'hui?",
    model="mlx-community/CosyVoice2-0.5B-4bit",
    ref_audio="reference.wav",
)

# CosyVoice2: instruct mode with style control
generate_audio(
    text="I have some exciting news to share with you!",
    model="mlx-community/CosyVoice2-0.5B-4bit",
    ref_audio="reference.wav",
    instruct_text="Speak with excitement and enthusiasm",
)

# CosyVoice2: voice conversion (convert source audio to target speaker)
generate_audio(
    text="",  # Not used in VC mode
    model="mlx-community/CosyVoice2-0.5B-4bit",
    ref_audio="target_speaker.wav",  # Target voice
    source_audio="source_speech.wav",
)

# Chatterbox: generate speech from reference audio
generate_audio(
    text="The quick brown fox jumped over the lazy dog.",
    model="mlx-community/Chatterbox-TTS-4bit",
    ref_audio="reference.wav",
)
```

