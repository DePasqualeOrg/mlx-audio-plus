# MLX Audio Plus

## Motivation

This fork removes a large amount of cruft (incompatibly licensed code and data that should not be included in the repo) from [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio). In addition to the models from that repo, this one includes improvements as well as the following new models ported to MLX in Python:

- TTS
  - [Chatterbox](https://github.com/resemble-ai/chatterbox)
  - [CosyVoice 2](https://github.com/FunAudioLLM/CosyVoice)
  - [CosyVoice 3](https://github.com/FunAudioLLM/CosyVoice)
- STT
  - [Fun-ASR](https://github.com/modelscope/FunASR)

Improvements to the upstream repo will continue to be merged here.

This repo also serves as the basis for Swift ports of models in [mlx-swift-audio](https://github.com/DePasqualeOrg/mlx-swift-audio).

## Installation

```bash
pip install mlx-audio-plus
```

## Usage

### CLI

```bash
# CosyVoice 3: cross-lingual mode (reference audio only)
mlx_audio.tts.generate --model mlx-community/Fun-CosyVoice3-0.5B-2512-4bit \
    --text "Hello, this is a test of text to speech." \
    --ref_audio reference.wav

# CosyVoice 3: zero-shot mode (with transcription for better quality)
mlx_audio.tts.generate --model mlx-community/Fun-CosyVoice3-0.5B-2512-4bit \
    --text "Hello, this is a test of text to speech." \
    --ref_audio reference.wav \
    --ref_text "This is what I said in the reference audio."

# CosyVoice 3: instruct mode with style control
mlx_audio.tts.generate --model mlx-community/Fun-CosyVoice3-0.5B-2512-4bit \
    --text "I have exciting news!" \
    --ref_audio reference.wav \
    --instruct_text "Speak with excitement and enthusiasm"

# CosyVoice 3: voice conversion
mlx_audio.tts.generate --model mlx-community/Fun-CosyVoice3-0.5B-2512-4bit \
    --ref_audio target_speaker.wav \
    --source_audio source_speech.wav

# Play audio directly instead of saving
mlx_audio.tts.generate --model mlx-community/Fun-CosyVoice3-0.5B-2512-4bit \
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

# CosyVoice 3: cross-lingual mode (reference audio only)
generate_audio(
    text="Hello, this is a test of text to speech.",
    model="mlx-community/Fun-CosyVoice3-0.5B-2512-4bit",
    ref_audio="reference.wav",
    file_prefix="output",  # Optional
    audio_format="wav",  # Optional
)

# CosyVoice 3: zero-shot mode (with transcription for better quality)
generate_audio(
    text="Bonjour, comment allez-vous aujourd'hui?",
    model="mlx-community/Fun-CosyVoice3-0.5B-2512-4bit",
    ref_audio="reference.wav",
    ref_text="This is what I said in the reference audio.",
)

# CosyVoice 3: instruct mode with style control
generate_audio(
    text="I have some exciting news to share with you!",
    model="mlx-community/Fun-CosyVoice3-0.5B-2512-4bit",
    ref_audio="reference.wav",
    instruct_text="Speak with excitement and enthusiasm",
)

# CosyVoice 3: voice conversion (convert source audio to target speaker)
generate_audio(
    model="mlx-community/Fun-CosyVoice3-0.5B-2512-4bit",
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

### Speech to text

```python
from mlx_audio.stt.models.funasr import Model

# Fun-ASR

# Load the model
model = Model.from_pretrained("mlx-community/Fun-ASR-Nano-2512-4bit")

# Basic transcription
result = model.generate("audio.wav")
print(result.text)

# Translation (speech to English text)
result = model.generate(
    "chinese_speech.wav",
    task="translate",
    target_language="en"
)

# Custom prompting for domain-specific content
result = model.generate(
    "medical_dictation.wav",
    initial_prompt="Medical consultation discussing cardiac symptoms."
)

# Streaming output
for chunk in model.generate("audio.wav", stream=True):
    print(chunk, end="", flush=True)
```

