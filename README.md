# MLX Audio Plus

In addition to the models from [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio), this package includes the following new models ported to MLX in Python:

- TTS
  - [Chatterbox](https://github.com/resemble-ai/chatterbox)
  - [CosyVoice2](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B)

### Installation

```bash
pip install mlx-audio-plus
```

### Usage

```python
from mlx_audio.tts.generate import generate_audio

# Chatterbox: generate speech from reference audio
generate_audio(
    text="The quick brown fox jumped over the lazy dog.",
    model="mlx-community/Chatterbox-TTS-4bit",
    ref_audio="reference.wav",
    file_prefix="output",
    audio_format="wav",
)

# Kokoro: efficient model with speed control and voice presets
generate_audio(
    text="The quick brown fox jumped over the lazy dog.",
    model="mlx-community/Kokoro-82M-bf16",
    voice="af_heart",
    speed=1.0,
    lang_code="a", # American English
    file_prefix="output",
    audio_format="wav",
)
```

