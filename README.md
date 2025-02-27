# MLX-Audio

A text-to-speech (TTS) and Speech-to-Speech (STS) library built on Apple's MLX framework, providing efficient speech synthesis on Apple Silicon.

## Features

- Fast inference on Apple Silicon (M series chips)
- Multiple language support
- Voice customization options
- Quantization support for optimized performance

## Installation

```bash
pip install mlx-audio
```

## Models

### Kokoro

Kokoro is a multilingual TTS model that supports various languages and voice styles.

#### Example Usage

```python
from tts.models.kokoro import KokoroModel, KokoroPipeline
from IPython.display import Audio
import soundfile as sf

# Initialize the model
model = KokoroModel(repo_id='prince-canuma/Kokoro-82M')

# Create a pipeline with American English
pipeline = KokoroPipeline(lang_code='a', model=model)

# Generate audio
text = "The MLX King lives. Let him cook!"
for _, _, audio in pipeline(text, voice='af_heart', speed=1, split_pattern=r'\n+'):
    # Display audio in notebook (if applicable)
    display(Audio(data=audio, rate=24000, autoplay=0))

    # Save audio to file
    sf.write('audio.wav', audio[0], 24000)
```

#### Language Options

- 🇺🇸 `'a'` - American English
- 🇬🇧 `'b'` - British English
- 🇯🇵 `'j'` - Japanese (requires `pip install misaki[ja]`)
- 🇨🇳 `'z'` - Mandarin Chinese (requires `pip install misaki[zh]`)

## Advanced Features

### Quantization

You can quantize models for improved performance:

```python
from tts.models.kokoro import KokoroModel
from tts.utils import quantize_model
import json
import mlx.core as mx

model = KokoroModel(repo_id='prince-canuma/Kokoro-82M')
config = model.config

# Quantize to 8-bit
weights, config = quantize_model(model, config, 64, 8)

# Save quantized model
with open('./8bit/config.json', 'w') as f:
    json.dump(config, f)

mx.save_safetensors("./8bit/kokoro-v1_0.safetensors", weights, metadata={"format": "mlx"})
```

## Requirements

- MLX
- Python 3.8+
- Apple Silicon Mac (for optimal performance)

## License

[Add your license information here]

## Acknowledgements

This project uses the Kokoro model architecture for text-to-speech synthesis.
