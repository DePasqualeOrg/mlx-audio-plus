# CLAUDE.md

This project includes speech-to-text, text-to-speech, and speech-to-speech models in MLX running on Apple Silicon.

## Project Structure

```
mlx-audio-plus/
├── mlx_audio/                 # Main package
│   ├── convert.py             # Universal model conversion (PyTorch → MLX)
│   ├── tts/                   # Text-to-speech
│   │   ├── generate.py        # CLI and API entry point
│   │   └── models/            # Chatterbox, CosyVoice2/3, Kokoro, Bark, etc.
│   ├── stt/                   # Speech-to-text
│   │   ├── generate.py        # CLI entry point
│   │   └── models/            # Whisper, FunASR, Parakeet, Voxtral, etc.
│   ├── sts/                   # Speech-to-speech
│   │   └── voice_pipeline.py  # Async pipeline (STT → LLM → TTS)
│   ├── codec/                 # Audio codecs (SNAC, Vocos, EnCodec, etc.)
│   └── server.py              # FastAPI inference server
├── docs/                      # Model documentation
├── examples/                  # Usage examples
└── tests/                     # Test suites
```

**Key entry points:**
- `mlx_audio.tts.generate` - TTS inference CLI/API
- `mlx_audio.stt.generate` - STT inference CLI
- `mlx_audio.convert` - Model conversion

**Model directories:** Each model in `tts/models/` and `stt/models/` has its own subdirectory containing model definitions and often a custom conversion script.

## Cloning Reference Repositories

When referring to code from other repositories that aren't available locally, for easy access to the source code, make a local clone of the repository in `/tmp` rather than using the web fetch or web search tools, which can be unreliable.

## Original Implementations of Models

The original implementations of the models are usually in PyTorch and can be downloaded from the original model's GitHub repository.

## Other Reference Repositories

- [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm): LLMs in MLX
- [ml-explore/mlx](https://github.com/ml-explore/mlx): MLX built-ins
- [huggingface/transformers](https://github.com/huggingface/transformers): models, tokenizers, and pipelines

## Hugging Face Model Repositories

When doing work that requires referring to a model's weights, config files, and other files in the model's Hugging Face repository, check for a local copy of that repository. You can clone the Hugging Face repository into a directory in `/tmp` if it does not already exist there.

Models that have been converted to MLX format are usually found in the mlx-community organization on Hugging Face. The original model weights (usually in PyTorch format) are found in the respective organizations' repositories on Hugging Face.

Cached repositories that have been downloaded from Hugging Face during weights conversion and inference are stored in `~/.cache/huggingface/hub/`.
