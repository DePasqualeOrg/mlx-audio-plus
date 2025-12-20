# Copyright © 2025 FunASR (original model implementation)
# Copyright © Anthony DePasquale (MLX port)
# Ported to MLX from https://github.com/modelscope/FunASR
# License: licenses/funasr.txt

"""
Fun-ASR model for MLX.

This module provides an implementation of the Fun-ASR speech recognition
model from Alibaba's Tongyi Lab, ported to Apple's MLX framework.

The model combines:
- SenseVoice audio encoder (SANM-based transformer)
- Audio adaptor (projects to LLM space)
- Qwen3 language model decoder

Features:
- Multilingual transcription (13+ languages)
- Speech-to-text translation
- Custom prompting for domain-specific recognition
- Streaming output support

Usage:
    from mlx_audio.stt.models.funasr import Model

    # Basic transcription
    model = Model.from_pretrained("path/to/model")
    result = model.generate("audio.wav")
    print(result.text)

    # Translation to English
    result = model.generate("chinese_audio.wav", task="translate")
    print(result.text)

    # With custom context
    result = model.generate(
        "meeting.wav",
        initial_prompt="Technical discussion about machine learning."
    )

    # Streaming output
    for chunk in model.generate("audio.wav", stream=True):
        print(chunk, end="")
"""

from .funasr import (
    SUPPORTED_LANGUAGES,
    TASK_TRANSCRIBE,
    TASK_TRANSLATE,
    FunASRConfig,
    Model,
    STTOutput,
)

__all__ = [
    "Model",
    "FunASRConfig",
    "STTOutput",
    "SUPPORTED_LANGUAGES",
    "TASK_TRANSCRIBE",
    "TASK_TRANSLATE",
]
