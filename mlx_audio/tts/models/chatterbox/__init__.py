# Ported from https://github.com/resemble-ai/chatterbox

from .chatterbox import ChatterboxTTS, Conditionals
from .config import ChatterboxConfig, T3Config
from .t3 import T3
from .s3gen import S3Token2Mel, S3Token2Wav
from .voice_encoder import VoiceEncoder, VoiceEncConfig
from .s3tokenizer import S3TokenizerV2
from .tokenizer import EnTokenizer

# Aliases for mlx_audio loading convention
Model = ChatterboxTTS
ModelConfig = ChatterboxConfig

__all__ = [
    "ChatterboxTTS",
    "Conditionals",
    "ChatterboxConfig",
    "T3Config",
    "T3",
    "S3Token2Mel",
    "S3Token2Wav",
    "VoiceEncoder",
    "VoiceEncConfig",
    "S3TokenizerV2",
    "EnTokenizer",
    "Model",
    "ModelConfig",
]
