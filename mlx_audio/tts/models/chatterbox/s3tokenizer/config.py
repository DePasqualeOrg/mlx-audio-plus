# Ported from https://github.com/xingchensong/S3Tokenizer

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """S3Tokenizer model configuration."""

    n_mels: int = 128
    n_audio_ctx: int = 1500
    n_audio_state: int = 1280
    n_audio_head: int = 20
    n_audio_layer: int = 6
    n_codebook_size: int = 3**8  # 6561 for V2

    use_sdpa: bool = False


# Constants
S3_SR = 16_000  # Sample rate for S3Tokenizer
S3_HOP = 160  # 100 frames/sec
S3_TOKEN_HOP = 640  # 25 tokens/sec
S3_TOKEN_RATE = 25
SPEECH_VOCAB_SIZE = 6561
