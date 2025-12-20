# Copyright © Xingchen Song (original model implementation)
# Copyright © Anthony DePasquale (MLX port)
# Ported to MLX from https://github.com/xingchensong/S3Tokenizer
# License: licenses/s3tokenizer.txt

from .model import ModelConfig as ModelConfigV1
from .model import S3Tokenizer
from .model_v2 import ModelConfig, S3TokenizerV2
from .model_v3 import S3TokenizerV3
from .utils import (
    log_mel_spectrogram,
    log_mel_spectrogram_compat,
    make_non_pad_mask,
    mask_to_bias,
    merge_tokenized_segments,
    padding,
)

# S3Tokenizer constants
S3_SR = 16_000  # Sample rate for S3Tokenizer
S3_HOP = 160  # 100 frames/sec
S3_TOKEN_HOP = 640  # 25 tokens/sec
S3_TOKEN_RATE = 25
SPEECH_VOCAB_SIZE = 6561  # 3^8 (V2/V3)
S3_V1_VOCAB_SIZE = 4096  # V1

__all__ = [
    "S3Tokenizer",
    "S3TokenizerV2",
    "S3TokenizerV3",
    "ModelConfig",
    "ModelConfigV1",
    "log_mel_spectrogram",
    "log_mel_spectrogram_compat",
    "make_non_pad_mask",
    "mask_to_bias",
    "padding",
    "merge_tokenized_segments",
    "S3_SR",
    "S3_HOP",
    "S3_TOKEN_HOP",
    "S3_TOKEN_RATE",
    "SPEECH_VOCAB_SIZE",
    "S3_V1_VOCAB_SIZE",
]
