# Re-export mel_spectrogram from shared s3gen module
# Uses optimized MLX-native implementation instead of numpy/librosa
from mlx_audio.codec.models.s3gen.mel import mel_spectrogram

__all__ = ["mel_spectrogram"]
