# Ported from https://github.com/resemble-ai/chatterbox

import mlx.core as mx
from mlx_audio.utils import mel_filters, stft


def _reflect_pad_1d(x: mx.array, pad_amount: int) -> mx.array:
    """Reflect pad a 1D array at both ends."""
    if pad_amount == 0:
        return x
    # Reflect at start: x[1:pad_amount+1][::-1]
    prefix = x[1:pad_amount + 1]
    prefix = prefix[::-1]
    # Reflect at end: x[-(pad_amount+1):-1][::-1]
    suffix = x[-(pad_amount + 1):-1]
    suffix = suffix[::-1]
    return mx.concatenate([prefix, x, suffix])


def mel_spectrogram(
    y: mx.array,
    n_fft: int = 1920,
    num_mels: int = 80,
    sampling_rate: int = 24000,
    hop_size: int = 480,
    win_size: int = 1920,
    fmin: int = 0,
    fmax: int = 8000,
    center: bool = False,
) -> mx.array:
    """
    Extract mel-spectrogram from waveform.

    Args:
        y: Waveform (B, T) or (T,)
        n_fft: FFT size
        num_mels: Number of mel bins
        sampling_rate: Sample rate
        hop_size: Hop size
        win_size: Window size
        fmin: Minimum frequency
        fmax: Maximum frequency
        center: Whether to center the window

    Returns:
        Mel-spectrogram (B, num_mels, T')
    """
    was_1d = len(y.shape) == 1
    if was_1d:
        y = mx.expand_dims(y, 0)

    # Pad signal with reflection (MLX doesn't support reflect mode in pad)
    pad_amount = (n_fft - hop_size) // 2
    y_padded = []
    for i in range(y.shape[0]):
        y_padded.append(_reflect_pad_1d(y[i], pad_amount))
    y = mx.stack(y_padded)

    # STFT - process each batch item since shared stft expects 1D
    # Use center=False since we already applied reflection padding above
    specs = []
    for i in range(y.shape[0]):
        spec = stft(
            y[i],  # 1D input
            window="hann",
            n_fft=n_fft,
            hop_length=hop_size,
            win_length=win_size,
            center=False,
        )
        specs.append(spec)
    # Stack: each spec is (T', F) -> stack to (B, T', F)
    spec = mx.stack(specs, axis=0)

    # Magnitude spectrogram
    magnitudes = mx.abs(spec)  # (B, T', F)

    # Create mel filterbank
    # Use slaney normalization to match librosa defaults (used by original PyTorch)
    filters = mel_filters(
        sample_rate=sampling_rate,
        n_fft=n_fft,
        n_mels=num_mels,
        f_min=fmin,
        f_max=fmax,
        norm="slaney",
        mel_scale="slaney",
    )

    # Apply mel filterbank: (B, T', F) @ (F, M) -> (B, T', M)
    mel_spec = magnitudes @ filters.T
    mel_spec = mx.transpose(mel_spec, [0, 2, 1])  # (B, M, T')

    # Log compression
    mel_spec = mx.log(mx.maximum(mel_spec, 1e-5))

    return mel_spec
