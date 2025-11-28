# Ported from https://github.com/xingchensong/S3Tokenizer

from typing import List
import mlx.core as mx

from mlx_audio.utils import stft


def make_non_pad_mask(lengths: mx.array, max_len: int = 0) -> mx.array:
    """Make mask tensor containing indices of non-padded part.

    1 for non-padded part and 0 for padded part.

    Args:
        lengths: Batch of lengths (B,).
        max_len: Maximum length. If 0, uses max of lengths.

    Returns:
        Mask tensor (B, T) where True indicates valid positions.

    Examples:
        >>> lengths = mx.array([5, 3, 2])
        >>> masks = make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1, 1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
    """
    batch_size = lengths.shape[0]
    max_len = max_len if max_len > 0 else int(mx.max(lengths).item())

    seq_range = mx.arange(max_len)
    seq_range_expand = mx.expand_dims(seq_range, 0)  # (1, T)
    seq_range_expand = mx.broadcast_to(seq_range_expand, (batch_size, max_len))

    seq_length_expand = mx.expand_dims(lengths, -1)  # (B, 1)
    mask = seq_range_expand < seq_length_expand
    return mask


def mask_to_bias(mask: mx.array, dtype: mx.Dtype) -> mx.array:
    """Convert bool-tensor to float-tensor for attention.

    Args:
        mask: Boolean mask tensor.
        dtype: Output dtype.

    Returns:
        Float mask where False positions have large negative values.
    """
    mask = mask.astype(dtype)
    # Attention mask bias: (1.0 - mask) * -1e10
    mask = (1.0 - mask) * -1.0e+10
    return mask


def librosa_mel_filters(
    sample_rate: int = 16000,
    n_fft: int = 400,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float = None,
) -> mx.array:
    """
    Generate mel filterbank matching librosa's default implementation.

    Librosa uses:
    - Slaney-style mel scale
    - Slaney normalization (area normalization)

    Args:
        sample_rate: Audio sample rate
        n_fft: FFT size
        n_mels: Number of mel bands
        fmin: Minimum frequency
        fmax: Maximum frequency (defaults to sample_rate/2)

    Returns:
        Mel filterbank (n_mels, n_fft//2+1)
    """
    import math

    if fmax is None:
        fmax = sample_rate / 2.0

    n_freqs = n_fft // 2 + 1

    # Slaney-style Hz to Mel conversion (matches librosa default)
    def hz_to_mel(freq):
        f_min = 0.0
        f_sp = 200.0 / 3.0
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = math.log(6.4) / 27.0

        if isinstance(freq, (int, float)):
            if freq < min_log_hz:
                return (freq - f_min) / f_sp
            else:
                return min_log_mel + math.log(freq / min_log_hz) / logstep
        else:
            mels = (freq - f_min) / f_sp
            log_region = freq >= min_log_hz
            mels = mx.where(
                log_region,
                min_log_mel + mx.log(mx.maximum(freq, min_log_hz) / min_log_hz) / logstep,
                mels
            )
            return mels

    def mel_to_hz(mel):
        f_min = 0.0
        f_sp = 200.0 / 3.0
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = math.log(6.4) / 27.0

        if isinstance(mel, (int, float)):
            if mel < min_log_mel:
                return f_min + f_sp * mel
            else:
                return min_log_hz * math.exp(logstep * (mel - min_log_mel))
        else:
            freqs = f_min + f_sp * mel
            log_region = mel >= min_log_mel
            freqs = mx.where(
                log_region,
                min_log_hz * mx.exp(logstep * (mel - min_log_mel)),
                freqs
            )
            return freqs

    # Create mel points
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = mx.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # FFT frequencies
    fft_freqs = mx.linspace(0, sample_rate / 2, n_freqs)

    # Create filterbank
    filterbank = mx.zeros((n_mels, n_freqs))

    for i in range(n_mels):
        f_left = float(hz_points[i])
        f_center = float(hz_points[i + 1])
        f_right = float(hz_points[i + 2])

        # Left slope
        left_slope = (fft_freqs - f_left) / (f_center - f_left + 1e-10)
        # Right slope
        right_slope = (f_right - fft_freqs) / (f_right - f_center + 1e-10)
        # Triangle
        triangle = mx.minimum(left_slope, right_slope)
        triangle = mx.maximum(triangle, 0.0)

        # Slaney normalization: divide by bandwidth
        enorm = 2.0 / (f_right - f_left + 1e-10)
        triangle = triangle * enorm

        filterbank[i] = triangle

    return filterbank


def log_mel_spectrogram(
    audio: mx.array,
    n_mels: int = 128,
    padding: int = 0,
) -> mx.array:
    """
    Compute the log-Mel spectrogram.

    This implementation matches the PyTorch S3Tokenizer which uses librosa's
    default mel filterbank (slaney norm, slaney mel scale).

    Args:
        audio: Audio waveform (T,) or (B, T) in 16 kHz
        n_mels: Number of Mel-frequency filters (80 or 128)
        padding: Number of zero samples to pad to the right

    Returns:
        Log-Mel spectrogram (n_mels, T') or (B, n_mels, T')
    """
    was_1d = len(audio.shape) == 1
    if was_1d:
        audio = mx.expand_dims(audio, 0)

    if padding > 0:
        audio = mx.pad(audio, [(0, 0), (0, padding)])

    # Process each batch item separately since stft expects 1D input
    # STFT with S3Tokenizer parameters
    specs = []
    for i in range(audio.shape[0]):
        spec = stft(
            audio[i],  # 1D input (T,)
            window="hann",
            n_fft=400,
            hop_length=160,
            win_length=400,
        )
        specs.append(spec)

    # Stack: each spec is (T', F) -> stack to (B, T', F)
    spec = mx.stack(specs, axis=0)

    # Magnitude squared (drop last frame to match PT)
    magnitudes = mx.abs(spec[:, :-1, :]) ** 2

    # Use librosa-style mel filterbank (slaney normalization)
    filters = librosa_mel_filters(
        sample_rate=16000,
        n_fft=400,
        n_mels=n_mels,
    )

    # Apply mel filterbank: (B, T, F) @ (F, M) -> (B, T, M)
    mel_spec = magnitudes @ filters.T
    mel_spec = mx.transpose(mel_spec, [0, 2, 1])  # (B, M, T)

    # Log compression with S3Tokenizer-style normalization
    log_spec = mx.log10(mx.maximum(mel_spec, 1e-10))
    log_spec = mx.maximum(log_spec, mx.max(log_spec) - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec.squeeze(0) if was_1d else log_spec


def padding(data: List[mx.array]):
    """Padding the data into batch data.

    Args:
        data: List of tensors, each with shape (n_mels, T_i)

    Returns:
        feats: Padded features (B, n_mels, T_max)
        feats_lengths: Feature lengths (B,)
    """
    feats_lengths = mx.array([s.shape[1] for s in data], dtype=mx.int32)
    max_len = int(mx.max(feats_lengths).item())

    # Pad each feature to max_len
    padded = []
    for s in data:
        if s.shape[1] < max_len:
            pad_amount = max_len - s.shape[1]
            s = mx.pad(s, [(0, 0), (0, pad_amount)])
        padded.append(s)

    feats = mx.stack(padded)
    return feats, feats_lengths


def merge_tokenized_segments(
    tokenized_segments: List[List[int]],
    overlap: int,
    token_rate: int
) -> List[int]:
    """
    Merges tokenized outputs by keeping the middle and dropping half of the overlapped tokens.

    Args:
        tokenized_segments: List of tokenized sequences.
        overlap: Overlapping duration in seconds (default: 4s).
        token_rate: Number of tokens per second.

    Returns:
        A single merged token sequence.
    """
    merged_tokens = []
    overlap_tokens = (overlap // 2) * token_rate

    for i, tokens in enumerate(tokenized_segments):
        left = 0 if i == 0 else overlap_tokens
        right = -overlap_tokens if i != len(tokenized_segments) - 1 else len(tokens)
        merged_tokens.extend(tokens[left:right])

    return merged_tokens
