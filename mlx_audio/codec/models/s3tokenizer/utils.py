# Copyright © Xingchen Song (original model implementation)
# Copyright © Anthony DePasquale (MLX port)
# Ported to MLX from https://github.com/xingchensong/S3Tokenizer
# License: licenses/s3tokenizer.txt

from typing import List

import mlx.core as mx

from mlx_audio.utils import hanning, mel_filters, stft


def log_mel_spectrogram(
    audio: mx.array,
    sample_rate: int = 16_000,
    n_mels: int = 128,
    n_fft: int = 400,
    hop_length: int = 160,
    padding: int = 0,
):
    """
    Compute the log-Mel spectrogram of audio.

    Note: This function does NOT drop the last STFT frame, which differs from
    the original PyTorch S3Tokenizer implementation. For strict compatibility
    with PyTorch S3Tokenizer outputs, use `log_mel_spectrogram_compat()` instead.

    Args:
        audio: Audio waveform (T,) in 16 kHz
        sample_rate: Sample rate (default 16000)
        n_mels: Number of Mel-frequency filters (default 128)
        n_fft: FFT size (default 400)
        hop_length: Hop length (default 160)
        padding: Number of zero samples to pad to the right

    Returns:
        Log-Mel spectrogram (n_mels, T')
    """
    if not isinstance(audio, mx.array):
        audio = mx.array(audio)

    if padding > 0:
        audio = mx.pad(audio, (0, padding))

    window = hanning(n_fft + 1)[:-1]
    freqs = stft(
        audio,
        window=window,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
    ).swapaxes(0, 1)
    magnitudes = freqs.abs() ** 2
    filters = mel_filters(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        norm="slaney",
        mel_scale="slaney",
    )
    mel_spec = filters @ magnitudes
    log_spec = mx.maximum(mel_spec, 1e-10).log10()
    log_spec = mx.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


def log_mel_spectrogram_compat(
    audio: mx.array,
    n_mels: int = 128,
    padding: int = 0,
) -> mx.array:
    """
    Compute the log-Mel spectrogram with PyTorch-compatible behavior.

    This implementation matches the PyTorch S3Tokenizer which uses:
    - torch.stft with return_complex=True
    - Drops the last frame (magnitudes[..., :-1])
    - librosa-style mel filterbank (slaney norm, slaney mel scale)

    This is used by Chatterbox and other TTS models that require exact
    compatibility with the original PyTorch S3Tokenizer behavior.

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

    # STFT with S3Tokenizer parameters
    specs = []
    for i in range(audio.shape[0]):
        spec = stft(
            audio[i],
            window="hann",
            n_fft=400,
            hop_length=160,
            win_length=400,
        )
        specs.append(spec)

    # Stack: each spec is (T', F) -> stack to (B, T', F)
    spec = mx.stack(specs, axis=0)

    # Magnitude squared - drop last frame to match PyTorch torch.stft behavior
    magnitudes = mx.abs(spec[:, :-1, :]) ** 2

    # Use slaney-style mel filterbank
    filters = mel_filters(
        sample_rate=16000,
        n_fft=400,
        n_mels=n_mels,
        norm="slaney",
        mel_scale="slaney",
    )

    # Apply mel filterbank: (B, T, F) @ (F, M) -> (B, T, M)
    mel_spec = magnitudes @ filters.T
    mel_spec = mx.transpose(mel_spec, [0, 2, 1])  # (B, M, T)

    # Log compression with S3Tokenizer-style normalization
    log_spec = mx.log10(mx.maximum(mel_spec, 1e-10))
    log_spec = mx.maximum(log_spec, mx.max(log_spec) - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec.squeeze(0) if was_1d else log_spec


def make_non_pad_mask(lengths: mx.array, max_len: int = 0) -> mx.array:
    """Make mask tensor containing indices of non-padded part.

    The sequences in a batch may have different lengths. To enable
    batch computing, padding is need to make all sequence in same
    size. To avoid the padding part pass value to context dependent
    block such as attention or convolution, this padding part is
    masked.

    1 for non-padded part and 0 for padded part.

    Parameters
    ----------
        lengths (mx.array): Batch of lengths (B,).
        max_len (int): Maximum length. If 0, use the maximum length in batch.

    Returns:
    -------
        mx.array: Mask tensor containing indices of padded part (B, max_T).

    Examples:
        >>> import mlx.core as mx
        >>> lengths = mx.array([5, 3, 2])
        >>> masks = make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1, 1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
    """
    batch_size = lengths.shape[0]
    max_len = max_len if max_len > 0 else int(mx.max(lengths).item())
    seq_range = mx.arange(0, max_len, dtype=mx.int32)
    seq_range_expand = mx.expand_dims(seq_range, axis=0)  # (1, max_len)
    seq_range_expand = mx.broadcast_to(seq_range_expand, (batch_size, max_len))
    seq_length_expand = mx.expand_dims(lengths, axis=-1)  # (B, 1)
    mask = seq_range_expand >= seq_length_expand
    return mx.logical_not(mask)


def mask_to_bias(mask: mx.array, dtype: mx.Dtype = mx.float32) -> mx.array:
    assert mask.dtype == mx.bool_, "Input mask must be boolean type"
    assert dtype in [
        mx.float32,
        mx.bfloat16,
        mx.float16,
    ], "dtype must be a floating point type"
    mask = mask.astype(dtype)
    mask = (1.0 - mask) * -1.0e10
    return mask


def padding(data: List[mx.array]) -> tuple[mx.array, mx.array]:
    """Padding the data into batch data

    Parameters
    ----------
        data: List[mx.array], shape of each array (128, T)

    Returns:
    -------
        Tuple of (padded_feats, feats_lengths)
        - padded_feats: shape (B, 128, max_T)
        - feats_lengths: shape (B,)
    """
    assert isinstance(data, list), "Input must be a list of arrays"

    feats_lengths = mx.array([s.shape[1] for s in data], dtype=mx.int32)

    max_len = max(s.shape[1] for s in data)
    batch_size = len(data)
    n_mels = data[0].shape[0]

    padded_feats = mx.zeros((batch_size, n_mels, max_len), dtype=data[0].dtype)

    for i, feat in enumerate(data):
        seq_len = feat.shape[1]
        padded_feats[i, :, :seq_len] = feat

    return padded_feats, feats_lengths


def merge_tokenized_segments(
    tokenized_segments: List[List[int]], overlap: int, token_rate: int
) -> List[int]:
    """
    Merges tokenized outputs by keeping the middle and dropping half of the overlapped tokens.

    Args:
        tokenized_segments: List of tokenized sequences.
        overlap: Overlapping duration in seconds.
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
