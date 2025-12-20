# Copyright © 2025 FunASR (original model implementation)
# Copyright © Anthony DePasquale (MLX port)
# Ported to MLX from https://github.com/modelscope/FunASR
# License: licenses/funasr.txt

"""
Audio preprocessing for Fun-ASR model.

Implements mel-filterbank feature extraction with Low Frame Rate (LFR) processing.
"""

import math
from typing import Union

import mlx.core as mx
import numpy as np

from mlx_audio.dsp import hamming, mel_filters, stft
from mlx_audio.stt.utils import load_audio

# Audio hyperparameters for Fun-ASR
SAMPLE_RATE = 16000
N_FFT = 400  # 25ms window at 16kHz
HOP_LENGTH = 160  # 10ms hop
N_MELS = 80

# LFR (Low Frame Rate) parameters
LFR_M = 7  # Stack every 7 frames
LFR_N = 6  # Subsample by factor of 6


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, mx.array],
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    sample_rate: int = SAMPLE_RATE,
) -> mx.array:
    """
    Compute the log-Mel spectrogram with hamming window.

    Parameters
    ----------
    audio : Union[str, np.ndarray, mx.array]
        The path to audio or audio waveform array
    n_mels : int
        The number of Mel-frequency filters
    n_fft : int
        FFT size
    hop_length : int
        Hop length for STFT
    sample_rate : int
        Sample rate of audio

    Returns
    -------
    mx.array, shape = (n_frames, n_mels)
        Log mel spectrogram features
    """
    if isinstance(audio, str):
        audio = load_audio(audio, sr=sample_rate)
    elif isinstance(audio, np.ndarray):
        audio = mx.array(audio)

    # Use hamming window (Fun-ASR uses hamming)
    window = hamming(n_fft)

    # Compute STFT
    freqs = stft(audio, window=window, n_fft=n_fft, hop_length=hop_length)

    # Compute power spectrum
    magnitudes = freqs[:-1, :].abs().square()

    # Apply mel filterbank
    filters = mel_filters(sample_rate, n_fft, n_mels, norm="slaney", mel_scale="htk")
    mel_spec = magnitudes @ filters.T

    # Log mel spectrogram
    log_spec = mx.log(mx.maximum(mel_spec, 1e-10))

    return log_spec


def apply_lfr(
    features: mx.array,
    lfr_m: int = LFR_M,
    lfr_n: int = LFR_N,
) -> mx.array:
    """
    Apply Low Frame Rate (LFR) processing to features.

    This stacks consecutive frames and subsamples to reduce the frame rate.
    Uses vectorized gather operations for efficiency.

    Parameters
    ----------
    features : mx.array, shape = (n_frames, n_mels)
        Input mel spectrogram features
    lfr_m : int
        Number of frames to stack (default: 7)
    lfr_n : int
        Subsampling factor (default: 6)

    Returns
    -------
    mx.array, shape = (ceil(n_frames / lfr_n), n_mels * lfr_m)
        LFR-processed features with stacked frames
    """
    T, n_mels = features.shape

    # Output length uses ceiling division
    T_lfr = int(math.ceil(T / lfr_n))

    # Left padding
    left_pad = (lfr_m - 1) // 2
    if left_pad > 0:
        left_padding = mx.broadcast_to(features[0:1], (left_pad, n_mels))
        features = mx.concatenate([left_padding, features], axis=0)

    # Right padding to ensure we have enough frames
    T_padded = features.shape[0]
    total_needed = (T_lfr - 1) * lfr_n + lfr_m
    if total_needed > T_padded:
        right_pad = total_needed - T_padded
        right_padding = mx.broadcast_to(features[-1:], (right_pad, n_mels))
        features = mx.concatenate([features, right_padding], axis=0)

    # Create indices for all output frames
    # Shape: (T_lfr, lfr_m)
    start_indices = mx.arange(T_lfr) * lfr_n
    offsets = mx.arange(lfr_m)
    # Broadcasting: (T_lfr, 1) + (lfr_m,) -> (T_lfr, lfr_m)
    indices = start_indices[:, None] + offsets[None, :]

    # Gather frames: features[indices] -> (T_lfr, lfr_m, n_mels)
    gathered = features[indices]

    # Reshape to (T_lfr, lfr_m * n_mels)
    return gathered.reshape(T_lfr, -1)


def apply_cmvn(
    features: mx.array,
    cmvn_mean: mx.array = None,
    cmvn_istd: mx.array = None,
) -> mx.array:
    """
    Apply Cepstral Mean and Variance Normalization (CMVN).

    Uses the formula: (features + mean) * istd
    where mean and istd come from precomputed statistics.

    If cmvn_mean and cmvn_istd are not provided, applies per-utterance
    normalization.

    Parameters
    ----------
    features : mx.array
        Input features
    cmvn_mean : mx.array, optional
        Additive shift (negative of mean)
    cmvn_istd : mx.array, optional
        Multiplicative scale (inverse std)

    Returns
    -------
    mx.array
        Normalized features
    """
    if cmvn_mean is None or cmvn_istd is None:
        # Per-utterance normalization
        mean = mx.mean(features, axis=0, keepdims=True)
        std = mx.std(features, axis=0, keepdims=True) + 1e-6
        return (features - mean) / std

    # Apply precomputed CMVN: (x + mean) * istd
    # Note: cmvn_mean is actually the negative mean (shift)
    return (features + cmvn_mean) * cmvn_istd


def preprocess_audio(
    audio: Union[str, np.ndarray, mx.array],
    n_mels: int = N_MELS,
    lfr_m: int = LFR_M,
    lfr_n: int = LFR_N,
    cmvn_mean: mx.array = None,
    cmvn_istd: mx.array = None,
    apply_normalization: bool = True,
) -> mx.array:
    """
    Full audio preprocessing pipeline for Fun-ASR.

    1. Compute log mel spectrogram
    2. Apply LFR (frame stacking and subsampling)
    3. Optionally apply CMVN

    Parameters
    ----------
    audio : Union[str, np.ndarray, mx.array]
        Input audio (path or waveform)
    n_mels : int
        Number of mel bins
    lfr_m : int
        LFR frame stacking count
    lfr_n : int
        LFR subsampling factor
    cmvn_mean : mx.array, optional
        Precomputed CMVN mean shift
    cmvn_istd : mx.array, optional
        Precomputed CMVN inverse std
    apply_normalization : bool
        Whether to apply CMVN normalization

    Returns
    -------
    mx.array, shape = (ceil(time / lfr_n), n_mels * lfr_m)
        Preprocessed audio features ready for the encoder
    """
    # Compute log mel spectrogram
    mel_features = log_mel_spectrogram(audio, n_mels=n_mels)

    # Apply LFR processing
    lfr_features = apply_lfr(mel_features, lfr_m=lfr_m, lfr_n=lfr_n)

    # Apply normalization
    if apply_normalization:
        lfr_features = apply_cmvn(lfr_features, cmvn_mean, cmvn_istd)

    return lfr_features


def compute_feature_lengths(
    audio_lengths: mx.array,
    hop_length: int = HOP_LENGTH,
    lfr_n: int = LFR_N,
) -> mx.array:
    """
    Compute output feature lengths after preprocessing.

    Parameters
    ----------
    audio_lengths : mx.array
        Lengths of input audio in samples
    hop_length : int
        Hop length for STFT
    lfr_n : int
        LFR subsampling factor

    Returns
    -------
    mx.array
        Output feature lengths
    """
    # Frames after STFT
    n_frames = audio_lengths // hop_length

    # Frames after LFR (ceiling division)
    out_len = (n_frames + lfr_n - 1) // lfr_n

    return out_len.astype(mx.int32)
