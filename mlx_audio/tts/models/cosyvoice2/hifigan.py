# Copyright (c) 2024 Alibaba Inc
# MLX port by mlx-audio contributors
#
# Licensed under the Apache License, Version 2.0

"""HiFi-GAN vocoder with built-in F0 predictor for CosyVoice2."""

import math
from typing import Dict, List

import mlx.core as mx
import mlx.nn as nn

# Import shared components from chatterbox
from ..chatterbox.s3gen.hifigan import (
    ResBlock,
    Snake,
    hann_window_periodic,
    istft,
    stft,
)


def linear_interpolate_1d(x: mx.array, scale_factor: float) -> mx.array:
    """
    Linear interpolation for 1D signals matching PyTorch's F.interpolate behavior.

    Args:
        x: Input array (B, T, C) or (B, T)
        scale_factor: Scale factor for interpolation

    Returns:
        Interpolated array
    """
    if len(x.shape) == 2:
        x = mx.expand_dims(x, -1)
        squeeze = True
    else:
        squeeze = False

    B, T, C = x.shape
    new_T = int(T * scale_factor)

    if new_T == 0:
        new_T = 1

    # PyTorch F.interpolate with align_corners=False uses:
    # source_index = (dest_index + 0.5) * (input_size / output_size) - 0.5
    indices = (mx.arange(new_T) + 0.5) * (T / new_T) - 0.5
    indices = mx.clip(indices, 0, T - 1.001)  # Avoid out of bounds

    idx_low = mx.floor(indices).astype(mx.int32)
    idx_high = mx.minimum(idx_low + 1, T - 1)
    weight_high = indices - idx_low.astype(mx.float32)
    weight_low = 1.0 - weight_high

    # Vectorized gather and interpolate
    # x shape: (B, T, C), indices shape: (new_T,)
    # Use advanced indexing
    low_vals = x[:, idx_low, :]  # (B, new_T, C)
    high_vals = x[:, idx_high, :]  # (B, new_T, C)

    # Reshape weights for broadcasting: (new_T,) -> (1, new_T, 1)
    weight_low = mx.reshape(weight_low, (1, -1, 1))
    weight_high = mx.reshape(weight_high, (1, -1, 1))

    out = low_vals * weight_low + high_vals * weight_high  # (B, new_T, C)

    if squeeze:
        out = out.squeeze(-1)

    return out


class SineGen2(nn.Module):
    """
    Sine generator for CosyVoice2 (24kHz version).

    This version uses interpolation for phase calculation,
    which is required for 24kHz sampling rate.
    """

    def __init__(
        self,
        sampling_rate: int,
        upsample_scale: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = harmonic_num + 1
        self.sampling_rate = sampling_rate
        self.voiced_threshold = voiced_threshold
        self.upsample_scale = upsample_scale

    def _f02uv(self, f0: mx.array) -> mx.array:
        """Generate UV (unvoiced) signal from F0."""
        return (f0 > self.voiced_threshold).astype(mx.float32)

    def _f02sine(self, f0_values: mx.array) -> mx.array:
        """
        Convert F0 values to sine waveforms with interpolation.

        Args:
            f0_values: (B, T, dim) where dim is fundamental + harmonics

        Returns:
            Sine waveforms (B, T, dim)
        """
        # Convert to normalized frequency (rad values)
        rad_values = (f0_values / self.sampling_rate) % 1

        # Initial phase noise (no noise for fundamental component)
        B, T, dim = rad_values.shape
        rand_ini = mx.random.uniform(shape=(B, dim))
        rand_ini = mx.concatenate([mx.zeros((B, 1)), rand_ini[:, 1:]], axis=1)
        rad_values = mx.array(rad_values)  # Make a copy
        # Add random initial phase to first time step
        rad_values_list = [rad_values[:, 0:1, :] + mx.expand_dims(rand_ini, 1)]
        rad_values_list.append(rad_values[:, 1:, :])
        rad_values = mx.concatenate(rad_values_list, axis=1)

        # Downsample, accumulate phase, then upsample
        # This matches the PyTorch interpolation behavior
        rad_downsampled = linear_interpolate_1d(rad_values, 1.0 / self.upsample_scale)

        # Cumulative sum for phase
        phase = mx.cumsum(rad_downsampled, axis=1) * 2 * math.pi

        # Upsample phase back
        phase = linear_interpolate_1d(phase * self.upsample_scale, self.upsample_scale)

        # Trim to match original length
        phase = phase[:, :T, :]

        # Generate sine
        sines = mx.sin(phase)

        return sines

    def __call__(self, f0: mx.array) -> tuple:
        """
        Generate sine waveforms from F0.

        Args:
            f0: F0 values (B, T, 1)

        Returns:
            Tuple of (sine_waves, uv, noise)
        """
        # Create harmonics: f0 * [1, 2, 3, ..., harmonic_num+1]
        harmonics = mx.arange(1, self.harmonic_num + 2).astype(mx.float32)
        fn = f0 * mx.reshape(harmonics, (1, 1, -1))  # (B, T, dim)

        # Generate sine waveforms
        sine_waves = self._f02sine(fn) * self.sine_amp

        # Generate UV signal
        uv = self._f02uv(f0)

        # Noise: for unvoiced similar to sine_amp, for voiced use noise_std
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * mx.random.normal(shape=sine_waves.shape)

        # Apply UV masking and add noise
        sine_waves = sine_waves * uv + noise

        return sine_waves, uv, noise


class SourceModuleHnNSF2(nn.Module):
    """
    Source module for CosyVoice2 (24kHz version).

    Uses SineGen2 with interpolation for proper 24kHz generation.
    """

    def __init__(
        self,
        sampling_rate: int,
        upsample_scale: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # Sine generator with interpolation
        self.l_sin_gen = SineGen2(
            sampling_rate,
            upsample_scale,
            harmonic_num,
            sine_amp,
            add_noise_std,
            voiced_threshold,
        )

        # Linear layer to merge harmonics
        self.l_linear = nn.Linear(harmonic_num + 1, 1)

    def __call__(self, x: mx.array) -> tuple:
        """
        Generate source signal from F0.

        Args:
            x: F0 values (B, T, 1)

        Returns:
            Tuple of (source, noise, uv)
        """
        # Generate sine waveforms
        sine_wavs, uv, _ = self.l_sin_gen(x)

        # Merge harmonics with tanh activation
        sine_merge = mx.tanh(self.l_linear(sine_wavs))

        # Generate noise for noise branch
        noise = mx.random.normal(shape=uv.shape) * self.sine_amp / 3

        return sine_merge, noise, uv


class CosyF0Predictor(nn.Module):
    """
    F0 predictor for CosyVoice2 HiFTGenerator (ConvRNNF0Predictor).

    Takes mel spectrogram and predicts F0 (fundamental frequency).
    Uses a series of 1D convolutions with ELU activation followed by a classifier.
    """

    def __init__(
        self,
        in_channels: int = 80,
        hidden_channels: int = 512,
        num_layers: int = 5,
        kernel_size: int = 3,
    ):
        """
        Args:
            in_channels: Number of mel channels
            hidden_channels: Hidden layer channels
            num_layers: Number of conv layers
            kernel_size: Convolution kernel size
        """
        super().__init__()

        # Build condnet as a sequential list of conv layers
        # Structure: Conv1d -> ELU -> Conv1d -> ELU -> ...
        # Weight keys: condnet_0, condnet_2, condnet_4, condnet_6, condnet_8
        # (indices 1, 3, 5, 7 are ELU activations, not stored)
        self.condnet_0 = nn.Conv1d(
            in_channels, hidden_channels, kernel_size, padding=kernel_size // 2
        )
        self.condnet_2 = nn.Conv1d(
            hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2
        )
        self.condnet_4 = nn.Conv1d(
            hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2
        )
        self.condnet_6 = nn.Conv1d(
            hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2
        )
        self.condnet_8 = nn.Conv1d(
            hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2
        )

        self.classifier = nn.Linear(hidden_channels, 1)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Predict F0 from mel spectrogram.

        Args:
            x: Mel spectrogram (B, C, T)

        Returns:
            F0 prediction (B, T)
        """
        # x is (B, C, T), transpose to (B, T, C) for Conv1d
        x = mx.swapaxes(x, 1, 2)

        # Apply condnet layers with ELU activation
        x = self.condnet_0(x)
        x = nn.elu(x)
        x = self.condnet_2(x)
        x = nn.elu(x)
        x = self.condnet_4(x)
        x = nn.elu(x)
        x = self.condnet_6(x)
        x = nn.elu(x)
        x = self.condnet_8(x)
        x = nn.elu(x)

        # Classifier: (B, T, hidden_channels) -> (B, T, 1) -> (B, T)
        f0 = self.classifier(x).squeeze(-1)

        # Use abs() to ensure non-negative F0 (matching original)
        f0 = mx.abs(f0)

        return f0


class CosyHiFTGenerator(nn.Module):
    """
    HiFi-GAN with Neural Source Filter (HiFT-Net) for CosyVoice2.

    This version has F0 predictor built-in as a submodule,
    matching the weight structure of CosyVoice2's hift.safetensors.
    """

    def __init__(
        self,
        in_channels: int = 80,
        base_channels: int = 512,
        nb_harmonics: int = 8,
        sampling_rate: int = 24000,
        nsf_alpha: float = 0.1,
        nsf_sigma: float = 0.003,
        nsf_voiced_threshold: float = 10,
        upsample_rates: List[int] = [8, 5, 3],
        upsample_kernel_sizes: List[int] = [16, 11, 7],
        istft_params: Dict[str, int] = {"n_fft": 16, "hop_len": 4},
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes: List[int] = [7, 7, 11],
        source_resblock_dilation_sizes: List[List[int]] = [
            [1, 3, 5],
            [1, 3, 5],
            [1, 3, 5],
        ],
        lrelu_slope: float = 0.1,
        audio_limit: float = 0.99,
        use_interpolation: bool = True,
    ):
        super().__init__()

        self.out_channels = 1
        self.nb_harmonics = nb_harmonics
        self.sampling_rate = sampling_rate
        self.istft_params = istft_params
        self.lrelu_slope = lrelu_slope
        self.audio_limit = audio_limit

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        upsample_scale = math.prod(upsample_rates) * istft_params["hop_len"]

        # Built-in F0 predictor
        self.f0_predictor = CosyF0Predictor(
            in_channels=in_channels,
            hidden_channels=base_channels,
        )

        # Neural Source Filter - use SourceModuleHnNSF2 for 24kHz
        self.m_source = SourceModuleHnNSF2(
            sampling_rate=sampling_rate,
            upsample_scale=upsample_scale,
            harmonic_num=nb_harmonics,
            sine_amp=nsf_alpha,
            add_noise_std=nsf_sigma,
            voiced_threshold=nsf_voiced_threshold,
        )

        # F0 upsampler
        self.f0_upsample_scale = upsample_scale

        # Pre-convolution
        self.conv_pre = nn.Conv1d(in_channels, base_channels, 7, stride=1, padding=3)

        # Upsampling layers
        self.ups = []
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(
                    base_channels // (2**i),
                    base_channels // (2 ** (i + 1)),
                    k,
                    stride=u,
                    padding=(k - u) // 2,
                )
            )

        # Source downsampling and resblocks
        self.source_downs = []
        self.source_resblocks = []
        downsample_rates = [1] + upsample_rates[::-1][:-1]
        downsample_cum_rates = []
        cum_prod = 1
        for rate in downsample_rates:
            cum_prod *= rate
            downsample_cum_rates.append(cum_prod)

        for i, (u, k, d) in enumerate(
            zip(
                downsample_cum_rates[::-1],
                source_resblock_kernel_sizes,
                source_resblock_dilation_sizes,
            )
        ):
            if u == 1:
                self.source_downs.append(
                    nn.Conv1d(
                        istft_params["n_fft"] + 2,
                        base_channels // (2 ** (i + 1)),
                        1,
                        stride=1,
                    )
                )
            else:
                self.source_downs.append(
                    nn.Conv1d(
                        istft_params["n_fft"] + 2,
                        base_channels // (2 ** (i + 1)),
                        u * 2,
                        stride=u,
                        padding=u // 2,
                    )
                )
            self.source_resblocks.append(
                ResBlock(base_channels // (2 ** (i + 1)), k, d)
            )

        # Residual blocks
        self.resblocks = []
        for i in range(len(self.ups)):
            ch = base_channels // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(ResBlock(ch, k, d))

        # Post-convolution
        ch = base_channels // (2 ** len(self.ups))
        self.conv_post = nn.Conv1d(
            ch, istft_params["n_fft"] + 2, 7, stride=1, padding=3
        )

        # STFT window
        self.stft_window = hann_window_periodic(istft_params["n_fft"])

    def _f0_upsample(self, f0: mx.array) -> mx.array:
        """Upsample F0 using nearest neighbor interpolation."""
        return mx.repeat(f0, self.f0_upsample_scale, axis=2)

    def _stft(self, x: mx.array) -> tuple:
        """Perform STFT on input signal."""
        return stft(
            x,
            self.istft_params["n_fft"],
            self.istft_params["hop_len"],
            self.stft_window,
        )

    def _istft(self, magnitude: mx.array, phase: mx.array) -> mx.array:
        """Perform inverse STFT."""
        return istft(
            magnitude,
            phase,
            self.istft_params["n_fft"],
            self.istft_params["hop_len"],
            self.stft_window,
        )

    def decode(self, x: mx.array, s: mx.array) -> mx.array:
        """
        Decode mel-spectrogram to waveform.

        Args:
            x: Mel-spectrogram (B, C, T)
            s: Source signal (B, 1, T_s)

        Returns:
            Generated waveform (B, T)
        """
        # STFT of source signal
        s_stft_real, s_stft_imag = self._stft(s.squeeze(1))
        s_stft = mx.concatenate([s_stft_real, s_stft_imag], axis=1)

        # Pre-convolution
        x = mx.swapaxes(x, 1, 2)
        x = self.conv_pre(x)
        x = mx.swapaxes(x, 1, 2)

        for i in range(self.num_upsamples):
            x = nn.leaky_relu(x, negative_slope=self.lrelu_slope)
            x = mx.swapaxes(x, 1, 2)
            x = self.ups[i](x)
            x = mx.swapaxes(x, 1, 2)

            if i == self.num_upsamples - 1:
                # Reflection pad
                x = mx.concatenate([x[:, :, 1:2], x], axis=2)

            # Source fusion
            si = mx.swapaxes(s_stft, 1, 2)
            si = self.source_downs[i](si)
            si = mx.swapaxes(si, 1, 2)
            si = self.source_resblocks[i](si)
            x = x + si

            # Apply residual blocks and average their outputs
            # Using mx.stack allows MLX's lazy evaluation to optimize the computation graph
            start_idx = i * self.num_kernels
            x = mx.mean(
                mx.stack(
                    [self.resblocks[start_idx + j](x) for j in range(self.num_kernels)],
                    axis=0,
                ),
                axis=0,
            )

        # Note: PyTorch CosyVoice2 uses default negative_slope=0.01 here (not 0.1)
        x = nn.leaky_relu(x, negative_slope=0.01)
        x = mx.swapaxes(x, 1, 2)
        x = self.conv_post(x)
        x = mx.swapaxes(x, 1, 2)

        # Split into magnitude and phase
        n_fft_half = self.istft_params["n_fft"] // 2 + 1
        magnitude = mx.exp(x[:, :n_fft_half, :])
        phase = mx.sin(x[:, n_fft_half:, :])

        # Inverse STFT
        x = self._istft(magnitude, phase)
        x = mx.clip(x, -self.audio_limit, self.audio_limit)

        return x

    def __call__(self, speech_feat: mx.array, cache_source: mx.array = None) -> tuple:
        """
        Generate waveform from mel-spectrogram.

        Args:
            speech_feat: Mel-spectrogram (B, C, T)
            cache_source: Cached source for streaming

        Returns:
            Tuple of (waveform, source)
        """
        if cache_source is None:
            cache_source = mx.zeros((1, 1, 0))

        # Predict F0 from mel
        f0 = self.f0_predictor(speech_feat)

        # Upsample F0
        s = self._f0_upsample(mx.expand_dims(f0, 1))
        s = mx.swapaxes(s, 1, 2)  # (B, T, 1)

        # Generate source from F0
        s, _, _ = self.m_source(s)
        s = mx.swapaxes(s, 1, 2)  # (B, 1, T)

        # Use cache to avoid glitch in streaming
        if cache_source.shape[2] != 0:
            cache_len = cache_source.shape[2]
            s = mx.concatenate([cache_source, s[:, :, cache_len:]], axis=2)

        # Decode to waveform
        generated_speech = self.decode(x=speech_feat, s=s)

        return generated_speech, s
