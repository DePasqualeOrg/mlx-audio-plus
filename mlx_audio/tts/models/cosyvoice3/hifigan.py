# Copyright © Anthony DePasquale
# Ported to MLX from https://github.com/FunAudioLLM/CosyVoice

"""
Causal HiFi-GAN vocoder for CosyVoice3.

This module implements the causal version of HiFi-GAN with Neural Source Filter
for streaming inference at 24kHz.
"""

import math
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .convolution import CausalConv1d, CausalConv1dDownSample, CausalConv1dUpsample


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
    indices = mx.clip(indices, 0, T - 1.001)

    idx_low = mx.floor(indices).astype(mx.int32)
    idx_high = mx.minimum(idx_low + 1, T - 1)
    weight_high = indices - idx_low.astype(mx.float32)
    weight_low = 1.0 - weight_high

    low_vals = x[:, idx_low, :]
    high_vals = x[:, idx_high, :]

    weight_low = mx.reshape(weight_low, (1, -1, 1))
    weight_high = mx.reshape(weight_high, (1, -1, 1))

    out = low_vals * weight_low + high_vals * weight_high

    if squeeze:
        out = out.squeeze(-1)

    return out


class Snake(nn.Module):
    """Snake activation function: x + sin^2(alpha * x) / alpha."""

    def __init__(self, channels: int, alpha_logscale: bool = False):
        super().__init__()
        self.alpha = mx.ones((1, channels, 1))
        self.alpha_logscale = alpha_logscale
        self.no_div_by_zero = 1e-9

    def __call__(self, x: mx.array) -> mx.array:
        """Apply Snake activation."""
        alpha = self.alpha
        if self.alpha_logscale:
            alpha = mx.exp(alpha)
        # Ensure alpha has shape (1, C, 1) for broadcasting with (B, C, T)
        if alpha.ndim == 1:
            alpha = mx.expand_dims(mx.expand_dims(alpha, 0), -1)
        return x + (mx.sin(alpha * x) ** 2) / (alpha + self.no_div_by_zero)


class ResBlock(nn.Module):
    """Residual block with Snake activation for HiFi-GAN."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: List[int] = [1, 3, 5],
        causal: bool = True,
    ):
        super().__init__()
        self.causal = causal
        self.convs1 = []
        self.convs2 = []
        self.activations1 = []
        self.activations2 = []

        for dilation in dilations:
            if causal:
                self.convs1.append(
                    CausalConv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=dilation,
                        causal_type="left",
                    )
                )
                self.convs2.append(
                    CausalConv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=1,
                        causal_type="left",
                    )
                )
            else:
                padding = (kernel_size * dilation - dilation) // 2
                self.convs1.append(
                    nn.Conv1d(channels, channels, kernel_size, padding=padding)
                )
                padding = (kernel_size - 1) // 2
                self.convs2.append(
                    nn.Conv1d(channels, channels, kernel_size, padding=padding)
                )

            self.activations1.append(Snake(channels, alpha_logscale=False))
            self.activations2.append(Snake(channels, alpha_logscale=False))

    def __call__(self, x: mx.array) -> mx.array:
        """Apply residual block."""
        for idx in range(len(self.convs1)):
            xt = self.activations1[idx](x)
            xt = self.convs1[idx](xt)
            xt = self.activations2[idx](xt)
            xt = self.convs2[idx](xt)
            x = xt + x
        return x


class SineGen2(nn.Module):
    """
    Sine generator for 24kHz with interpolation-based phase calculation.

    This version handles the 24kHz sampling rate properly by using
    interpolation for phase accumulation.
    """

    def __init__(
        self,
        sampling_rate: int,
        upsample_scale: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0,
        causal: bool = True,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = harmonic_num + 1
        self.sampling_rate = sampling_rate
        self.voiced_threshold = voiced_threshold
        self.upsample_scale = upsample_scale
        self.causal = causal

    def _f02uv(self, f0: mx.array) -> mx.array:
        """Generate UV (unvoiced) signal from F0."""
        return (f0 > self.voiced_threshold).astype(mx.float32)

    def _f02sine(self, f0_values: mx.array) -> mx.array:
        """
        Convert F0 values to sine waveforms.

        Args:
            f0_values: (B, T, dim)

        Returns:
            Sine waveforms (B, T, dim)
        """
        # Normalized frequency
        rad_values = (f0_values / self.sampling_rate) % 1

        # Initial phase noise (no noise for fundamental component)
        B, T, dim = rad_values.shape
        rand_ini = mx.random.uniform(shape=(B, dim))
        rand_ini = mx.concatenate([mx.zeros((B, 1)), rand_ini[:, 1:]], axis=1)

        # Add random initial phase to first timestep
        rad_values_first = rad_values[:, 0:1, :] + mx.expand_dims(rand_ini, 1)
        rad_values = mx.concatenate([rad_values_first, rad_values[:, 1:, :]], axis=1)

        # Downsample for phase calculation
        rad_downsampled = linear_interpolate_1d(rad_values, 1.0 / self.upsample_scale)

        # Cumulative phase
        phase = mx.cumsum(rad_downsampled, axis=1) * 2 * math.pi

        # Upsample phase - use nearest for causal mode
        phase = mx.repeat(phase, self.upsample_scale, axis=1)
        phase = phase * self.upsample_scale

        # Trim to match original length
        phase = phase[:, :T, :]

        # Generate sine
        sines = mx.sin(phase)

        return sines

    def __call__(self, f0: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Generate sine waveforms from F0.

        Args:
            f0: F0 values (B, T, 1)

        Returns:
            Tuple of (sine_waves, uv, noise)
        """
        # Create harmonics
        harmonics = mx.arange(1, self.harmonic_num + 2).astype(mx.float32)
        fn = f0 * mx.reshape(harmonics, (1, 1, -1))

        # Generate sine waveforms
        sine_waves = self._f02sine(fn) * self.sine_amp

        # Generate UV signal
        uv = self._f02uv(f0)

        # Noise generation
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * mx.random.normal(shape=sine_waves.shape)

        # Apply UV masking and add noise
        sine_waves = sine_waves * uv + noise

        return sine_waves, uv, noise


class SourceModuleHnNSF2(nn.Module):
    """Neural Source Filter module for 24kHz with causal support."""

    def __init__(
        self,
        sampling_rate: int,
        upsample_scale: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshold: float = 0,
        causal: bool = True,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        self.l_sin_gen = SineGen2(
            sampling_rate,
            upsample_scale,
            harmonic_num,
            sine_amp,
            add_noise_std,
            voiced_threshold,
            causal=causal,
        )

        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.causal = causal

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Generate source signal from F0.

        Args:
            x: F0 values (B, T, 1)

        Returns:
            Tuple of (source, noise, uv)
        """
        sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = mx.tanh(self.l_linear(sine_wavs))
        noise = mx.random.normal(shape=uv.shape) * self.sine_amp / 3
        return sine_merge, noise, uv


class CausalConvRNNF0Predictor(nn.Module):
    """
    Causal F0 predictor for streaming inference.

    Uses causal convolutions with right lookahead for the first layer
    and left causal for subsequent layers.
    """

    def __init__(
        self,
        num_class: int = 1,
        in_channels: int = 80,
        cond_channels: int = 512,
    ):
        super().__init__()
        self.num_class = num_class

        # First conv has right lookahead (kernel_size=4)
        self.condnet_0 = CausalConv1d(
            in_channels, cond_channels, kernel_size=4, causal_type="right"
        )
        # Subsequent convs are left causal (kernel_size=3)
        self.condnet_2 = CausalConv1d(
            cond_channels, cond_channels, kernel_size=3, causal_type="left"
        )
        self.condnet_4 = CausalConv1d(
            cond_channels, cond_channels, kernel_size=3, causal_type="left"
        )
        self.condnet_6 = CausalConv1d(
            cond_channels, cond_channels, kernel_size=3, causal_type="left"
        )
        self.condnet_8 = CausalConv1d(
            cond_channels, cond_channels, kernel_size=3, causal_type="left"
        )

        self.classifier = nn.Linear(cond_channels, num_class)

    def __call__(self, x: mx.array, finalize: bool = True) -> mx.array:
        """
        Predict F0 from mel spectrogram.

        Args:
            x: Mel spectrogram (B, C, T)
            finalize: Whether this is the final chunk

        Returns:
            F0 prediction (B, T)
        """
        # First conv with optional lookahead handling
        if finalize:
            x = self.condnet_0(x)
        else:
            # Split input and context for lookahead
            causal_padding = self.condnet_0.causal_padding
            x_main = x[:, :, :-causal_padding]
            x_context = x[:, :, -causal_padding:]
            x = self.condnet_0(x_main, cache=x_context)

        x = nn.elu(x)

        # Apply remaining layers
        x = self.condnet_2(x)
        x = nn.elu(x)
        x = self.condnet_4(x)
        x = nn.elu(x)
        x = self.condnet_6(x)
        x = nn.elu(x)
        x = self.condnet_8(x)
        x = nn.elu(x)

        # Transpose and classify
        x = mx.swapaxes(x, 1, 2)  # (B, T, C)
        f0 = self.classifier(x).squeeze(-1)  # (B, T)
        f0 = mx.abs(f0)

        return f0


def hann_window_periodic(length: int) -> mx.array:
    """Create a periodic Hann window."""
    n = mx.arange(length, dtype=mx.float32)
    return 0.5 - 0.5 * mx.cos(2 * math.pi * n / length)


def stft(
    x: mx.array, n_fft: int, hop_len: int, window: mx.array
) -> Tuple[mx.array, mx.array]:
    """
    Compute Short-Time Fourier Transform.

    Args:
        x: Input signal (B, T)
        n_fft: FFT size
        hop_len: Hop length
        window: Window function

    Returns:
        Tuple of (real, imag) components (B, n_fft//2+1, T_frames)
    """
    B, T = x.shape

    # Pad signal
    pad_len = n_fft // 2
    x = mx.pad(x, [(0, 0), (pad_len, pad_len)])

    # Frame the signal using strided view (vectorized)
    n_frames = (x.shape[1] - n_fft) // hop_len + 1

    # Use mx.as_strided to extract all frames at once (per batch element)
    # This replaces the Python loop over n_frames with vectorized operations
    frames_list = []
    for b in range(B):
        # Extract frames for this batch element using strided view
        frames_b = mx.as_strided(x[b], shape=(n_frames, n_fft), strides=(hop_len, 1))
        frames_list.append(frames_b * window)
    frames = mx.stack(frames_list, axis=0)  # (B, n_frames, n_fft)

    # FFT
    spec = mx.fft.rfft(frames, axis=-1)
    real = mx.real(spec)
    imag = mx.imag(spec)

    # Transpose to (B, freq, time)
    real = mx.swapaxes(real, 1, 2)
    imag = mx.swapaxes(imag, 1, 2)

    return real, imag


def istft(
    magnitude: mx.array, phase: mx.array, n_fft: int, hop_len: int, window: mx.array
) -> mx.array:
    """
    Compute inverse STFT with proper window normalization.

    Matches PyTorch's torch.istft behavior for overlap-add reconstruction.

    Args:
        magnitude: Magnitude spectrum (B, n_fft//2+1, T_frames)
        phase: Phase spectrum (B, n_fft//2+1, T_frames)
        n_fft: FFT size
        hop_len: Hop length
        window: Window function

    Returns:
        Reconstructed signal (B, T)
    """
    # Clip magnitude
    magnitude = mx.clip(magnitude, a_min=0.0, a_max=1e2)

    # Reconstruct complex spectrum
    real = magnitude * mx.cos(phase)
    imag = magnitude * mx.sin(phase)

    # Transpose to (B, time, freq)
    real = mx.swapaxes(real, 1, 2)
    imag = mx.swapaxes(imag, 1, 2)

    # Create complex array
    spec = real + 1j * imag

    # Inverse FFT
    frames = mx.fft.irfft(spec, n=n_fft, axis=-1)

    # Apply window
    frames = frames * window

    # Overlap-add with window normalization (vectorized)
    B, n_frames, _ = frames.shape
    output_len = n_fft + (n_frames - 1) * hop_len
    window_sq = window * window

    # Compute all frame indices at once
    frame_offsets = mx.arange(n_frames) * hop_len
    indices = frame_offsets[:, None] + mx.arange(n_fft)  # (n_frames, n_fft)
    indices_flat = indices.flatten()

    # Compute window sum once (same for all batches)
    window_updates = mx.tile(window_sq, (n_frames,))
    window_sum = mx.zeros((output_len,))
    window_sum = window_sum.at[indices_flat].add(window_updates)
    window_sum = mx.expand_dims(window_sum, 0)  # (1, output_len)

    # Overlap-add for each batch element (vectorized over frames)
    output_list = []
    for b in range(B):
        out_b = mx.zeros((output_len,))
        updates = frames[b].flatten()  # (n_frames * n_fft,)
        out_b = out_b.at[indices_flat].add(updates)
        output_list.append(out_b)
    output = mx.stack(output_list, axis=0)  # (B, output_len)

    # Normalize by window sum (avoid division by zero)
    window_sum = mx.maximum(window_sum, 1e-8)
    output = output / window_sum

    # Remove padding
    pad_len = n_fft // 2
    output = output[:, pad_len:-pad_len]

    return output


class CausalHiFTGenerator(nn.Module):
    """
    Causal HiFi-GAN with Neural Source Filter for CosyVoice3.

    This version uses causal convolutions throughout for streaming inference.
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
        source_resblock_kernel_sizes: List[int] = [7, 11],
        source_resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5]],
        lrelu_slope: float = 0.1,
        audio_limit: float = 0.99,
        conv_pre_look_right: int = 4,
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
        self.upsample_rates = upsample_rates
        self.conv_pre_look_right = conv_pre_look_right

        upsample_scale = math.prod(upsample_rates) * istft_params["hop_len"]

        # Causal F0 predictor
        self.f0_predictor = CausalConvRNNF0Predictor(
            in_channels=in_channels,
            cond_channels=base_channels,
        )

        # Neural Source Filter
        self.m_source = SourceModuleHnNSF2(
            sampling_rate=sampling_rate,
            upsample_scale=upsample_scale,
            harmonic_num=nb_harmonics,
            sine_amp=nsf_alpha,
            add_noise_std=nsf_sigma,
            voiced_threshold=nsf_voiced_threshold,
            causal=True,
        )

        self.f0_upsample_scale = upsample_scale

        # Pre-convolution with right lookahead
        self.conv_pre = CausalConv1d(
            in_channels,
            base_channels,
            kernel_size=conv_pre_look_right + 1,
            causal_type="right",
        )

        # Upsampling layers (causal)
        self.ups = []
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                CausalConv1dUpsample(
                    base_channels // (2**i),
                    base_channels // (2 ** (i + 1)),
                    k,
                    u,
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
                    CausalConv1d(
                        istft_params["n_fft"] + 2,
                        base_channels // (2 ** (i + 1)),
                        kernel_size=1,
                        causal_type="left",
                    )
                )
            else:
                self.source_downs.append(
                    CausalConv1dDownSample(
                        istft_params["n_fft"] + 2,
                        base_channels // (2 ** (i + 1)),
                        u * 2,
                        u,
                    )
                )
            self.source_resblocks.append(
                ResBlock(base_channels // (2 ** (i + 1)), k, d, causal=True)
            )

        # Residual blocks
        self.resblocks = []
        for i in range(len(self.ups)):
            ch = base_channels // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d, causal=True))

        # Post-convolution
        ch = base_channels // (2 ** len(self.ups))
        self.conv_post = CausalConv1d(
            ch, istft_params["n_fft"] + 2, kernel_size=7, causal_type="left"
        )

        # STFT window
        self.stft_window = hann_window_periodic(istft_params["n_fft"])

    def _f0_upsample(self, f0: mx.array) -> mx.array:
        """Upsample F0 using nearest neighbor."""
        return mx.repeat(f0, self.f0_upsample_scale, axis=2)

    def _stft(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """Compute STFT."""
        return stft(
            x,
            self.istft_params["n_fft"],
            self.istft_params["hop_len"],
            self.stft_window,
        )

    def _istft(self, magnitude: mx.array, phase: mx.array) -> mx.array:
        """Compute inverse STFT."""
        return istft(
            magnitude,
            phase,
            self.istft_params["n_fft"],
            self.istft_params["hop_len"],
            self.stft_window,
        )

    def decode(self, x: mx.array, s: mx.array, finalize: bool = True) -> mx.array:
        """
        Decode mel-spectrogram to waveform.

        Args:
            x: Mel-spectrogram (B, C, T)
            s: Source signal (B, 1, T_s)
            finalize: Whether this is the final chunk

        Returns:
            Generated waveform (B, T)
        """
        # STFT of source
        s_stft_real, s_stft_imag = self._stft(s.squeeze(1))

        # Handle lookahead for non-final chunks
        if finalize:
            x = self.conv_pre(x)
        else:
            causal_padding = self.conv_pre.causal_padding
            x_main = x[:, :, :-causal_padding]
            x_context = x[:, :, -causal_padding:]
            x = self.conv_pre(x_main, cache=x_context)
            # Trim source STFT
            trim_len = int(math.prod(self.upsample_rates) * self.conv_pre_look_right)
            s_stft_real = s_stft_real[:, :, :-trim_len]
            s_stft_imag = s_stft_imag[:, :, :-trim_len]

        s_stft = mx.concatenate([s_stft_real, s_stft_imag], axis=1)

        for i in range(self.num_upsamples):
            x = nn.leaky_relu(x, negative_slope=self.lrelu_slope)
            x = self.ups[i](x)

            if i == self.num_upsamples - 1:
                # Reflection pad
                x = mx.concatenate([x[:, :, 1:2], x], axis=2)

            # Source fusion
            si = self.source_downs[i](s_stft)
            si = self.source_resblocks[i](si)
            x = x + si

            # Apply residual blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs = xs + self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = nn.leaky_relu(x)
        x = self.conv_post(x)

        # Split into magnitude and phase
        n_fft_half = self.istft_params["n_fft"] // 2 + 1
        magnitude = mx.exp(x[:, :n_fft_half, :])
        phase = mx.sin(x[:, n_fft_half:, :])

        # Inverse STFT
        x = self._istft(magnitude, phase)

        if not finalize:
            trim_len = int(
                math.prod(self.upsample_rates) * self.istft_params["hop_len"]
            )
            x = x[:, :-trim_len]

        x = mx.clip(x, -self.audio_limit, self.audio_limit)
        return x

    def __call__(
        self, speech_feat: mx.array, finalize: bool = True
    ) -> Tuple[mx.array, mx.array]:
        """
        Generate waveform from mel-spectrogram.

        Args:
            speech_feat: Mel-spectrogram (B, C, T)
            finalize: Whether this is the final chunk

        Returns:
            Tuple of (waveform, source)
        """
        # Predict F0
        f0 = self.f0_predictor(speech_feat, finalize=finalize)

        # Upsample F0
        s = self._f0_upsample(mx.expand_dims(f0, 1))
        s = mx.swapaxes(s, 1, 2)  # (B, T, 1)

        # Generate source from F0
        s, _, _ = self.m_source(s)
        s = mx.swapaxes(s, 1, 2)  # (B, 1, T)

        # Decode to waveform
        if finalize:
            generated_speech = self.decode(x=speech_feat, s=s, finalize=finalize)
        else:
            # Trim mel for non-final
            causal_padding = self.f0_predictor.condnet_0.causal_padding
            mel_trimmed = speech_feat[:, :, :-causal_padding]
            generated_speech = self.decode(x=mel_trimmed, s=s, finalize=finalize)

        return generated_speech, s
