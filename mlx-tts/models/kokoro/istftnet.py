import mlx.core as mx
import mlx.nn as nn
import librosa
import numpy as np
from scipy.signal import get_window
from typing import List, Tuple, Optional

def init_weights(m, mean=0.0, std=0.01):
    if isinstance(m, nn.Conv1d):
        m.weight = mx.random.normal(mean, std, m.weight.shape)

def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)

class WeightNorm:
    """MLX implementation of weight normalization"""
    def __init__(self, weight: mx.array, dim: int):
        self.dim = dim
        self.g = mx.sqrt(mx.sum(weight * weight, axis=dim, keepdims=True))
        self.v = weight

    def __call__(self) -> mx.array:
        return self.g * self.v / mx.sqrt(mx.sum(self.v * self.v, axis=self.dim, keepdims=True))

class Conv1d(nn.Module):
    """Conv1d with weight normalization"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                            stride=stride, padding=padding, dilation=dilation, groups=groups)
        self.weight_norm = WeightNorm(self.conv.weight, dim=0)

    def __call__(self, x):
        self.conv.weight = self.weight_norm()
        return self.conv(x)

class AdaIN1d(nn.Module):
    def __init__(self, style_dim: int, num_features: int):
        super().__init__()
        self.norm = nn.InstanceNorm(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def __call__(self, x: mx.array, s: mx.array) -> mx.array:
        h = self.fc(s)
        h = mx.expand_dims(h, axis=2)  # Equivalent to view(..., 1)
        gamma, beta = mx.split(h, 2, axis=1)
        return (1 + gamma) * self.norm(x) + beta

class AdaINResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3,
                 dilation: Tuple[int, int, int] = (1, 3, 5), style_dim: int = 64):
        super().__init__()
        self.convs1 = [
            Conv1d(channels, channels, kernel_size, 1,
                  padding=get_padding(kernel_size, dilation[i]),
                  dilation=dilation[i])
            for i in range(3)
        ]
        self.convs2 = [
            Conv1d(channels, channels, kernel_size, 1,
                  padding=get_padding(kernel_size, 1),
                  dilation=1)
            for _ in range(3)
        ]
        self.adain1 = [AdaIN1d(style_dim, channels) for _ in range(3)]
        self.adain2 = [AdaIN1d(style_dim, channels) for _ in range(3)]
        self.alpha1 = [mx.ones((1, channels, 1)) for _ in range(len(self.convs1))]
        self.alpha2 = [mx.ones((1, channels, 1)) for _ in range(len(self.convs2))]

    def __call__(self, x: mx.array, s: mx.array) -> mx.array:
        for c1, c2, n1, n2, a1, a2 in zip(self.convs1, self.convs2,
                                         self.adain1, self.adain2,
                                         self.alpha1, self.alpha2):
            xt = n1(x, s)
            xt = xt + (1 / a1) * (mx.sin(a1 * xt) ** 2)  # Snake1D
            xt = c1(xt)
            xt = n2(xt, s)
            xt = xt + (1 / a2) * (mx.sin(a2 * xt) ** 2)  # Snake1D
            xt = c2(xt)
            x = xt + x
        return x

class MLXSTFT:
    def __init__(self, filter_length=800, hop_length=200, win_length=800, window='hann'):
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length

        # Get the window exactly as PyTorch does
        self.window = get_window(window, win_length, fftbins=True).astype(np.float32)

        # Pad window if needed
        if win_length < filter_length:
            pad_start = (filter_length - win_length) // 2
            pad_end = filter_length - win_length - pad_start
            self.window = np.pad(self.window, (pad_start, pad_end))

    def transform(self, input_data):
        # Convert to numpy and ensure 2D
        audio_np = np.array(input_data)
        if audio_np.ndim == 1:
            audio_np = audio_np[None, :]

        magnitudes = []
        phases = []

        for batch_idx in range(audio_np.shape[0]):
            # Compute STFT using librosa
            stft = librosa.stft(
                audio_np[batch_idx],
                n_fft=self.filter_length,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=True,
                pad_mode='reflect'
            )

            # Get magnitude
            magnitude = np.abs(stft)

            # Get phase, matching PyTorch's initialization and accumulation
            phase = np.angle(stft)

            magnitudes.append(magnitude)
            phases.append(phase)

        magnitudes = np.stack(magnitudes, axis=0)
        phases = np.stack(phases, axis=0)

        return mx.array(magnitudes), mx.array(phases)

    def inverse(self, magnitude, phase):
        magnitude_np = np.array(magnitude)
        phase_np = np.array(phase)

        reconstructed = []

        for batch_idx in range(magnitude_np.shape[0]):
            # Unwrap phases for reconstruction
            phase_cont = np.unwrap(phase_np[batch_idx], axis=1)

            # Combine magnitude and phase
            stft = magnitude_np[batch_idx] * np.exp(1j * phase_cont)

            # Inverse STFT using librosa
            audio = librosa.istft(
                stft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=True,
                length=None
            )

            reconstructed.append(audio)

        # Stack and reshape to match PyTorch output shape
        reconstructed = np.stack(reconstructed, axis=0)[:, None, :]

        return mx.array(reconstructed)

    def __call__(self, input_data: mx.array) -> mx.array:
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return mx.expand_dims(reconstruction, axis=-2)

class SineGen(nn.Module):
    def __init__(self, samp_rate: int, upsample_scale: int, harmonic_num: int = 0,
                 sine_amp: float = 0.1, noise_std: float = 0.003,
                 voiced_threshold: float = 0, flag_for_pulse: bool = False):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        self.upsample_scale = upsample_scale

    def _f02uv(self, f0: mx.array) -> mx.array:
        return mx.array(f0 > self.voiced_threshold, dtype=mx.float32)

    def _f02sine(self, f0_values: mx.array) -> mx.array:
        # Convert to F0 in rad
        rad_values = mx.mod(f0_values / self.sampling_rate, 1)

        # Initial phase noise
        rand_ini = mx.random.uniform(shape=(f0_values.shape[0], f0_values.shape[2]))
        rand_ini = mx.index_update(rand_ini, (slice(None), 0), 0)
        rad_values = mx.index_update(rad_values[:, 0, :], slice(None),
                                   rad_values[:, 0, :] + rand_ini)

        if not self.flag_for_pulse:
            # Interpolate rad_values
            rad_values_t = mx.transpose(rad_values, (0, 2, 1))
            rad_values_t = mx.interpolate(rad_values_t,
                                        scale_factor=1/self.upsample_scale,
                                        mode="linear")
            rad_values = mx.transpose(rad_values_t, (0, 2, 1))

            # Calculate phase
            phase = mx.cumsum(rad_values, axis=1) * 2 * np.pi
            phase_t = mx.transpose(phase, (0, 2, 1)) * self.upsample_scale
            phase_t = mx.interpolate(phase_t,
                                   scale_factor=self.upsample_scale,
                                   mode="linear")
            phase = mx.transpose(phase_t, (0, 2, 1))
            return mx.sin(phase)
        else:
            # Pulse train generation logic would go here
            raise NotImplementedError("Pulse train generation not yet implemented")

    def __call__(self, f0: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        f0_buf = mx.zeros((f0.shape[0], f0.shape[1], self.dim))
        # Fundamental component
        fn = f0 * mx.array(range(1, self.harmonic_num + 2))[None, None, :]

        # Generate sine waveforms
        sine_waves = self._f02sine(fn) * self.sine_amp

        # Generate UV signal
        uv = self._f02uv(f0)

        # Generate noise
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * mx.random.normal(sine_waves.shape)

        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise