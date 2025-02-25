from typing import Optional, List, Union, Tuple

import mlx.core as mx
import mlx.nn as nn
import librosa
import numpy as np
from scipy.signal import get_window
from typing import List, Tuple, Optional
import time
from ..interpolate import interpolate

def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def compute_norm(x: mx.array,
                p: int,
                dim: Optional[Union[int, List[int]]] = None,
                keepdim: bool = False) -> mx.array:
    """
    Compute the p-norm of a tensor along specified dimensions.

    Args:
        x: Input array
        p: Order of the norm (1 or 2)
        dim: Dimension(s) along which to compute the norm
        keepdim: Whether to keep the reduced dimensions

    Returns:
        MLX array containing the computed norm
    """
    if p not in [1, 2]:
        raise ValueError("Only p-norms with p of 1 or 2 are supported")

    # Handle dimension input
    if dim is None:
        dim = tuple(range(x.ndim))
    elif isinstance(dim, int):
        dim = (dim,)

    if p == 1:
        # L1 norm
        return mx.sum(mx.abs(x), axis=dim, keepdims=keepdim)
    else:
        # L2 norm
        return mx.sqrt(mx.sum(x * x, axis=dim, keepdims=keepdim))

def weight_norm(weight_v: mx.array,
                weight_g: mx.array,
                dim: Optional[int] = None) -> mx.array:
    """
    Applies weight normalization to the input tensor.

    Weight normalization reparameterizes weight vectors in a neural network
    as a magnitude scalar times a direction vector: w = g * v/||v||

    Args:
        weight_v: Weight direction tensor (v)
        weight_g: Weight magnitude tensor (g)
        dim: Dimension along which to normalize. If None, normalize over all dims
            except dim=-1

    Returns:
        Normalized weight tensor
    """
    rank = len(weight_v.shape)

    if dim is not None:
        # Adjust negative dim
        if dim < -1:
            dim += rank

        # Create list of axes to normalize over
        axes = list(range(rank))
        if dim != -1:
            axes.remove(dim)
    else:
        # Default behavior: normalize over all dimensions
        axes = list(range(rank))

    # Compute L2 norm of v along specified axes
    norm_v = compute_norm(weight_v, p=2, dim=axes, keepdim=True)

    # Normalize and scale by g: w = g * (v / ||v||)
    normalized_weight = weight_v / (norm_v + 1e-7)  # Add epsilon for numerical stability
    return normalized_weight * weight_g



class ConvWeighted(nn.Module):
    """Conv1d with weight normalization"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 1, dilation: int = 1, groups: int = 1, bias: bool = True, transpose_g: bool = False, encode: bool = False):
        super().__init__()

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.transpose_g = transpose_g

         # Initialize weight and direction vectors
        weight_shape_g = (out_channels, 1, 1)
        weight_shape_v = (out_channels, kernel_size, in_channels)
        weight_g = mx.ones(shape=weight_shape_g)
        weight_v = mx.ones(shape=weight_shape_v)
        # Store parameters
        self.weight_g = mx.array(weight_g)
        self.weight_v = mx.array(weight_v)
        if encode:
            self.bias = mx.zeros(in_channels)
        else:
            self.bias = mx.zeros(out_channels) if bias else None


    def __call__(self, x, conv, transpose_weight: bool = False, transpose_shape: Tuple[int, int, int] = None):

        weight = weight_norm(self.weight_v, self.weight_g, dim=0)

        if self.bias is not None:
            bias = self.bias.reshape(1, 1, -1)
        else:
            bias = None
        try:
            if self.bias is not None:
                return conv(x, weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups) + bias
            else:
                return conv(x, weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

        except Exception as e:
            print(f"Error: {e}")
            print(f"x.shape: {x.shape}, weight.shape: {weight.shape}")
            print(f"transpose_weight: {transpose_weight}")
            try:
                if self.bias is not None:
                    return conv(x, weight.T, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups) + bias
                else:
                    return conv(x, weight.T, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
            except Exception as e:
                print(f"Error: {e}")
                print(f"x.shape: {x.shape}, weight.shape: {weight.shape}")
                raise e

class AdaIN1d(nn.Module):
    def __init__(self, style_dim: int, num_features: int):
        super().__init__()
        self.norm = nn.InstanceNorm(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def __call__(self, x: mx.array, s: mx.array) -> mx.array:
        h = self.fc(s)
        h = mx.expand_dims(h, axis=2)  # Equivalent to view(..., 1)
        gamma, beta = mx.split(h, 2, axis=1)
        x = (1 + gamma) * self.norm(x) + beta
        return x

class AdaINResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3,
                 dilation: Tuple[int, int, int] = (1, 3, 5), style_dim: int = 64):
        super().__init__()
        self.convs1 = [
            ConvWeighted(channels, channels, kernel_size, 1,
                  padding=get_padding(kernel_size, dilation[i]),
                  dilation=dilation[i])
            for i in range(3)
        ]
        self.convs2 = [
            ConvWeighted(channels, channels, kernel_size, 1,
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

            xt = xt.swapaxes(2, 1)
            xt = c1(xt, mx.conv1d)
            xt = xt.swapaxes(2, 1)

            xt = n2(xt, s)
            xt = xt + (1 / a2) * (mx.sin(a2 * xt) ** 2)  # Snake1D

            xt = xt.swapaxes(2, 1)
            xt = c2(xt, mx.conv1d)
            xt = xt.swapaxes(2, 1)

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


class SineGen():
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
        """ f0_values: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1
        # initial phase noise (no noise for fundamental component)
        rand_ini = mx.random.normal((f0_values.shape[0], f0_values.shape[2]))
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
            rad_values = interpolate(rad_values.transpose(0, 2, 1), scale_factor=1/self.upsample_scale, mode="linear").transpose(0, 2, 1)
            phase = mx.cumsum(rad_values, axis=1) * 2 * np.pi
            phase = interpolate(phase.transpose(0, 2, 1) * self.upsample_scale, scale_factor=self.upsample_scale, mode="linear").transpose(0, 2, 1)
            sines = mx.sin(phase)
        else:
            # If necessary, make sure that the first time step of every
            # voiced segments is sin(pi) or cos(0)
            # This is used for pulse-train generation
            # identify the last time step in unvoiced segments
            uv = self._f02uv(f0_values)
            uv_1 = mx.roll(uv, shifts=-1, axis=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)
            # get the instantanouse phase
            tmp_cumsum = mx.cumsum(rad_values, axis=1)
            # different batch needs to be processed differently
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                # stores the accumulation of i.phase within
                # each voiced segments
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum
            # rad_values - tmp_cumsum: remove the accumulation of i.phase
            # within the previous voiced segment.
            i_phase = mx.cumsum(rad_values - tmp_cumsum, axis=1)
            # get the sines
            sines = mx.cos(i_phase * 2 * np.pi)
        return sines

    def __call__(self, f0: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        f0_buf = mx.zeros((f0.shape[0], f0.shape[1], self.dim))

        # Fundamental component
        fn = f0 * mx.arange(1, self.harmonic_num + 2)[None, None, :]

        # Generate sine waveforms
        sine_waves = self._f02sine(fn) * self.sine_amp


        # Generate UV signal
        uv = self._f02uv(f0)

        # Generate noise
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * mx.random.normal(sine_waves.shape)

        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise



class SourceModuleHnNSF(nn.Module):
    """ SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """
    def __init__(self, sampling_rate, upsample_scale, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0):
        super(SourceModuleHnNSF, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        # to produce sine waveforms
        self.l_sin_gen = SineGen(sampling_rate, upsample_scale, harmonic_num,
                                 sine_amp, add_noise_std, voiced_threshod)
        # to merge source harmonics into a single excitation
        self.l_linear = nn.Linear(harmonic_num + 1, 1)

    def __call__(self, x):
        """
        Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        # source for harmonic branch
        sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = mx.tanh(self.l_linear(sine_wavs))
        # source for noise branch, in the same shape as uv
        noise = mx.random.normal(uv.shape) * self.sine_amp / 3
        return sine_merge, noise, uv

class ReflectionPad1d(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def __call__(self, x):
        return mx.pad(x, ((0, 0), (0, 0), (self.padding[0], self.padding[1])))

def leaky_relu(x, negative_slope=0.01):
    return mx.where(x > 0, x, x * negative_slope)

class Generator(nn.Module):
    def __init__(self, style_dim, resblock_kernel_sizes, upsample_rates, upsample_initial_channel, resblock_dilation_sizes, upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        upsample_rates = mx.array(upsample_rates)
        self.m_source = SourceModuleHnNSF(
                    sampling_rate=24000,
                    upsample_scale=mx.prod(upsample_rates) * gen_istft_hop_size,
                    harmonic_num=8, voiced_threshod=10)
        self.f0_upsamp = nn.Upsample(scale_factor=mx.prod(upsample_rates) * gen_istft_hop_size)
        self.noise_convs = []
        self.noise_res = []
        self.ups = []
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(ConvWeighted(upsample_initial_channel//(2**(i+1)), upsample_initial_channel//(2**i),
                                   int(k), int(u), padding=int((k-u)//2), encode=True))
        self.resblocks = []
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2 **( i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes,resblock_dilation_sizes)):
                self.resblocks.append(AdaINResBlock1(ch, k, d, style_dim))
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            if i + 1 < len(upsample_rates):
                stride_f0 = int(mx.prod(upsample_rates[i + 1:]))
                self.noise_convs.append(nn.Conv1d(
                    gen_istft_n_fft + 2, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0, padding=(stride_f0+1) // 2))
                self.noise_res.append(AdaINResBlock1(c_cur, 7, [1,3,5], style_dim))
            else:
                self.noise_convs.append(nn.Conv1d(gen_istft_n_fft + 2, c_cur, kernel_size=1))
                self.noise_res.append(AdaINResBlock1(c_cur, 11, [1,3,5], style_dim))
        self.post_n_fft = gen_istft_n_fft
        self.conv_post = ConvWeighted(ch, self.post_n_fft + 2, 7, 1, padding=3)
        self.reflection_pad = ReflectionPad1d((1, 0))
        self.stft = MLXSTFT(filter_length=gen_istft_n_fft, hop_length=gen_istft_hop_size, win_length=gen_istft_n_fft)

    def __call__(self, x, s, f0):
        f0 = self.f0_upsamp(f0[:, None].transpose(0, 2, 1))  # bs,n,t
        har_source, noi_source, uv = self.m_source(f0)
        har_source = mx.squeeze(har_source.transpose(0, 2, 1), axis=1)
        har_spec, har_phase = self.stft.transform(har_source)
        har = mx.concatenate([har_spec, har_phase], axis=1)
        har = har.swapaxes(2, 1)
        for i in range(self.num_upsamples):
            x = leaky_relu(x, negative_slope=0.1)
            x_source = self.noise_convs[i](har)
            x_source = x_source.swapaxes(2, 1)
            x_source = self.noise_res[i](x_source, s)


            x = x.swapaxes(2, 1)
            x = self.ups[i](x, mx.conv_transpose1d)
            x = x.swapaxes(2, 1)


            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            x = x + x_source


            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x, s)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x, s)
            x = xs / self.num_kernels

        x = leaky_relu(x, negative_slope=0.01)

        x = x.swapaxes(2, 1)
        x = self.conv_post(x, mx.conv1d)
        x = x.swapaxes(2, 1)

        spec = mx.exp(x[:,:self.post_n_fft // 2 + 1, :])
        phase = mx.sin(x[:, self.post_n_fft // 2 + 1:, :])
        result = self.stft.inverse(spec, phase)
        return result


class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type
        self.interpolate = nn.Upsample(scale_factor=2, mode='nearest', align_corners=True)

    def __call__(self, x):
        if self.layer_type == 'none':
            return x
        else:
            return self.interpolate(x)



class AdainResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, actv=nn.LeakyReLU(0.2), upsample='none', dropout_p=0.0, bias=False, conv_type=None):
        super().__init__()
        self.actv = actv
        self.dim_in = dim_in
        self.conv_type = conv_type
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim, bias)
        self.dropout = nn.Dropout(dropout_p)
        if upsample == 'none':
            self.pool = nn.Identity()
        else:
            self.pool = ConvWeighted(1, dim_in, kernel_size=3, stride=2, padding=1, groups=dim_in)


    def _build_weights(self, dim_in, dim_out, style_dim, bias: bool = False):
        self.conv1 = ConvWeighted(dim_in, dim_out, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvWeighted(dim_out, dim_out, kernel_size=3, stride=1, padding=1)
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = ConvWeighted(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False)

    def _shortcut(self, x):
        x = x.swapaxes(2, 1)
        x = self.upsample(x)
        x = x.swapaxes(2, 1)

        if self.learned_sc:
            x = x.swapaxes(2, 1)
            x = self.conv1x1(x, mx.conv1d)
            x = x.swapaxes(2, 1)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)

        # Manually implement grouped ConvTranspose1d since MLX doesn't support groups
        if self.upsample_type == 'none':
            x = self.pool(x)  # [B*C, 1, L*2]
        else:
            x = x.swapaxes(2, 1)
            # TODO: Replace with official MLX implementation
            x = self.pool(x, mx.conv_transpose1d)
            x = mx.pad(x, ((0, 0), (1, 0), (0, 0)))
            x = x.swapaxes(2, 1)

        x = x.swapaxes(2, 1)
        x = self.conv1(self.dropout(x), mx.conv1d)
        x = x.swapaxes(2, 1)

        x = self.norm2(x, s)
        x = self.actv(x)

        x = x.swapaxes(2, 1)
        x = self.conv2(x, mx.conv1d)
        x = x.swapaxes(2, 1)
        return x

    def __call__(self, x, s):
        print("x.shape before residual", x.shape , "s.shape", s.shape)
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / np.sqrt(2)
        return out


class Decoder(nn.Module):
    def __init__(self, dim_in, style_dim, dim_out,
                 resblock_kernel_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 resblock_dilation_sizes,
                 upsample_kernel_sizes,
                 gen_istft_n_fft, gen_istft_hop_size):
        super().__init__()
        self.encode = AdainResBlk1d(dim_in + 2, 1024, style_dim, conv_type=mx.conv1d)
        self.decode = []
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim, conv_type=mx.conv1d))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim, conv_type=mx.conv1d))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim, conv_type=mx.conv1d))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 512, style_dim, upsample=True, conv_type=mx.conv1d))
        self.F0_conv = ConvWeighted(1, 1, kernel_size=3, stride=2, padding=1, groups=1)
        self.N_conv = ConvWeighted(1, 1, kernel_size=3, stride=2, padding=1, groups=1)
        self.asr_res = [ConvWeighted(512, 64, kernel_size=1, padding=0)]
        self.generator = Generator(style_dim, resblock_kernel_sizes, upsample_rates,
                                   upsample_initial_channel, resblock_dilation_sizes,
                                   upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size)

    def __call__(self, asr, F0_curve, N, s):
        s = mx.array(s)
        F0 = self.F0_conv(F0_curve[:, None, :].swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)
        N = self.N_conv(N[:, None, :].swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)
        x = mx.concatenate([asr, F0, N], axis=1)
        x = self.encode(x, s)
        asr_res = self.asr_res[0](asr.swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)
        res = True
        # for block in self.decode: # Not working in MLX
        #     if res:
        #         x = mx.concatenate([x, asr_res, F0, N], axis=1)
        #     x = block(x, s)
        #     # Check if this block has upsampling
        #     if hasattr(block, 'upsample_type') and block.upsample_type != "none":
        #         res = False
        # x = self.generator(x, s, F0_curve) # Not working in MLX
        return x, s, F0_curve, asr_res, F0, N