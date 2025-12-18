# Copyright © Anthony DePasquale
# Ported to MLX from https://github.com/FunAudioLLM/CosyVoice

"""
Causal convolution modules for CosyVoice3.

These modules implement causal convolutions used in the HiFi-GAN vocoder
and other components for streaming inference.
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution supporting both left (past) and right (future) causal types.

    - 'left': Standard causal convolution (past context only)
    - 'right': Lookahead convolution (future context for streaming)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal_type: str = "left",
    ):
        """
        Initialize CausalConv1d.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride (must be 1)
            dilation: Dilation factor
            groups: Number of groups for grouped convolution
            bias: Whether to use bias
            causal_type: 'left' for past-only, 'right' for future context
        """
        super().__init__()
        assert stride == 1, "CausalConv1d only supports stride=1"
        assert causal_type in ["left", "right"]

        self.causal_padding = (
            int((kernel_size * dilation - dilation) / 2) * 2 + (kernel_size + 1) % 2
        )
        self.causal_type = causal_type

        # MLX Conv1d: input (B, T, C), kernel (out_c, kW, in_c)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def __call__(self, x: mx.array, cache: Optional[mx.array] = None) -> mx.array:
        """
        Apply causal convolution.

        Args:
            x: Input tensor (B, C, T) in channel-first format
            cache: Optional cache for streaming (B, C, cache_len)

        Returns:
            Output tensor (B, C, T)
        """
        # Transpose to (B, T, C) for MLX Conv1d
        x = mx.swapaxes(x, 1, 2)
        input_timestep = x.shape[1]

        # Create or use cache
        if cache is None or cache.size == 0:
            cache = mx.zeros((x.shape[0], self.causal_padding, x.shape[2]))
        else:
            cache = mx.swapaxes(cache, 1, 2)

        # Apply causal padding
        if self.causal_type == "left":
            x = mx.concatenate([cache, x], axis=1)
        else:  # right
            x = mx.concatenate([x, cache], axis=1)

        # Apply convolution
        x = self.conv(x)

        # Verify output shape
        assert x.shape[1] == input_timestep

        # Transpose back to (B, C, T)
        return mx.swapaxes(x, 1, 2)


class CausalConv1dDownSample(nn.Module):
    """Causal 1D convolution with downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        """
        Initialize CausalConv1dDownSample.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            stride: Downsampling stride (must be > 1)
            dilation: Dilation factor (must be 1)
            groups: Number of groups
            bias: Whether to use bias
        """
        super().__init__()
        assert stride != 1 and dilation == 1
        assert kernel_size % stride == 0

        self.causal_padding = stride - 1
        self.stride = stride

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def __call__(self, x: mx.array, cache: Optional[mx.array] = None) -> mx.array:
        """
        Apply causal downsampling convolution.

        Args:
            x: Input tensor (B, C, T)
            cache: Optional cache (B, C, cache_len)

        Returns:
            Downsampled output (B, C, T//stride)
        """
        # Transpose to (B, T, C) for MLX
        x = mx.swapaxes(x, 1, 2)

        # Apply causal padding
        if cache is None or cache.size == 0:
            x = mx.pad(x, [(0, 0), (self.causal_padding, 0), (0, 0)])
        else:
            cache = mx.swapaxes(cache, 1, 2)
            x = mx.concatenate([cache, x], axis=1)

        # Apply convolution
        x = self.conv(x)

        # Transpose back to (B, C, T)
        return mx.swapaxes(x, 1, 2)


class CausalConv1dUpsample(nn.Module):
    """Causal 1D convolution with upsampling (using interpolation + conv)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        """
        Initialize CausalConv1dUpsample.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            stride: Upsampling factor
            dilation: Dilation factor (must be 1)
            groups: Number of groups
            bias: Whether to use bias
        """
        super().__init__()
        assert dilation == 1

        self.causal_padding = kernel_size - 1
        self.upsample_factor = stride

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def __call__(self, x: mx.array, cache: Optional[mx.array] = None) -> mx.array:
        """
        Apply causal upsampling convolution.

        Args:
            x: Input tensor (B, C, T)
            cache: Optional cache (B, C, cache_len)

        Returns:
            Upsampled output (B, C, T*stride)
        """
        # Transpose to (B, T, C) for MLX
        x = mx.swapaxes(x, 1, 2)

        # Upsample using nearest neighbor (repeat)
        x = mx.repeat(x, self.upsample_factor, axis=1)
        input_timestep = x.shape[1]

        # Apply causal padding
        if cache is None or cache.size == 0:
            x = mx.pad(x, [(0, 0), (self.causal_padding, 0), (0, 0)])
        else:
            cache = mx.swapaxes(cache, 1, 2)
            x = mx.concatenate([cache, x], axis=1)

        # Apply convolution
        x = self.conv(x)

        # Verify output shape
        assert input_timestep == x.shape[1]

        # Transpose back to (B, C, T)
        return mx.swapaxes(x, 1, 2)


class PreLookaheadLayer(nn.Module):
    """
    Pre-lookahead layer for CosyVoice3 flow model.

    This simple layer replaces the complex encoder in CosyVoice3,
    using just two causal convolutions with residual connection.

    Original PyTorch implementation works on channel-first (B, C, T) internally.
    """

    def __init__(
        self,
        in_channels: int,
        channels: int,
        pre_lookahead_len: int = 3,
    ):
        """
        Initialize PreLookaheadLayer.

        Args:
            in_channels: Number of input channels
            channels: Number of hidden channels
            pre_lookahead_len: Lookahead length (kernel_size - 1)
        """
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.pre_lookahead_len = pre_lookahead_len

        # First conv with lookahead
        # kernel_size = pre_lookahead_len + 1
        self.conv1 = nn.Conv1d(
            in_channels, channels, kernel_size=pre_lookahead_len + 1, padding=0
        )

        # Second conv (left causal, kernel_size=3)
        self.conv2 = nn.Conv1d(channels, in_channels, kernel_size=3, padding=0)

    def __call__(
        self, inputs: mx.array, context: Optional[mx.array] = None
    ) -> mx.array:
        """
        Apply pre-lookahead layer.

        Args:
            inputs: Input tensor (B, T, D)
            context: Optional lookahead context (B, pre_lookahead_len, D)

        Returns:
            Output tensor (B, T, D) with residual connection
        """
        # Look ahead padding (on right side)
        if context is None or context.shape[1] == 0:
            # Non-streaming: pad with zeros on right
            outputs = mx.pad(inputs, [(0, 0), (0, self.pre_lookahead_len), (0, 0)])
        else:
            # Streaming: concatenate context and pad remaining
            assert context.shape[1] == self.pre_lookahead_len
            outputs = mx.concatenate([inputs, context], axis=1)
            remaining_pad = self.pre_lookahead_len - context.shape[1]
            if remaining_pad > 0:
                outputs = mx.pad(outputs, [(0, 0), (0, remaining_pad), (0, 0)])

        # First convolution + leaky_relu (matching PyTorch F.leaky_relu)
        outputs = self.conv1(outputs)
        outputs = nn.leaky_relu(outputs)

        # Second convolution with left causal padding (kernel_size - 1 = 2)
        outputs = mx.pad(outputs, [(0, 0), (2, 0), (0, 0)])
        outputs = self.conv2(outputs)

        # Residual connection
        outputs = outputs + inputs

        return outputs
