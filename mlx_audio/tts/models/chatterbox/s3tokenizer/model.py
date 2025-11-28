# Ported from https://github.com/xingchensong/S3Tokenizer

from typing import Optional, Tuple
import mlx.core as mx
import mlx.nn as nn

from .utils import make_non_pad_mask, mask_to_bias


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    scaling: Optional[float] = None
) -> Tuple[mx.array, mx.array]:
    """Precompute frequency tensor for rotary embeddings."""
    freqs = 1.0 / (
        theta ** (mx.arange(0, dim, 2)[: (dim // 2)].astype(mx.float32) / dim)
    )
    t = mx.arange(end)
    if scaling is not None:
        t = t * scaling
    freqs = mx.outer(t, freqs).astype(mx.float32)
    cos_freqs = mx.cos(freqs)
    sin_freqs = mx.sin(freqs)
    cos_freqs = mx.concatenate([cos_freqs, cos_freqs], axis=-1)
    sin_freqs = mx.concatenate([sin_freqs, sin_freqs], axis=-1)
    return cos_freqs, sin_freqs


def apply_rotary_emb(
    xq: mx.array,
    xk: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> Tuple[mx.array, mx.array]:
    """Apply rotary embeddings to query and key tensors."""
    # Expand dimensions for broadcasting
    cos = mx.expand_dims(mx.expand_dims(cos, axis=0), axis=2)
    sin = mx.expand_dims(mx.expand_dims(sin, axis=0), axis=2)

    D = xq.shape[-1]
    # Split and rotate
    xq_half_l, xq_half_r = xq[..., : D // 2], xq[..., D // 2:]
    xq_rotated = mx.concatenate([-xq_half_r, xq_half_l], axis=-1)

    xk_half_l, xk_half_r = xk[..., : D // 2], xk[..., D // 2:]
    xk_rotated = mx.concatenate([-xk_half_r, xk_half_l], axis=-1)

    # Apply rotation
    xq_out = xq * cos + xq_rotated * sin
    xk_out = xk * cos + xk_rotated * sin

    return xq_out, xk_out


class FSMNMultiHeadAttention(nn.Module):
    """Multi-head attention with FSMN (Feedforward Sequential Memory Network)."""

    def __init__(
        self,
        n_state: int,
        n_head: int,
        kernel_size: int = 31,
    ):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

        self.fsmn_block = nn.Conv1d(
            in_channels=n_state,
            out_channels=n_state,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            groups=n_state,
            bias=False,
        )
        self.left_padding = (kernel_size - 1) // 2
        self.right_padding = kernel_size - 1 - self.left_padding

    def forward_fsmn(
        self, inputs: mx.array, mask: Optional[mx.array] = None
    ) -> mx.array:
        """Apply FSMN memory block."""
        b, t, n, d = inputs.shape
        inputs = inputs.reshape(b, t, -1)

        if mask is not None and mask.shape[2] > 0:
            inputs = inputs * mask

        x = mx.pad(inputs, [(0, 0), (self.left_padding, self.right_padding), (0, 0)])
        x = self.fsmn_block(x)
        x = x + inputs

        if mask is not None:
            x = x * mask

        return x

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        mask_pad: Optional[mx.array] = None,
        freqs_cis: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        B, T, D = x.shape
        scale = (D // self.n_head) ** -0.25

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.reshape(B, T, self.n_head, -1)
        k = k.reshape(B, T, self.n_head, -1)
        v = v.reshape(B, T, self.n_head, -1)

        if freqs_cis is not None:
            cos, sin = freqs_cis
            q, k = apply_rotary_emb(q, k, cos[:T], sin[:T])

        fsm_memory = self.forward_fsmn(v, mask_pad)

        q = q.transpose(0, 2, 1, 3) * scale
        k = k.transpose(0, 2, 1, 3) * scale
        v = v.transpose(0, 2, 1, 3)

        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=1, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, T, D)

        return self.out(output) + fsm_memory, None


class ResidualAttentionBlock(nn.Module):
    """Residual attention block with FSMN."""

    def __init__(
        self,
        n_state: int,
        n_head: int,
        kernel_size: int = 31,
    ):
        super().__init__()

        self.attn = FSMNMultiHeadAttention(n_state, n_head, kernel_size)
        self.attn_ln = nn.LayerNorm(n_state, eps=1e-6)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_mlp), nn.GELU(), nn.Linear(n_mlp, n_state)
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        mask_pad: Optional[mx.array] = None,
        freqs_cis: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        x = x + self.attn(self.attn_ln(x), mask=mask, mask_pad=mask_pad, freqs_cis=freqs_cis)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoderV2(nn.Module):
    """Audio encoder with convolutional frontend and transformer blocks."""

    def __init__(
        self,
        n_mels: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        stride: int,
    ):
        super().__init__()
        self.stride = stride

        self.conv1 = nn.Conv1d(
            in_channels=n_mels,
            out_channels=n_state,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=n_state,
            out_channels=n_state,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self._freqs_cis = precompute_freqs_cis(64, 1024 * 2)

        self.blocks = [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]

    def __call__(self, x: mx.array, x_len: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Args:
            x: Mel spectrogram (B, n_mels, T)
            x_len: Length of each audio (B,)

        Returns:
            x: Encoded features (B, T', n_state)
            x_len: Output lengths (B,)
        """
        # First conv with mask
        mask = make_non_pad_mask(x_len)
        mask = mx.expand_dims(mask, axis=1)  # (B, 1, T)

        x = x.transpose(0, 2, 1)  # (B, T, n_mels)
        mask_transposed = mask.transpose(0, 2, 1)  # (B, T, 1)

        x = self.conv1(x * mask_transposed)
        x = nn.gelu(x)
        x_len = (x_len + 2 - 1 * (3 - 1) - 1) // self.stride + 1

        # Second conv with mask
        mask = make_non_pad_mask(x_len)
        mask_transposed = mx.expand_dims(mask, axis=-1)  # (B, T, 1)

        x = self.conv2(x * mask_transposed)
        x = nn.gelu(x)
        x_len = (x_len + 2 - 1 * (3 - 1) - 1) // 2 + 1

        # Prepare masks for attention
        mask = make_non_pad_mask(x_len)
        mask_pad = mx.expand_dims(mask, axis=-1)  # (B, T, 1)
        mask = mask_to_bias(mask, x.dtype)
        mask = mx.expand_dims(mask, axis=1)  # (B, 1, T)

        for block in self.blocks:
            x = block(x, mask, mask_pad, self._freqs_cis)

        return x, x_len


class FSQCodebook(nn.Module):
    """Finite Scalar Quantization codebook."""

    def __init__(self, dim: int, level: int = 3):
        super().__init__()
        self.project_down = nn.Linear(dim, 8)
        self.level = level

    def encode(self, x: mx.array) -> mx.array:
        """Encode continuous features to discrete indices."""
        x_shape = x.shape

        # Flatten to (N, D)
        x = x.reshape(-1, x.shape[-1])

        # Project down to 8 dimensions
        h = self.project_down(x).astype(mx.float32)

        # Quantize: tanh -> scale -> round
        h = mx.tanh(h)
        h = h * 0.9990000128746033
        h = mx.round(h) + 1  # Values in {0, 1, 2}

        # Convert to index using base-3 encoding
        powers = mx.power(
            self.level,
            mx.arange(8).astype(mx.float32)
        )
        mu = mx.sum(h * powers, axis=-1)

        # Reshape back
        ind = mu.reshape(x_shape[0], x_shape[1]).astype(mx.int32)
        return ind


class FSQVectorQuantization(nn.Module):
    """FSQ Vector quantization for S3TokenizerV2."""

    def __init__(self, dim: int, codebook_size: int):
        super().__init__()
        assert 3**8 == codebook_size
        # Note: Use 'codebook' without underscore so tree_flatten includes it in parameters
        self.codebook = FSQCodebook(dim=dim, level=3)
        self.codebook_size = codebook_size

    def encode(self, x: mx.array) -> mx.array:
        """Encode features to discrete tokens."""
        return self.codebook.encode(x)
