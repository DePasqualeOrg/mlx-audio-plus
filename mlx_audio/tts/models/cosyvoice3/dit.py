# Copyright © Anthony DePasquale
# Ported to MLX from https://github.com/FunAudioLLM/CosyVoice

"""
Diffusion Transformer (DiT) implementation for CosyVoice3.

This module implements the DiT architecture used in CosyVoice3 for
mel-spectrogram generation via flow matching.
"""

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class SinusPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def __call__(self, x: mx.array, scale: float = 1000) -> mx.array:
        """
        Generate sinusoidal embeddings.

        Args:
            x: Input tensor (B,) of timesteps
            scale: Scale factor for embeddings

        Returns:
            Embeddings (B, dim)
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = mx.exp(mx.arange(half_dim, dtype=mx.float32) * -emb)
        emb = scale * mx.expand_dims(x, 1) * mx.expand_dims(emb, 0)
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
        return emb


class TimestepEmbedding(nn.Module):
    """Timestep embedding using sinusoidal position embedding + MLP."""

    def __init__(self, dim: int, freq_embed_dim: int = 256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp_0 = nn.Linear(freq_embed_dim, dim)
        self.time_mlp_2 = nn.Linear(dim, dim)

    def __call__(self, timestep: mx.array) -> mx.array:
        """
        Generate timestep embeddings.

        Args:
            timestep: Timestep values (B,)

        Returns:
            Embeddings (B, dim)
        """
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.astype(timestep.dtype)
        time = self.time_mlp_0(time_hidden)
        time = nn.silu(time)
        time = self.time_mlp_2(time)
        return time


class CausalConvPositionEmbedding(nn.Module):
    """Causal convolutional position embedding for streaming."""

    def __init__(self, dim: int, kernel_size: int = 31, groups: int = 16):
        super().__init__()
        assert kernel_size % 2 != 0
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=0)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=0)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """
        Apply causal convolutional position embedding.

        Args:
            x: Input (B, N, D)
            mask: Optional mask (B, N)

        Returns:
            Position-embedded output (B, N, D)
        """
        if mask is not None:
            mask = mx.expand_dims(mask, -1)
            x = mx.where(mask, x, mx.zeros_like(x))

        # Apply causal padding and convolutions
        # Pad on the left for causal behavior
        x = mx.pad(x, [(0, 0), (self.kernel_size - 1, 0), (0, 0)])
        x = self.conv1(x)
        x = nn.mish(x)
        x = mx.pad(x, [(0, 0), (self.kernel_size - 1, 0), (0, 0)])
        x = self.conv2(x)
        out = nn.mish(x)

        if mask is not None:
            out = mx.where(mask, out, mx.zeros_like(out))

        return out


class InputEmbedding(nn.Module):
    """Input embedding for combining noised audio, condition, mu, and speaker."""

    def __init__(
        self,
        mel_dim: int,
        text_dim: int,
        out_dim: int,
        spk_dim: Optional[int] = None,
    ):
        super().__init__()
        spk_dim = 0 if spk_dim is None else spk_dim
        self.spk_dim = spk_dim
        self.proj = nn.Linear(mel_dim * 2 + text_dim + spk_dim, out_dim)
        self.conv_pos_embed = CausalConvPositionEmbedding(dim=out_dim)

    def __call__(
        self,
        x: mx.array,
        cond: mx.array,
        text_embed: mx.array,
        spks: mx.array,
    ) -> mx.array:
        """
        Combine inputs and apply position embedding.

        Args:
            x: Noised input audio (B, N, mel_dim)
            cond: Condition audio (B, N, mel_dim)
            text_embed: Text/mu embeddings (B, N, text_dim)
            spks: Speaker embeddings (B, spk_dim)

        Returns:
            Combined embeddings (B, N, out_dim)
        """
        to_cat = [x, cond, text_embed]
        if self.spk_dim > 0:
            # Repeat speaker embedding for each time step
            spks = mx.broadcast_to(
                mx.expand_dims(spks, 1), (spks.shape[0], x.shape[1], spks.shape[-1])
            )
            to_cat.append(spks)

        x = self.proj(mx.concatenate(to_cat, axis=-1))
        x = self.conv_pos_embed(x) + x
        return x


class GRN(nn.Module):
    """Global Response Normalization layer."""

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = mx.zeros((1, 1, dim))
        self.beta = mx.zeros((1, 1, dim))

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply Global Response Normalization.

        Args:
            x: Input (B, N, D)

        Returns:
            Normalized output (B, N, D)
        """
        gx = mx.sqrt(mx.sum(x * x, axis=1, keepdims=True))
        nx = gx / (mx.mean(gx, axis=-1, keepdims=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        approximate: str = "tanh",
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        # Using GELU with approximate='tanh' for compatibility
        self.ff_0_0 = nn.Linear(dim, inner_dim)
        self.ff_1 = nn.Dropout(dropout)
        self.ff_2 = nn.Linear(inner_dim, dim_out)
        self.approximate = approximate

    def __call__(self, x: mx.array) -> mx.array:
        """Apply feed-forward network."""
        x = self.ff_0_0(x)
        # Apply GELU with tanh approximation
        x = nn.gelu_approx(x) if self.approximate == "tanh" else nn.gelu(x)
        x = self.ff_1(x)
        x = self.ff_2(x)
        return x


class AdaLayerNormZero(nn.Module):
    """Adaptive Layer Normalization with zero initialization for DiT blocks."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, affine=False, eps=1e-6)

    def __call__(
        self, x: mx.array, emb: mx.array
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        """
        Apply adaptive layer normalization.

        Args:
            x: Input (B, N, D)
            emb: Conditioning embedding (B, D)

        Returns:
            Tuple of (normalized_x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        """
        emb = self.linear(nn.silu(emb))
        # Split into 6 parts
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mx.split(
            emb, 6, axis=1
        )

        x = self.norm(x) * (1 + mx.expand_dims(scale_msa, 1)) + mx.expand_dims(
            shift_msa, 1
        )
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroFinal(nn.Module):
    """Final adaptive layer normalization (only shift and scale, no gate)."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, affine=False, eps=1e-6)

    def __call__(self, x: mx.array, emb: mx.array) -> mx.array:
        """
        Apply final adaptive layer normalization.

        Args:
            x: Input (B, N, D)
            emb: Conditioning embedding (B, D)

        Returns:
            Normalized output (B, N, D)
        """
        emb = self.linear(nn.silu(emb))
        scale, shift = mx.split(emb, 2, axis=1)
        x = self.norm(x) * (1 + mx.expand_dims(scale, 1)) + mx.expand_dims(shift, 1)
        return x


def rotate_half(x: mx.array) -> mx.array:
    """
    Rotate half the hidden dims of the input (for RoPE).

    Matches x_transformers implementation: treats consecutive pairs
    and rotates each pair independently.

    Input:  [a, b, c, d, e, f, ...]
    Pairs:  [(a,b), (c,d), (e,f), ...]
    Output: [-b, a, -d, c, -f, e, ...]
    """
    # Reshape to pairs: (..., d) -> (..., d//2, 2)
    shape = x.shape
    x = x.reshape(*shape[:-1], shape[-1] // 2, 2)
    # Split pairs
    x1 = x[..., 0]  # First element of each pair
    x2 = x[..., 1]  # Second element of each pair
    # Stack as (-x2, x1) interleaved
    rotated = mx.stack([-x2, x1], axis=-1)
    # Flatten back: (..., d//2, 2) -> (..., d)
    return rotated.reshape(shape)


def apply_rotary_pos_emb(t: mx.array, freqs: mx.array, scale: float = 1.0) -> mx.array:
    """
    Apply rotary position embedding to input tensor.

    Matches x_transformers implementation exactly.

    Args:
        t: Input tensor to rotate (B, N, D) or (B, H, N, D)
        freqs: Frequency tensor (1, N, D) - raw angles with pairs interleaved
        scale: Scaling factor (default 1.0)

    Returns:
        Tensor with rotary embeddings applied
    """
    rot_dim = freqs.shape[-1]
    seq_len = t.shape[-2]
    orig_dtype = t.dtype

    # Slice freqs to match sequence length
    freqs = freqs[:, -seq_len:, :]

    # Handle 4D tensor (B, H, N, D)
    if t.ndim == 4 and freqs.ndim == 3:
        # (1, N, D) -> (1, 1, N, D)
        freqs = mx.expand_dims(freqs, 1)

    # Partial rotary embeddings (GPT-J style)
    t_rot, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]

    # Apply rotation: t * cos + rotate_half(t) * sin
    t_rot = (t_rot * mx.cos(freqs) * scale) + (
        rotate_half(t_rot) * mx.sin(freqs) * scale
    )

    out = mx.concatenate([t_rot, t_unrotated], axis=-1)
    return out.astype(orig_dtype)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding module.

    Matches x_transformers implementation exactly.
    """

    def __init__(
        self,
        dim: int,
        use_xpos: bool = False,
        scale_base: int = 512,
        interpolation_factor: float = 1.0,
        base: float = 10000.0,
        base_rescale_factor: float = 1.0,
    ):
        super().__init__()
        # Rescale base for longer sequences (NTK-aware scaling)
        base *= base_rescale_factor ** (dim / (dim - 2))

        # Compute inv_freq - stored as non-learnable (frozen) attribute
        inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
        self._inv_freq = (
            inv_freq  # Use underscore to avoid it being registered as parameter
        )
        self.interpolation_factor = interpolation_factor
        self.use_xpos = use_xpos
        self.scale_base = scale_base

        if use_xpos:
            scale = (mx.arange(0, dim, 2).astype(mx.float32) + 0.4 * dim) / (1.4 * dim)
            self._scale = scale
        else:
            self._scale = None

        self._cache = {}

    def forward_from_seq_len(self, seq_len: int) -> Tuple[mx.array, mx.array]:
        """
        Get rotary embeddings for a given sequence length.

        Args:
            seq_len: Sequence length

        Returns:
            Tuple of (frequencies, scale)
        """
        if seq_len not in self._cache:
            t = mx.arange(seq_len, dtype=mx.float32)
            self._cache[seq_len] = self._forward(t)
        return self._cache[seq_len]

    def _forward(self, t: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Compute rotary embeddings.

        Args:
            t: Position indices (N,) or (B, N)

        Returns:
            Tuple of (frequencies, scale)
        """
        if t.ndim == 1:
            t = mx.expand_dims(t, 0)  # (1, N)

        # Compute frequencies: (B, N, dim/2)
        freqs = mx.einsum("bi,j->bij", t, self._inv_freq) / self.interpolation_factor

        # Stack and interleave: each angle appears twice for the pair
        # (B, N, dim/2) -> (B, N, dim/2, 2) -> (B, N, dim)
        freqs = mx.stack([freqs, freqs], axis=-1)
        freqs = freqs.reshape(freqs.shape[0], freqs.shape[1], -1)

        if self._scale is None:
            return freqs, mx.array(1.0)

        # Compute xpos scale
        max_pos = int(mx.max(t).item()) + 1
        power = (t - (max_pos // 2)) / self.scale_base
        scale = self._scale ** mx.expand_dims(power, -1)
        scale = mx.stack([scale, scale], axis=-1)
        scale = scale.reshape(scale.shape[0], scale.shape[1], -1)

        return freqs, scale


class Attention(nn.Module):
    """Multi-head attention with rotary position embedding support."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = dim_head * heads
        self.dropout = dropout

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)

        self.to_out_0 = nn.Linear(self.inner_dim, dim)
        self.to_out_1 = nn.Dropout(dropout)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        rope: Optional[Tuple[mx.array, Optional[mx.array]]] = None,
    ) -> mx.array:
        """
        Apply multi-head attention.

        Args:
            x: Input (B, N, D)
            mask: Attention mask (B, N) or (B, 1, N, N)
            rope: Rotary position embedding tuple (freqs, scale)

        Returns:
            Attention output (B, N, D)
        """
        B, N, _ = x.shape

        # Project to Q, K, V - shape (B, N, inner_dim) where inner_dim = heads * head_dim
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        # Apply rotary position embedding BEFORE reshaping to multi-head format
        # This matches PyTorch x_transformers behavior exactly
        # RoPE rotates only the first dim_head dimensions of the flattened representation
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale = xpos_scale if xpos_scale is not None else 1.0
            k_xpos_scale = (1.0 / xpos_scale) if xpos_scale is not None else 1.0
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        # Reshape for multi-head attention: (B, N, heads, head_dim)
        query = query.reshape(B, N, self.heads, self.dim_head)
        key = key.reshape(B, N, self.heads, self.dim_head)
        value = value.reshape(B, N, self.heads, self.dim_head)

        # Transpose for attention: (B, heads, N, head_dim)
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)

        # Prepare attention mask for scaled_dot_product_attention
        # Convert from boolean mask to additive mask (False -> -inf, True -> 0)
        attn_mask = None
        if mask is not None:
            if mask.ndim == 2:
                # (B, N) -> (B, 1, 1, N)
                input_mask = mx.expand_dims(mx.expand_dims(mask, 1), 1)
            elif mask.ndim == 3:
                input_mask = mx.expand_dims(mask, 1)
            else:
                input_mask = mask
            # Convert boolean mask to additive mask: False -> -inf, True -> 0
            neg_inf = mx.array(float("-inf"), dtype=query.dtype)
            attn_mask = mx.where(
                input_mask, mx.zeros(input_mask.shape, dtype=query.dtype), neg_inf
            )

        # Use optimized fused attention kernel
        scale = 1.0 / math.sqrt(self.dim_head)
        out = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=scale, mask=attn_mask
        )

        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(B, N, self.inner_dim)
        out = out.astype(query.dtype)

        # Output projection
        out = self.to_out_0(out)
        out = self.to_out_1(out)

        # Apply mask to output
        if mask is not None:
            if mask.ndim == 2:
                out_mask = mx.expand_dims(mask, -1)
            elif mask.ndim == 4:
                # Use the last row of mask for output masking
                out_mask = mx.expand_dims(mask[:, 0, -1, :], -1)
            else:
                out_mask = mx.expand_dims(mask, -1)
            out = mx.where(out_mask, out, mx.zeros_like(out))

        return out


class DiTBlock(nn.Module):
    """Diffusion Transformer Block with adaptive layer norm."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = Attention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )
        self.ff_norm = nn.LayerNorm(dim, affine=False, eps=1e-6)
        self.ff = FeedForward(
            dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh"
        )

    def __call__(
        self,
        x: mx.array,
        t: mx.array,
        mask: Optional[mx.array] = None,
        rope: Optional[Tuple[mx.array, Optional[mx.array]]] = None,
    ) -> mx.array:
        """
        Apply DiT block.

        Args:
            x: Input (B, N, D)
            t: Time embedding (B, D)
            mask: Attention mask
            rope: Rotary position embedding

        Returns:
            Output (B, N, D)
        """
        # Pre-norm and modulation for attention
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        # Attention
        attn_output = self.attn(x=norm, mask=mask, rope=rope)

        # Process attention output with gate
        x = x + mx.expand_dims(gate_msa, 1) * attn_output

        # FFN with modulation
        ff_norm = self.ff_norm(x) * (1 + mx.expand_dims(scale_mlp, 1)) + mx.expand_dims(
            shift_mlp, 1
        )
        ff_output = self.ff(ff_norm)
        x = x + mx.expand_dims(gate_mlp, 1) * ff_output

        return x


def subsequent_chunk_mask(size: int, chunk_size: int) -> mx.array:
    """
    Create mask for subsequent steps with chunk size (for streaming encoder).

    This matches PyTorch's subsequent_chunk_mask:
    - Position i can attend to all positions j where j < (chunk_idx[i] + 1) * chunk_size
    - Each position can see up to the end of its current chunk

    Args:
        size: Sequence length
        chunk_size: Chunk size

    Returns:
        Attention mask (size, size) where True means can attend

    Examples:
        >>> subsequent_chunk_mask(4, 2)
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
    """
    pos_idx = mx.arange(size)
    # block_value[i] = (i // chunk_size + 1) * chunk_size = end of chunk for position i
    block_value = (pos_idx // chunk_size + 1) * chunk_size
    # Position j can be attended from position i if j < block_value[i]
    # ret[i, j] = pos_idx[j] < block_value[i]
    ret = mx.expand_dims(pos_idx, 0) < mx.expand_dims(block_value, 1)
    return ret


def add_optional_chunk_mask(
    x: mx.array,
    mask: mx.array,
    use_dynamic_chunk: bool,
    use_dynamic_left_chunk: bool,
    decoding_chunk_size: int,
    static_chunk_size: int,
    num_decoding_left_chunks: int,
) -> mx.array:
    """
    Create chunk-based attention mask for streaming inference.

    Args:
        x: Input tensor (B, N, D)
        mask: Base mask (B, N) where True/non-zero means valid position
        use_dynamic_chunk: Use dynamic chunk size
        use_dynamic_left_chunk: Use dynamic left chunk
        decoding_chunk_size: Chunk size for decoding
        static_chunk_size: Static chunk size (>0 for streaming)
        num_decoding_left_chunks: Number of left chunks to attend to

    Returns:
        Attention mask (B, 1, N, N) where True means can attend
    """
    B, N, _ = x.shape

    # Convert mask to boolean (matches PyTorch's mask.bool())
    if mask is not None:
        mask = mask.astype(mx.bool_)

    if static_chunk_size > 0:
        # Streaming mode: create chunk-based mask
        chunk_masks = subsequent_chunk_mask(N, static_chunk_size)  # (N, N)
        chunk_masks = mx.expand_dims(chunk_masks, 0)  # (1, N, N)
        # Apply base mask: mask is (B, N), expand to (B, 1, N)
        if mask is not None:
            chunk_masks = mx.expand_dims(mask, 1) & chunk_masks  # (B, N, N)
        else:
            chunk_masks = mx.broadcast_to(chunk_masks, (B, N, N))
    else:
        # Non-streaming: use base mask or full attention
        if mask is not None:
            chunk_masks = mask  # (B, N)
        else:
            chunk_masks = mx.ones((B, N), dtype=mx.bool_)

    # Ensure correct output shape and handle zero-sum case
    if chunk_masks.ndim == 2:
        # (B, N) -> broadcast to (B, N, N) by expanding
        chunk_masks = mx.expand_dims(chunk_masks, 1)  # (B, 1, N)
        chunk_masks = mx.broadcast_to(chunk_masks, (B, N, N))

    # Safety fix: if any row is all-False, force to True (prevents NaN from softmax)
    # Use pure GPU operation to avoid CPU-GPU sync from conditional check
    row_sums = mx.sum(chunk_masks.astype(mx.float32), axis=-1, keepdims=True)
    all_false_rows = row_sums == 0
    chunk_masks = mx.where(
        mx.broadcast_to(all_false_rows, chunk_masks.shape),
        mx.ones_like(chunk_masks),
        chunk_masks,
    )

    # Add head dimension: (B, N, N) -> (B, 1, N, N)
    chunk_masks = mx.expand_dims(chunk_masks, 1)

    return chunk_masks


class DiT(nn.Module):
    """
    Diffusion Transformer for CosyVoice3.

    This is the main estimator that replaces the U-Net style decoder
    used in CosyVoice2.
    """

    def __init__(
        self,
        dim: int = 1024,
        depth: int = 22,
        heads: int = 16,
        dim_head: int = 64,
        dropout: float = 0.1,
        ff_mult: int = 2,
        mel_dim: int = 80,
        mu_dim: Optional[int] = None,
        long_skip_connection: bool = False,
        spk_dim: Optional[int] = None,
        out_channels: Optional[int] = None,
        static_chunk_size: int = 50,
        num_decoding_left_chunks: int = -1,
    ):
        """
        Initialize DiT.

        Args:
            dim: Model dimension
            depth: Number of transformer blocks
            heads: Number of attention heads
            dim_head: Dimension per head
            dropout: Dropout rate
            ff_mult: Feed-forward expansion multiplier
            mel_dim: Mel spectrogram dimension
            mu_dim: Mu (condition) dimension (defaults to mel_dim)
            long_skip_connection: Use long skip connection
            spk_dim: Speaker embedding dimension
            out_channels: Output channels (defaults to mel_dim)
            static_chunk_size: Chunk size for streaming
            num_decoding_left_chunks: Number of left chunks for streaming
        """
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if mu_dim is None:
            mu_dim = mel_dim
        self.input_embed = InputEmbedding(mel_dim, mu_dim, dim, spk_dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = [
            DiTBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                ff_mult=ff_mult,
                dropout=dropout,
            )
            for _ in range(depth)
        ]

        self.long_skip_connection = (
            nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None
        )

        self.norm_out = AdaLayerNormZeroFinal(dim)
        self.proj_out = nn.Linear(
            dim, mel_dim if out_channels is None else out_channels
        )
        self.out_channels = out_channels
        self.static_chunk_size = static_chunk_size
        self.num_decoding_left_chunks = num_decoding_left_chunks

    def __call__(
        self,
        x: mx.array,
        mask: mx.array,
        mu: mx.array,
        t: mx.array,
        spks: Optional[mx.array] = None,
        cond: Optional[mx.array] = None,
        streaming: bool = False,
    ) -> mx.array:
        """
        Forward pass of DiT.

        Args:
            x: Noised input (B, mel_dim, N) - note channel-first input
            mask: Mask (B, N)
            mu: Mu/condition (B, mel_dim, N)
            t: Timestep (B,) or scalar
            spks: Speaker embedding (B, D)
            cond: Condition audio (B, mel_dim, N)
            streaming: Whether in streaming mode

        Returns:
            Output (B, mel_dim, N) - channel-first
        """
        # Transpose from channel-first to sequence-first
        x = mx.swapaxes(x, 1, 2)  # (B, N, mel_dim)
        mu = mx.swapaxes(mu, 1, 2)  # (B, N, mu_dim)
        cond = mx.swapaxes(cond, 1, 2)  # (B, N, mel_dim)

        # spks stays as (B, D) - InputEmbedding expects (B, D)

        B, N, _ = x.shape
        if t.ndim == 0:
            t = mx.broadcast_to(t, (B,))

        # Time embedding
        t = self.time_embed(t)

        # Input embedding
        x = self.input_embed(x, cond, mu, spks)

        # Rotary embeddings
        rope = self.rotary_embed.forward_from_seq_len(N)

        if self.long_skip_connection is not None:
            residual = x

        # Create attention mask
        if streaming:
            attn_mask = add_optional_chunk_mask(
                x, mask, False, False, 0, self.static_chunk_size, -1
            )
        else:
            attn_mask = add_optional_chunk_mask(x, mask, False, False, 0, 0, -1)

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, t, mask=attn_mask, rope=rope)

        # Long skip connection
        if self.long_skip_connection is not None:
            x = self.long_skip_connection(mx.concatenate([x, residual], axis=-1))

        # Final normalization and projection
        x = self.norm_out(x, t)
        output = self.proj_out(x)

        # Transpose back to channel-first
        output = mx.swapaxes(output, 1, 2)  # (B, mel_dim, N)

        return output
