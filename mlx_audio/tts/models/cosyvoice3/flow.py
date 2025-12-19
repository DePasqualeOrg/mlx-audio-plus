# Copyright © Anthony DePasquale
# Ported to MLX from https://github.com/FunAudioLLM/CosyVoice

"""
Flow matching module for CosyVoice3.

This module implements the flow matching decoder using DiT (Diffusion Transformer)
as the estimator, replacing the U-Net style decoder used in CosyVoice2.
"""

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .convolution import PreLookaheadLayer
from .dit import DiT


def make_pad_mask(lengths: mx.array, max_len: Optional[int] = None) -> mx.array:
    """
    Create a padding mask from sequence lengths.

    Args:
        lengths: Sequence lengths (B,)
        max_len: Maximum length (if None, use max of lengths)

    Returns:
        Boolean mask (B, max_len) where True indicates padding
    """
    if max_len is None:
        max_len = int(mx.max(lengths).item())
    positions = mx.arange(max_len)
    mask = positions[None, :] >= lengths[:, None]
    return mask


class CosyVoice3ConditionalCFM(nn.Module):
    """
    Conditional Flow Matching module for CosyVoice3 with DiT estimator.

    This implements the ODE-based flow matching algorithm with classifier-free
    guidance for mel-spectrogram generation.
    """

    def __init__(
        self,
        estimator: nn.Module,
        sigma_min: float = 1e-6,
        t_scheduler: str = "cosine",
        inference_cfg_rate: float = 0.7,
        rand_noise: Optional[mx.array] = None,
    ):
        """
        Initialize CosyVoice3ConditionalCFM.

        Args:
            estimator: DiT model for estimating the velocity field
            sigma_min: Minimum noise level
            t_scheduler: Time scheduler type ('cosine' or 'linear')
            inference_cfg_rate: Classifier-free guidance rate
            rand_noise: Pre-computed random noise for deterministic generation
        """
        super().__init__()
        self.estimator = estimator
        self.sigma_min = sigma_min
        self.t_scheduler = t_scheduler
        self.inference_cfg_rate = inference_cfg_rate
        # Pre-computed noise for deterministic generation (matches PyTorch CausalConditionalCFM)
        # Use underscore prefix to avoid it being treated as a learnable parameter
        self._rand_noise = rand_noise

    def _cosine_schedule(self, t: mx.array) -> mx.array:
        """Apply cosine schedule to timestep."""
        return 1 - mx.cos(t * math.pi / 2)

    def __call__(
        self,
        mu: mx.array,
        mask: mx.array,
        spks: mx.array,
        cond: mx.array,
        n_timesteps: int = 10,
        streaming: bool = False,
    ) -> Tuple[mx.array, None]:
        """
        Generate mel-spectrogram using flow matching.

        Args:
            mu: Condition/text embedding (B, mel_dim, N)
            mask: Mask (B, 1, N)
            spks: Speaker embedding (B, D)
            cond: Conditioning audio (B, mel_dim, N)
            n_timesteps: Number of ODE solver steps
            streaming: Whether in streaming mode

        Returns:
            Tuple of (generated_mel, None)
        """
        # Use pre-computed noise for deterministic generation (matches PyTorch)
        B, mel_dim, N = mu.shape
        if self._rand_noise is not None:
            # Slice pre-computed noise to match required size
            z = self._rand_noise[:, :, :N].astype(mu.dtype)
        else:
            # Fallback: generate noise with fixed seed
            mx.random.seed(0)
            z = mx.random.normal(shape=(B, mel_dim, N))

        # Solve ODE
        output = self.solve_euler(
            z=z,
            mu=mu,
            mask=mask,
            spks=spks,
            cond=cond,
            n_timesteps=n_timesteps,
            streaming=streaming,
        )

        return output, None

    def solve_euler(
        self,
        z: mx.array,
        mu: mx.array,
        mask: mx.array,
        spks: mx.array,
        cond: mx.array,
        n_timesteps: int = 10,
        streaming: bool = False,
    ) -> mx.array:
        """
        Solve the ODE using Euler method with classifier-free guidance.

        Uses batched computation for efficiency: conditional and unconditional
        paths are batched together in a single forward pass through the estimator.

        Args:
            z: Initial noise (B, mel_dim, N)
            mu: Condition embedding (B, mel_dim, N)
            mask: Mask (B, 1, N)
            spks: Speaker embedding (B, D)
            cond: Conditioning audio (B, mel_dim, N)
            n_timesteps: Number of steps
            streaming: Whether in streaming mode

        Returns:
            Generated mel-spectrogram (B, mel_dim, N)
        """
        # Create time span with cosine schedule
        t_span = mx.linspace(0, 1, n_timesteps + 1)
        if self.t_scheduler == "cosine":
            t_span = 1 - mx.cos(t_span * 0.5 * math.pi)

        x = z
        B = mu.shape[0]

        # Squeeze mask for DiT
        mask_squeeze = mask.squeeze(1)  # (B, N)

        # Pre-allocate batched tensors for CFG (batch size 2*B)
        # First B samples: conditional, Last B samples: unconditional
        mu_zeros = mx.zeros_like(mu)
        spks_zeros = mx.zeros_like(spks)
        cond_zeros = mx.zeros_like(cond)

        for step in range(1, n_timesteps + 1):
            t = t_span[step - 1]
            dt = t_span[step] - t_span[step - 1]

            # Batch conditional and unconditional inputs together
            # This matches PyTorch's efficient batched CFG computation
            x_batched = mx.concatenate([x, x], axis=0)  # (2B, mel_dim, N)
            mask_batched = mx.concatenate([mask_squeeze, mask_squeeze], axis=0)
            mu_batched = mx.concatenate([mu, mu_zeros], axis=0)  # cond, then uncond
            spks_batched = mx.concatenate([spks, spks_zeros], axis=0)
            cond_batched = mx.concatenate([cond, cond_zeros], axis=0)
            t_batched = mx.broadcast_to(t, (2 * B,))

            # Single batched forward pass through estimator
            dphi_dt_batched = self.estimator(
                x=x_batched,
                mask=mask_batched,
                mu=mu_batched,
                t=t_batched,
                spks=spks_batched,
                cond=cond_batched,
                streaming=streaming,
            )

            # Split back into conditional and unconditional
            dphi_dt_cond = dphi_dt_batched[:B]
            dphi_dt_uncond = dphi_dt_batched[B:]

            # Classifier-free guidance
            dphi_dt = (
                1.0 + self.inference_cfg_rate
            ) * dphi_dt_cond - self.inference_cfg_rate * dphi_dt_uncond

            # Euler step
            x = x + dt * dphi_dt

            # Force evaluation to prevent computation graph explosion
            mx.eval(x)

        return x.astype(mx.float32)

    def compute_loss(
        self,
        x1: mx.array,
        mask: mx.array,
        mu: mx.array,
        spks: mx.array,
        cond: mx.array,
        streaming: bool = False,
    ) -> Tuple[mx.array, mx.array]:
        """
        Compute training loss.

        Args:
            x1: Target mel-spectrogram (B, mel_dim, N)
            mask: Mask (B, 1, N)
            mu: Condition embedding (B, mel_dim, N)
            spks: Speaker embedding (B, D)
            cond: Conditioning audio (B, mel_dim, N)
            streaming: Whether in streaming mode

        Returns:
            Tuple of (loss, predicted_output)
        """
        B, mel_dim, N = x1.shape

        # Sample timestep
        t = mx.random.uniform(shape=(B,))

        # Sample noise
        z = mx.random.normal(shape=x1.shape)

        # Interpolate between noise and target
        if self.t_scheduler == "cosine":
            t_scheduled = self._cosine_schedule(t)
        else:
            t_scheduled = t

        t_exp = mx.expand_dims(mx.expand_dims(t_scheduled, 1), 2)
        x_t = (1 - (1 - self.sigma_min) * t_exp) * z + t_exp * x1

        # Target velocity
        u_t = x1 - (1 - self.sigma_min) * z

        # Predict velocity
        mask_squeeze = mask.squeeze(1)
        pred = self.estimator(
            x=x_t,
            mask=mask_squeeze,
            mu=mu,
            t=t_scheduled,
            spks=spks,
            cond=cond,
            streaming=streaming,
        )

        # Compute loss (L1)
        loss = mx.mean(mx.abs(pred - u_t) * mask)

        return loss, pred


class CausalMaskedDiffWithDiT(nn.Module):
    """
    CosyVoice3 flow model with DiT-based decoder.

    This replaces the encoder + U-Net decoder of CosyVoice2 with:
    - PreLookaheadLayer for simple token processing
    - DiT-based CFM decoder for mel generation
    """

    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        vocab_size: int = 6561,
        input_frame_rate: int = 25,
        token_mel_ratio: int = 2,
        pre_lookahead_len: int = 3,
        pre_lookahead_layer: Optional[PreLookaheadLayer] = None,
        decoder: Optional[CosyVoice3ConditionalCFM] = None,
        n_timesteps: int = 10,
    ):
        """
        Initialize CausalMaskedDiffWithDiT.

        Args:
            input_size: Token embedding dimension
            output_size: Mel spectrogram dimension
            spk_embed_dim: Speaker embedding dimension
            vocab_size: Speech token vocabulary size
            input_frame_rate: Input frame rate
            token_mel_ratio: Ratio of mel frames per token
            pre_lookahead_len: Lookahead length for streaming
            pre_lookahead_layer: PreLookaheadLayer instance
            decoder: CosyVoice3ConditionalCFM instance
            n_timesteps: Number of ODE solver steps
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.input_frame_rate = input_frame_rate
        self.token_mel_ratio = token_mel_ratio
        self.pre_lookahead_len = pre_lookahead_len
        self.n_timesteps = n_timesteps

        # Token embedding
        self.input_embedding = nn.Embedding(vocab_size, input_size)

        # Speaker embedding projection
        self.spk_embed_affine_layer = nn.Linear(spk_embed_dim, output_size)

        # Pre-lookahead layer (simple conv instead of complex encoder)
        self.pre_lookahead_layer = pre_lookahead_layer or PreLookaheadLayer(
            input_size, input_size, pre_lookahead_len
        )

        # DiT-based decoder
        self.decoder = decoder

    def inference(
        self,
        token: mx.array,
        token_len: mx.array,
        prompt_token: mx.array,
        prompt_token_len: mx.array,
        prompt_feat: mx.array,
        prompt_feat_len: mx.array,
        embedding: mx.array,
        streaming: bool = False,
        finalize: bool = True,
    ) -> Tuple[mx.array, None]:
        """
        Generate mel-spectrogram from speech tokens.

        Args:
            token: Speech token IDs (1, T_token)
            token_len: Token length (1,)
            prompt_token: Prompt speech tokens (1, T_prompt)
            prompt_token_len: Prompt token length (1,)
            prompt_feat: Prompt mel features (1, T_mel, mel_dim)
            prompt_feat_len: Prompt feature length (1,)
            embedding: Speaker embedding (1, spk_dim)
            streaming: Whether in streaming mode
            finalize: Whether this is the final chunk

        Returns:
            Tuple of (generated_mel, None)
        """
        assert token.shape[0] == 1, "Batch size must be 1 for inference"

        # Speaker embedding projection
        embedding = embedding / mx.sqrt(
            mx.sum(embedding * embedding, axis=-1, keepdims=True) + 1e-8
        )
        embedding = self.spk_embed_affine_layer(embedding)

        # Concatenate prompt and input tokens
        token = mx.concatenate([prompt_token, token], axis=1)
        token_len = prompt_token_len + token_len

        # Create mask and embed tokens
        mask = ~make_pad_mask(token_len, token.shape[1])
        mask = mx.expand_dims(mask, -1).astype(mx.float32)
        token_emb = self.input_embedding(mx.clip(token, 0, self.vocab_size - 1)) * mask

        # Apply pre-lookahead layer
        if finalize:
            h = self.pre_lookahead_layer(token_emb)
        else:
            # Split for streaming: main tokens and lookahead context
            h = self.pre_lookahead_layer(
                token_emb[:, : -self.pre_lookahead_len, :],
                context=token_emb[:, -self.pre_lookahead_len :, :],
            )

        # Upsample by token_mel_ratio
        h = mx.repeat(h, self.token_mel_ratio, axis=1)

        mel_len1 = prompt_feat.shape[1]
        mel_len2 = h.shape[1] - prompt_feat.shape[1]

        # Prepare conditioning: concatenate prompt features with zeros
        zeros_part = mx.zeros((1, mel_len2, self.output_size), dtype=h.dtype)
        cond = mx.concatenate([prompt_feat, zeros_part], axis=1)

        # Transpose to channel-first for decoder
        cond = mx.swapaxes(cond, 1, 2)  # (B, mel_dim, N)
        h = mx.swapaxes(h, 1, 2)  # (B, input_size, N)

        # Create mask for decoder
        total_len = mel_len1 + mel_len2
        mask = mx.ones((1, 1, total_len), dtype=mx.float32)

        # Generate mel using flow matching
        feat, _ = self.decoder(
            mu=h,
            mask=mask,
            spks=embedding,
            cond=cond,
            n_timesteps=self.n_timesteps,
            streaming=streaming,
        )

        # Remove prompt portion
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2

        return feat.astype(mx.float32), None


def build_flow_model(
    input_size: int = 512,
    output_size: int = 80,
    spk_embed_dim: int = 192,
    vocab_size: int = 6561,
    input_frame_rate: int = 25,
    token_mel_ratio: int = 2,
    pre_lookahead_len: int = 3,
    dit_dim: int = 1024,
    dit_depth: int = 22,
    dit_heads: int = 16,
    dit_dim_head: int = 64,
    dit_ff_mult: int = 2,
    dit_dropout: float = 0.1,
    cfm_sigma_min: float = 1e-6,
    cfm_t_scheduler: str = "cosine",
    cfm_inference_cfg_rate: float = 0.7,
    n_timesteps: int = 10,
    static_chunk_size: int = 50,
    rand_noise: Optional[mx.array] = None,
) -> CausalMaskedDiffWithDiT:
    """
    Build the complete flow model for CosyVoice3.

    Args:
        input_size: Token embedding dimension
        output_size: Mel dimension
        spk_embed_dim: Speaker embedding dimension
        vocab_size: Speech token vocabulary size
        input_frame_rate: Input frame rate
        token_mel_ratio: Mel frames per token
        pre_lookahead_len: Lookahead length
        dit_dim: DiT model dimension
        dit_depth: Number of DiT blocks
        dit_heads: Number of attention heads
        dit_dim_head: Dimension per head
        dit_ff_mult: Feed-forward multiplier
        dit_dropout: Dropout rate
        cfm_sigma_min: CFM minimum sigma
        cfm_t_scheduler: Time scheduler type
        cfm_inference_cfg_rate: CFG rate
        n_timesteps: ODE solver steps
        static_chunk_size: Chunk size for streaming

    Returns:
        CausalMaskedDiffWithDiT instance
    """
    # Build pre-lookahead layer
    # in_channels: input_size (80 in config), channels: dit_dim (1024)
    # PreLookaheadLayer outputs same dimension as input (in_channels)
    pre_lookahead_layer = PreLookaheadLayer(input_size, dit_dim, pre_lookahead_len)

    # Build DiT estimator
    # mu comes from pre_lookahead_layer which outputs input_size
    # In CosyVoice3 config, input_size=output_size=80
    dit = DiT(
        dim=dit_dim,
        depth=dit_depth,
        heads=dit_heads,
        dim_head=dit_dim_head,
        ff_mult=dit_ff_mult,
        dropout=dit_dropout,
        mel_dim=output_size,
        mu_dim=input_size,  # mu comes from pre_lookahead_layer with input_size output
        spk_dim=output_size,  # speaker embedding after projection
        out_channels=output_size,
        static_chunk_size=static_chunk_size,
    )

    # Build CFM decoder
    decoder = CosyVoice3ConditionalCFM(
        estimator=dit,
        sigma_min=cfm_sigma_min,
        t_scheduler=cfm_t_scheduler,
        inference_cfg_rate=cfm_inference_cfg_rate,
        rand_noise=rand_noise,
    )

    # Build full flow model
    flow_model = CausalMaskedDiffWithDiT(
        input_size=input_size,
        output_size=output_size,
        spk_embed_dim=spk_embed_dim,
        vocab_size=vocab_size,
        input_frame_rate=input_frame_rate,
        token_mel_ratio=token_mel_ratio,
        pre_lookahead_len=pre_lookahead_len,
        pre_lookahead_layer=pre_lookahead_layer,
        decoder=decoder,
        n_timesteps=n_timesteps,
    )

    return flow_model
