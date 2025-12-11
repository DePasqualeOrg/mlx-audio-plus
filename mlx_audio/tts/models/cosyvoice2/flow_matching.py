# CosyVoice2-specific flow matching module

import math

import mlx.core as mx
import mlx.nn as nn

from ..chatterbox.s3gen.matcha.flow_matching import BASECFM, CFMParams

# Default CFM parameters
CFM_PARAMS = CFMParams()


class CosyVoice2ConditionalCFM(BASECFM):
    """Conditional Flow Matching with Classifier-Free Guidance for CosyVoice2."""

    # Mel output channels
    MEL_CHANNELS = 80

    def __init__(
        self,
        in_channels: int = 240,
        cfm_params: CFMParams = CFM_PARAMS,
        n_spks: int = 1,
        spk_emb_dim: int = 80,
        estimator: nn.Module = None,
    ):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        self.estimator = estimator

    def __call__(
        self,
        mu: mx.array,
        mask: mx.array,
        n_timesteps: int,
        temperature: float = 1.0,
        spks: mx.array = None,
        cond: mx.array = None,
        streaming: bool = False,
    ) -> tuple:
        """
        Forward diffusion with dynamically generated noise.

        Args:
            mu: Encoder output (B, C, T)
            mask: Mask (B, 1, T)
            n_timesteps: Number of diffusion steps
            temperature: Noise temperature
            spks: Speaker embeddings (B, spk_dim)
            cond: Conditioning (B, C, T)
            streaming: Whether to use streaming (chunk-based) attention

        Returns:
            Tuple of (generated_mel, None)
        """
        B = mu.shape[0]
        T = mu.shape[2]
        # Generate noise dynamically with the correct shape
        z = mx.random.normal((B, self.MEL_CHANNELS, T)) * temperature

        # Time span with cosine schedule
        t_span = mx.linspace(0, 1, n_timesteps + 1)
        if self.t_scheduler == "cosine":
            t_span = 1 - mx.cos(t_span * 0.5 * math.pi)

        result = self.solve_euler(
            z,
            t_span=t_span,
            mu=mu,
            mask=mask,
            spks=spks,
            cond=cond,
            streaming=streaming,
        )
        return result, None

    def solve_euler(
        self,
        x: mx.array,
        t_span: mx.array,
        mu: mx.array,
        mask: mx.array,
        spks: mx.array,
        cond: mx.array,
        streaming: bool = False,
    ) -> mx.array:
        """
        Euler solver with Classifier-Free Guidance.

        Args:
            x: Starting noise (B, C, T)
            t_span: Time steps
            mu: Encoder output (B, C, T)
            mask: Mask (B, 1, T)
            spks: Speaker embeddings (B, spk_dim)
            cond: Conditioning (B, C, T)
            streaming: Whether to use streaming attention

        Returns:
            Generated mel spectrogram (B, C, T)
        """
        t = mx.expand_dims(t_span[0], 0)
        dt = t_span[1] - t_span[0]

        # Pre-allocate CFG inputs outside the loop to avoid redundant allocations
        T_len = x.shape[2]
        B = x.shape[0]

        # Pre-compute static inputs for CFG (these don't change during iteration)
        mask_in = mx.concatenate([mask, mask], axis=0)
        zeros_mu = mx.zeros_like(mu)

        # Pre-compute spks_in (doesn't change during iteration)
        if spks is not None:
            spks_in = mx.concatenate([spks, mx.zeros_like(spks)], axis=0)
        else:
            spks_in = mx.zeros((2, self.spk_emb_dim))

        # Pre-compute cond_in (doesn't change during iteration)
        if cond is not None:
            cond_in = mx.concatenate([cond, mx.zeros_like(cond)], axis=0)
        else:
            cond_in = mx.zeros((2, self.n_feats, T_len))

        for step in range(1, len(t_span)):
            # Prepare inputs for CFG - only x and t change each iteration
            x_in = mx.concatenate([x, x], axis=0)
            mu_in = mx.concatenate([mu, zeros_mu], axis=0)
            t_in = mx.concatenate([t, t], axis=0)

            # Forward estimator with streaming parameter
            dphi_dt = self.estimator(
                x_in, mask_in, mu_in, t_in, spks_in, cond_in, streaming=streaming
            )

            # Split and apply CFG
            dphi_dt_cond = dphi_dt[:B]
            dphi_dt_uncond = dphi_dt[B:]
            dphi_dt = (
                1.0 + self.inference_cfg_rate
            ) * dphi_dt_cond - self.inference_cfg_rate * dphi_dt_uncond

            x = x + dt * dphi_dt
            t = t + dt

            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return x
