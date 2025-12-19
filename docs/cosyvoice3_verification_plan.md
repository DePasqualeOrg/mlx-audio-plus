# CosyVoice 3 MLX Port Verification Plan

## Summary

Verified the CosyVoice 3 MLX port against the original PyTorch implementation at https://github.com/FunAudioLLM/CosyVoice.

## Components Verified

### 1. LLM Module (llm/llm.py)
**Status: CORRECT**
- Qwen2Encoder wrapper correctly implemented
- Speech embedding indices match PyTorch (sos=token_size+0, eos=token_size+1, task_id=token_size+2, fill_token=token_size+3)
- Extended vocabulary (+200 tokens) correctly handled
- Sampling methods (RAS, top-k) implemented correctly

### 2. DiT Module (dit.py)
**Status: CORRECT**
- TimestepEmbedding with sinusoidal encoding matches PyTorch
- CausalConvPositionEmbedding correctly implemented
- RotaryEmbedding matches x_transformers implementation
- GRN (Global Response Normalization) correctly uses L2 norm over time axis
- AdaLayerNormZero and AdaLayerNormZeroFinal match PyTorch
- DiTBlock with attention and feed-forward correctly structured
- Full DiT model (22 layers, 1024 dim, 16 heads) matches config

### 3. Flow Module (flow.py)
**Status: MOSTLY CORRECT - ONE EFFICIENCY ISSUE**
- CosyVoice3ConditionalCFM correctly implements ODE-based flow matching
- Time scheduler (cosine) correctly implemented
- Classifier-free guidance rate (0.7) applied correctly
- Pre-computed random noise support added

**Issue Found**: CFM solve_euler computes conditional and unconditional paths separately instead of batching them together. PyTorch batches into a single forward pass with batch size 2.

### 4. HiFi-GAN Vocoder (hifigan.py)
**Status: CORRECT**
- CausalHiFTGenerator correctly implements causal HiFi-GAN
- Snake activation correctly implemented: x + sin^2(alpha*x)/alpha
- SourceModuleHnNSF2 with SineGen2 matches PyTorch implementation
- CausalConvRNNF0Predictor correctly predicts F0
- STFT/ISTFT implementations correct with proper window handling
- Upsample rates [8, 5, 3] and kernel sizes [16, 11, 7] match config

### 5. Convolution Module (convolution.py)
**Status: CORRECT**
- CausalConv1d correctly handles left (past) and right (future) causal types
- Causal padding formula matches: (kernel_size * dilation - dilation) / 2 * 2 + (kernel_size + 1) % 2
- CausalConv1dDownSample correctly downsamples with causal padding
- CausalConv1dUpsample correctly upsamples with nearest neighbor interpolation
- PreLookaheadLayer correctly applies lookahead with residual connection

### 6. Main CosyVoice3 Class (cosyvoice3.py)
**Status: CORRECT**
- All synthesis modes (zero-shot, cross-lingual, instruct, VC, streaming) match PyTorch API
- Token-to-mel pipeline correctly implemented
- Speaker embedding correctly extracted and processed
- Mel spectrogram generation with proper alignment

## Issues Found

### 1. CFM Batching Efficiency (PERFORMANCE)
**File**: `flow.py`, lines 164-192
**Issue**: The MLX implementation computes conditional and unconditional DiT estimator calls separately in the `solve_euler` method, while PyTorch batches them together with batch size 2 for a single forward pass.

**Current MLX code**:
```python
# Two separate forward passes
dphi_dt_cond = self.estimator(x, mask, mu, t, spks, cond, streaming)
dphi_dt_uncond = self.estimator(x, mask, mu_zeros, t, spks_zeros, cond_zeros, streaming)
```

**PyTorch code**:
```python
# Single batched forward pass
x_in = torch.zeros([2, 80, x.size(2)], ...)
x_in[:] = x
mu_in[0] = mu  # conditional
# mu_in[1] = 0  # unconditional (left as zeros)
dphi_dt = self.forward_estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in, ...)
dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, ...)
```

**Impact**: ~2x slowdown in the flow matching decoder portion of inference.

**Fix**: Batch conditional and unconditional inputs together and run a single forward pass.

**Status**: FIXED - Implemented batched CFG computation in `solve_euler`.

## Efficiency Improvements Implemented

### 1. Batched CFG Computation (IMPLEMENTED)
Changed `solve_euler` to batch conditional and unconditional paths together in a single forward pass through the DiT estimator, matching the PyTorch implementation. This provides approximately 2x speedup in the flow matching decoder portion of inference.

**Implementation**:
```python
# Batch conditional and unconditional inputs together
x_batched = mx.concatenate([x, x], axis=0)  # (2B, mel_dim, N)
mu_batched = mx.concatenate([mu, mu_zeros], axis=0)
spks_batched = mx.concatenate([spks, spks_zeros], axis=0)
cond_batched = mx.concatenate([cond, cond_zeros], axis=0)

# Single batched forward pass through estimator
dphi_dt_batched = self.estimator(x_batched, mask_batched, mu_batched, t_batched, spks_batched, cond_batched, streaming)

# Split back into conditional and unconditional
dphi_dt_cond = dphi_dt_batched[:B]
dphi_dt_uncond = dphi_dt_batched[B:]
```

## Conclusion

The CosyVoice 3 MLX port is accurate and well-implemented. The main efficiency improvement (batched CFG computation) has been applied. The model correctly implements:
- DiT-based flow matching decoder
- Causal HiFi-GAN vocoder with Neural Source Filter
- Qwen2-based speech token LLM
- All synthesis modes (zero-shot, cross-lingual, instruct, VC, streaming)
