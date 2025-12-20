# Copyright © Anthony DePasquale
# Ported to MLX from https://github.com/FunAudioLLM/CosyVoice

"""
CosyVoice3 Language Model implementation for MLX.

This module implements the Qwen2-based speech token language model
used in CosyVoice3 for generating speech tokens from text.

Key differences from CosyVoice2:
- Unified speech_embedding (no separate llm_embedding)
- Different special token indices (sos = speech_token_size + 0, etc.)
- No bias on llm_decoder
- Extended vocabulary (+200)
"""

from typing import Callable, Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache


class Qwen2Encoder(nn.Module):
    """
    Wrapper around mlx-lm's Qwen2 model for CosyVoice3.

    Provides full sequence forward and single-step forward with KV cache.
    """

    def __init__(self, model: nn.Module):
        """
        Initialize Qwen2Encoder.

        Args:
            model: The loaded Qwen2 model from mlx-lm
        """
        super().__init__()
        self.model = model

    @property
    def embed_tokens(self) -> nn.Embedding:
        """Access the token embedding layer."""
        return self.model.model.embed_tokens

    def __call__(
        self,
        xs: mx.array,
        xs_lens: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """
        Full forward pass with attention mask.

        Args:
            xs: Input embeddings (B, T, D)
            xs_lens: Sequence lengths (B,)

        Returns:
            Tuple of (hidden_states, mask)
        """
        B, T, D = xs.shape
        positions = mx.arange(T)
        mask = positions[None, :] < xs_lens[:, None]

        hidden_states = self.model.model(
            inputs=None,
            input_embeddings=xs,
            cache=None,
        )

        return hidden_states, mask[:, None, :]

    def forward_one_step(
        self,
        xs: mx.array,
        cache: Optional[List[KVCache]] = None,
    ) -> Tuple[mx.array, List[KVCache]]:
        """
        Single step forward with KV cache.

        Args:
            xs: Input embeddings (B, T, D)
            cache: List of KVCache objects

        Returns:
            Tuple of (hidden_states, new_cache)
        """
        if cache is None:
            cache = [KVCache() for _ in range(len(self.model.model.layers))]

        hidden_states = self.model.model(
            inputs=None,
            input_embeddings=xs,
            cache=cache,
        )

        return hidden_states, cache


class CosyVoice3LM(nn.Module):
    """
    Qwen2-based Language Model for CosyVoice3 speech token generation.

    Key differences from Qwen2LM (CosyVoice2):
    - Uses unified speech_embedding for all tokens including special tokens
    - Special token indices: sos = speech_token_size + 0, eos = speech_token_size + 1, etc.
    - llm_decoder has no bias and extended vocabulary (+200)
    """

    def __init__(
        self,
        llm_input_size: int = 896,
        llm_output_size: int = 896,
        speech_token_size: int = 6561,
        extended_vocab_size: int = 200,
        llm: Qwen2Encoder = None,
        sampling: Callable = None,
        mix_ratio: List[int] = None,
    ):
        """
        Initialize CosyVoice3LM.

        Args:
            llm_input_size: Input dimension of the LLM
            llm_output_size: Output dimension of the LLM
            speech_token_size: Size of speech token vocabulary (6561)
            extended_vocab_size: Extended vocabulary size (200)
            llm: Qwen2Encoder instance
            sampling: Sampling function for token generation
            mix_ratio: [text_ratio, speech_ratio] for bidirectional streaming
        """
        super().__init__()
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size
        self.extended_vocab_size = extended_vocab_size

        # CosyVoice3 special token indices (unified in speech_embedding)
        self.sos = speech_token_size + 0
        self.eos_token = speech_token_size + 1
        self.task_id = speech_token_size + 2
        self.fill_token = speech_token_size + 3

        # LLM backbone
        self.llm = llm

        # Output projection: no bias, extended vocabulary
        self.llm_decoder = nn.Linear(
            llm_output_size, speech_token_size + extended_vocab_size, bias=False
        )

        # Unified speech token embedding (includes special tokens)
        self.speech_embedding = nn.Embedding(
            speech_token_size + extended_vocab_size, llm_input_size
        )

        # Sampling function
        self.sampling = sampling

        # Mix ratio for bidirectional streaming
        self.mix_ratio = mix_ratio or [5, 15]

        # Stop token IDs for generation (all extended vocab tokens)
        self.stop_token_ids = [
            speech_token_size + i for i in range(extended_vocab_size)
        ]

    def sampling_ids(
        self,
        weighted_scores: mx.array,
        decoded_tokens: List[int],
        sampling: int,
        ignore_eos: bool = True,
    ) -> int:
        """
        Sample token IDs with optional EOS rejection.

        Args:
            weighted_scores: Log probabilities (vocab_size,)
            decoded_tokens: Previously decoded tokens
            sampling: Top-k sampling parameter
            ignore_eos: Whether to reject EOS tokens

        Returns:
            Sampled token ID
        """
        num_trials, max_trials = 0, 100
        while True:
            top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
            # If not ignoring EOS, or sampled token is valid speech token, accept it
            if (not ignore_eos) or (top_ids < self.speech_token_size):
                break
            num_trials += 1
            if num_trials > max_trials:
                raise RuntimeError(
                    f"sampling reaches max_trials {max_trials} and still get special token "
                    "when ignore_eos is True, check your input!"
                )
        return top_ids

    def inference(
        self,
        text: mx.array,
        text_len: mx.array,
        prompt_text: mx.array,
        prompt_text_len: mx.array,
        prompt_speech_token: mx.array,
        prompt_speech_token_len: mx.array,
        embedding: mx.array,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
    ) -> Generator[int, None, None]:
        """
        Generate speech tokens autoregressively.

        Args:
            text: Text token IDs (1, T_text)
            text_len: Text length (1,)
            prompt_text: Prompt text token IDs (1, T_prompt_text)
            prompt_text_len: Prompt text length (1,)
            prompt_speech_token: Prompt speech tokens (1, T_prompt_speech)
            prompt_speech_token_len: Prompt speech token length (1,)
            embedding: Speaker embedding (not used, kept for API compat)
            sampling: Top-k sampling parameter
            max_token_text_ratio: Maximum speech/text token ratio
            min_token_text_ratio: Minimum speech/text token ratio

        Yields:
            Generated speech token IDs one by one
        """
        # Concatenate prompt and input text
        text = mx.concatenate([prompt_text, text], axis=1)
        text_len = text_len + prompt_text_len

        # Embed text tokens using Qwen2's embedding
        text_emb = self.llm.embed_tokens(text)

        # Get special token embeddings from unified speech_embedding
        sos_emb = self.speech_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.speech_embedding.weight[self.task_id].reshape(1, 1, -1)

        # Embed prompt speech tokens if provided
        if prompt_speech_token_len.item() != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = mx.zeros(
                (1, 0, self.llm_input_size), dtype=text_emb.dtype
            )

        # Construct initial LM input: [sos, text, task_id, prompt_speech]
        lm_input = mx.concatenate(
            [sos_emb, text_emb, task_id_emb, prompt_speech_token_emb], axis=1
        )

        # Calculate min/max generation length
        min_len = int((text_len.item() - prompt_text_len.item()) * min_token_text_ratio)
        max_len = int((text_len.item() - prompt_text_len.item()) * max_token_text_ratio)

        # Generate tokens
        yield from self._inference_loop(lm_input, sampling, min_len, max_len)

    def _inference_loop(
        self,
        lm_input: mx.array,
        sampling: int,
        min_len: int,
        max_len: int,
    ) -> Generator[int, None, None]:
        """
        Core inference loop with KV caching.

        Args:
            lm_input: Initial input embeddings (1, T, D)
            sampling: Top-k sampling parameter
            min_len: Minimum generation length
            max_len: Maximum generation length

        Yields:
            Generated speech token IDs
        """
        out_tokens = []
        cache = None

        for i in range(max_len):
            # Forward one step
            y_pred, cache = self.llm.forward_one_step(lm_input, cache=cache)

            # Pipeline: evaluate async while preparing next iteration
            mx.async_eval(y_pred, cache)

            # Get logits for last position
            logits = self.llm_decoder(y_pred[:, -1, :])
            logp = mx.log(mx.softmax(logits, axis=-1))

            # Sample next token
            top_ids = self.sampling_ids(
                logp.reshape(-1),
                out_tokens,
                sampling,
                ignore_eos=(i < min_len),
            )

            # Check for any stop token (EOS or any extended vocab token)
            if top_ids in self.stop_token_ids:
                break

            # Yield the token
            yield top_ids
            out_tokens.append(top_ids)

            # Prepare input for next step
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)

    def inference_bistream(
        self,
        text: Generator[mx.array, None, None],
        prompt_text: mx.array,
        prompt_text_len: mx.array,
        prompt_speech_token: mx.array,
        prompt_speech_token_len: mx.array,
        embedding: mx.array,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
    ) -> Generator[int, None, None]:
        """
        Bidirectional streaming inference.

        Generates speech tokens while receiving text tokens in a streaming fashion.

        Args:
            text: Generator yielding text token chunks
            prompt_text: Prompt text token IDs
            prompt_text_len: Prompt text length
            prompt_speech_token: Prompt speech tokens
            prompt_speech_token_len: Prompt speech token length
            embedding: Speaker embedding (not used)
            sampling: Top-k sampling parameter
            max_token_text_ratio: Maximum speech/text token ratio
            min_token_text_ratio: Minimum speech/text token ratio

        Yields:
            Generated speech token IDs
        """
        # Get special token embeddings
        sos_emb = self.speech_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.speech_embedding.weight[self.task_id].reshape(1, 1, -1)

        # Embed prompt speech tokens if provided
        if prompt_speech_token_len.item() != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = mx.zeros(
                (1, 0, self.llm_input_size), dtype=sos_emb.dtype
            )

        # Initialize LM input with SOS
        lm_input = sos_emb

        # Initialize caches
        out_tokens = []
        cache = None
        text_cache = self.llm.embed_tokens(prompt_text)
        next_fill_index = (
            int(prompt_speech_token.shape[1] / self.mix_ratio[1]) + 1
        ) * self.mix_ratio[1] - prompt_speech_token.shape[1]

        # Process streaming text
        for this_text in text:
            # Add new text to cache
            text_cache = mx.concatenate(
                [text_cache, self.llm.embed_tokens(this_text)], axis=1
            )

            # Process prompt speech tokens first
            while prompt_speech_token_emb.shape[1] != 0:
                if text_cache.shape[1] >= self.mix_ratio[0]:
                    lm_input_text = text_cache[:, : self.mix_ratio[0]]
                    lm_input_speech = prompt_speech_token_emb[:, : self.mix_ratio[1]]
                    lm_input = mx.concatenate(
                        [lm_input, lm_input_text, lm_input_speech], axis=1
                    )
                    text_cache = text_cache[:, self.mix_ratio[0] :]
                    prompt_speech_token_emb = prompt_speech_token_emb[
                        :, self.mix_ratio[1] :
                    ]
                else:
                    break

            # Generate speech when no prompt speech remains
            if prompt_speech_token_emb.shape[1] == 0:
                if (len(out_tokens) != 0 and out_tokens[-1] == self.fill_token) or (
                    len(out_tokens) == 0 and lm_input.shape[1] == 1
                ):
                    if text_cache.shape[1] >= self.mix_ratio[0]:
                        lm_input_text = text_cache[:, : self.mix_ratio[0]]
                        if len(out_tokens) != 0 and out_tokens[-1] == self.fill_token:
                            lm_input = lm_input_text
                        else:
                            lm_input = mx.concatenate([lm_input, lm_input_text], axis=1)
                        text_cache = text_cache[:, self.mix_ratio[0] :]
                    else:
                        continue

                # Generate speech tokens
                while True:
                    y_pred, cache = self.llm.forward_one_step(lm_input, cache=cache)
                    logits = self.llm_decoder(y_pred[:, -1, :])
                    logp = mx.log(mx.softmax(logits, axis=-1))

                    # Handle fill token scheduling
                    if next_fill_index != -1 and len(out_tokens) == next_fill_index:
                        top_ids = self.fill_token
                        next_fill_index += self.mix_ratio[1] + 1
                    else:
                        top_ids = self.sampling_ids(
                            logp.reshape(-1), out_tokens, sampling, ignore_eos=True
                        )

                    if top_ids == self.fill_token:
                        next_fill_index = len(out_tokens) + self.mix_ratio[1] + 1

                    out_tokens.append(top_ids)

                    if top_ids >= self.speech_token_size:
                        if top_ids == self.fill_token:
                            break
                        else:
                            raise ValueError(f"should not get token {top_ids}")

                    yield top_ids
                    lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)

        # Final decode with remaining text
        lm_input = mx.concatenate([lm_input, text_cache, task_id_emb], axis=1)

        while True:
            y_pred, cache = self.llm.forward_one_step(lm_input, cache=cache)
            logits = self.llm_decoder(y_pred[:, -1, :])
            logp = mx.log(mx.softmax(logits, axis=-1))

            top_ids = self.sampling_ids(
                logp.reshape(-1), out_tokens, sampling, ignore_eos=False
            )
            out_tokens.append(top_ids)

            if top_ids >= self.speech_token_size:
                if top_ids == self.eos_token:
                    break
                else:
                    raise ValueError(f"should not get token {top_ids}")

            yield top_ids
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)


def nucleus_sampling(
    logits: mx.array,
    top_p: float = 0.8,
    top_k: int = 25,
) -> int:
    """
    Nucleus (top-p) sampling with top-k cutoff.

    Args:
        logits: Log probabilities (vocab_size,)
        top_p: Cumulative probability threshold
        top_k: Maximum number of tokens to consider

    Returns:
        Sampled token ID
    """
    probs = mx.softmax(logits)
    sorted_indices = mx.argsort(-probs)
    sorted_probs = probs[sorted_indices]
    cumsum_probs = mx.cumsum(sorted_probs)

    below_threshold = cumsum_probs < top_p
    n_tokens = min(int(mx.sum(below_threshold).item()) + 1, top_k)

    top_indices = sorted_indices[:n_tokens]
    top_probs = sorted_probs[:n_tokens]
    top_probs = top_probs / mx.sum(top_probs)

    idx = mx.random.categorical(mx.log(top_probs))
    return int(top_indices[idx].item())


def ras_sampling(
    logits: mx.array,
    decoded_tokens: List[int],
    sampling: int,
    top_p: float = 0.8,
    top_k: int = 25,
    win_size: int = 10,
    tau_r: float = 0.1,
) -> int:
    """
    Repetition-Aware Sampling (RAS).

    Uses nucleus sampling but falls back to random sampling if
    repetition is detected in recent tokens.

    Args:
        logits: Log probabilities (vocab_size,)
        decoded_tokens: Previously decoded tokens
        sampling: Not used, kept for API compatibility
        top_p: Cumulative probability threshold
        top_k: Maximum number of tokens to consider
        win_size: Window size for repetition detection
        tau_r: Repetition threshold ratio

    Returns:
        Sampled token ID
    """
    top_ids = nucleus_sampling(logits, top_p=top_p, top_k=top_k)

    if len(decoded_tokens) > 0:
        recent_tokens = decoded_tokens[-win_size:]
        rep_num = sum(1 for t in recent_tokens if t == top_ids)

        if rep_num >= win_size * tau_r:
            probs = mx.softmax(logits)
            top_ids = int(mx.random.categorical(mx.log(probs)).item())

    return top_ids


def top_k_sampling(
    logits: mx.array,
    decoded_tokens: List[int],
    top_k: int = 25,
) -> int:
    """
    Simple top-k sampling from logits.

    Args:
        logits: Log probabilities (vocab_size,)
        decoded_tokens: Previously decoded tokens (unused, for API compat)
        top_k: Number of top tokens to sample from

    Returns:
        Sampled token ID
    """
    top_k_indices = mx.argpartition(-logits, kth=top_k - 1)[:top_k]
    top_k_values = logits[top_k_indices]
    probs = mx.softmax(top_k_values)
    idx = mx.random.categorical(mx.log(probs))
    return int(top_k_indices[idx].item())
