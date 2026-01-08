# Copyright © Anthony DePasquale
# Ported to MLX from https://github.com/FunAudioLLM/CosyVoice

"""
CosyVoice3 Text-to-Speech model for MLX.

This module provides the main CosyVoice3 model class that integrates:
- Qwen2-based language model for speech token generation
- DiT-based flow matching for mel-spectrogram generation
- Causal HiFi-GAN vocoder for waveform synthesis
"""

import logging
import math
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import CosyVoice3Config, ModelConfig
from .flow import CausalMaskedDiffWithDiT, build_flow_model
from .hifigan import CausalHiFTGenerator
from .llm import CosyVoice3LM, Qwen2Encoder, ras_sampling, top_k_sampling

logger = logging.getLogger(__name__)

# FSQ silent and breath tokens (from PyTorch CosyVoice3Model)
# These are filtered during streaming to avoid excessive pauses
SILENT_TOKENS = {1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323}
MAX_SILENT_TOKEN_NUM = 5


class CosyVoice3(nn.Module):
    """
    CosyVoice3 Text-to-Speech model.

    This model generates high-quality speech from text using:
    1. Language model (LLM) to generate speech tokens from text
    2. Flow matching with DiT to generate mel-spectrograms from tokens
    3. Causal HiFi-GAN vocoder to synthesize waveforms

    Key features:
    - 9 languages supported
    - 18+ Chinese dialects/accents
    - Bi-streaming support (150ms latency)
    - Pronunciation inpainting
    - Instruction-following for style control
    """

    def __init__(
        self,
        config: CosyVoice3Config,
        llm: Optional[CosyVoice3LM] = None,
        flow: Optional[CausalMaskedDiffWithDiT] = None,
        hifigan: Optional[CausalHiFTGenerator] = None,
    ):
        """
        Initialize CosyVoice3.

        Args:
            config: Model configuration
            llm: Language model instance
            flow: Flow matching model instance
            hifigan: HiFi-GAN vocoder instance
        """
        super().__init__()
        self.config = config
        self.llm = llm
        self.flow = flow
        self.hifigan = hifigan

        # Lazy-loaded components
        self._tokenizer = None
        self._speech_tokenizer = None
        self._speaker_encoder = None

    @property
    def sample_rate(self) -> int:
        """Audio sample rate (24kHz)."""
        return self.config.hifigan.sampling_rate

    def generate_tokens(
        self,
        text: mx.array,
        text_len: mx.array,
        prompt_text: mx.array,
        prompt_text_len: mx.array,
        prompt_speech_token: mx.array,
        prompt_speech_token_len: mx.array,
        embedding: mx.array,
        sampling: int = 25,
        max_token_text_ratio: float = 20.0,
        min_token_text_ratio: float = 2.0,
    ) -> Generator[int, None, None]:
        """
        Generate speech tokens from text using the LLM.

        Args:
            text: Input text token IDs (1, T)
            text_len: Text length (1,)
            prompt_text: Prompt text token IDs (1, T_p)
            prompt_text_len: Prompt text length (1,)
            prompt_speech_token: Prompt speech tokens (1, T_s)
            prompt_speech_token_len: Speech token length (1,)
            embedding: Speaker embedding (unused, for API compat)
            sampling: Top-k sampling parameter
            max_token_text_ratio: Maximum speech/text ratio
            min_token_text_ratio: Minimum speech/text ratio

        Yields:
            Generated speech token IDs
        """
        if self.llm is None:
            raise RuntimeError("LLM not initialized")

        yield from self.llm.inference(
            text=text,
            text_len=text_len,
            prompt_text=prompt_text,
            prompt_text_len=prompt_text_len,
            prompt_speech_token=prompt_speech_token,
            prompt_speech_token_len=prompt_speech_token_len,
            embedding=embedding,
            sampling=sampling,
            max_token_text_ratio=max_token_text_ratio,
            min_token_text_ratio=min_token_text_ratio,
        )

    def tokens_to_mel(
        self,
        tokens: mx.array,
        token_len: mx.array,
        prompt_token: mx.array,
        prompt_token_len: mx.array,
        prompt_feat: mx.array,
        prompt_feat_len: mx.array,
        embedding: mx.array,
        finalize: bool = True,
        n_timesteps: Optional[int] = None,
        streaming: bool = False,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """
        Convert speech tokens to mel spectrogram using flow matching.

        Args:
            tokens: Speech tokens (1, T)
            token_len: Token length (1,)
            prompt_token: Prompt speech tokens (1, T_p)
            prompt_token_len: Prompt token length (1,)
            prompt_feat: Prompt mel features (1, T_mel, D)
            prompt_feat_len: Prompt feature length (1,)
            embedding: Speaker embedding (1, D_spk)
            finalize: Whether this is the final chunk
            n_timesteps: Number of diffusion steps
            streaming: Whether in streaming mode

        Returns:
            Tuple of (mel_spectrogram, flow_cache)
        """
        if self.flow is None:
            raise RuntimeError("Flow module not initialized")

        return self.flow.inference(
            token=tokens,
            token_len=token_len,
            prompt_token=prompt_token,
            prompt_token_len=prompt_token_len,
            prompt_feat=prompt_feat,
            prompt_feat_len=prompt_feat_len,
            embedding=embedding,
            streaming=streaming,
            finalize=finalize,
        )

    def mel_to_audio(
        self,
        mel: mx.array,
        finalize: bool = True,
    ) -> mx.array:
        """
        Convert mel spectrogram to audio waveform.

        Args:
            mel: Mel spectrogram (1, D, T)
            finalize: Whether this is the final chunk

        Returns:
            Audio waveform (1, T_audio)
        """
        if self.hifigan is None:
            raise RuntimeError("HiFi-GAN not initialized")

        audio, _ = self.hifigan(mel, finalize=finalize)
        return audio

    def synthesize(
        self,
        text: mx.array,
        text_len: mx.array,
        prompt_text: mx.array,
        prompt_text_len: mx.array,
        prompt_speech_token: mx.array,
        prompt_speech_token_len: mx.array,
        prompt_mel: mx.array,
        prompt_mel_len: mx.array,
        speaker_embedding: mx.array,
        sampling: int = 25,
        n_timesteps: int = 10,
        max_token_text_ratio: float = 20.0,
        min_token_text_ratio: float = 2.0,
    ) -> mx.array:
        """
        Full TTS pipeline: text -> audio.

        Args:
            text: Input text token IDs (1, T)
            text_len: Text length (1,)
            prompt_text: Prompt text token IDs (1, T_p)
            prompt_text_len: Prompt text length (1,)
            prompt_speech_token: Prompt speech tokens (1, T_s)
            prompt_speech_token_len: Prompt speech token length (1,)
            prompt_mel: Prompt mel spectrogram (1, T_mel, D)
            prompt_mel_len: Prompt mel length (1,)
            speaker_embedding: Speaker embedding (1, D_spk)
            sampling: Top-k sampling parameter
            n_timesteps: Number of flow matching steps
            max_token_text_ratio: Maximum speech/text ratio
            min_token_text_ratio: Minimum speech/text ratio

        Returns:
            Audio waveform (1, T_audio)
        """
        # Step 1: Generate speech tokens
        tokens = []
        for token in self.generate_tokens(
            text=text,
            text_len=text_len,
            prompt_text=prompt_text,
            prompt_text_len=prompt_text_len,
            prompt_speech_token=prompt_speech_token,
            prompt_speech_token_len=prompt_speech_token_len,
            embedding=speaker_embedding,
            sampling=sampling,
            max_token_text_ratio=max_token_text_ratio,
            min_token_text_ratio=min_token_text_ratio,
        ):
            tokens.append(token)

        if len(tokens) == 0:
            raise RuntimeError("No tokens generated")

        # Convert to array
        token_array = mx.array([tokens], dtype=mx.int32)
        token_len = mx.array([len(tokens)], dtype=mx.int32)

        # Step 2: Convert tokens to mel spectrogram
        mel, _ = self.tokens_to_mel(
            tokens=token_array,
            token_len=token_len,
            prompt_token=prompt_speech_token,
            prompt_token_len=prompt_speech_token_len,
            prompt_feat=prompt_mel,
            prompt_feat_len=prompt_mel_len,
            embedding=speaker_embedding,
            finalize=True,
            n_timesteps=n_timesteps,
        )

        # Step 3: Convert mel to audio
        audio = self.mel_to_audio(mel)

        return audio

    def _tokens_to_audio(
        self,
        tokens: List[int],
        prompt_speech_token: mx.array,
        prompt_speech_token_len: mx.array,
        prompt_mel: mx.array,
        prompt_mel_len: mx.array,
        speaker_embedding: mx.array,
        n_timesteps: int = 10,
    ) -> mx.array:
        """Convert generated tokens to audio via flow + vocoder."""
        if len(tokens) == 0:
            raise RuntimeError("No tokens generated")

        token_array = mx.array([tokens], dtype=mx.int32)
        token_len = mx.array([len(tokens)], dtype=mx.int32)

        mel, _ = self.tokens_to_mel(
            tokens=token_array,
            token_len=token_len,
            prompt_token=prompt_speech_token,
            prompt_token_len=prompt_speech_token_len,
            prompt_feat=prompt_mel,
            prompt_feat_len=prompt_mel_len,
            embedding=speaker_embedding,
            finalize=True,
            n_timesteps=n_timesteps,
        )

        return self.mel_to_audio(mel)

    def synthesize_cross_lingual(
        self,
        text: mx.array,
        text_len: mx.array,
        prompt_speech_token: mx.array,
        prompt_speech_token_len: mx.array,
        prompt_mel: mx.array,
        prompt_mel_len: mx.array,
        speaker_embedding: mx.array,
        sampling: int = 25,
        n_timesteps: int = 10,
        max_token_text_ratio: float = 20.0,
        min_token_text_ratio: float = 2.0,
    ) -> mx.array:
        """
        Cross-lingual TTS pipeline (no reference transcription needed).

        In this mode:
        - LLM receives NO prompt_text and NO prompt_speech_token
        - LLM generates speech tokens based purely on the input text
        - Flow model still uses prompt_speech_token and prompt_mel for speaker identity

        This is useful when:
        - You don't have a transcription of the reference audio
        - The reference audio is in a different language than the target text

        Args:
            text: Input text token IDs (1, T)
            text_len: Text length (1,)
            prompt_speech_token: Speech tokens from reference (used by flow only)
            prompt_speech_token_len: Speech token length (1,)
            prompt_mel: Mel spectrogram from reference (1, T_mel, D)
            prompt_mel_len: Prompt mel length (1,)
            speaker_embedding: Speaker embedding from reference (1, D_spk)
            sampling: Top-k sampling parameter
            n_timesteps: Number of flow matching steps
            max_token_text_ratio: Maximum speech/text ratio
            min_token_text_ratio: Minimum speech/text ratio

        Returns:
            Audio waveform (1, T_audio)
        """
        # Generate tokens WITHOUT prompt context (LLM gets no reference)
        empty = mx.zeros((1, 0), dtype=mx.int32)
        empty_len = mx.array([0], dtype=mx.int32)

        tokens = list(
            self.generate_tokens(
                text=text,
                text_len=text_len,
                prompt_text=empty,
                prompt_text_len=empty_len,
                prompt_speech_token=empty,
                prompt_speech_token_len=empty_len,
                embedding=speaker_embedding,
                sampling=sampling,
                max_token_text_ratio=max_token_text_ratio,
                min_token_text_ratio=min_token_text_ratio,
            )
        )

        # Flow model uses prompt for speaker identity
        return self._tokens_to_audio(
            tokens,
            prompt_speech_token,
            prompt_speech_token_len,
            prompt_mel,
            prompt_mel_len,
            speaker_embedding,
            n_timesteps,
        )

    def synthesize_instruct(
        self,
        text: mx.array,
        text_len: mx.array,
        instruct_text: mx.array,
        instruct_text_len: mx.array,
        prompt_speech_token: mx.array,
        prompt_speech_token_len: mx.array,
        prompt_mel: mx.array,
        prompt_mel_len: mx.array,
        speaker_embedding: mx.array,
        sampling: int = 25,
        n_timesteps: int = 10,
        max_token_text_ratio: float = 20.0,
        min_token_text_ratio: float = 2.0,
    ) -> mx.array:
        """
        Instruct-mode TTS pipeline with style control.

        In this mode:
        - LLM receives instruct_text (style instruction) but NO speech tokens
        - Flow model uses prompt_speech_token and prompt_mel for speaker identity

        This allows controlling the style of speech generation with instructions
        like "Speak slowly and calmly" or "Read with excitement".

        Args:
            text: Input text token IDs (1, T)
            text_len: Text length (1,)
            instruct_text: Style instruction as token IDs, should end with <|endofprompt|> (1, T_i)
            instruct_text_len: Instruct text length (1,)
            prompt_speech_token: Speech tokens from reference (used by flow only)
            prompt_speech_token_len: Speech token length (1,)
            prompt_mel: Mel spectrogram from reference (1, T_mel, D)
            prompt_mel_len: Prompt mel length (1,)
            speaker_embedding: Speaker embedding from reference (1, D_spk)
            sampling: Top-k sampling parameter
            n_timesteps: Number of flow matching steps
            max_token_text_ratio: Maximum speech/text ratio
            min_token_text_ratio: Minimum speech/text ratio

        Returns:
            Audio waveform (1, T_audio)
        """
        # Generate tokens with instruct text but no speech reference
        empty = mx.zeros((1, 0), dtype=mx.int32)
        empty_len = mx.array([0], dtype=mx.int32)

        tokens = list(
            self.generate_tokens(
                text=text,
                text_len=text_len,
                prompt_text=instruct_text,
                prompt_text_len=instruct_text_len,
                prompt_speech_token=empty,
                prompt_speech_token_len=empty_len,
                embedding=speaker_embedding,
                sampling=sampling,
                max_token_text_ratio=max_token_text_ratio,
                min_token_text_ratio=min_token_text_ratio,
            )
        )

        return self._tokens_to_audio(
            tokens,
            prompt_speech_token,
            prompt_speech_token_len,
            prompt_mel,
            prompt_mel_len,
            speaker_embedding,
            n_timesteps,
        )

    def synthesize_vc(
        self,
        source_speech_token: mx.array,
        source_speech_token_len: mx.array,
        prompt_speech_token: mx.array,
        prompt_speech_token_len: mx.array,
        prompt_mel: mx.array,
        prompt_mel_len: mx.array,
        speaker_embedding: mx.array,
        n_timesteps: int = 10,
    ) -> mx.array:
        """
        Voice conversion pipeline.

        In this mode:
        - No LLM is used - source speech tokens are used directly
        - Flow model converts source tokens to target speaker's voice

        Args:
            source_speech_token: Speech tokens from source audio (1, T_s)
            source_speech_token_len: Source token length (1,)
            prompt_speech_token: Speech tokens from reference (1, T_p)
            prompt_speech_token_len: Prompt token length (1,)
            prompt_mel: Mel spectrogram from reference (1, T_mel, D)
            prompt_mel_len: Prompt mel length (1,)
            speaker_embedding: Speaker embedding from reference (1, D_spk)
            n_timesteps: Number of flow matching steps

        Returns:
            Audio waveform (1, T_audio)
        """
        # Voice conversion uses source tokens directly (no LLM generation)
        mel, _ = self.tokens_to_mel(
            tokens=source_speech_token,
            token_len=source_speech_token_len,
            prompt_token=prompt_speech_token,
            prompt_token_len=prompt_speech_token_len,
            prompt_feat=prompt_mel,
            prompt_feat_len=prompt_mel_len,
            embedding=speaker_embedding,
            finalize=True,
            n_timesteps=n_timesteps,
        )

        audio = self.mel_to_audio(mel)
        return audio

    def synthesize_streaming(
        self,
        text: mx.array,
        text_len: mx.array,
        prompt_text: mx.array,
        prompt_text_len: mx.array,
        prompt_speech_token: mx.array,
        prompt_speech_token_len: mx.array,
        prompt_mel: mx.array,
        prompt_mel_len: mx.array,
        speaker_embedding: mx.array,
        sampling: int = 25,
        n_timesteps: int = 10,
        chunk_size: int = 25,
        max_token_text_ratio: float = 20.0,
        min_token_text_ratio: float = 2.0,
        filter_silent_tokens: bool = True,
    ) -> Generator[mx.array, None, None]:
        """
        Streaming speech synthesis with chunked output.

        Generates audio in chunks as tokens are produced, reducing latency.
        This implementation closely follows PyTorch CosyVoice3Model.tts().

        Args:
            text: Text token IDs (1, T)
            text_len: Text length (1,)
            prompt_text: Prompt text tokens (1, T_p)
            prompt_text_len: Prompt text length (1,)
            prompt_speech_token: Prompt speech tokens (1, T_s)
            prompt_speech_token_len: Prompt speech token length (1,)
            prompt_mel: Prompt mel spectrogram (1, T_mel, D)
            prompt_mel_len: Prompt mel length (1,)
            speaker_embedding: Speaker embedding (1, D_spk)
            sampling: Top-k sampling parameter
            n_timesteps: Number of flow matching steps
            chunk_size: Number of tokens per chunk (default 25, must match training)
            max_token_text_ratio: Maximum speech/text ratio
            min_token_text_ratio: Minimum speech/text ratio
            filter_silent_tokens: Whether to filter consecutive silent tokens

        Yields:
            Audio waveform chunks
        """
        pre_lookahead_len = self.flow.pre_lookahead_len
        token_mel_ratio = self.flow.token_mel_ratio

        # Calculate prompt token padding to align first chunk to chunk_size boundary
        # This matches PyTorch: prompt_token_pad = ceil(prompt_len / chunk_size) * chunk_size - prompt_len
        prompt_len = int(prompt_speech_token_len[0].item())
        prompt_token_pad = (
            int(math.ceil(prompt_len / chunk_size) * chunk_size) - prompt_len
        )

        # State for streaming
        speech_tokens: List[int] = []
        token_offset = 0
        mel_cache: Optional[mx.array] = None
        speech_offset = 0

        # Silent token filtering state
        cur_silent_token_num = 0

        for token in self.generate_tokens(
            text=text,
            text_len=text_len,
            prompt_text=prompt_text,
            prompt_text_len=prompt_text_len,
            prompt_speech_token=prompt_speech_token,
            prompt_speech_token_len=prompt_speech_token_len,
            embedding=speaker_embedding,
            sampling=sampling,
            max_token_text_ratio=max_token_text_ratio,
            min_token_text_ratio=min_token_text_ratio,
        ):
            # Filter consecutive silent tokens (matches PyTorch llm_job)
            if filter_silent_tokens and token in SILENT_TOKENS:
                cur_silent_token_num += 1
                if cur_silent_token_num > MAX_SILENT_TOKEN_NUM:
                    continue
            else:
                cur_silent_token_num = 0

            speech_tokens.append(token)

            # Calculate current chunk size (first chunk includes padding)
            this_chunk_size = (
                chunk_size + prompt_token_pad if token_offset == 0 else chunk_size
            )

            # Process chunk when enough tokens accumulated
            # Matches PyTorch: len(tokens) - token_offset >= this_chunk_size + pre_lookahead_len
            if len(speech_tokens) - token_offset >= this_chunk_size + pre_lookahead_len:
                # Take all tokens up to current position + lookahead
                end_idx = token_offset + this_chunk_size + pre_lookahead_len
                chunk_tokens = mx.array(
                    speech_tokens[:end_idx], dtype=mx.int32
                ).reshape(1, -1)
                chunk_len = mx.array([end_idx], dtype=mx.int32)

                # Generate mel via flow
                mel, _ = self.tokens_to_mel(
                    tokens=chunk_tokens,
                    token_len=chunk_len,
                    prompt_token=prompt_speech_token,
                    prompt_token_len=prompt_speech_token_len,
                    prompt_feat=prompt_mel,
                    prompt_feat_len=prompt_mel_len,
                    embedding=speaker_embedding,
                    finalize=False,
                    n_timesteps=n_timesteps,
                    streaming=True,
                )

                # Slice mel to get only new portion (from token_offset)
                mel_new = mel[:, :, token_offset * token_mel_ratio :]

                # Accumulate mel (matches PyTorch caching strategy)
                if mel_cache is not None:
                    mel_cache = mx.concatenate([mel_cache, mel_new], axis=2)
                else:
                    mel_cache = mel_new

                # Generate audio from accumulated mel
                audio = self.mel_to_audio(mel_cache, finalize=False)

                # Extract only new audio (from speech_offset)
                if audio.shape[-1] > speech_offset:
                    chunk_audio = audio[:, speech_offset:]
                    speech_offset += chunk_audio.shape[-1]
                    yield chunk_audio.squeeze(0)

                # Advance token offset
                token_offset += this_chunk_size

        # Final chunk - process remaining tokens
        # Note: PyTorch uses streaming=False for final chunk (full attention)
        if len(speech_tokens) > token_offset:
            chunk_tokens = mx.array(speech_tokens, dtype=mx.int32).reshape(1, -1)
            chunk_len = mx.array([len(speech_tokens)], dtype=mx.int32)

            mel, _ = self.tokens_to_mel(
                tokens=chunk_tokens,
                token_len=chunk_len,
                prompt_token=prompt_speech_token,
                prompt_token_len=prompt_speech_token_len,
                prompt_feat=prompt_mel,
                prompt_feat_len=prompt_mel_len,
                embedding=speaker_embedding,
                finalize=True,
                n_timesteps=n_timesteps,
                streaming=False,
            )

            # Slice mel to get only new portion
            mel_new = mel[:, :, token_offset * token_mel_ratio :]

            # Accumulate mel
            if mel_cache is not None:
                mel_cache = mx.concatenate([mel_cache, mel_new], axis=2)
            else:
                mel_cache = mel_new

            # Generate final audio
            audio = self.mel_to_audio(mel_cache, finalize=True)

            # Extract only new audio
            if audio.shape[-1] > speech_offset:
                yield audio.squeeze(0)[speech_offset:]


def load_cosyvoice3(
    model_path: str,
    dtype: mx.Dtype = mx.float16,
) -> CosyVoice3:
    """
    Load a CosyVoice3 model from a directory.

    Expects a single model.safetensors with prefixes:
    - qwen2.* : Qwen2 LLM weights
    - llm.* : Speech embedding and decoder weights
    - flow.* : Flow/DiT weights
    - hifigan.* : HiFi-GAN weights

    Args:
        model_path: Path to model directory
        dtype: Data type for model weights

    Returns:
        Loaded CosyVoice3 model
    """
    import json

    from mlx.utils import tree_unflatten

    model_path = Path(model_path)

    # Load config
    config_path = model_path / "config.json"
    if config_path.exists():
        config = CosyVoice3Config.from_pretrained(model_path)
        with open(config_path) as f:
            raw_config = json.load(f)
            quantization_config = raw_config.get("quantization")
    else:
        config = CosyVoice3Config()
        quantization_config = None

    # Load all weights from consolidated model.safetensors
    consolidated_path = model_path / "model.safetensors"
    if not consolidated_path.exists():
        raise FileNotFoundError(
            f"model.safetensors not found in {model_path}. "
            "Please run the conversion script."
        )

    logger.info(f"Loading weights from {consolidated_path}")
    all_weights = mx.load(str(consolidated_path))

    # Load Qwen2 model
    from mlx_lm.models.qwen2 import Model as Qwen2Model
    from mlx_lm.models.qwen2 import ModelArgs

    qwen2_args = ModelArgs(
        model_type="qwen2",
        hidden_size=896,
        intermediate_size=4864,
        num_attention_heads=14,
        num_hidden_layers=24,
        num_key_value_heads=2,
        vocab_size=151936,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        tie_word_embeddings=True,
    )
    qwen2_model = Qwen2Model(qwen2_args)

    # Apply quantization if needed
    if quantization_config:
        q_bits = quantization_config.get("bits", 4)
        q_group_size = quantization_config.get("group_size", 64)

        def should_quantize(path, module):
            if isinstance(module, nn.Linear) and "model.layers" in path:
                return True
            return False

        nn.quantize(
            qwen2_model,
            bits=q_bits,
            group_size=q_group_size,
            class_predicate=should_quantize,
        )

    # Load Qwen2 weights (qwen2.* prefix)
    qwen2_weights = {
        k[6:]: v
        for k, v in all_weights.items()
        if k.startswith("qwen2.") and k != "qwen2.lm_head.weight"
    }
    if qwen2_weights:
        qwen2_model.load_weights(list(qwen2_weights.items()))

    qwen2_encoder = Qwen2Encoder(qwen2_model)

    # Build LLM
    llm = CosyVoice3LM(
        llm_input_size=config.llm.llm_input_size,
        llm_output_size=config.llm.llm_output_size,
        speech_token_size=config.llm.speech_token_size,
        extended_vocab_size=config.llm.extended_vocab_size,
        llm=qwen2_encoder,
        sampling=ras_sampling,
        mix_ratio=config.llm.mix_ratio,
    )

    # Load speech weights (llm.* prefix)
    llm_weights = {k[4:]: v for k, v in all_weights.items() if k.startswith("llm.")}
    if llm_weights:
        llm.update(tree_unflatten(list(llm_weights.items())))

    # Load pre-computed random noise for deterministic generation
    # This matches PyTorch CausalConditionalCFM which uses fixed noise
    rand_noise = None
    rand_noise_path = model_path / "rand_noise.npy"
    if rand_noise_path.exists():
        import numpy as np

        rand_noise_np = np.load(str(rand_noise_path))
        rand_noise = mx.array(rand_noise_np)
        logger.info(f"Loaded pre-computed noise from {rand_noise_path}")

    # Build flow model
    flow = build_flow_model(
        input_size=config.flow.input_size,
        output_size=config.flow.output_size,
        spk_embed_dim=config.flow.spk_embed_dim,
        vocab_size=config.flow.vocab_size,
        token_mel_ratio=config.flow.token_mel_ratio,
        pre_lookahead_len=config.flow.pre_lookahead_len,
        dit_dim=config.flow.dit.dim,
        dit_depth=config.flow.dit.depth,
        dit_heads=config.flow.dit.heads,
        dit_dim_head=config.flow.dit.dim_head,
        dit_ff_mult=config.flow.dit.ff_mult,
        static_chunk_size=config.flow.dit.static_chunk_size,
        cfm_sigma_min=config.flow.cfm_sigma_min,
        cfm_t_scheduler=config.flow.cfm_t_scheduler,
        cfm_inference_cfg_rate=config.flow.cfm_inference_cfg_rate,
        n_timesteps=config.flow.n_timesteps,
        rand_noise=rand_noise,
    )

    # Load flow weights (flow.* prefix)
    # Filter out rotary_embed.inv_freq which is computed at runtime
    flow_weights = {
        k[5:]: v
        for k, v in all_weights.items()
        if k.startswith("flow.") and "rotary_embed.inv_freq" not in k
    }
    if flow_weights:
        flow.load_weights(list(flow_weights.items()))

    # Build HiFi-GAN
    hifigan = CausalHiFTGenerator(
        in_channels=config.hifigan.in_channels,
        base_channels=config.hifigan.base_channels,
        nb_harmonics=config.hifigan.nb_harmonics,
        sampling_rate=config.hifigan.sampling_rate,
        upsample_rates=config.hifigan.upsample_rates,
        upsample_kernel_sizes=config.hifigan.upsample_kernel_sizes,
        istft_params={
            "n_fft": config.hifigan.istft_n_fft,
            "hop_len": config.hifigan.istft_hop_len,
        },
        resblock_kernel_sizes=config.hifigan.resblock_kernel_sizes,
        resblock_dilation_sizes=config.hifigan.resblock_dilation_sizes,
        source_resblock_kernel_sizes=config.hifigan.source_resblock_kernel_sizes,
        source_resblock_dilation_sizes=config.hifigan.source_resblock_dilation_sizes,
        conv_pre_look_right=config.hifigan.conv_pre_look_right,
    )

    # Load HiFi-GAN weights (hifigan.* prefix)
    # Use strict=False because stft_window is computed at init, not saved
    hifigan_weights = {
        k[8:]: v for k, v in all_weights.items() if k.startswith("hifigan.")
    }
    if hifigan_weights:
        hifigan.load_weights(list(hifigan_weights.items()), strict=False)

    # Create model
    model = CosyVoice3(
        config=config,
        llm=llm,
        flow=flow,
        hifigan=hifigan,
    )

    # Set model to eval mode (disables dropout)
    model.eval()

    logger.info("CosyVoice3 model loaded successfully")
    return model


class Model(nn.Module):
    """
    CosyVoice3 Model wrapper for compatibility with generate API.

    This provides a standardized interface for text-to-speech generation,
    matching the API used by other TTS models in mlx-audio.
    """

    def __init__(self, config: "ModelConfig" = None):
        """
        Initialize Model wrapper.

        Args:
            config: ModelConfig instance with model_path set
        """
        super().__init__()
        self.config = config or ModelConfig()
        self._sample_rate = self.config.sample_rate
        self._model: Optional[CosyVoice3] = None
        self._tokenizer = None
        self._s3_tokenizer = None
        self._speaker_encoder = None

    @property
    def sample_rate(self) -> int:
        """Audio sample rate."""
        return self._sample_rate

    def model_type(self) -> str:
        """Return model type identifier."""
        return "cosyvoice3"

    def sanitize(self, weights: dict) -> dict:
        """Sanitize weights for loading."""
        return weights

    def load_weights(self, weights: list, strict: bool = True) -> None:
        """Load model weights (handled by load_cosyvoice3)."""
        pass

    def _ensure_model_loaded(self) -> None:
        """Ensure the CosyVoice3 model is loaded."""
        if self._model is None:
            if self.config.model_path is None:
                raise RuntimeError("model_path not set in config")
            self._model = load_cosyvoice3(self.config.model_path)

    def _ensure_tokenizers_loaded(self) -> None:
        """Ensure tokenizers are loaded."""
        if self._tokenizer is None:
            model_path = Path(self.config.model_path)

            # Load text tokenizer (Qwen2 tokenizer)
            tokenizer_path = model_path / "tokenizer"
            if not tokenizer_path.exists():
                tokenizer_path = model_path / "qwen2"
            if not tokenizer_path.exists():
                tokenizer_path = model_path

            # Check for tokenizer files (fast or slow format)
            has_tokenizer = (tokenizer_path / "tokenizer.json").exists() or (
                tokenizer_path / "vocab.json"
            ).exists()
            if has_tokenizer:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

                # Add CosyVoice3-specific special tokens
                special_tokens = {
                    "additional_special_tokens": [
                        "<|endofprompt|>",
                        "[breath]",
                        "<strong>",
                        "</strong>",
                        "[noise]",
                        "[laughter]",
                        "[cough]",
                        "[clucking]",
                        "[accent]",
                        "[quick_breath]",
                        "<laughter>",
                        "</laughter>",
                        "[hissing]",
                        "[sigh]",
                        "[vocalized-noise]",
                        "[lipsmack]",
                        "[mn]",
                    ]
                }
                self._tokenizer.add_special_tokens(special_tokens)
            else:
                raise RuntimeError(
                    f"No tokenizer found in {tokenizer_path}. "
                    "Expected tokenizer.json or vocab.json"
                )

        if self._s3_tokenizer is None:
            # Load S3 speech tokenizer V3 from mlx-community/S3TokenizerV3
            from mlx_audio.codec.models.s3tokenizer import S3TokenizerV3

            self._s3_tokenizer = S3TokenizerV3.from_pretrained()

        if self._speaker_encoder is None:
            from ..cosyvoice2.speaker_encoder import CAMPlusSpeakerEncoder

            model_path = Path(self.config.model_path)
            consolidated_path = model_path / "model.safetensors"

            if consolidated_path.exists():
                raw_weights = mx.load(str(consolidated_path))
                campplus_weights = {
                    k[9:]: v
                    for k, v in raw_weights.items()
                    if k.startswith("campplus.")
                }
                if campplus_weights:
                    self._speaker_encoder = CAMPlusSpeakerEncoder()
                    self._speaker_encoder.model.load_weights(
                        list(campplus_weights.items())
                    )
                    self._speaker_encoder.model.eval()
                    self._speaker_encoder._loaded = True

            if self._speaker_encoder is None:
                campplus_safetensors = model_path / "campplus.safetensors"
                campplus_npz = model_path / "campplus.npz"

                if campplus_safetensors.exists():
                    self._speaker_encoder = CAMPlusSpeakerEncoder(
                        str(campplus_safetensors)
                    )
                elif campplus_npz.exists():
                    self._speaker_encoder = CAMPlusSpeakerEncoder(str(campplus_npz))
                else:
                    self._speaker_encoder = CAMPlusSpeakerEncoder(str(model_path))

    def generate(
        self,
        text: str,
        ref_audio: Optional[mx.array] = None,
        ref_text: Optional[str] = None,
        instruct_text: Optional[str] = None,
        source_audio: Optional[mx.array] = None,
        voice: Optional[str] = None,
        speed: float = 1.0,
        lang_code: str = "a",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        verbose: bool = True,
        stream: bool = False,
        streaming_interval: float = 2.0,
        stt_model: Optional[str] = "mlx-community/whisper-large-v3-turbo-4bit",
        **kwargs,
    ):
        """
        Generate speech from text or convert source audio.

        CosyVoice3 supports multiple inference modes (ref_audio required for all):
        1. Voice Conversion (VC): source_audio + ref_audio - Convert source to target voice
        2. Zero-shot: ref_audio + ref_text - Clone voice with semantic alignment
        3. Cross-lingual: ref_audio only - Clone voice for different language
        4. Instruct: ref_audio + instruct_text - Control style with instructions

        When ref_text is not provided and stt_model is set, the reference audio will
        be automatically transcribed using Whisper for best quality zero-shot cloning.
        Set stt_model=None to skip transcription and use cross-lingual mode instead.

        Args:
            text: Input text to synthesize (ignored in VC mode)
            ref_audio: Reference audio for voice cloning (REQUIRED, at 24kHz, max 30s)
            ref_text: Transcript of reference audio (for zero-shot mode).
                If not provided and stt_model is set, will auto-transcribe.
            instruct_text: Style instructions (for instruct mode)
            source_audio: Source audio for voice conversion (at 24kHz)
            voice: Ignored (CosyVoice3 uses reference audio)
            speed: Ignored
            lang_code: Ignored (auto-detected)
            temperature: Ignored
            max_tokens: Maximum tokens to generate
            verbose: Whether to print status
            stream: Whether to stream output
            streaming_interval: Ignored
            stt_model: Whisper model for auto-transcription (default: whisper-large-v3-turbo-4bit).
                Set to None to skip transcription and use cross-lingual mode.

        Yields:
            GenerationResult with generated audio
        """
        import time

        import numpy as np
        from scipy.signal import resample

        from mlx_audio.codec.models.s3gen.mel import (
            mel_spectrogram as cosyvoice_mel_spectrogram,
        )
        from mlx_audio.codec.models.s3tokenizer import (
            log_mel_spectrogram_compat as log_mel_spectrogram,
        )

        from ..base import GenerationResult

        start_time = time.time()

        # Validate ref_audio is provided
        if ref_audio is None:
            raise ValueError(
                "ref_audio is required for CosyVoice3. "
                "Provide reference audio for speaker conditioning."
            )

        # Load model and tokenizers
        self._ensure_model_loaded()
        self._ensure_tokenizers_loaded()

        # Tokenize text
        if self._tokenizer is None:
            raise RuntimeError("Text tokenizer not loaded")

        text_tokens = self._tokenizer.encode(text, add_special_tokens=False)
        text_array = mx.array([text_tokens], dtype=mx.int32)
        text_len = mx.array([len(text_tokens)], dtype=mx.int32)

        # Process reference audio
        ref_audio_np = (
            np.array(ref_audio) if isinstance(ref_audio, mx.array) else ref_audio
        )

        # Truncate reference audio to max 30 seconds
        max_ref_samples = int(30 * self._sample_rate)
        if len(ref_audio_np) > max_ref_samples:
            ref_audio_np = ref_audio_np[:max_ref_samples]

        # Trim silence from reference audio
        import librosa

        ref_audio_np, _ = librosa.effects.trim(
            ref_audio_np,
            top_db=60,
            frame_length=int(0.025 * self._sample_rate),
            hop_length=int(0.0125 * self._sample_rate),
        )

        # Resample to 16kHz for S3 tokenizer
        ref_audio_16k = resample(
            ref_audio_np,
            int(len(ref_audio_np) * 16000 / self._sample_rate),
        )
        ref_audio_16k = mx.array(ref_audio_16k, dtype=mx.float32)

        # Get mel spectrogram for S3 tokenizer (128 mels)
        mel_128 = log_mel_spectrogram(ref_audio_16k, n_mels=128)
        mel_128 = mx.expand_dims(mel_128, 0)
        mel_len_128 = mx.array([mel_128.shape[2]])

        # Get speech tokens
        speech_tokens, speech_token_lens = self._s3_tokenizer(mel_128, mel_len_128)
        prompt_speech_token = speech_tokens
        prompt_speech_token_len = speech_token_lens

        # Get mel spectrogram for flow model (80 mels) at 24kHz
        ref_audio_24k = mx.array(ref_audio_np, dtype=mx.float32)
        mel_80 = cosyvoice_mel_spectrogram(
            ref_audio_24k,
            n_fft=1920,
            num_mels=80,
            sampling_rate=24000,
            hop_size=480,
            win_size=1920,
            fmin=0,
            fmax=8000,
            center=False,
        )
        mel_80 = mx.swapaxes(mel_80, 1, 2)  # (1, T, 80)

        # Align mel and token lengths
        token_len = int(prompt_speech_token_len[0])
        max_mel_len = int(mel_80.shape[1])

        if max_mel_len < token_len * 2:
            token_len = max_mel_len // 2

        mel_len = token_len * 2
        prompt_mel = mel_80[:, :mel_len, :]
        prompt_mel_len = mx.array([mel_len], dtype=mx.int32)
        prompt_speech_token = prompt_speech_token[:, :token_len]
        prompt_speech_token_len = mx.array([token_len], dtype=mx.int32)

        # Auto-transcribe reference audio if ref_text not provided
        # Skip for VC mode (source_audio) and instruct mode (instruct_text)
        if (
            not ref_text
            and stt_model is not None
            and source_audio is None
            and instruct_text is None
        ):
            if verbose:
                logger.info("Transcribing reference audio with Whisper...")
            from mlx_audio.stt.models.whisper import Model as Whisper

            whisper = Whisper.from_pretrained(path_or_hf_repo=stt_model)
            ref_text = whisper.generate(ref_audio_16k).text
            if verbose:
                logger.info(f"Transcription: {ref_text}")
            # Clean up whisper model to free memory
            del whisper
            mx.clear_cache()

        # Tokenize reference text (for zero-shot mode)
        if ref_text:
            prompt_text_tokens = self._tokenizer.encode(
                ref_text, add_special_tokens=False
            )
            prompt_text = mx.array([prompt_text_tokens], dtype=mx.int32)
            prompt_text_len = mx.array([len(prompt_text_tokens)], dtype=mx.int32)
        else:
            prompt_text = mx.zeros((1, 0), dtype=mx.int32)
            prompt_text_len = mx.array([0], dtype=mx.int32)

        # Extract speaker embedding using CAMPlus
        speaker_embedding = self._speaker_encoder(ref_audio_16k, sample_rate=16000)

        # Process source audio for voice conversion mode
        source_speech_token = None
        source_speech_token_len = None
        if source_audio is not None:
            source_audio_np = (
                np.array(source_audio)
                if isinstance(source_audio, mx.array)
                else source_audio
            )

            max_source_samples = int(30 * self._sample_rate)
            if len(source_audio_np) > max_source_samples:
                source_audio_np = source_audio_np[:max_source_samples]

            source_audio_16k = resample(
                source_audio_np,
                int(len(source_audio_np) * 16000 / self._sample_rate),
            )
            source_audio_16k = mx.array(source_audio_16k, dtype=mx.float32)

            source_mel_128 = log_mel_spectrogram(source_audio_16k, n_mels=128)
            source_mel_128 = mx.expand_dims(source_mel_128, 0)
            source_mel_len_128 = mx.array([source_mel_128.shape[2]])

            source_speech_token, source_speech_token_len = self._s3_tokenizer(
                source_mel_128, source_mel_len_128
            )

        # Generate audio using appropriate mode
        try:
            if source_audio is not None:
                audio = self._model.synthesize_vc(
                    source_speech_token=source_speech_token,
                    source_speech_token_len=source_speech_token_len,
                    prompt_speech_token=prompt_speech_token,
                    prompt_speech_token_len=prompt_speech_token_len,
                    prompt_mel=prompt_mel,
                    prompt_mel_len=prompt_mel_len,
                    speaker_embedding=speaker_embedding,
                )
            elif ref_text:
                audio = self._model.synthesize(
                    text=text_array,
                    text_len=text_len,
                    prompt_text=prompt_text,
                    prompt_text_len=prompt_text_len,
                    prompt_speech_token=prompt_speech_token,
                    prompt_speech_token_len=prompt_speech_token_len,
                    prompt_mel=prompt_mel,
                    prompt_mel_len=prompt_mel_len,
                    speaker_embedding=speaker_embedding,
                    sampling=25,
                    max_token_text_ratio=20.0,
                    min_token_text_ratio=2.0,
                )
            elif instruct_text:
                instruct_with_marker = instruct_text + "<|endofprompt|>"
                instruct_tokens = self._tokenizer.encode(
                    instruct_with_marker, add_special_tokens=False
                )
                instruct_array = mx.array([instruct_tokens], dtype=mx.int32)
                instruct_len = mx.array([len(instruct_tokens)], dtype=mx.int32)

                audio = self._model.synthesize_instruct(
                    text=text_array,
                    text_len=text_len,
                    instruct_text=instruct_array,
                    instruct_text_len=instruct_len,
                    prompt_speech_token=prompt_speech_token,
                    prompt_speech_token_len=prompt_speech_token_len,
                    prompt_mel=prompt_mel,
                    prompt_mel_len=prompt_mel_len,
                    speaker_embedding=speaker_embedding,
                    sampling=25,
                    max_token_text_ratio=20.0,
                    min_token_text_ratio=2.0,
                )
            else:
                audio = self._model.synthesize_cross_lingual(
                    text=text_array,
                    text_len=text_len,
                    prompt_speech_token=prompt_speech_token,
                    prompt_speech_token_len=prompt_speech_token_len,
                    prompt_mel=prompt_mel,
                    prompt_mel_len=prompt_mel_len,
                    speaker_embedding=speaker_embedding,
                    sampling=25,
                    max_token_text_ratio=20.0,
                    min_token_text_ratio=2.0,
                )
        except Exception as e:
            raise RuntimeError(f"Audio generation failed: {e}")

        # Squeeze audio to 1D
        audio_out = audio.squeeze()
        num_samples = audio_out.shape[0]

        end_time = time.time()
        processing_time = end_time - start_time
        audio_duration_secs = num_samples / self._sample_rate

        mins = int(audio_duration_secs // 60)
        secs = audio_duration_secs % 60
        duration_str = f"{mins:02d}:{secs:06.3f}"

        result = GenerationResult(
            audio=audio_out,
            samples=num_samples,
            sample_rate=self._sample_rate,
            segment_idx=0,
            token_count=len(text_tokens),
            audio_samples={
                "samples": num_samples,
                "samples-per-sec": (
                    num_samples / processing_time if processing_time > 0 else 0
                ),
            },
            audio_duration=duration_str,
            real_time_factor=(
                processing_time / audio_duration_secs if audio_duration_secs > 0 else 0
            ),
            prompt={
                "tokens-per-sec": (
                    len(text_tokens) / processing_time if processing_time > 0 else 0
                )
            },
            processing_time_seconds=processing_time,
            peak_memory_usage=(
                mx.get_peak_memory() / 1e9 if hasattr(mx, "get_peak_memory") else 0
            ),
        )

        yield result
