# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#               2025 Alibaba Inc (authors: Xiang Lyu, Yabin Li)
# MLX port by mlx-audio contributors
#
# Licensed under the Apache License, Version 2.0

"""
CosyVoice2 TTS model for MLX.

This module provides the main CosyVoice2 class that integrates:
- Qwen2LM for speech token generation
- Flow Matching for mel spectrogram generation
- HiFi-GAN for waveform synthesis
"""

from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .config import CosyVoice2Config
from .llm import Qwen2Encoder, Qwen2LM, ras_sampling


class CosyVoice2(nn.Module):
    """
    CosyVoice2 Text-to-Speech model.

    This model generates high-quality speech from text using:
    1. Qwen2-based LLM for speech token generation
    2. Flow Matching for mel spectrogram synthesis
    3. HiFi-GAN for waveform generation

    Supports:
    - Zero-shot voice cloning with prompt audio
    - Streaming generation
    - Multiple languages (Chinese, English, Japanese, Korean)
    """

    def __init__(
        self,
        config: CosyVoice2Config = None,
        llm: Qwen2LM = None,
        flow: nn.Module = None,
        hifigan: nn.Module = None,
    ):
        """
        Initialize CosyVoice2.

        Args:
            config: Model configuration
            llm: Qwen2LM instance for speech token generation
            flow: Flow matching module (CausalMaskedDiffWithXvec)
            hifigan: HiFi-GAN vocoder (HiFTGenerator)
        """
        super().__init__()
        self.config = config or CosyVoice2Config()
        self.llm = llm
        self.flow = flow
        self.hifigan = hifigan

    def generate_tokens(
        self,
        text: mx.array,
        text_len: mx.array,
        prompt_text: mx.array,
        prompt_text_len: mx.array,
        prompt_speech_token: mx.array,
        prompt_speech_token_len: mx.array,
        embedding: Optional[mx.array] = None,
        sampling: int = 25,
        max_token_text_ratio: float = 20.0,
        min_token_text_ratio: float = 2.0,
    ) -> Generator[int, None, None]:
        """
        Generate speech tokens from text.

        Args:
            text: Input text token IDs (1, T)
            text_len: Text length (1,)
            prompt_text: Prompt text token IDs for voice cloning (1, T_p)
            prompt_text_len: Prompt text length (1,)
            prompt_speech_token: Prompt speech tokens for voice cloning (1, T_s)
            prompt_speech_token_len: Prompt speech token length (1,)
            embedding: Optional speaker embedding (unused, for API compat)
            sampling: Top-k sampling parameter
            max_token_text_ratio: Maximum speech/text token ratio
            min_token_text_ratio: Minimum speech/text token ratio

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
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """
        Convert speech tokens to mel spectrogram.

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
            finalize=finalize,
            n_timesteps=n_timesteps,
        )

    def mel_to_audio(
        self,
        mel: mx.array,
        f0: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Convert mel spectrogram to audio waveform.

        Args:
            mel: Mel spectrogram (1, D, T)
            f0: Optional F0 contour

        Returns:
            Audio waveform (1, T_audio)
        """
        if self.hifigan is None:
            raise RuntimeError("HiFi-GAN not initialized")

        # CosyHiFTGenerator has built-in F0 predictor, doesn't need external f0
        # It expects (B, n_mels, T) and returns (audio, source)
        audio, _ = self.hifigan(mel)
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

    def synthesize_zero_shot(
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
        Zero-shot voice cloning TTS pipeline.

        This mode requires a transcription of the reference audio (prompt_text).
        The LLM receives both the prompt text and prompt speech tokens, allowing
        it to learn the semantic alignment between text and speech.

        Args:
            text: Input text token IDs (1, T)
            text_len: Text length (1,)
            prompt_text: Transcription of reference audio as token IDs (1, T_p)
            prompt_text_len: Prompt text length (1,)
            prompt_speech_token: Speech tokens from reference audio (1, T_s)
            prompt_speech_token_len: Speech token length (1,)
            prompt_mel: Mel spectrogram from reference audio (1, T_mel, D)
            prompt_mel_len: Prompt mel length (1,)
            speaker_embedding: Speaker embedding from reference audio (1, D_spk)
            sampling: Top-k sampling parameter
            n_timesteps: Number of flow matching steps
            max_token_text_ratio: Maximum speech/text ratio
            min_token_text_ratio: Minimum speech/text ratio

        Returns:
            Audio waveform (1, T_audio)
        """
        # In zero-shot mode, LLM receives:
        # - prompt_text (transcription of reference)
        # - prompt_speech_token (speech tokens from reference)
        # This provides semantic alignment for voice cloning
        return self.synthesize(
            text=text,
            text_len=text_len,
            prompt_text=prompt_text,
            prompt_text_len=prompt_text_len,
            prompt_speech_token=prompt_speech_token,
            prompt_speech_token_len=prompt_speech_token_len,
            prompt_mel=prompt_mel,
            prompt_mel_len=prompt_mel_len,
            speaker_embedding=speaker_embedding,
            sampling=sampling,
            n_timesteps=n_timesteps,
            max_token_text_ratio=max_token_text_ratio,
            min_token_text_ratio=min_token_text_ratio,
        )

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
        # Step 1: Generate speech tokens WITHOUT prompt context
        # In cross-lingual mode, LLM receives:
        # - Empty prompt_text
        # - Empty prompt_speech_token
        tokens = []
        empty_prompt_text = mx.zeros((1, 0), dtype=mx.int32)
        empty_prompt_text_len = mx.array([0], dtype=mx.int32)
        empty_speech_token = mx.zeros((1, 0), dtype=mx.int32)
        empty_speech_token_len = mx.array([0], dtype=mx.int32)

        for token in self.generate_tokens(
            text=text,
            text_len=text_len,
            prompt_text=empty_prompt_text,
            prompt_text_len=empty_prompt_text_len,
            prompt_speech_token=empty_speech_token,
            prompt_speech_token_len=empty_speech_token_len,
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
        # Flow model STILL uses prompt_speech_token and prompt_mel for speaker identity
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
        # Step 1: Generate speech tokens with instruct context
        # In instruct mode, LLM receives:
        # - instruct_text as prompt_text (style instructions)
        # - NO prompt_speech_token (unlike zero-shot)
        tokens = []
        empty_speech_token = mx.zeros((1, 0), dtype=mx.int32)
        empty_speech_token_len = mx.array([0], dtype=mx.int32)

        for token in self.generate_tokens(
            text=text,
            text_len=text_len,
            prompt_text=instruct_text,
            prompt_text_len=instruct_text_len,
            prompt_speech_token=empty_speech_token,
            prompt_speech_token_len=empty_speech_token_len,
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
        # Flow model STILL uses prompt_speech_token and prompt_mel for speaker identity
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
        Voice Conversion (VC) pipeline: convert source speech to target speaker voice.

        In this mode:
        - No LLM inference - source speech tokens are used directly
        - Flow model uses prompt_speech_token and prompt_mel for target speaker identity
        - Speaker embedding provides additional speaker characteristics

        This converts the content from source audio to the voice of the prompt speaker.

        Args:
            source_speech_token: Speech tokens from source audio to convert (1, T_s)
            source_speech_token_len: Source token length (1,)
            prompt_speech_token: Speech tokens from target speaker reference (1, T_p)
            prompt_speech_token_len: Prompt token length (1,)
            prompt_mel: Mel spectrogram from target speaker reference (1, T_mel, D)
            prompt_mel_len: Prompt mel length (1,)
            speaker_embedding: Speaker embedding from target speaker (1, D_spk)
            n_timesteps: Number of flow matching steps

        Returns:
            Audio waveform (1, T_audio)
        """
        # VC mode: use source speech tokens directly (no LLM generation)
        # The flow model converts these tokens to target speaker voice
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

        # Convert mel to audio
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
        chunk_size: int = 50,
        max_token_text_ratio: float = 20.0,
        min_token_text_ratio: float = 2.0,
    ) -> Generator[mx.array, None, None]:
        """
        Streaming TTS pipeline: yields audio chunks as they're generated.

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
            chunk_size: Number of tokens per chunk
            max_token_text_ratio: Maximum speech/text ratio
            min_token_text_ratio: Minimum speech/text ratio

        Yields:
            Audio waveform chunks
        """
        token_buffer = []

        for token in self.generate_tokens(
            text=text,
            text_len=text_len,
            prompt_text=prompt_text,
            prompt_text_len=prompt_text_len,
            prompt_speech_token=prompt_speech_token,
            prompt_speech_token_len=prompt_speech_token_len,
            sampling=sampling,
            max_token_text_ratio=max_token_text_ratio,
            min_token_text_ratio=min_token_text_ratio,
        ):
            token_buffer.append(token)

            if len(token_buffer) >= chunk_size:
                # Process chunk
                token_array = mx.array([token_buffer], dtype=mx.int32)
                token_len = mx.array([len(token_buffer)], dtype=mx.int32)

                mel, _ = self.tokens_to_mel(
                    tokens=token_array,
                    token_len=token_len,
                    prompt_token=prompt_speech_token,
                    prompt_token_len=prompt_speech_token_len,
                    prompt_feat=prompt_mel,
                    prompt_feat_len=prompt_mel_len,
                    embedding=speaker_embedding,
                    finalize=False,
                    n_timesteps=n_timesteps,
                )

                audio = self.mel_to_audio(mel)
                yield audio

                # Reset buffer
                token_buffer = []

        # Process remaining tokens
        if token_buffer:
            token_array = mx.array([token_buffer], dtype=mx.int32)
            token_len = mx.array([len(token_buffer)], dtype=mx.int32)

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

            audio = self.mel_to_audio(mel)
            yield audio


def load_cosyvoice2(
    model_path: Union[str, Path],
    qwen2_path: Optional[Union[str, Path]] = None,
) -> CosyVoice2:
    """
    Load a CosyVoice2 model from a directory.

    Args:
        model_path: Path to the model directory containing weights
        qwen2_path: Optional path to Qwen2 model (defaults to Qwen/Qwen2-0.5B-Instruct)

    Returns:
        Loaded CosyVoice2 model
    """
    model_path = Path(model_path)

    # Load configuration
    config = CosyVoice2Config.from_pretrained(model_path)

    # Load raw config.json to check for quantization
    import json

    config_json_path = model_path / "config.json"
    quantization_config = None
    if config_json_path.exists():
        with open(config_json_path) as f:
            raw_config = json.load(f)
            quantization_config = raw_config.get("quantization")

    # Load all weights from consolidated model.safetensors
    consolidated_path = model_path / "model.safetensors"
    if not consolidated_path.exists():
        raise FileNotFoundError(
            f"model.safetensors not found in {model_path}. "
            "Please run the conversion script."
        )

    all_weights = mx.load(str(consolidated_path))

    # Load Qwen2 model
    from mlx_lm.models.qwen2 import Model as Qwen2Model
    from mlx_lm.models.qwen2 import ModelArgs

    # Qwen2-0.5B-Instruct config
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

    # Apply quantization to Qwen2 model if needed
    if quantization_config:
        q_bits = quantization_config.get("bits", 4)
        q_group_size = quantization_config.get("group_size", 64)

        # Only quantize Linear layers in transformer blocks (same as conversion)
        def should_quantize(path, module):
            if isinstance(module, nn.Linear):
                if "model.layers" in path:
                    return True
            return False

        nn.quantize(
            qwen2_model,
            group_size=q_group_size,
            bits=q_bits,
            class_predicate=should_quantize,
        )

    # Load Qwen2 weights (qwen2.* prefix)
    # Skip lm_head.weight since it's tied to embeddings in Qwen2-0.5B
    qwen2_weights = {
        k[6:]: v
        for k, v in all_weights.items()
        if k.startswith("qwen2.") and k != "qwen2.lm_head.weight"
    }
    if qwen2_weights:
        qwen2_model.load_weights(list(qwen2_weights.items()))

    encoder = Qwen2Encoder(qwen2_model)

    # Create LLM
    llm = Qwen2LM(
        llm_input_size=config.llm.llm_input_size,
        llm_output_size=config.llm.llm_output_size,
        speech_token_size=config.llm.speech_token_size,
        llm=encoder,
        sampling=ras_sampling,
        mix_ratio=config.llm.mix_ratio,
    )

    # Load LLM weights (speech embeddings and decoder) - llm.* prefix
    from mlx.utils import tree_unflatten

    llm_weights = {k[4:]: v for k, v in all_weights.items() if k.startswith("llm.")}
    if llm_weights:
        llm.update(tree_unflatten(list(llm_weights.items())))

    # Import flow components from chatterbox
    from ..chatterbox.s3gen.decoder import ConditionalDecoder
    from ..chatterbox.s3gen.flow import CausalMaskedDiffWithXvec
    from ..chatterbox.s3gen.transformer import UpsampleConformerEncoder

    # Use CosyVoice2-specific flow matching (with PyTorch noise compatibility)
    from .flow_matching import CFM_PARAMS, CosyVoice2ConditionalCFM

    # Create encoder
    flow_encoder = UpsampleConformerEncoder(
        input_size=config.flow.encoder_input_size,
        output_size=config.flow.encoder_output_size,
        attention_heads=config.flow.encoder_attention_heads,
        linear_units=config.flow.encoder_linear_units,
        num_blocks=config.flow.encoder_num_blocks,
        num_up_blocks=config.flow.encoder_num_up_blocks,
        dropout_rate=config.flow.encoder_dropout_rate,
        positional_dropout_rate=config.flow.encoder_positional_dropout_rate,
        attention_dropout_rate=config.flow.encoder_attention_dropout_rate,
        normalize_before=config.flow.encoder_normalize_before,
        macaron_style=config.flow.encoder_macaron_style,
        use_cnn_module=config.flow.encoder_use_cnn_module,
        cnn_module_kernel=config.flow.encoder_cnn_module_kernel,
        causal=config.flow.encoder_causal,
        upsample_stride=config.flow.encoder_upsample_stride,
        static_chunk_size=config.flow.encoder_static_chunk_size,
        pos_enc_layer_type=config.flow.encoder_pos_enc_layer_type,
    )

    # Create decoder
    flow_decoder_estimator = ConditionalDecoder(
        in_channels=config.flow.decoder_in_channels,
        out_channels=config.flow.decoder_out_channel,
        channels=config.flow.decoder_channels,
        dropout=config.flow.decoder_dropout,
        attention_head_dim=config.flow.decoder_attention_head_dim,
        n_blocks=config.flow.decoder_n_blocks,
        num_mid_blocks=config.flow.decoder_num_mid_blocks,
        num_heads=config.flow.decoder_num_heads,
        act_fn=config.flow.decoder_act_fn,
        static_chunk_size=config.flow.decoder_static_chunk_size,
        num_decoding_left_chunks=config.flow.decoder_num_decoding_left_chunks,
    )

    # Flow decoder - random noise is generated at runtime by MLX
    flow_decoder = CosyVoice2ConditionalCFM(
        in_channels=config.flow.cfm_in_channels,  # 240 for CosyVoice2
        cfm_params=CFM_PARAMS,
        n_spks=1,
        spk_emb_dim=config.flow.output_size,  # Projected speaker embedding dim (80)
        estimator=flow_decoder_estimator,
    )

    # Create flow module
    flow = CausalMaskedDiffWithXvec(
        input_size=config.flow.input_size,
        output_size=config.flow.output_size,
        spk_embed_dim=config.flow.spk_embed_dim,
        vocab_size=config.flow.vocab_size,
        input_frame_rate=config.flow.input_frame_rate,
        token_mel_ratio=config.flow.token_mel_ratio,
        pre_lookahead_len=config.flow.pre_lookahead_len,
        n_timesteps=config.flow.n_timesteps,
        encoder=flow_encoder,
        decoder=flow_decoder,
    )

    # Load flow weights - flow.* prefix
    flow_weights = {k[5:]: v for k, v in all_weights.items() if k.startswith("flow.")}
    if flow_weights:
        flow.update(tree_unflatten(list(flow_weights.items())))

    # Import and create CosyVoice2-specific HiFi-GAN with built-in F0 predictor
    from .hifigan import CosyHiFTGenerator

    hifigan = CosyHiFTGenerator(
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
        use_interpolation=config.hifigan.use_interpolation,
    )

    # Load HiFi-GAN weights - hift.* prefix
    # Remap indexed-style keys to list-style keys for CosyHiFTGenerator
    import re

    def remap_hifigan_key(key):
        key = re.sub(
            r"(ups|resblocks|source_downs|source_resblocks)_(\d+)", r"\1.\2", key
        )
        key = re.sub(r"(convs1|convs2|activations1|activations2)_(\d+)", r"\1.\2", key)
        return key

    hift_weights = {k[5:]: v for k, v in all_weights.items() if k.startswith("hift.")}
    if hift_weights:
        hift_weights = {remap_hifigan_key(k): v for k, v in hift_weights.items()}
        hifigan.update(tree_unflatten(list(hift_weights.items())))

    # Create model
    model = CosyVoice2(
        config=config,
        llm=llm,
        flow=flow,
        hifigan=hifigan,
    )

    # Set to eval mode (disable dropout, etc.)
    model.eval()

    return model


class Model(nn.Module):
    """
    CosyVoice2 model wrapper for mlx_audio.tts.generate API compatibility.

    This wrapper provides the standard interface expected by generate_audio().
    """

    def __init__(self, config: "ModelConfig" = None):
        """
        Initialize Model wrapper.

        Args:
            config: ModelConfig instance with model_path set
        """
        super().__init__()
        from .config import ModelConfig

        self.config = config or ModelConfig()
        self._sample_rate = self.config.sample_rate
        self._model: Optional[CosyVoice2] = None
        self._tokenizer = None
        self._s3_tokenizer = None
        self._speaker_encoder = None

    @property
    def sample_rate(self) -> int:
        """Audio sample rate."""
        return self._sample_rate

    def model_type(self) -> str:
        """Return model type identifier."""
        return "cosyvoice2"

    def sanitize(self, weights: dict) -> dict:
        """
        Sanitize weights for loading.

        CosyVoice2 uses custom loading via load_cosyvoice2(),
        so this is a pass-through.
        """
        return weights

    def load_weights(self, weights: list, strict: bool = True) -> None:
        """
        Load model weights.

        For CosyVoice2, the actual model loading happens in generate()
        when model_path is available.
        """
        # CosyVoice2 uses load_cosyvoice2() which handles weight loading
        # We'll load the model lazily in generate()
        pass

    def _ensure_model_loaded(self) -> None:
        """Ensure the CosyVoice2 model is loaded."""
        if self._model is None:
            if self.config.model_path is None:
                raise RuntimeError("model_path not set in config")
            self._model = load_cosyvoice2(self.config.model_path)

    def _ensure_tokenizers_loaded(self) -> None:
        """Ensure tokenizers are loaded."""
        if self._tokenizer is None:
            from pathlib import Path

            model_path = Path(self.config.model_path)

            # Load text tokenizer (Qwen2 tokenizer)
            # Try tokenizer/ subdirectory first (backwards compat), then root directory
            tokenizer_path = model_path / "tokenizer"
            if not tokenizer_path.exists():
                tokenizer_path = model_path

            if (tokenizer_path / "tokenizer.json").exists():
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

                # Add CosyVoice2-specific special tokens
                # These are dynamically added at runtime in the original implementation
                special_tokens = {
                    "additional_special_tokens": [
                        "<|endofprompt|>",  # For instruct mode
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
            # Load S3 speech tokenizer with pretrained weights
            from ..chatterbox.s3tokenizer import S3TokenizerV2

            self._s3_tokenizer = S3TokenizerV2.from_pretrained(
                "speech_tokenizer_v2_25hz"
            )

        if self._speaker_encoder is None:
            # Load CAMPlus speaker encoder (pure MLX, no ONNX dependency)
            from pathlib import Path

            from .speaker_encoder import CAMPlusSpeakerEncoder

            model_path = Path(self.config.model_path)
            consolidated_path = model_path / "model.safetensors"

            # First try to load from consolidated model.safetensors
            if consolidated_path.exists():
                raw_weights = mx.load(str(consolidated_path))
                # Extract campplus.* weights and strip prefix
                campplus_weights = {
                    k[9:]: v
                    for k, v in raw_weights.items()
                    if k.startswith("campplus.")
                }
                if campplus_weights:
                    self._speaker_encoder = CAMPlusSpeakerEncoder()
                    # Sanitize and load weights
                    campplus_weights = self._speaker_encoder.model.sanitize(
                        campplus_weights
                    )
                    self._speaker_encoder.model.load_weights(
                        list(campplus_weights.items())
                    )
                    self._speaker_encoder.model.eval()
                    self._speaker_encoder._loaded = True

            # Fall back to separate files
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
                    # Try loading from model directory (will look for campplus.* files)
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
        **kwargs,
    ):
        """
        Generate speech from text or convert source audio.

        CosyVoice2 supports multiple inference modes (ref_audio required for all):
        1. Voice Conversion (VC): source_audio + ref_audio - Convert source to target voice
        2. Zero-shot: ref_audio + ref_text - Clone voice with semantic alignment
        3. Cross-lingual: ref_audio only - Clone voice for different language
        4. Instruct: ref_audio + instruct_text - Control style with instructions

        Args:
            text: Input text to synthesize (ignored in VC mode)
            ref_audio: Reference audio for voice cloning (REQUIRED, at 24kHz, max 30s)
            ref_text: Transcript of reference audio (for zero-shot mode)
            instruct_text: Style instructions (for instruct mode)
                e.g., "Speak slowly and clearly", "Read with excitement"
            source_audio: Source audio for voice conversion (at 24kHz)
                When provided with ref_audio, enables VC mode
            voice: Ignored (CosyVoice2 uses reference audio)
            speed: Ignored (not supported)
            lang_code: Ignored (auto-detected)
            temperature: Ignored (uses sampling parameter)
            max_tokens: Maximum tokens to generate
            verbose: Whether to print status
            stream: Whether to stream output
            streaming_interval: Ignored

        Yields:
            GenerationResult with generated audio
        """
        import time

        import numpy as np
        from scipy.signal import resample

        from ..base import GenerationResult
        from ..chatterbox.s3gen.mel import mel_spectrogram as cosyvoice2_mel_spectrogram
        from ..chatterbox.s3tokenizer import log_mel_spectrogram

        start_time = time.time()

        # Validate ref_audio is provided
        if ref_audio is None:
            raise ValueError(
                "ref_audio is required for CosyVoice2. "
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

        # Process reference audio (required for all modes)
        ref_audio_np = (
            np.array(ref_audio) if isinstance(ref_audio, mx.array) else ref_audio
        )

        # Truncate reference audio to max 30 seconds (CosyVoice2 limit)
        # Speech tokenizer doesn't support audio longer than 30s
        max_ref_samples = int(30 * self._sample_rate)
        if len(ref_audio_np) > max_ref_samples:
            ref_audio_np = ref_audio_np[:max_ref_samples]

        # Trim silence from reference audio (matches PyTorch implementation)
        # This prevents incomplete words at the end from bleeding into generated audio
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
        mel_128 = mx.expand_dims(mel_128, 0)  # Add batch dim
        mel_len_128 = mx.array([mel_128.shape[2]])

        # Get speech tokens
        speech_tokens, speech_token_lens = self._s3_tokenizer(mel_128, mel_len_128)
        prompt_speech_token = speech_tokens
        prompt_speech_token_len = speech_token_lens

        # Get mel spectrogram for flow model (80 mels) at 24kHz
        # CosyVoice2 uses n_fft=1920, hop_size=480, win_size=1920, fmin=0, fmax=8000
        ref_audio_24k = mx.array(ref_audio_np, dtype=mx.float32)
        mel_80 = cosyvoice2_mel_spectrogram(
            ref_audio_24k,
            n_fft=1920,
            num_mels=80,
            sampling_rate=24000,
            hop_size=480,
            win_size=1920,
            fmin=0,
            fmax=8000,
            center=False,
        )  # (1, 80, T)
        mel_80 = mx.swapaxes(mel_80, 1, 2)  # (1, T, 80) for flow model

        # Align mel and token lengths: mel_len must equal token_len * 2 exactly
        # This is critical for correct trimming in flow.inference()
        token_len = int(prompt_speech_token_len[0])
        max_mel_len = int(mel_80.shape[1])

        # Ensure mel_len = token_len * 2 (trim tokens if mel is shorter)
        if max_mel_len < token_len * 2:
            # Mel is shorter, trim tokens to match
            token_len = max_mel_len // 2

        mel_len = token_len * 2  # Exact alignment
        prompt_mel = mel_80[:, :mel_len, :]
        prompt_mel_len = mx.array([mel_len], dtype=mx.int32)
        prompt_speech_token = prompt_speech_token[:, :token_len]
        prompt_speech_token_len = mx.array([token_len], dtype=mx.int32)

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

            # Truncate source audio to max 30 seconds (CosyVoice2 limit)
            max_source_samples = int(30 * self._sample_rate)
            if len(source_audio_np) > max_source_samples:
                source_audio_np = source_audio_np[:max_source_samples]

            # Resample to 16kHz for S3 tokenizer
            source_audio_16k = resample(
                source_audio_np,
                int(len(source_audio_np) * 16000 / self._sample_rate),
            )
            source_audio_16k = mx.array(source_audio_16k, dtype=mx.float32)

            # Get mel spectrogram for S3 tokenizer (128 mels)
            source_mel_128 = log_mel_spectrogram(source_audio_16k, n_mels=128)
            source_mel_128 = mx.expand_dims(source_mel_128, 0)  # Add batch dim
            source_mel_len_128 = mx.array([source_mel_128.shape[2]])

            # Get speech tokens from source audio
            source_speech_token, source_speech_token_len = self._s3_tokenizer(
                source_mel_128, source_mel_len_128
            )

        # Generate audio using appropriate mode based on available inputs
        # Priority: VC > zero-shot > instruct > cross-lingual
        try:
            if source_audio is not None:
                # Voice Conversion mode: convert source speech to target speaker voice
                # Uses source speech tokens directly (no LLM), flow converts to target voice
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
                # Zero-shot mode: have both reference audio AND transcription
                # LLM receives prompt_text + prompt_speech_token for semantic alignment
                audio = self._model.synthesize_zero_shot(
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
                # Instruct mode: reference audio + style instructions
                # LLM receives instruct_text but NO speech tokens
                # Tokenize instruct text with <|endofprompt|> marker
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
                # Cross-lingual mode: have reference audio but NO transcription
                # LLM receives only text, flow uses prompt for speaker identity
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

        # Format duration string
        mins = int(audio_duration_secs // 60)
        secs = audio_duration_secs % 60
        duration_str = f"{mins:02d}:{secs:06.3f}"

        # Create result
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
