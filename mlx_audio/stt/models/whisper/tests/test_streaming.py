# Copyright © Prince Canuma (https://github.com/Blaizzy/mlx-audio)
"""Tests for AlignAtt streaming transcription."""

import numpy as np
import pytest


class TestStreamingConfig:
    """Test StreamingConfig dataclass."""

    def test_default_values(self):
        """StreamingConfig has sensible defaults."""
        from mlx_audio.stt.models.whisper.streaming import StreamingConfig

        config = StreamingConfig()
        assert config.frame_threshold == 25  # ~0.5s lookahead
        assert config.min_chunk_duration == 0.5  # minimum audio before processing
        assert config.emit_partial is True

    def test_custom_values(self):
        """StreamingConfig accepts custom values."""
        from mlx_audio.stt.models.whisper.streaming import StreamingConfig

        config = StreamingConfig(
            frame_threshold=10, min_chunk_duration=1.0, emit_partial=False
        )
        assert config.frame_threshold == 10
        assert config.min_chunk_duration == 1.0
        assert config.emit_partial is False


class TestStreamingResult:
    """Test StreamingResult dataclass."""

    def test_creation(self):
        """StreamingResult holds transcription state."""
        from mlx_audio.stt.models.whisper.streaming import StreamingResult

        result = StreamingResult(
            text="hello world",
            tokens=[1, 2, 3],
            is_final=False,
            start_time=0.0,
            end_time=1.5,
        )
        assert result.text == "hello world"
        assert result.tokens == [1, 2, 3]
        assert result.is_final is False
        assert result.start_time == 0.0
        assert result.end_time == 1.5

    def test_final_result(self):
        """StreamingResult can mark final emission."""
        from mlx_audio.stt.models.whisper.streaming import StreamingResult

        result = StreamingResult(
            text="complete sentence",
            tokens=[1, 2, 3, 4],
            is_final=True,
            start_time=0.0,
            end_time=2.0,
        )
        assert result.is_final is True


class TestInferenceWithCrossAttention:
    """Test Inference class returns cross-attention weights."""

    @pytest.fixture
    def mock_model(self):
        """Create a minimal mock model for testing."""
        from mlx_audio.stt.models.whisper.whisper import Model, ModelDimensions

        dims = ModelDimensions(
            n_mels=80,
            n_audio_ctx=1500,
            n_audio_state=384,
            n_audio_head=6,
            n_audio_layer=4,
            n_vocab=51865,
            n_text_ctx=448,
            n_text_state=384,
            n_text_head=6,
            n_text_layer=4,
        )
        return Model(dims)

    def test_logits_with_cross_qk(self, mock_model):
        """Inference.logits_with_cross_qk returns attention weights."""
        import mlx.core as mx

        from mlx_audio.stt.models.whisper.decoding import Inference

        inference = Inference(mock_model)

        # Create dummy inputs
        tokens = mx.array([[50258]])  # SOT token
        audio_features = mx.zeros((1, 1500, 384))

        logits, cross_qk = inference.logits_with_cross_qk(tokens, audio_features)

        assert logits.shape[0] == 1  # batch size
        assert logits.shape[-1] == 51865  # vocab size
        assert cross_qk is not None
        assert len(cross_qk) == 4  # n_text_layer


class TestGetMostAttendedFrame:
    """Test attention frame extraction helper."""

    def test_single_head(self):
        """Extract most attended frame from single attention head."""
        import mlx.core as mx

        from mlx_audio.stt.models.whisper.streaming import get_most_attended_frame

        # Simulate attention: token attends most to frame 10
        cross_qk = [mx.zeros((1, 6, 5, 100))]  # 1 layer, 6 heads, 5 tokens, 100 frames
        cross_qk[0][:, :, -1, 10] = 1.0  # Last token attends to frame 10

        alignment_heads = mx.array([[0, 0]])  # Use layer 0, head 0
        frame = get_most_attended_frame(cross_qk, alignment_heads)

        assert frame == 10

    def test_multiple_heads_averaged(self):
        """Average attention across multiple alignment heads."""
        import mlx.core as mx

        from mlx_audio.stt.models.whisper.streaming import get_most_attended_frame

        # Two heads attend to different frames
        cross_qk = [mx.zeros((1, 6, 5, 100))]
        cross_qk[0][:, 0, -1, 10] = 1.0  # Head 0 → frame 10
        cross_qk[0][:, 1, -1, 20] = 1.0  # Head 1 → frame 20

        alignment_heads = mx.array([[0, 0], [0, 1]])  # Use heads 0 and 1
        frame = get_most_attended_frame(cross_qk, alignment_heads)

        # Average of 10 and 20 = 15, argmax should be near that
        assert 10 <= frame <= 20


class TestShouldEmit:
    """Test AlignAtt emission decision logic."""

    def test_emit_when_attention_near_end(self):
        """Emit when attention is within threshold of audio end."""
        from mlx_audio.stt.models.whisper.streaming import StreamingConfig, should_emit

        config = StreamingConfig(frame_threshold=25)

        # Attention at frame 80, content has 100 frames → distance = 20 < 25 → emit
        assert (
            should_emit(most_attended_frame=80, content_frames=100, config=config)
            is True
        )

    def test_no_emit_when_attention_far_from_end(self):
        """Don't emit when attention is far from audio end."""
        from mlx_audio.stt.models.whisper.streaming import StreamingConfig, should_emit

        config = StreamingConfig(frame_threshold=25)

        # Attention at frame 50, content has 100 frames → distance = 50 > 25 → don't emit
        assert (
            should_emit(most_attended_frame=50, content_frames=100, config=config)
            is False
        )

    def test_emit_at_exact_threshold(self):
        """Emit when exactly at threshold boundary."""
        from mlx_audio.stt.models.whisper.streaming import StreamingConfig, should_emit

        config = StreamingConfig(frame_threshold=25)

        # Attention at frame 75, content has 100 frames → distance = 25 == 25 → emit
        assert (
            should_emit(most_attended_frame=75, content_frames=100, config=config)
            is True
        )


class TestStreamingDecoder:
    """Test StreamingDecoder class."""

    @pytest.fixture
    def whisper_model(self):
        """Load a small Whisper model for integration tests."""
        from mlx_audio.stt.utils import load_model

        # Use tiny model for fast tests
        return load_model("mlx-community/whisper-tiny-asr-fp16")

    def test_initialization(self, whisper_model):
        """StreamingDecoder initializes with model and config."""
        from mlx_audio.stt.models.whisper.streaming import (
            StreamingConfig,
            StreamingDecoder,
        )

        config = StreamingConfig(frame_threshold=25)
        decoder = StreamingDecoder(whisper_model, config)

        assert decoder.model is whisper_model
        assert decoder.config.frame_threshold == 25

    def test_reset_clears_state(self, whisper_model):
        """reset() clears accumulated state."""
        from mlx_audio.stt.models.whisper.streaming import (
            StreamingConfig,
            StreamingDecoder,
        )

        decoder = StreamingDecoder(whisper_model, StreamingConfig())
        decoder._emitted_tokens = [1, 2, 3]  # Simulate accumulated state
        decoder.reset()

        assert decoder._emitted_tokens == []


class TestDecodeChunk:
    """Test streaming decode_chunk method."""

    @pytest.fixture
    def whisper_model(self):
        """Load a small Whisper model."""
        from mlx_audio.stt.utils import load_model

        return load_model("mlx-community/whisper-tiny-asr-fp16")

    @pytest.fixture
    def sample_audio(self):
        """Create sample audio for testing."""
        # 1 second of silence at 16kHz
        return np.zeros(16000, dtype=np.float32)

    def test_decode_chunk_returns_result(self, whisper_model, sample_audio):
        """decode_chunk returns StreamingResult."""
        from mlx_audio.stt.models.whisper.audio import log_mel_spectrogram
        from mlx_audio.stt.models.whisper.streaming import (
            StreamingConfig,
            StreamingDecoder,
            StreamingResult,
        )

        decoder = StreamingDecoder(whisper_model, StreamingConfig())
        mel = log_mel_spectrogram(sample_audio)

        result = decoder.decode_chunk(mel)

        assert isinstance(result, StreamingResult)
        assert isinstance(result.text, str)
        assert isinstance(result.is_final, bool)

    def test_decode_chunk_accumulates_audio(self, whisper_model, sample_audio):
        """Multiple chunks accumulate for better context."""
        from mlx_audio.stt.models.whisper.audio import log_mel_spectrogram
        from mlx_audio.stt.models.whisper.streaming import (
            StreamingConfig,
            StreamingDecoder,
        )

        decoder = StreamingDecoder(whisper_model, StreamingConfig())

        # First chunk
        mel1 = log_mel_spectrogram(sample_audio)
        result1 = decoder.decode_chunk(mel1)

        # Second chunk (should have more context)
        mel2 = log_mel_spectrogram(sample_audio)
        result2 = decoder.decode_chunk(mel2)

        # Decoder should track state across chunks
        assert result2 is not None


class TestModuleExports:
    """Test that streaming components are properly exported."""

    def test_import_from_whisper_module(self):
        """Streaming classes importable from whisper module."""
        from mlx_audio.stt.models.whisper import (
            StreamingConfig,
            StreamingDecoder,
            StreamingResult,
        )

        assert StreamingConfig is not None
        assert StreamingResult is not None
        assert StreamingDecoder is not None


class TestSharedHelpers:
    """Test shared helper methods."""

    def test_get_suppress_tokens_helper(self):
        """get_suppress_tokens returns consistent token set."""
        from mlx_audio.stt.models.whisper.decoding import get_suppress_tokens

        # Mock tokenizer with required properties for get_suppress_tokens
        class MockTokenizer:
            non_speech_tokens = (100, 101, 102)
            transcribe = 50358
            translate = 50357
            sot = 50258
            sot_prev = 50361
            sot_lm = 50360
            no_speech = 50362

        tokenizer = MockTokenizer()
        tokens = get_suppress_tokens(tokenizer)

        assert isinstance(tokens, (list, tuple, set))
        assert len(tokens) > 0
        # Should include non-speech tokens
        assert any(t in tokens for t in tokenizer.non_speech_tokens)
