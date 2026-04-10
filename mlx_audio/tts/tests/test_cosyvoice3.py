# Copyright © Anthony DePasquale

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np

from mlx_audio.tts.generate import generate_audio
from mlx_audio.tts.models.base import GenerationResult
from mlx_audio.tts.models.cosyvoice3.config import FlowConfig
from mlx_audio.tts.models.cosyvoice3.cosyvoice3 import CosyVoice3, Model
from mlx_audio.tts.models.cosyvoice3.flow import (
    CausalMaskedDiffWithDiT,
    build_flow_model,
)
from mlx_audio.tts.models.cosyvoice3.llm.llm import CosyVoice3LM, ras_sampling


class TestCosyVoice3Defaults(unittest.TestCase):
    def test_flow_defaults_match_original_pytorch(self):
        """CosyVoice3 flow defaults should match the original PyTorch config."""
        config = FlowConfig()
        self.assertEqual(config.input_size, 80)
        self.assertEqual(config.pre_lookahead_channels, 1024)

        flow_init = inspect.signature(CausalMaskedDiffWithDiT.__init__)
        build_flow = inspect.signature(build_flow_model)
        self.assertEqual(flow_init.parameters["input_size"].default, 80)
        self.assertEqual(build_flow.parameters["input_size"].default, 80)


class TestCosyVoice3Sampling(unittest.TestCase):
    def test_sampling_ids_masks_only_stop_index_during_min_length(self):
        """ignore_eos should mask only speech_token_size, matching PyTorch."""

        captured = {}

        def sampling(scores, decoded_tokens, sampling):
            captured["masked_stop"] = float(scores[8].item())
            captured["next_token"] = float(scores[9].item())
            return 9

        lm = CosyVoice3LM(
            llm_input_size=4,
            llm_output_size=4,
            speech_token_size=8,
            extended_vocab_size=5,
            sampling=sampling,
        )

        sampled = lm.sampling_ids(mx.zeros((13,), dtype=mx.float32), [], 25)

        self.assertTrue(np.isneginf(captured["masked_stop"]))
        self.assertEqual(captured["next_token"], 0.0)
        self.assertEqual(sampled, 9)

    def test_ras_sampling_masks_repeated_token_before_resampling(self):
        """RAS should ban the repeated token before the fallback sample."""
        logits = mx.full((10,), -1e9, dtype=mx.float32)
        logits[5] = 0.0
        logits[7] = -0.2

        sampled = ras_sampling(
            logits,
            decoded_tokens=[5] * 10,
            sampling=25,
            win_size=10,
            tau_r=0.1,
        )

        self.assertEqual(sampled, 7)


class TestCosyVoice3Model(unittest.TestCase):
    def test_normalize_text_spells_english_numbers(self):
        """English digit runs should be spelled out like the original frontend."""
        model = Model()
        self.assertEqual(
            model._normalize_text("I have 2 apples and 15 oranges"),
            "I have two apples and fifteen oranges",
        )

    def test_normalize_text_preserves_control_tags(self):
        """Inline control tags should bypass normalization like the original frontend."""
        model = Model()
        tagged = "You are a helpful assistant.<|endofprompt|>I have 2 apples."
        self.assertEqual(model._normalize_text(tagged), tagged)

    def test_generate_rejects_reference_audio_longer_than_30_seconds(self):
        """Reference audio should fail fast instead of being silently clipped."""
        model = Model()
        model._ensure_model_loaded = MagicMock()
        model._ensure_tokenizers_loaded = MagicMock()
        model._tokenizer = MagicMock()
        model._tokenizer.encode.return_value = [1, 2, 3]

        ref_audio = mx.zeros((model.sample_rate * 31,), dtype=mx.float32)

        with self.assertRaisesRegex(
            ValueError,
            "reference audio longer than 30 seconds",
        ):
            list(model.generate("hello", ref_audio=ref_audio, ref_text="reference"))

    @patch(
        "mlx_audio.codec.models.s3gen.mel.mel_spectrogram",
        return_value=mx.zeros((1, 80, 16), dtype=mx.float32),
    )
    @patch(
        "mlx_audio.codec.models.s3tokenizer.log_mel_spectrogram_compat",
        return_value=mx.zeros((128, 8), dtype=mx.float32),
    )
    @patch(
        "mlx_audio.tts.models.cosyvoice3.cosyvoice3.resample_poly",
        side_effect=lambda audio, up, down: np.zeros(
            int(np.ceil(len(audio) * up / down)), dtype=np.float32
        ),
    )
    def test_generate_uses_cross_lingual_when_ref_text_missing(
        self,
        mock_resample,
        mock_log_mel,
        mock_flow_mel,
    ):
        """Missing ref_text should stay cross-lingual even if stt_model is set."""
        del mock_resample, mock_log_mel, mock_flow_mel

        model = Model()
        model._ensure_model_loaded = MagicMock()
        model._ensure_tokenizers_loaded = MagicMock()
        model._tokenizer = MagicMock()
        model._tokenizer.encode.return_value = [10, 11, 12]
        model._s3_tokenizer = MagicMock(
            return_value=(
                mx.array([[1, 2, 3, 4, 5, 6]], dtype=mx.int32),
                mx.array([6], dtype=mx.int32),
            )
        )
        model._speaker_encoder = MagicMock(return_value=mx.zeros((1, 192)))
        model._model = MagicMock()
        model._model.synthesize.return_value = mx.ones((1, 32), dtype=mx.float32)
        model._model.synthesize_cross_lingual.return_value = mx.zeros(
            (1, 32), dtype=mx.float32
        )

        results = list(
            model.generate(
                "hello",
                ref_audio=mx.zeros((24000,), dtype=mx.float32),
                ref_text=None,
                stt_model="mlx-community/whisper-large-v3-turbo-asr-fp16",
                verbose=False,
            )
        )

        self.assertEqual(len(results), 1)
        model._model.synthesize.assert_not_called()
        model._model.synthesize_cross_lingual.assert_called_once()

    def test_split_text_for_inference_matches_original_chunking(self):
        """Long sentence groups should split before synthesis like the original frontend."""
        model = Model()
        model._tokenizer = MagicMock()

        def encode(text, add_special_tokens=False):
            del add_special_tokens
            if text == "First sentence.":
                return list(range(61))
            if text == " Second sentence.":
                return list(range(61))
            if text == "First sentence. Second sentence.":
                return list(range(122))
            return list(range(max(1, len(text.split()))))

        model._tokenizer.encode.side_effect = encode

        chunks = model._split_text_for_inference("First sentence. Second sentence.")

        self.assertEqual(chunks, ["First sentence.", " Second sentence."])

    def test_streaming_hop_grows_like_original_pytorch(self):
        """Streaming should grow chunk hops 25 -> 50 -> 100 instead of staying fixed."""
        model = CosyVoice3(
            config=SimpleNamespace(hifigan=SimpleNamespace(sampling_rate=24000))
        )
        model.flow = SimpleNamespace(pre_lookahead_len=3, token_mel_ratio=2)
        model.generate_tokens = lambda **kwargs: iter(range(130))

        token_lens = []

        def tokens_to_mel(**kwargs):
            token_len = int(kwargs["token_len"][0].item())
            token_lens.append((token_len, kwargs["finalize"]))
            mel = mx.zeros((1, 80, token_len * 2), dtype=mx.float32)
            return mel, None

        model.tokens_to_mel = tokens_to_mel
        model.mel_to_audio = lambda mel, finalize=True: mx.zeros(
            (1, mel.shape[2] * 4), dtype=mx.float32
        )

        list(
            model.synthesize_streaming(
                text=mx.array([[1, 2]], dtype=mx.int32),
                text_len=mx.array([2], dtype=mx.int32),
                prompt_text=mx.zeros((1, 0), dtype=mx.int32),
                prompt_text_len=mx.array([0], dtype=mx.int32),
                prompt_speech_token=mx.zeros((1, 0), dtype=mx.int32),
                prompt_speech_token_len=mx.array([0], dtype=mx.int32),
                prompt_mel=mx.zeros((1, 0, 80), dtype=mx.float32),
                prompt_mel_len=mx.array([0], dtype=mx.int32),
                speaker_embedding=mx.zeros((1, 192), dtype=mx.float32),
                chunk_size=25,
            )
        )

        self.assertEqual(token_lens[:2], [(28, False), (78, False)])


class FakeCosyVoice3Model:
    def __init__(self):
        self.sample_rate = 24000
        self.called_kwargs = None

    def model_type(self):
        return "cosyvoice3"

    def generate(
        self,
        text,
        ref_audio=None,
        ref_text=None,
        instruct_text=None,
        stt_model=None,
        seed=None,
        **kwargs,
    ):
        self.called_kwargs = {
            "text": text,
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "instruct_text": instruct_text,
            "stt_model": stt_model,
            "seed": seed,
            **kwargs,
        }
        yield GenerationResult(
            audio=mx.zeros((32,), dtype=mx.float32),
            samples=32,
            sample_rate=24000,
            segment_idx=0,
            token_count=2,
            audio_samples={"samples": 32, "samples-per-sec": 3200.0},
            audio_duration="00:00.001",
            real_time_factor=0.1,
            prompt={"tokens-per-sec": 200.0},
            processing_time_seconds=0.01,
            peak_memory_usage=0.0,
        )


class TestCosyVoice3GenerateAudio(unittest.TestCase):
    @patch("mlx_audio.tts.generate.audio_write")
    @patch("mlx_audio.tts.generate.load_audio", return_value=mx.zeros((24000,)))
    @patch("mlx_audio.tts.generate.load_model")
    @patch("mlx_audio.tts.generate.os.path.exists", return_value=True)
    def test_generate_audio_does_not_helper_autotranscribe_models_with_stt_support(
        self,
        mock_exists,
        mock_load_model,
        mock_load_audio,
        mock_audio_write,
    ):
        """The public helper should defer STT-capable models to their own routing."""
        del mock_exists, mock_load_audio, mock_audio_write

        fake_model = FakeCosyVoice3Model()
        mock_load_model.return_value = fake_model

        generate_audio(
            text="hello",
            model="mlx-community/Fun-CosyVoice3-0.5B-2512",
            ref_audio="reference.wav",
            ref_text=None,
            stt_model=None,
            verbose=False,
        )

        self.assertIsNotNone(fake_model.called_kwargs)
        self.assertIsNone(fake_model.called_kwargs["ref_text"])
        self.assertIsNone(fake_model.called_kwargs["stt_model"])
        self.assertIn("instruct_text", fake_model.called_kwargs)

    @patch("mlx_audio.tts.generate.mx.random.seed")
    @patch("mlx_audio.tts.generate.audio_write")
    @patch("mlx_audio.tts.generate.load_audio", return_value=mx.zeros((24000,)))
    @patch("mlx_audio.tts.generate.load_model")
    @patch("mlx_audio.tts.generate.os.path.exists", return_value=True)
    def test_generate_audio_applies_and_forwards_seed(
        self,
        mock_exists,
        mock_load_model,
        mock_load_audio,
        mock_audio_write,
        mock_seed,
    ):
        del mock_exists, mock_load_audio, mock_audio_write

        fake_model = FakeCosyVoice3Model()
        mock_load_model.return_value = fake_model

        generate_audio(
            text="hello",
            model="mlx-community/Fun-CosyVoice3-0.5B-2512",
            ref_audio="reference.wav",
            ref_text=None,
            stt_model=None,
            verbose=False,
            seed=123,
        )

        mock_seed.assert_called_once_with(123)
        self.assertEqual(fake_model.called_kwargs["seed"], 123)


if __name__ == "__main__":
    unittest.main()
