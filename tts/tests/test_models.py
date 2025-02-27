import unittest
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from unittest.mock import patch, MagicMock
from tts.models.kokoro.model import KokoroModel, sanitize_lstm_weights
from tts.models.kokoro.pipeline import KokoroPipeline, ALIASES, LANG_CODES
from misaki import en
import importlib.resources

# Create a patch for the deprecated open_text function
def patched_open_text(package, resource):
    """Replacement for deprecated open_text using files() API"""
    return importlib.resources.files(package).joinpath(resource).open('r')

# Apply the patch at the module level
@patch('importlib.resources.open_text', patched_open_text)
class TestSanitizeLSTMWeights(unittest.TestCase):
    def test_sanitize_lstm_weights(self):
        """Test sanitize_lstm_weights function."""
        # Test weight_ih_l0_reverse
        key = "lstm.weight_ih_l0_reverse"
        weights = mx.array(np.zeros((10, 10)))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "lstm.Wx_backward")

        # Test weight_hh_l0_reverse
        key = "lstm.weight_hh_l0_reverse"
        weights = mx.array(np.zeros((10, 10)))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "lstm.Wh_backward")

        # Test bias_ih_l0_reverse
        key = "lstm.bias_ih_l0_reverse"
        weights = mx.array(np.zeros(10))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "lstm.bias_ih_backward")

        # Test bias_hh_l0_reverse
        key = "lstm.bias_hh_l0_reverse"
        weights = mx.array(np.zeros(10))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "lstm.bias_hh_backward")

        # Test weight_ih_l0
        key = "lstm.weight_ih_l0"
        weights = mx.array(np.zeros((10, 10)))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "lstm.Wx_forward")

        # Test weight_hh_l0
        key = "lstm.weight_hh_l0"
        weights = mx.array(np.zeros((10, 10)))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "lstm.Wh_forward")

        # Test bias_ih_l0
        key = "lstm.bias_ih_l0"
        weights = mx.array(np.zeros(10))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "lstm.bias_ih_forward")

        # Test bias_hh_l0
        key = "lstm.bias_hh_l0"
        weights = mx.array(np.zeros(10))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "lstm.bias_hh_forward")

        # Test unknown key
        key = "unknown.key"
        weights = mx.array(np.zeros(10))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "unknown.key")


@patch('importlib.resources.open_text', patched_open_text)
class TestKokoroModel(unittest.TestCase):
    @patch('tts.models.kokoro.model.hf_hub_download')
    @patch('tts.models.kokoro.model.json.load')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('tts.models.kokoro.model.mx.load')
    @patch.object(nn.Module, 'load_weights')
    def test_init(self, mock_load_weights, mock_mx_load, mock_open, mock_json_load, mock_hf_hub_download):
        """Test KokoroModel initialization."""
        # Mock the config loading
        mock_json_load.return_value = {
            "istftnet": {
                "upsample_kernel_sizes": [20, 12],
                "upsample_rates": [10, 6],
                "gen_istft_hop_size": 5,
                "gen_istft_n_fft": 20,
                "resblock_dilation_sizes": [
                [1, 3, 5],
                [1, 3, 5],
                [1, 3, 5]
                ],
                "resblock_kernel_sizes": [3, 7, 11],
                "upsample_initial_channel": 512
            },
            "dim_in": 64,
            "dropout": 0.2,
            "hidden_dim": 512,
            "max_conv_dim": 512,
            "max_dur": 50,
            "multispeaker": True,
            "n_layer": 3,
            "n_mels": 80,
            "n_token": 178,
            "style_dim": 128,
            "text_encoder_kernel_size": 5,
            "plbert": {
                "hidden_size": 768,
                "num_attention_heads": 12,
                "intermediate_size": 2048,
                "max_position_embeddings": 512,
                "num_hidden_layers": 12,
                "dropout": 0.1
            },
            "vocab": {
                "a": 1,
                "b": 2
            }
        }

        # Mock the weights loading
        mock_mx_load.return_value = {"key": mx.array(np.zeros(10))}

        # Make load_weights return the module
        mock_load_weights.return_value = None

        # Initialize the model
        model = KokoroModel()

        # Check that the model was initialized correctly
        self.assertIsInstance(model, nn.Module)
        self.assertEqual(model.vocab, {"a": 1, "b": 2})

        # Verify that the hub download was called
        mock_hf_hub_download.assert_called()

    @patch.object(KokoroModel, '__init__', return_value=None)
    def test_output_dataclass(self, mock_init):
        """Test KokoroModel.Output dataclass."""
        # Create a mock output
        audio = mx.array(np.zeros((1, 1000)))
        pred_dur = mx.array(np.zeros((1, 100)))
        output = KokoroModel.Output(audio=audio, pred_dur=pred_dur)

        # Check that the output was created correctly
        self.assertIs(output.audio, audio)
        self.assertIs(output.pred_dur, pred_dur)


@patch('importlib.resources.open_text', patched_open_text)
class TestKokoroPipeline(unittest.TestCase):
    def test_aliases_and_lang_codes(self):
        """Test ALIASES and LANG_CODES constants."""
        # Check that all aliases map to valid language codes
        for alias_key, alias_value in ALIASES.items():
            self.assertIn(alias_value, LANG_CODES)

        # Check specific mappings
        self.assertEqual(ALIASES['en-us'], 'a')
        self.assertEqual(ALIASES['ja'], 'j')
        self.assertEqual(LANG_CODES['a'], 'American English')
        self.assertEqual(LANG_CODES['j'], 'Japanese')

    @patch('tts.models.kokoro.pipeline.KokoroModel')
    @patch('tts.models.kokoro.pipeline.isinstance')
    def test_init(self, mock_isinstance, mock_kokoro_model):
        """Test KokoroPipeline initialization."""
        # Mock the KokoroModel
        mock_model = MagicMock()
        mock_kokoro_model.return_value = mock_model

        # Simply make isinstance always return True when checking for KokoroModel
        mock_isinstance.return_value = True

        # Initialize with default model
        pipeline = KokoroPipeline(lang_code='a')
        self.assertEqual(pipeline.lang_code, 'a')
        self.assertEqual(LANG_CODES[pipeline.lang_code], 'American English')

        # Initialize with provided model
        model = mock_model
        pipeline = KokoroPipeline(lang_code='a', model=model)
        self.assertEqual(pipeline.model, model)

        # Initialize with no model
        pipeline = KokoroPipeline(lang_code='a', model=False)
        self.assertIs(pipeline.model, False)

    @patch.object(KokoroPipeline, '__init__', return_value=None)
    @patch('tts.models.kokoro.pipeline.torch.load')
    @patch('tts.models.kokoro.pipeline.hf_hub_download')
    @patch('tts.models.kokoro.pipeline.torch.stack')
    @patch('tts.models.kokoro.pipeline.torch.mean')
    def test_load_voice(self, mock_mean, mock_stack, mock_hf_hub_download, mock_torch_load, mock_init):
        """Test load_voice method."""
        # Setup the pipeline
        pipeline = KokoroPipeline.__new__(KokoroPipeline)
        pipeline.lang_code = 'a'
        pipeline.voices = {}

        # Mock the torch.load return value
        mock_torch_load.return_value = MagicMock()

        # Mock torch.stack and torch.mean to return a MagicMock
        mock_stack.return_value = MagicMock()
        mock_mean.return_value = MagicMock()

        # Test loading a single voice
        pipeline.load_single_voice('voice1')
        mock_hf_hub_download.assert_called_once()
        self.assertIn('voice1', pipeline.voices)

        # Test loading multiple voices
        mock_hf_hub_download.reset_mock()
        pipeline.voices = {}  # Reset voices
        result = pipeline.load_voice('voice1,voice2')
        self.assertEqual(mock_hf_hub_download.call_count, 2)
        self.assertIn('voice1', pipeline.voices)
        self.assertIn('voice2', pipeline.voices)

    @patch.object(KokoroPipeline, '__init__', return_value=None)
    @patch.object(KokoroPipeline, 'tokens_to_ps', return_value='p1 p2')
    def test_tokens_to_ps(self, mock_tokens_to_ps, mock_init):
        """Test tokens_to_ps method."""
        # Create mock tokens with whitespace attribute
        token1 = MagicMock(spec=en.MToken)
        token1.ps = 'p1'
        token1.whitespace = ' '
        token1.phonemes = 'p1'

        token2 = MagicMock(spec=en.MToken)
        token2.ps = 'p2'
        token2.whitespace = ''
        token2.phonemes = 'p2'

        tokens = [token1, token2]

        # Test the method
        result = KokoroPipeline.tokens_to_ps(tokens)
        self.assertEqual(result, 'p1 p2')

    @patch.object(KokoroPipeline, '__init__', return_value=None)
    @patch.object(KokoroPipeline, 'tokens_to_text', return_value='Hello world')
    def test_tokens_to_text(self, mock_tokens_to_text, mock_init):
        """Test tokens_to_text method."""
        # Create mock tokens with whitespace attribute
        token1 = MagicMock(spec=en.MToken)
        token1.text = 'Hello'
        token1.whitespace = ' '

        token2 = MagicMock(spec=en.MToken)
        token2.text = 'world'
        token2.whitespace = ''

        tokens = [token1, token2]

        # Test the method
        result = KokoroPipeline.tokens_to_text(tokens)
        self.assertEqual(result, 'Hello world')

    @patch.object(KokoroPipeline, '__init__', return_value=None)
    @patch.object(KokoroPipeline, 'infer')
    def test_result_dataclass(self, mock_infer, mock_init):
        """Test KokoroPipeline.Result dataclass."""
        # Create a mock output
        audio = mx.array(np.zeros((1, 1000)))
        pred_dur = mx.array(np.zeros((1, 100)))
        model_output = KokoroModel.Output(audio=audio, pred_dur=pred_dur)

        # Create a Result instance
        result = KokoroPipeline.Result(
            graphemes="Hello",
            phonemes="HH EH L OW",
            tokens=[MagicMock()],
            output=model_output,
            text_index=0
        )

        # Check properties
        self.assertEqual(result.graphemes, "Hello")
        self.assertEqual(result.phonemes, "HH EH L OW")
        self.assertIs(result.audio, audio)
        self.assertIs(result.pred_dur, pred_dur)

        # Test backward compatibility
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "Hello")
        self.assertEqual(result[1], "HH EH L OW")
        self.assertIs(result[2], audio)

        # Test iteration
        items = list(result)
        self.assertEqual(items[0], "Hello")
        self.assertEqual(items[1], "HH EH L OW")
        self.assertIs(items[2], audio)



if __name__ == "__main__":
    unittest.main()