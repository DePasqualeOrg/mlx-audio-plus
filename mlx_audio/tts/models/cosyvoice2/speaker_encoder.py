# CAMPlus Speaker Encoder for CosyVoice2
# Uses the existing MLX CAMPlus implementation from Chatterbox

from typing import Optional

import mlx.core as mx

# Import the existing MLX CAMPlus implementation
from ..chatterbox.s3gen.xvector import CAMPPlus, kaldi_fbank


class CAMPlusSpeakerEncoder:
    """
    CAMPlus speaker encoder that extracts 192-dim speaker embeddings.

    Uses the pure MLX implementation from Chatterbox (no ONNX dependency).
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize CAMPlus speaker encoder.

        Args:
            model_path: Path to CAMPlus weights file (.safetensors or .npz).
                       If None, model will be initialized but not loaded.
        """
        self.embedding_dim = 192
        self.model = CAMPPlus(
            feat_dim=80,
            embedding_size=self.embedding_dim,
            growth_rate=32,
            bn_size=4,
            init_channels=128,
            config_str="batchnorm-relu",
            memory_efficient=True,
            output_level="segment",
        )
        self._loaded = False

        if model_path is not None:
            self.load_weights(model_path)

    def load_weights(self, model_path: str) -> None:
        """
        Load model weights from file.

        Args:
            model_path: Path to weights file (.safetensors, .npz, or directory)
        """
        from pathlib import Path

        import mlx.core as mx

        path = Path(model_path)

        if path.is_dir():
            # Look for campplus weights in directory
            safetensors_path = path / "campplus.safetensors"
            npz_path = path / "campplus.npz"
            if safetensors_path.exists():
                path = safetensors_path
            elif npz_path.exists():
                path = npz_path
            else:
                print(
                    f"Warning: No campplus weights found in {model_path}. "
                    "Speaker embeddings will be zeros."
                )
                return

        try:
            if str(path).endswith(".safetensors"):
                weights = mx.load(str(path))
            elif str(path).endswith(".npz"):
                weights = dict(mx.load(str(path)))
            else:
                # Try loading as safetensors first, then npz
                try:
                    weights = mx.load(str(path))
                except Exception:
                    weights = dict(mx.load(str(path)))

            # Sanitize weights (handles PyTorch -> MLX conversion)
            weights = self.model.sanitize(weights)
            self.model.load_weights(list(weights.items()))
            # Set eval mode for inference (BatchNorm uses running stats)
            self.model.eval()
            self._loaded = True
        except Exception as e:
            print(
                f"Warning: Failed to load CAMPlus weights from {model_path}: {e}. "
                "Speaker embeddings will be zeros."
            )

    def __call__(self, audio: mx.array, sample_rate: int = 16000) -> mx.array:
        """
        Extract speaker embedding from audio.

        Args:
            audio: Audio waveform at 16kHz (can be (T,) or (B, T))
            sample_rate: Sample rate (should be 16000)

        Returns:
            Speaker embedding (1, 192) or (B, 192)
        """
        if not self._loaded:
            # Return zeros if model not loaded
            if audio.ndim == 1:
                return mx.zeros((1, self.embedding_dim))
            else:
                return mx.zeros((audio.shape[0], self.embedding_dim))

        # Ensure 1D or 2D
        if audio.ndim > 2:
            audio = audio.squeeze()

        # Use the model's inference method which handles fbank extraction
        embedding = self.model.inference(audio)

        # Ensure output is (B, embedding_dim)
        if embedding.ndim == 1:
            embedding = mx.expand_dims(embedding, 0)

        return embedding

    def extract_fbank(self, audio: mx.array, sample_rate: int = 16000) -> mx.array:
        """
        Extract 80-dim filter bank features from audio.

        This uses the Kaldi-compatible fbank implementation from Chatterbox.

        Args:
            audio: Audio waveform (samples,) at 16kHz
            sample_rate: Sample rate (should be 16000)

        Returns:
            Filter bank features (T, 80)
        """
        # Ensure 1D
        if audio.ndim > 1:
            audio = audio.squeeze()

        return kaldi_fbank(audio, sample_rate=sample_rate, num_mel_bins=80)
