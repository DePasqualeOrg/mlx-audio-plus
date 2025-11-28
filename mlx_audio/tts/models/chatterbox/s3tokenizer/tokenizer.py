# Ported from https://github.com/xingchensong/S3Tokenizer

import hashlib
import os
import urllib.request
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .model import AudioEncoderV2, FSQVectorQuantization
from .utils import merge_tokenized_segments

# S3Tokenizer model URLs and checksums (from original s3tokenizer package)
_MODELS = {
    "speech_tokenizer_v1": (
        "https://www.modelscope.cn/models/iic/cosyvoice-300m/resolve/master/speech_tokenizer_v1.onnx",
        "23b5a723ed9143aebfd9ffda14ac4c21231f31c35ef837b6a13bb9e5488abb1e",
    ),
    "speech_tokenizer_v1_25hz": (
        "https://www.modelscope.cn/models/iic/CosyVoice-300M-25Hz/resolve/master/speech_tokenizer_v1.onnx",
        "56285ddd4a83e883ee0cb9f8d69c1089b53a94b1f78ff7e4a0224a27eb4cb486",
    ),
    "speech_tokenizer_v2_25hz": (
        "https://www.modelscope.cn/models/iic/CosyVoice2-0.5B/resolve/master/speech_tokenizer_v2.onnx",
        "d43342aa12163a80bf07bffb94c9de2e120a8df2f9917cd2f642e7f4219c6f71",
    ),
}


def _download_onnx(name: str, cache_dir: Optional[str] = None) -> str:
    """
    Download S3Tokenizer ONNX checkpoint.

    Args:
        name: Model name (e.g., "speech_tokenizer_v2_25hz")
        cache_dir: Directory to cache downloads (default: ~/.cache/s3tokenizer)

    Returns:
        Path to downloaded ONNX file
    """
    if name not in _MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(_MODELS.keys())}")

    url, expected_sha256 = _MODELS[name]

    if cache_dir is None:
        cache_dir = os.path.join(
            os.getenv("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")),
            "s3tokenizer"
        )

    os.makedirs(cache_dir, exist_ok=True)
    download_target = os.path.join(cache_dir, f"{name}.onnx")

    # Check if already downloaded with correct checksum
    if os.path.isfile(download_target):
        with open(download_target, "rb") as f:
            model_bytes = f.read()
        if hashlib.sha256(model_bytes).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(
                f"{download_target} exists but SHA256 mismatch; re-downloading"
            )

    # Download
    print(f"Downloading {name} from ModelScope...")
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        total = int(source.info().get("Content-Length", 0))
        downloaded = 0
        block_size = 8192
        while True:
            buffer = source.read(block_size)
            if not buffer:
                break
            output.write(buffer)
            downloaded += len(buffer)
            if total > 0:
                percent = downloaded * 100 // total
                print(f"\rDownloading: {percent}%", end="", flush=True)
        print()  # newline

    # Verify checksum
    with open(download_target, "rb") as f:
        model_bytes = f.read()
    if hashlib.sha256(model_bytes).hexdigest() != expected_sha256:
        os.remove(download_target)
        raise RuntimeError("Downloaded file failed SHA256 verification")

    return download_target


def _onnx_to_mlx_weights(onnx_path: str) -> Dict[str, mx.array]:
    """
    Convert ONNX weights to MLX format.

    This uses the s3tokenizer package's onnx2torch utility if available,
    otherwise falls back to direct ONNX parsing.

    Args:
        onnx_path: Path to ONNX file

    Returns:
        Dictionary of MLX arrays
    """
    try:
        # Try using s3tokenizer's conversion utility
        from s3tokenizer.utils import onnx2torch
        import torch

        pytorch_weights = onnx2torch(onnx_path, None, False)

        # Convert PyTorch tensors to MLX arrays
        mlx_weights = {}
        for key, value in pytorch_weights.items():
            if isinstance(value, torch.Tensor):
                mlx_weights[key] = mx.array(value.cpu().numpy())
            else:
                mlx_weights[key] = mx.array(value)

        return mlx_weights

    except ImportError:
        # Fallback: direct ONNX parsing
        try:
            import onnx
            from onnx import numpy_helper

            model = onnx.load(onnx_path)
            mlx_weights = {}

            for initializer in model.graph.initializer:
                array = numpy_helper.to_array(initializer)
                mlx_weights[initializer.name] = mx.array(array)

            return mlx_weights

        except ImportError:
            raise ImportError(
                "Either 's3tokenizer' or 'onnx' package required to load S3Tokenizer weights. "
                "Install with: pip install s3tokenizer  OR  pip install onnx"
            )


class S3TokenizerV2(nn.Module):
    """S3 Tokenizer V2 implementation for MLX (inference-only).

    Converts mel spectrograms to discrete speech tokens using an audio encoder
    and finite scalar quantization.

    Args:
        name: Model name (e.g., "speech_tokenizer_v2_25hz")
        config: Model configuration
    """

    def __init__(self, name: str = "speech_tokenizer_v2_25hz", config: ModelConfig = None):
        super().__init__()
        self.name = name

        if config is None:
            config = ModelConfig()

        # Ensure V2 uses FSQ with 3^8 codebook
        if 'v1' not in name:
            assert 'v2' in name
            config.n_codebook_size = 3**8

        self.config = config

        # Audio encoder
        self.encoder = AudioEncoderV2(
            self.config.n_mels,
            self.config.n_audio_state,
            self.config.n_audio_head,
            self.config.n_audio_layer,
            stride=2,
        )

        # FSQ quantizer
        self.quantizer = FSQVectorQuantization(
            self.config.n_audio_state,
            self.config.n_codebook_size,
        )

    @classmethod
    def from_pretrained(
        cls,
        name: str = "speech_tokenizer_v2_25hz",
        cache_dir: Optional[str] = None,
    ) -> "S3TokenizerV2":
        """
        Load pretrained S3TokenizerV2 from ModelScope.

        Downloads ONNX weights and converts them to MLX format.

        Args:
            name: Model name. Options:
                - "speech_tokenizer_v2_25hz" (default, recommended)
                - "speech_tokenizer_v1_25hz"
                - "speech_tokenizer_v1"
            cache_dir: Directory to cache downloads (default: ~/.cache/s3tokenizer)

        Returns:
            Initialized S3TokenizerV2 with loaded weights
        """
        # Download ONNX checkpoint
        onnx_path = _download_onnx(name, cache_dir)

        # Convert ONNX to MLX weights
        weights = _onnx_to_mlx_weights(onnx_path)

        # Create model
        model = cls(name=name)

        # Sanitize and load weights
        weights = model.sanitize(weights)
        # Use strict=False because ONNX conversion may have extra/missing keys
        # For best results, use bundled weights from s3gen.safetensors instead
        model.load_weights(list(weights.items()), strict=False)

        return model

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """
        Sanitize PyTorch weights for MLX.

        Handles:
        - Conv1d weight transposition (PyTorch: out, in, kernel -> MLX: out, kernel, in)
        - MLP key renaming (PyTorch mlp.0/mlp.2 -> MLX mlp.layers.0/mlp.layers.2)
        - Skipping computed buffers (_mel_filters, freqs_cis)
        - Removing unused keys (onnx::* intermediate nodes)

        This method is idempotent - it checks shapes before transposing to support
        both PyTorch-format and pre-converted MLX-format weights.
        """
        from mlx.utils import tree_flatten
        import re

        new_weights = {}

        # Get expected shapes from model for idempotent transposition
        curr_weights = dict(tree_flatten(self.parameters()))

        for key, value in weights.items():
            new_key = key

            # Skip computed buffers and dynamic embeddings
            if "freqs_cis" in key or "_mel_filters" in key:
                continue

            # Skip ONNX intermediate nodes (raw ONNX weight names)
            if key.startswith("onnx::"):
                continue

            # Quantizer key mappings:
            # - quantizer._codebook -> quantizer.codebook (PyTorch private attr)
            # - quantizer.fsq_codebook -> quantizer.codebook (existing HuggingFace format)
            new_key = new_key.replace('quantizer._codebook.', 'quantizer.codebook.')
            new_key = new_key.replace('quantizer.fsq_codebook.', 'quantizer.codebook.')

            # PyTorch Sequential uses mlp.0, mlp.2; MLX uses mlp.layers.0, mlp.layers.2
            new_key = re.sub(r'\.mlp\.(\d+)\.', r'.mlp.layers.\1.', new_key)

            # Conv1d weights need transposition (idempotent)
            # Only transpose if shape doesn't match expected MLX format
            if ".conv1." in new_key or ".conv2." in new_key or ".fsmn_block." in new_key:
                if "weight" in new_key and value.ndim == 3:
                    # PyTorch Conv1d: (out_channels, in_channels, kernel_size)
                    # MLX Conv1d: (out_channels, kernel_size, in_channels)
                    if new_key in curr_weights and value.shape != curr_weights[new_key].shape:
                        value = value.swapaxes(1, 2)

            new_weights[new_key] = value

        return new_weights

    def __call__(
        self,
        mel: mx.array,
        mel_len: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """Quantize mel spectrogram to tokens."""
        return self.quantize(mel, mel_len)

    def quantize(
        self,
        mel: mx.array,
        mel_len: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """
        Quantize mel spectrogram to tokens, with automatic long audio handling.

        Args:
            mel: Mel spectrogram tensor (B, n_mels, T)
            mel_len: Mel length tensor (B,)

        Returns:
            code: Quantized tokens (B, T')
            code_len: Token length (B,)
        """
        # Check if any audio exceeds 30 seconds
        # At 16kHz with hop_length=160: 30s = 3000 frames
        max_frames = 3000
        long_audio_mask = mel_len > max_frames

        if mx.any(long_audio_mask):
            # Has long audio - need special processing
            return self._quantize_mixed_batch(mel, mel_len, long_audio_mask, max_frames)
        else:
            # All short audio - use simple path
            hidden, code_len = self.encoder(mel, mel_len)
            code = self.quantizer.encode(hidden)
            return code, code_len

    def _quantize_mixed_batch(
        self,
        mel: mx.array,
        mel_len: mx.array,
        long_audio_mask: mx.array,
        max_frames: int
    ) -> Tuple[mx.array, mx.array]:
        """
        Handle mixed batch with both short and long audio.

        For long audio, uses sliding window approach with 30s windows
        and 4s overlap.
        """
        batch_size = mel.shape[0]

        # Sliding window parameters
        sample_rate = 16000
        hop_length = 160
        window_size = 30  # seconds
        overlap = 4  # seconds

        frames_per_window = window_size * sample_rate // hop_length  # 3000
        frames_per_overlap = overlap * sample_rate // hop_length  # 400
        frames_per_stride = frames_per_window - frames_per_overlap  # 2600

        # Collect all segments
        all_segments = []
        all_segments_len = []
        segment_info = []

        for batch_idx in range(batch_size):
            audio_mel = mel[batch_idx]
            audio_mel_len = int(mel_len[batch_idx].item())
            is_long_audio = bool(long_audio_mask[batch_idx].item())

            if not is_long_audio:
                # Short audio: process as single segment
                segment = audio_mel[:, :audio_mel_len]
                seg_len = audio_mel_len

                if seg_len < frames_per_window:
                    pad_size = frames_per_window - seg_len
                    segment = mx.pad(segment, [(0, 0), (0, pad_size)])

                all_segments.append(segment)
                all_segments_len.append(seg_len)
                segment_info.append({
                    'batch_idx': batch_idx,
                    'is_long_audio': False,
                    'segment_idx': 0,
                    'total_segments': 1
                })
            else:
                # Long audio: split into segments
                start = 0
                segment_idx = 0
                while start < audio_mel_len:
                    end = min(start + frames_per_window, audio_mel_len)
                    segment = audio_mel[:, start:end]
                    seg_len = segment.shape[1]

                    if seg_len < frames_per_window:
                        pad_size = frames_per_window - seg_len
                        segment = mx.pad(segment, [(0, 0), (0, pad_size)])

                    all_segments.append(segment)
                    all_segments_len.append(seg_len)
                    segment_info.append({
                        'batch_idx': batch_idx,
                        'is_long_audio': True,
                        'segment_idx': segment_idx,
                        'total_segments': None
                    })

                    segment_idx += 1
                    start += frames_per_stride

                # Update total_segments
                total_segments = segment_idx
                for info in segment_info:
                    if info['batch_idx'] == batch_idx and info['is_long_audio']:
                        info['total_segments'] = total_segments

        if not all_segments:
            return (
                mx.zeros((batch_size, 0), dtype=mx.int32),
                mx.zeros((batch_size,), dtype=mx.int32)
            )

        # Process all segments
        unified_batch_mel = mx.stack(all_segments)
        unified_batch_lens = mx.array(all_segments_len, dtype=mx.int32)

        hidden, code_len = self.encoder(unified_batch_mel, unified_batch_lens)
        codes = self.quantizer.encode(hidden)

        # Reorganize results
        results = {}

        for seg_idx, info in enumerate(segment_info):
            batch_idx = info['batch_idx']
            is_long_audio = info['is_long_audio']

            seg_code_len = int(code_len[seg_idx].item())
            segment_code = codes[seg_idx, :seg_code_len].tolist()

            if not is_long_audio:
                results[batch_idx] = (mx.array(segment_code, dtype=mx.int32), len(segment_code))
            else:
                if batch_idx not in results:
                    results[batch_idx] = []
                results[batch_idx].append(segment_code)

        # Merge long audio segments
        for batch_idx in range(batch_size):
            if bool(long_audio_mask[batch_idx].item()):
                audio_codes = results[batch_idx]
                token_rate = 25  # V2 uses 25Hz
                merged_codes = merge_tokenized_segments(audio_codes, overlap=overlap, token_rate=token_rate)
                results[batch_idx] = (mx.array(merged_codes, dtype=mx.int32), len(merged_codes))

        # Build output
        max_code_len = max(info[1] for info in results.values())

        output_codes = mx.zeros((batch_size, max_code_len), dtype=mx.int32)
        output_codes_len = mx.zeros((batch_size,), dtype=mx.int32)

        for batch_idx, (code_tensor, code_len_val) in results.items():
            # MLX doesn't support item assignment, so we need to rebuild
            pass  # Will handle this differently

        # Rebuild using list comprehension
        output_list = []
        len_list = []
        for batch_idx in range(batch_size):
            code_tensor, code_len_val = results[batch_idx]
            if code_tensor.shape[0] < max_code_len:
                code_tensor = mx.pad(code_tensor, [(0, max_code_len - code_tensor.shape[0])])
            output_list.append(code_tensor)
            len_list.append(code_len_val)

        output_codes = mx.stack(output_list)
        output_codes_len = mx.array(len_list, dtype=mx.int32)

        return output_codes, output_codes_len

    def quantize_simple(
        self,
        mel: mx.array,
        mel_len: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """
        Simple quantization without long audio handling.

        Use this for audio clips under 30 seconds.
        """
        hidden, code_len = self.encoder(mel, mel_len)
        code = self.quantizer.encode(hidden)
        return code, code_len
