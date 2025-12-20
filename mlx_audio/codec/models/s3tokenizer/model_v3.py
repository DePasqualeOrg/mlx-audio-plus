# Copyright © Xingchen Song (original model implementation)
# Copyright © Anthony DePasquale (MLX port)
# Ported to MLX from https://github.com/xingchensong/S3Tokenizer
# License: licenses/s3tokenizer.txt

"""
S3TokenizerV3 implementation for MLX.

V3 is architecturally identical to V2, but with 12 transformer blocks instead of 6.
This was reverse-engineered from speech_tokenizer_v3.onnx.

Architecture:
- n_mels: 128
- n_audio_state: 1280
- n_audio_head: 20
- n_audio_layer: 12 (vs 6 in V2)
- n_codebook_size: 6561 (3^8)
- FSMN kernel_size: 31
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten

from .model_v2 import (
    AudioEncoderV2,
    FSMNMultiHeadAttention,
    FSQVectorQuantization,
    ResidualAttentionBlock,
    apply_rotary_emb,
    precompute_freqs_cis,
)
from .utils import make_non_pad_mask, mask_to_bias, merge_tokenized_segments


@dataclass
class ModelConfigV3:
    """Configuration for S3TokenizerV3."""

    n_mels: int = 128
    n_audio_ctx: int = 1500
    n_audio_state: int = 1280
    n_audio_head: int = 20
    n_audio_layer: int = 12  # V3 has 12 layers (vs 6 in V2)
    n_codebook_size: int = 3**8  # 6561


class AudioEncoderV3(nn.Module):
    """Audio encoder for V3 with 12 transformer blocks."""

    def __init__(
        self,
        n_mels: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        stride: int,
    ):
        super().__init__()
        self.stride = stride

        self.conv1 = nn.Conv1d(
            in_channels=n_mels,
            out_channels=n_state,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=n_state,
            out_channels=n_state,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # Precompute rotary embeddings
        self._freqs_cis = precompute_freqs_cis(64, 1024 * 2)

        # 12 transformer blocks for V3
        self.blocks = [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]

    def __call__(self, x: mx.array, x_len: mx.array) -> Tuple[mx.array, mx.array]:
        """
        x : mx.array, shape = (batch_size, n_mels, T)
            the mel spectrogram of the audio
        x_len: mx.array, shape = (batch_size,)
            length of each audio in x
        """
        T = x.shape[-1]
        mask = make_non_pad_mask(x_len, T)
        mask = mx.expand_dims(mask, axis=1)  # (B, 1, T)

        x = x.transpose(0, 2, 1)  # (B, T, n_mels)
        mask_transposed = mask.transpose(0, 2, 1)  # (B, T, 1)

        x = self.conv1(x * mask_transposed)
        x = nn.gelu(x)
        x_len = (x_len + 2 - 1 * (3 - 1) - 1) // self.stride + 1
        x_slen = (T + 2 - 1 * (3 - 1) - 1) // self.stride + 1

        mask = make_non_pad_mask(x_len, x_slen)
        mask_transposed = mx.expand_dims(mask, axis=-1)  # (B, T, 1)

        x = self.conv2(x * mask_transposed)
        x = nn.gelu(x)
        x_len = (x_len + 2 - 1 * (3 - 1) - 1) // 2 + 1
        x_slen = (x_slen + 2 - 1 * (3 - 1) - 1) // 2 + 1

        mask = make_non_pad_mask(x_len, x_slen)
        mask_pad = mx.expand_dims(mask, axis=-1)  # (B, T, 1)
        mask = mask_to_bias(mask, x.dtype)
        mask = mx.expand_dims(mask, axis=1)  # (B, 1, T)

        for block in self.blocks:
            x = block(x, mask, mask_pad, self._freqs_cis)

        return x, x_len


class S3TokenizerV3(nn.Module):
    """S3 tokenizer V3 implementation for MLX.

    V3 has the same architecture as V2 but with 12 transformer blocks
    instead of 6.

    Args:
        config (ModelConfigV3): Model configuration
    """

    def __init__(
        self, name: str = "speech_tokenizer_v3", config: ModelConfigV3 = ModelConfigV3()
    ):
        super().__init__()
        self.config = config
        self.encoder = AudioEncoderV3(
            self.config.n_mels,
            self.config.n_audio_state,
            self.config.n_audio_head,
            self.config.n_audio_layer,
            2,  # stride
        )
        self.quantizer = FSQVectorQuantization(
            self.config.n_audio_state,
            self.config.n_codebook_size,
        )

    def __call__(self, mel: mx.array, mel_len: mx.array) -> Tuple[mx.array, mx.array]:
        return self.quantize(mel, mel_len)

    def quantize(self, mel: mx.array, mel_len: mx.array) -> Tuple[mx.array, mx.array]:
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
        max_frames: int,
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
                segment_info.append(
                    {
                        "batch_idx": batch_idx,
                        "is_long_audio": False,
                        "segment_idx": 0,
                        "total_segments": 1,
                    }
                )
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
                    segment_info.append(
                        {
                            "batch_idx": batch_idx,
                            "is_long_audio": True,
                            "segment_idx": segment_idx,
                            "total_segments": None,
                        }
                    )

                    segment_idx += 1
                    start += frames_per_stride

                # Update total_segments
                total_segments = segment_idx
                for info in segment_info:
                    if info["batch_idx"] == batch_idx and info["is_long_audio"]:
                        info["total_segments"] = total_segments

        if not all_segments:
            return (
                mx.zeros((batch_size, 0), dtype=mx.int32),
                mx.zeros((batch_size,), dtype=mx.int32),
            )

        # Process all segments
        unified_batch_mel = mx.stack(all_segments)
        unified_batch_lens = mx.array(all_segments_len, dtype=mx.int32)

        hidden, code_len = self.encoder(unified_batch_mel, unified_batch_lens)
        codes = self.quantizer.encode(hidden)

        # Reorganize results
        results = {}

        for seg_idx, info in enumerate(segment_info):
            batch_idx = info["batch_idx"]
            is_long_audio = info["is_long_audio"]

            seg_code_len = int(code_len[seg_idx].item())
            segment_code = codes[seg_idx, :seg_code_len].tolist()

            if not is_long_audio:
                results[batch_idx] = (
                    mx.array(segment_code, dtype=mx.int32),
                    len(segment_code),
                )
            else:
                if batch_idx not in results:
                    results[batch_idx] = []
                results[batch_idx].append(segment_code)

        # Merge long audio segments
        for batch_idx in range(batch_size):
            if bool(long_audio_mask[batch_idx].item()):
                audio_codes = results[batch_idx]
                token_rate = 25  # V3 uses 25Hz like V2
                merged_codes = merge_tokenized_segments(
                    audio_codes, overlap=overlap, token_rate=token_rate
                )
                results[batch_idx] = (
                    mx.array(merged_codes, dtype=mx.int32),
                    len(merged_codes),
                )

        # Build output
        max_code_len = max(info[1] for info in results.values())

        # Rebuild using list comprehension
        output_list = []
        len_list = []
        for batch_idx in range(batch_size):
            code_tensor, code_len_val = results[batch_idx]
            if code_tensor.shape[0] < max_code_len:
                code_tensor = mx.pad(
                    code_tensor, [(0, max_code_len - code_tensor.shape[0])]
                )
            output_list.append(code_tensor)
            len_list.append(code_len_val)

        output_codes = mx.stack(output_list)
        output_codes_len = mx.array(len_list, dtype=mx.int32)

        return output_codes, output_codes_len

    def quantize_simple(
        self, mel: mx.array, mel_len: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """
        Simple quantization without long audio handling.

        Use this for audio clips under 30 seconds.
        """
        hidden, code_len = self.encoder(mel, mel_len)
        code = self.quantizer.encode(hidden)
        return code, code_len

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Sanitize weights from ONNX format to MLX format."""
        new_weights = {}

        # Get expected shapes from model for idempotent transposition
        curr_weights = dict(tree_flatten(self.parameters()))

        for key, value in weights.items():
            new_key = key

            # Skip computed buffers and dynamic embeddings
            if "freqs_cis" in key or "_mel_filters" in key:
                continue

            # Skip ONNX intermediate nodes
            if key.startswith("onnx::"):
                continue

            # Quantizer key mappings
            new_key = new_key.replace("quantizer._codebook.", "quantizer.fsq_codebook.")
            new_key = new_key.replace("quantizer.codebook.", "quantizer.fsq_codebook.")

            # PyTorch Sequential uses mlp.0, mlp.2; MLX uses mlp.layers.0, mlp.layers.2
            new_key = re.sub(r"\.mlp\.(\d+)\.", r".mlp.layers.\1.", new_key)

            # Conv1d weights need transposition (idempotent)
            if (
                ".conv1." in new_key
                or ".conv2." in new_key
                or ".fsmn_block." in new_key
            ):
                if "weight" in new_key and value.ndim == 3:
                    # PyTorch Conv1d: (out_channels, in_channels, kernel_size)
                    # MLX Conv1d: (out_channels, kernel_size, in_channels)
                    if (
                        new_key in curr_weights
                        and value.shape != curr_weights[new_key].shape
                    ):
                        value = value.swapaxes(1, 2)

            new_weights[new_key] = value

        return new_weights

    @classmethod
    def from_pretrained(
        cls,
        name: str = "speech_tokenizer_v3",
        repo_id: str = "mlx-community/S3TokenizerV3",
    ) -> "S3TokenizerV3":
        """Load a pretrained S3TokenizerV3 model."""
        path = fetch_from_hub(repo_id)
        if path is None:
            raise ValueError(f"Could not find model {path}")

        model = S3TokenizerV3(name)
        model_path = path / "model.safetensors"
        weights = mx.load(model_path.as_posix(), format="safetensors")
        model.load_weights(list(weights.items()))
        mx.eval(model.parameters())

        return model

    @classmethod
    def from_onnx(cls, onnx_path: str) -> "S3TokenizerV3":
        """
        Load S3TokenizerV3 from an ONNX file.

        This extracts weights from the ONNX model and loads them into MLX.

        Args:
            onnx_path: Path to speech_tokenizer_v3.onnx

        Returns:
            S3TokenizerV3 model loaded with weights
        """
        import numpy as np
        import onnx

        onnx_model = onnx.load(onnx_path)

        # Extract weights from ONNX
        weights = {}
        initializer_map = {init.name: init for init in onnx_model.graph.initializer}

        for node in onnx_model.graph.node:
            for input_name in node.input:
                if input_name not in initializer_map:
                    continue

                initializer = initializer_map[input_name]
                weight_array = onnx.numpy_helper.to_array(initializer).copy()

                # Map ONNX names to our model names
                weight_name = _map_onnx_name_v3(input_name, node)
                if weight_name:
                    # Handle LayerNorm specially
                    if node.op_type == "LayerNormalization":
                        ln_name = node.name.replace("/LayerNormalization", "")
                        ln_inputs = node.input
                        scale_name = ln_inputs[1]
                        bias_name = ln_inputs[2]

                        if input_name == scale_name:
                            weights[ln_name + ".weight"] = mx.array(weight_array)
                        elif input_name == bias_name:
                            weights[ln_name + ".bias"] = mx.array(weight_array)
                    else:
                        # Transpose linear weights
                        if len(weight_array.shape) == 2:
                            weight_array = weight_array.T
                        weights[weight_name] = mx.array(weight_array)

        # Create model and load weights
        model = cls()
        weights = model.sanitize(weights)
        model.load_weights(list(weights.items()))
        mx.eval(model.parameters())

        return model


def _map_onnx_name_v3(input_name: str, node) -> Optional[str]:
    """Map ONNX weight names to MLX model names for V3."""
    # Conv weights
    if input_name == "onnx::Conv_3986":
        return "encoder.conv1.weight"
    elif input_name == "onnx::Conv_3987":
        return "encoder.conv1.bias"
    elif input_name == "onnx::Conv_3988":
        return "encoder.conv2.weight"
    elif input_name == "onnx::Conv_3989":
        return "encoder.conv2.bias"

    # Quantizer - project_down layer (Linear 1280 -> 8)
    if input_name == "quantizer.project_in.bias":
        return "quantizer.fsq_codebook.project_down.bias"
    if input_name == "onnx::MatMul_4618":
        # This is the quantizer's project_down weight (shape: [1280, 8])
        return "quantizer.fsq_codebook.project_down.weight"
    if "project_down" in input_name or "project_in" in input_name:
        if "weight" in input_name.lower() or "MatMul" in node.name:
            return "quantizer.fsq_codebook.project_down.weight"

    # Block weights - handled by node name
    if "blocks" in input_name:
        # These are FSMN weights with explicit names
        return "encoder." + input_name

    # Use node name for transformer block weights
    if "blocks" in node.name:
        new_name = (
            node.name[1:]
            .replace("/", ".")
            .replace("MatMul", "weight")
            .replace("Add_1", "bias")
            .replace("Mul", "weight")
            .replace("Add", "bias")
            .replace("mlp.mlp", "mlp")
            .replace("fsmn_block.Conv", "fsmn_block.weight")
        )
        return f"encoder.{new_name}"

    return None


def fetch_from_hub(hf_repo: str) -> Path:
    """Fetch model from Hugging Face Hub."""
    model_path = Path(
        snapshot_download(
            repo_id=hf_repo,
            allow_patterns=["*.safetensors", "*.json"],
        )
    )
    return model_path
