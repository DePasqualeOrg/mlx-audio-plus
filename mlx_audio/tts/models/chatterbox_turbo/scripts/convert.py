#!/usr/bin/env python3
# Copyright © Anthony DePasquale
"""
Convert Chatterbox Turbo weights from PyTorch to MLX format.

All weights are combined into a single model.safetensors file with prefixes:
- ve.* : VoiceEncoder weights
- t3.* : T3 Turbo model (GPT2 backbone + conditioning + embeddings)
- s3gen.* : S3Gen decoder (flow matching + HiFi-GAN vocoder)

The S3Tokenizer is converted separately and uploaded to its own repo, as it's
shared between multiple TTS models (Chatterbox, Chatterbox Turbo, CosyVoice 2, etc.).

Quantization:
- Only T3 GPT2 transformer layers (t3.tfmr.h.*) are quantized
- VoiceEncoder, S3Gen, embeddings, and output heads remain in full precision
- This preserves audio quality while reducing model size

Usage:
    # From the mlx-audio-plus root directory:

    # Convert Chatterbox Turbo to fp16
    python mlx_audio/tts/models/chatterbox_turbo/scripts/convert.py

    # Convert to 4-bit quantized
    python mlx_audio/tts/models/chatterbox_turbo/scripts/convert.py --quantize

    # Convert with custom output directory
    python mlx_audio/tts/models/chatterbox_turbo/scripts/convert.py \\
        --output-dir ./my-chatterbox-turbo

    # Upload to Hugging Face
    python mlx_audio/tts/models/chatterbox_turbo/scripts/convert.py \\
        --quantize --upload-repo mlx-community/chatterbox-turbo-4bit

Requirements (for conversion only):
    pip install torch safetensors huggingface_hub mlx

After conversion, the model only needs:
    pip install mlx mlx-lm
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

# Add mlx-audio-plus root to path for local imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "ResembleAI/chatterbox-turbo"


def download_chatterbox_turbo_weights(cache_dir: Path = None) -> Path:
    """Download Chatterbox Turbo weights from Hugging Face.

    Args:
        cache_dir: Optional custom cache directory. If None, uses the default
            HF cache (~/.cache/huggingface/hub).
    """
    from huggingface_hub import snapshot_download

    logger.info("Downloading Chatterbox Turbo weights from Hugging Face...")
    ckpt_dir = Path(
        snapshot_download(
            repo_id=DEFAULT_MODEL_ID,
            allow_patterns=[
                "ve.safetensors",
                "t3_turbo_v1.safetensors",
                "s3gen_meanflow.safetensors",
                "conds.pt",
                # Tokenizer files
                "vocab.json",
                "merges.txt",
                "added_tokens.json",
                "special_tokens_map.json",
                "tokenizer_config.json",
            ],
            cache_dir=cache_dir,
        )
    )
    logger.info(f"Downloaded to: {ckpt_dir}")
    return ckpt_dir


def load_pytorch_safetensors(path: Path) -> Dict[str, np.ndarray]:
    """Load PyTorch safetensors and convert to numpy."""
    import torch
    from safetensors.torch import load_file

    state_dict = load_file(path)
    return {k: v.cpu().numpy() for k, v in state_dict.items()}


def load_pytorch_checkpoint(path: Path) -> Dict[str, np.ndarray]:
    """Load PyTorch .pt file and convert to numpy.

    Handles nested dictionaries by flattening with dot-separated keys.
    Skips None values.
    """
    import torch

    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    result = {}

    def flatten(d: dict, prefix: str = ""):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, torch.Tensor):
                result[key] = v.detach().cpu().numpy()
            elif isinstance(v, dict):
                flatten(v, key)
            elif v is None:
                # Skip None values
                pass
            else:
                result[key] = np.array(v)

    flatten(state_dict)
    return result


def numpy_to_mlx(weights: Dict[str, np.ndarray]) -> Dict:
    """Convert numpy arrays to MLX arrays for sanitization."""
    import mlx.core as mx

    return {k: mx.array(v) for k, v in weights.items()}


def remap_voice_encoder_keys(weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Remap PyTorch VoiceEncoder LSTM weights to MLX format.

    PyTorch uses stacked LSTM: lstm.weight_ih_l0, lstm.weight_hh_l0, etc.
    MLX uses separate LSTMs: lstm1.Wx, lstm1.Wh, lstm1.bias, etc.

    PyTorch and MLX both use shape (4*hidden, input) for LSTM weights.
    PyTorch LSTM gates order: input, forget, cell, output (i, f, g, o)
    MLX LSTM gates order: input, forget, output, cell (i, f, o, g)
    """
    new_weights = {}

    # Layer index mapping: l0 -> lstm1, l1 -> lstm2, l2 -> lstm3
    layer_map = {"l0": "lstm1", "l1": "lstm2", "l2": "lstm3"}

    # Track which bias pairs we've processed
    bias_pairs = {}

    for key, value in weights.items():
        if "num_batches_tracked" in key:
            continue

        new_key = key

        # Handle LSTM weights
        if key.startswith("lstm."):
            for py_suffix, mlx_prefix in layer_map.items():
                if f"weight_ih_{py_suffix}" in key:
                    # No transpose needed - both PyTorch and MLX use (4*hidden, input)
                    # Reorder gates: PyTorch (i,f,g,o) -> MLX (i,f,o,g)
                    hidden_size = value.shape[0] // 4
                    gates = np.split(value, 4, axis=0)  # [i, f, g, o]
                    value = np.concatenate(
                        [gates[0], gates[1], gates[3], gates[2]], axis=0
                    )
                    new_key = f"{mlx_prefix}.Wx"
                    new_weights[new_key] = value
                    continue
                elif f"weight_hh_{py_suffix}" in key:
                    # No transpose needed - both use (4*hidden, hidden)
                    # Reorder gates
                    hidden_size = value.shape[0] // 4
                    gates = np.split(value, 4, axis=0)
                    value = np.concatenate(
                        [gates[0], gates[1], gates[3], gates[2]], axis=0
                    )
                    new_key = f"{mlx_prefix}.Wh"
                    new_weights[new_key] = value
                    continue
                elif f"bias_ih_{py_suffix}" in key:
                    # Store for later combining with bias_hh
                    if py_suffix not in bias_pairs:
                        bias_pairs[py_suffix] = {}
                    bias_pairs[py_suffix]["ih"] = value
                    continue
                elif f"bias_hh_{py_suffix}" in key:
                    if py_suffix not in bias_pairs:
                        bias_pairs[py_suffix] = {}
                    bias_pairs[py_suffix]["hh"] = value
                    continue
            # Skip unhandled LSTM keys
            continue

        # Pass through other keys unchanged
        new_weights[new_key] = value

    # Combine bias pairs
    for py_suffix, biases in bias_pairs.items():
        if "ih" in biases and "hh" in biases:
            combined = biases["ih"] + biases["hh"]
            # Reorder gates
            hidden_size = combined.shape[0] // 4
            gates = np.split(combined, 4)
            combined = np.concatenate([gates[0], gates[1], gates[3], gates[2]])
            mlx_prefix = layer_map[py_suffix]
            new_weights[f"{mlx_prefix}.bias"] = combined

    return new_weights


def mlx_to_numpy(weights: Dict) -> Dict[str, np.ndarray]:
    """Convert MLX arrays back to numpy for saving."""
    return {k: np.array(v) for k, v in weights.items()}


def remap_t3_keys(weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Remap PyTorch T3 (GPT-2 based) weights to MLX format.

    GPT-2 uses Conv1d for attention/MLP projections with weights (in, out, 1).
    When squeezed they become (in, out) but MLX Linear expects (out, in).
    """
    new_weights = {}

    for key, value in weights.items():
        new_key = key

        # Skip num_batches_tracked
        if "num_batches_tracked" in key:
            continue

        # Transpose GPT-2 Conv1d weights for attention and MLP
        if value.ndim == 2:
            # c_attn, c_proj, c_fc weights need transposition
            if ".attn.c_attn.weight" in key:
                value = value.T
            elif ".attn.c_proj.weight" in key:
                value = value.T
            elif ".mlp.c_fc.weight" in key:
                value = value.T
            elif ".mlp.c_proj.weight" in key:
                value = value.T

        new_weights[new_key] = value

    return new_weights


def remap_s3gen_keys(weights: Dict[str, np.ndarray], model) -> Dict[str, np.ndarray]:
    """
    Remap PyTorch S3Gen weight keys to MLX format.

    This transforms PyTorch weight keys to match the MLX model's expected structure:
    - flow.X -> X (strip flow. prefix)
    - down_blocks.N.0 -> down_blocks.N.resnet
    - down_blocks.N.1.M -> down_blocks.N.transformer_blocks.M
    - down_blocks.N.2 -> down_blocks.N.downsample
    - block.0 -> block.0.conv (CausalConv1d wrapper)
    - block.2 -> block.1 (skip Transpose layers)
    - mlp.1 -> mlp.0 (skip Mish activation)
    - speaker_encoder/CAMPPlus key remapping
    - HiFi-GAN weight normalization -> regular weights
    - Conv weight transposition
    """
    import re

    import mlx.core as mx
    from mlx.utils import tree_flatten

    # Get expected shapes from model
    curr_weights = dict(tree_flatten(model.parameters()))

    new_weights = {}

    # === Handle weight normalization parametrizations ===
    # PyTorch uses weight_norm with parametrizations: weight = g * v / ||v||
    # where g = original0 and v = original1
    weight_norm_pairs = {}
    for key in weights:
        if ".parametrizations.weight.original0" in key:
            base_key = key.replace(".parametrizations.weight.original0", "")
            if base_key not in weight_norm_pairs:
                weight_norm_pairs[base_key] = {}
            weight_norm_pairs[base_key]["g"] = weights[key]
        elif ".parametrizations.weight.original1" in key:
            base_key = key.replace(".parametrizations.weight.original1", "")
            if base_key not in weight_norm_pairs:
                weight_norm_pairs[base_key] = {}
            weight_norm_pairs[base_key]["v"] = weights[key]

    # Compute and add combined weights
    for base_key, params in weight_norm_pairs.items():
        if "g" in params and "v" in params:
            g = params["g"]  # (out_channels, 1, 1) or similar
            v = params["v"]  # (out_channels, in_channels, kernel) or similar
            # weight = g * v / ||v||
            # Compute norm over all dims except batch (dim 0)
            norm_dims = tuple(range(1, v.ndim))
            v_norm = np.linalg.norm(v, axis=norm_dims, keepdims=True)
            weight = g * v / (v_norm + 1e-8)
            weights[base_key + ".weight"] = weight

    for key, value in weights.items():
        new_key = key

        # Skip num_batches_tracked
        if "num_batches_tracked" in key:
            continue

        # Skip parametrizations keys (already combined above)
        if ".parametrizations." in key:
            continue

        # Strip 'flow.' prefix
        if new_key.startswith("flow."):
            new_key = new_key[5:]

        # === Speaker encoder (CAMPPlus) remapping ===
        if "speaker_encoder." in new_key:
            # Strip xvector. prefix within speaker_encoder
            new_key = new_key.replace("speaker_encoder.xvector.", "speaker_encoder.")

            # Map tdnn.nonlinear.batchnorm -> tdnn.bn
            new_key = new_key.replace("tdnn.nonlinear.batchnorm", "tdnn.bn")

            # Map block{N}.tdnnd{M} -> blocks.{N-1}.layers.{M-1}
            match = re.search(r"block(\d+)\.tdnnd(\d+)", new_key)
            if match:
                block_idx = int(match.group(1)) - 1
                layer_idx = int(match.group(2)) - 1
                old = f"block{match.group(1)}.tdnnd{match.group(2)}"
                new = f"blocks.{block_idx}.layers.{layer_idx}"
                new_key = new_key.replace(old, new)

            # Map transit{N} -> transits.{N-1}
            match = re.search(r"transit(\d+)", new_key)
            if match:
                transit_idx = int(match.group(1)) - 1
                new_key = new_key.replace(
                    f"transit{match.group(1)}", f"transits.{transit_idx}"
                )

            # Map nonlinear.batchnorm -> bn (for transit/tdnn)
            new_key = new_key.replace("nonlinear.batchnorm", "bn")
            # Map nonlinear1.batchnorm -> bn1
            new_key = new_key.replace("nonlinear1.batchnorm", "bn1")
            # Map nonlinear2.batchnorm -> bn2
            new_key = new_key.replace("nonlinear2.batchnorm", "bn2")
            # Map out_nonlinear.batchnorm -> out_bn
            new_key = new_key.replace("out_nonlinear.batchnorm", "out_bn")
            # Map dense.nonlinear.batchnorm -> dense.bn
            new_key = new_key.replace("dense.nonlinear.batchnorm", "dense.bn")

            # Map shortcut.0 -> shortcut_conv
            new_key = new_key.replace("shortcut.0", "shortcut_conv")
            # Map shortcut.1 -> shortcut_bn
            new_key = new_key.replace("shortcut.1", "shortcut_bn")

        # === Decoder block naming ===
        # down_blocks.X.0 -> down_blocks.X.resnet
        new_key = re.sub(r"down_blocks\.(\d+)\.0\.", r"down_blocks.\1.resnet.", new_key)
        # down_blocks.X.1.Y -> down_blocks.X.transformer_blocks.Y
        new_key = re.sub(
            r"down_blocks\.(\d+)\.1\.(\d+)\.",
            r"down_blocks.\1.transformer_blocks.\2.",
            new_key,
        )
        # down_blocks.X.2 -> down_blocks.X.downsample
        new_key = re.sub(
            r"down_blocks\.(\d+)\.2\.", r"down_blocks.\1.downsample.", new_key
        )

        # mid_blocks.X.0 -> mid_blocks.X.resnet
        new_key = re.sub(r"mid_blocks\.(\d+)\.0\.", r"mid_blocks.\1.resnet.", new_key)
        # mid_blocks.X.1.Y -> mid_blocks.X.transformer_blocks.Y
        new_key = re.sub(
            r"mid_blocks\.(\d+)\.1\.(\d+)\.",
            r"mid_blocks.\1.transformer_blocks.\2.",
            new_key,
        )

        # up_blocks.X.0 -> up_blocks.X.resnet
        new_key = re.sub(r"up_blocks\.(\d+)\.0\.", r"up_blocks.\1.resnet.", new_key)
        # up_blocks.X.1.Y -> up_blocks.X.transformer_blocks.Y
        new_key = re.sub(
            r"up_blocks\.(\d+)\.1\.(\d+)\.",
            r"up_blocks.\1.transformer_blocks.\2.",
            new_key,
        )
        # up_blocks.X.2 -> up_blocks.X.upsample
        new_key = re.sub(r"up_blocks\.(\d+)\.2\.", r"up_blocks.\1.upsample.", new_key)

        # === ResNet block structure ===
        # CausalConv1d wraps conv: block.0 -> block.0.conv.conv (two levels!)
        new_key = re.sub(
            r"\.block1\.block\.0\.", r".block1.block.0.conv.conv.", new_key
        )
        new_key = re.sub(
            r"\.block2\.block\.0\.", r".block2.block.0.conv.conv.", new_key
        )
        # LayerNorm: block.2 -> block.1 (skip Transpose at indices 1,3)
        new_key = re.sub(r"\.block1\.block\.2\.", r".block1.block.1.", new_key)
        new_key = re.sub(r"\.block2\.block\.2\.", r".block2.block.1.", new_key)
        # MLP: mlp.1 -> mlp.0 (Mish removed, Linear shifted)
        new_key = re.sub(r"\.mlp\.1\.", r".mlp.0.", new_key)
        # res_conv: add .conv wrapper
        new_key = re.sub(r"\.res_conv\.(weight|bias)$", r".res_conv.conv.\1", new_key)

        # === Downsample/Upsample conv wrapping ===
        # Downsample/upsample direct weights -> conv.conv wrapper (two levels)
        new_key = re.sub(
            r"\.downsample\.(weight|bias)$", r".downsample.conv.conv.\1", new_key
        )
        new_key = re.sub(
            r"\.upsample\.(weight|bias)$", r".upsample.conv.conv.\1", new_key
        )

        # === Final block ===
        new_key = re.sub(
            r"\.final_block\.block\.0\.", r".final_block.block.0.conv.conv.", new_key
        )
        new_key = re.sub(
            r"\.final_block\.block\.2\.", r".final_block.block.1.", new_key
        )
        # final_proj: add .conv wrapper
        new_key = re.sub(
            r"\.final_proj\.(weight|bias)$", r".final_proj.conv.\1", new_key
        )

        # === Transformer feedforward network ===
        # ff.net.2 -> ff.net.1 (skip GELU at index 1)
        new_key = re.sub(r"\.ff\.net\.2\.", r".ff.net.1.", new_key)

        # === Encoder mappings ===
        # embed.out.0 -> embed.linear
        new_key = re.sub(r"\.embed\.out\.0\.", r".embed.linear.", new_key)
        # embed.out.1 -> embed.norm
        new_key = re.sub(r"\.embed\.out\.1\.", r".embed.norm.", new_key)
        # up_embed.out.0 -> up_embed.linear
        new_key = re.sub(r"\.up_embed\.out\.0\.", r".up_embed.linear.", new_key)
        # up_embed.out.1 -> up_embed.norm
        new_key = re.sub(r"\.up_embed\.out\.1\.", r".up_embed.norm.", new_key)

        # === HiFi-GAN (mel2wav) mappings ===
        # conv_pre/conv_post: add .conv wrapper
        new_key = re.sub(
            r"mel2wav\.conv_pre\.(weight|bias)$", r"mel2wav.conv_pre.conv.\1", new_key
        )
        new_key = re.sub(
            r"mel2wav\.conv_post\.(weight|bias)$", r"mel2wav.conv_post.conv.\1", new_key
        )
        # resblocks.X.convs1/2.Y: add .conv wrapper
        new_key = re.sub(
            r"mel2wav\.resblocks\.(\d+)\.convs1\.(\d+)\.(weight|bias)$",
            r"mel2wav.resblocks.\1.convs1.\2.conv.\3",
            new_key,
        )
        new_key = re.sub(
            r"mel2wav\.resblocks\.(\d+)\.convs2\.(\d+)\.(weight|bias)$",
            r"mel2wav.resblocks.\1.convs2.\2.conv.\3",
            new_key,
        )
        # source_resblocks.X.convs1/2.Y: add .conv wrapper
        new_key = re.sub(
            r"mel2wav\.source_resblocks\.(\d+)\.convs1\.(\d+)\.(weight|bias)$",
            r"mel2wav.source_resblocks.\1.convs1.\2.conv.\3",
            new_key,
        )
        new_key = re.sub(
            r"mel2wav\.source_resblocks\.(\d+)\.convs2\.(\d+)\.(weight|bias)$",
            r"mel2wav.source_resblocks.\1.convs2.\2.conv.\3",
            new_key,
        )
        # f0_predictor.condnet.X: add .conv wrapper and remap indices
        # First handle the condnet index remapping (0,2,4,6,8 -> 0,1,2,3,4)
        has_pytorch_indices = any(
            ".condnet.6." in k or ".condnet.8." in k for k in weights.keys()
        )
        if has_pytorch_indices:

            def remap_condnet_idx(match):
                idx_map = {"0": "0", "2": "1", "4": "2", "6": "3", "8": "4"}
                return f".condnet.{idx_map[match.group(1)]}."

            new_key = re.sub(r"\.condnet\.([02468])\.", remap_condnet_idx, new_key)
        # Then add .conv wrapper
        new_key = re.sub(
            r"mel2wav\.f0_predictor\.condnet\.(\d+)\.(weight|bias)$",
            r"mel2wav.f0_predictor.condnet.\1.conv.\2",
            new_key,
        )
        # source_downs.X: add .conv wrapper
        new_key = re.sub(
            r"mel2wav\.source_downs\.(\d+)\.(weight|bias)$",
            r"mel2wav.source_downs.\1.conv.\2",
            new_key,
        )
        # ups.X: add .conv wrapper
        new_key = re.sub(
            r"mel2wav\.ups\.(\d+)\.(weight|bias)$", r"mel2wav.ups.\1.conv.\2", new_key
        )

        # === Conv weight transposition ===
        if "weight" in new_key and value.ndim == 3:
            if new_key in curr_weights:
                expected_shape = curr_weights[new_key].shape
                if value.shape != expected_shape:
                    if ".ups." in new_key:
                        # ConvTranspose1d: PyTorch (in, out, kernel) -> MLX (out, kernel, in)
                        value = value.transpose(1, 2, 0)
                    else:
                        # Conv1d: PyTorch (out, in, kernel) -> MLX (out, kernel, in)
                        value = value.swapaxes(1, 2)
        elif "weight" in new_key and value.ndim == 4:
            if new_key in curr_weights:
                expected_shape = curr_weights[new_key].shape
                if value.shape != expected_shape:
                    # Conv2d: PyTorch (out, in, H, W) -> MLX (out, H, W, in)
                    value = value.transpose(0, 2, 3, 1)

        # Only include keys that exist in the model
        if new_key in curr_weights:
            new_weights[new_key] = value

    return new_weights


def save_mlx_safetensors(weights: Dict[str, np.ndarray], path: Path):
    """Save weights as MLX-compatible safetensors (for non-quantized weights)."""
    from safetensors.numpy import save_file

    # Ensure all values are numpy arrays with correct dtype
    clean_weights = {}
    for k, v in weights.items():
        if isinstance(v, np.ndarray):
            # Keep original dtype but ensure it's a supported type
            if v.dtype == np.float64:
                v = v.astype(np.float32)
            clean_weights[k] = v
        else:
            clean_weights[k] = np.array(v)

    save_file(clean_weights, path)
    logger.info(f"Saved: {path} ({len(clean_weights)} tensors)")


def save_mlx_quantized(weights: Dict, path: Path):
    """Save quantized MLX weights directly using mx.save_safetensors.

    This preserves the packed uint32 format for quantized weights.
    """
    import mlx.core as mx

    mx.save_safetensors(str(path), weights, metadata={"format": "mlx"})
    logger.info(f"Saved: {path} ({len(weights)} tensors)")


def create_tokenizer_json(ckpt_dir: Path, output_dir: Path):
    """Create a unified tokenizer.json from separate tokenizer files.

    Combines vocab.json, merges.txt, added_tokens.json, special_tokens_map.json,
    and tokenizer_config.json into a single tokenizer.json in HuggingFace format.
    """
    logger.info("Creating tokenizer.json...")

    # Load vocab
    with open(ckpt_dir / "vocab.json", "r") as f:
        vocab = json.load(f)

    # Load merges
    with open(ckpt_dir / "merges.txt", "r") as f:
        merges_content = f.read()
        # Skip header line if present
        lines = merges_content.strip().split("\n")
        if lines[0].startswith("#"):
            lines = lines[1:]
        merges = lines

    # Load added_tokens if present
    added_tokens = []
    added_tokens_path = ckpt_dir / "added_tokens.json"
    if added_tokens_path.exists():
        with open(added_tokens_path, "r") as f:
            added_tokens_dict = json.load(f)
            for token, idx in added_tokens_dict.items():
                added_tokens.append(
                    {
                        "id": idx,
                        "content": token,
                        "single_word": False,
                        "lstrip": False,
                        "rstrip": False,
                        "normalized": False,
                        "special": True,
                    }
                )

    # Load tokenizer config
    with open(ckpt_dir / "tokenizer_config.json", "r") as f:
        tokenizer_config = json.load(f)

    # Build the unified tokenizer.json
    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True,
        },
        "post_processor": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": False,
            "use_regex": True,
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": tokenizer_config.get("unk_token"),
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": merges,
        },
    }

    # Save tokenizer.json
    output_path = output_dir / "tokenizer.json"
    with open(output_path, "w") as f:
        json.dump(tokenizer_json, f, indent=2)
    logger.info(f"Created: {output_path}")

    # Also copy tokenizer_config.json for compatibility
    import shutil

    shutil.copy(
        ckpt_dir / "tokenizer_config.json", output_dir / "tokenizer_config.json"
    )
    logger.info("Copied: tokenizer_config.json")


def quantize_t3_backbone(model, bits: int = 4, group_size: int = 64) -> int:
    """
    Selectively quantize the T3 GPT2 backbone transformer layers.

    Only quantizes t3.tfmr.h.* (GPT2 transformer blocks containing attention and MLP).
    Other components are kept in full precision:
    - VoiceEncoder (ve.*): Small LSTM, sensitive to quantization
    - S3Gen (s3gen.*): Audio quality sensitive vocoder
    - Embeddings: text_emb, speech_emb, wte, wpe need precision
    - Output heads: text_head, speech_head need precision
    - Layer norms: Small and precision-critical

    Args:
        model: Model instance to quantize
        bits: Quantization bits (default: 4)
        group_size: Quantization group size (default: 64)

    Returns:
        Number of layers quantized
    """
    import mlx.nn as nn

    quantized_count = [0]

    def should_quantize(path, module):
        """Only quantize Linear layers in T3 GPT2 transformer blocks."""
        if not isinstance(module, nn.Linear):
            return False

        # Quantize transformer layers: tfmr.h.N.attn.* and tfmr.h.N.mlp.*
        # Handle both "t3.tfmr.h" (full model) and "tfmr.h" (T3 only)
        if "tfmr.h." in path:
            quantized_count[0] += 1
            return True
        return False

    nn.quantize(
        model, bits=bits, group_size=group_size, class_predicate=should_quantize
    )
    return quantized_count[0]


def generate_readme(
    path: Path,
    upload_repo: str = None,
    quantize: bool = False,
    q_bits: int = 4,
):
    """Generate README.md model card for Chatterbox Turbo on Hugging Face."""
    from mlx_audio.version import __version__

    # Use upload_repo if provided, otherwise use output directory name
    model_name = upload_repo if upload_repo else path.name

    quant_info = ""
    if quantize:
        quant_info = f"\n\nThis model uses **{q_bits}-bit quantization** for the T3 GPT2 backbone, reducing memory usage while maintaining audio quality."

    # For local paths, show how to use with local path
    if upload_repo:
        model_path_example = upload_repo
    else:
        model_path_example = f"/path/to/{path.name}"

    card_text = f"""---
library_name: mlx-audio-plus
base_model:
- ResembleAI/chatterbox-turbo
tags:
- mlx
- tts
- chatterbox
pipeline_tag: text-to-speech
language:
- en
---

# {model_name}

This model was converted to MLX format from [ResembleAI/chatterbox-turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) using [mlx-audio-plus](https://github.com/DePasqualeOrg/mlx-audio-plus) version **{__version__}**.{quant_info}

**Note:** This model requires the S3Tokenizer weights from [mlx-community/S3TokenizerV2](https://huggingface.co/mlx-community/S3TokenizerV2), which will be downloaded automatically.

## Use with mlx-audio-plus

```bash
pip install -U mlx-audio-plus
```

### Command line

```bash
mlx_audio.tts --model {model_path_example} --text "Hello, this is Chatterbox Turbo on MLX!" --ref_audio reference.wav
```

### Python

```python
from mlx_audio.tts.generate import generate_audio

generate_audio(
    text="Hello, this is Chatterbox Turbo on MLX!",
    model="{model_path_example}",
    ref_audio="reference.wav",
    file_prefix="output",
)
```
"""
    card_path = path / "README.md"
    with open(card_path, "w") as f:
        f.write(card_text)
    logger.info(f"Created: {card_path}")


def upload_to_hub(path: Path, upload_repo: str):
    """Upload converted model to Hugging Face Hub."""
    from huggingface_hub import HfApi

    logger.info(f"\nUploading to {upload_repo}...")
    api = HfApi()
    api.create_repo(repo_id=upload_repo, exist_ok=True)
    api.upload_folder(
        folder_path=str(path),
        repo_id=upload_repo,
        repo_type="model",
    )
    logger.info(f"Upload successful! Visit https://huggingface.co/{upload_repo}")


def convert_from_source(
    model_id: str = DEFAULT_MODEL_ID,
    output_dir: Path = None,
    cache_dir: Path = None,
    quantize: bool = False,
    q_bits: int = 4,
    q_group_size: int = 64,
    dtype: str = "float16",
    upload_repo: str = None,
    dry_run: bool = False,
) -> None:
    """
    Convert Chatterbox Turbo PyTorch weights to MLX format.

    This function is called by the central conversion utility when it detects
    a Chatterbox Turbo model, or can be called directly.

    Args:
        model_id: Hugging Face model ID (default: ResembleAI/chatterbox-turbo)
        output_dir: Output directory for MLX weights
        cache_dir: Optional custom cache directory for downloads
        quantize: Whether to apply selective quantization to T3 backbone
        q_bits: Quantization bits (default: 4)
        q_group_size: Quantization group size (default: 64)
        dtype: Data type for weights (float16 or float32)
        upload_repo: Hugging Face repo to upload to
        dry_run: Generate files but skip upload
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten

    if output_dir is None:
        suffix = f"{q_bits}bit" if quantize else "fp16"
        output_dir = Path(f"./ChatterboxTurbo-TTS-{suffix}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np_dtype = np.float16 if dtype == "float16" else np.float32

    # Download weights
    ckpt_dir = download_chatterbox_turbo_weights(cache_dir)

    # Import model components for their sanitize methods
    from mlx_audio.tts.models.chatterbox_turbo.models.s3gen import S3Gen
    from mlx_audio.tts.models.chatterbox_turbo.models.t3 import T3
    from mlx_audio.tts.models.chatterbox_turbo.models.voice_encoder import VoiceEncoder

    # Combined weights dict with prefixes
    all_weights = {}

    # Convert VoiceEncoder
    logger.info("\nConverting VoiceEncoder...")
    ve_weights = load_pytorch_safetensors(ckpt_dir / "ve.safetensors")
    # Remap PyTorch stacked LSTM to MLX separate LSTMs
    ve_weights = remap_voice_encoder_keys(ve_weights)
    for k, v in ve_weights.items():
        all_weights[f"ve.{k}"] = v.astype(np_dtype) if v.dtype == np.float32 else v
    logger.info(f"  Added {len(ve_weights)} VoiceEncoder weights")

    # Convert T3 Turbo
    logger.info("\nConverting T3 Turbo...")
    t3_weights = load_pytorch_safetensors(ckpt_dir / "t3_turbo_v1.safetensors")
    # Transpose GPT-2 Conv1d weights for MLX Linear layers
    t3_weights = remap_t3_keys(t3_weights)
    for k, v in t3_weights.items():
        all_weights[f"t3.{k}"] = v.astype(np_dtype) if v.dtype == np.float32 else v
    logger.info(f"  Added {len(t3_weights)} T3 Turbo weights")

    # Convert S3Gen (meanflow version for Turbo)
    logger.info("\nConverting S3Gen (meanflow)...")
    s3gen_weights = load_pytorch_safetensors(ckpt_dir / "s3gen_meanflow.safetensors")
    # Filter out tokenizer.* keys - S3Tokenizer is in a separate repo
    s3gen_weights = {
        k: v for k, v in s3gen_weights.items() if not k.startswith("tokenizer.")
    }
    s3gen = S3Gen(meanflow=True)  # Turbo uses meanflow=True
    # Remap PyTorch keys to MLX format (handles flow. prefix, block structure, conv transposition)
    s3gen_weights = remap_s3gen_keys(s3gen_weights, s3gen)
    for k, v in s3gen_weights.items():
        all_weights[f"s3gen.{k}"] = v.astype(np_dtype) if v.dtype == np.float32 else v
    logger.info(f"  Added {len(s3gen_weights)} S3Gen weights")

    # Note: S3Tokenizer is NOT included - it's in a separate repo
    logger.info(
        "\nNote: S3Tokenizer weights are loaded separately from mlx-community/S3TokenizerV2"
    )

    # Apply quantization if requested
    if quantize:
        logger.info(f"\nApplying {q_bits}-bit quantization to T3 backbone...")

        # Only T3 is quantized - create a fresh T3 model instance
        t3_model = T3()

        all_weights_mx = numpy_to_mlx(all_weights)

        # Split weights by component
        ve_w = {k: v for k, v in all_weights_mx.items() if k.startswith("ve.")}
        t3_w = {k[3:]: v for k, v in all_weights_mx.items() if k.startswith("t3.")}
        s3gen_w = {k: v for k, v in all_weights_mx.items() if k.startswith("s3gen.")}

        # Only load T3 into a model for quantization
        # VoiceEncoder and S3Gen are kept as-is (not quantized)
        t3_model.load_weights(list(t3_w.items()), strict=False)
        mx.eval(t3_model.parameters())

        # Get original size
        orig_size = sum(v.nbytes for v in all_weights_mx.values())
        logger.info(f"  Original size: {orig_size / 1e9:.2f} GB")

        # Apply selective quantization to T3 backbone only
        num_quantized = quantize_t3_backbone(
            t3_model, bits=q_bits, group_size=q_group_size
        )
        mx.eval(t3_model.parameters())
        logger.info(f"  Quantized {num_quantized} Linear layers in T3 backbone")

        # Collect weights with prefixes
        # VoiceEncoder and S3Gen keep their original (sanitized) weights
        # T3 gets the quantized weights from the model
        new_weights = {}
        for k, v in ve_w.items():
            new_weights[k] = v
        for k, v in dict(tree_flatten(t3_model.parameters())).items():
            new_weights[f"t3.{k}"] = v
        for k, v in s3gen_w.items():
            new_weights[k] = v

        new_size = sum(v.nbytes for v in new_weights.values())
        logger.info(f"  New size: {new_size / 1e9:.2f} GB")
        logger.info(f"  Reduction: {(1 - new_size / orig_size) * 100:.1f}%")

        # Save quantized weights
        logger.info("\nSaving combined model.safetensors (quantized)...")
        save_mlx_quantized(new_weights, output_dir / "model.safetensors")
    else:
        # Save non-quantized weights
        logger.info("\nSaving combined model.safetensors...")
        save_mlx_safetensors(all_weights, output_dir / "model.safetensors")

    # Create tokenizer.json from separate tokenizer files
    create_tokenizer_json(ckpt_dir, output_dir)

    # Convert and save conds.pt to conds.safetensors
    # The flattened keys are already prefixed with "t3." and "gen." from the nested dict structure
    logger.info("\nConverting conds.pt to conds.safetensors...")
    conds = load_pytorch_checkpoint(ckpt_dir / "conds.pt")
    save_mlx_safetensors(conds, output_dir / "conds.safetensors")

    # Create config.json
    logger.info("\nCreating config.json...")
    config = {
        "model_type": "chatterbox_turbo",
        "version": "1.0",
        "dtype": dtype,
    }
    if quantize:
        config["quantization"] = {
            "bits": q_bits,
            "group_size": q_group_size,
            "quantized_components": ["t3.tfmr.h"],
        }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Generate README
    logger.info("\nGenerating README.md...")
    generate_readme(output_dir, upload_repo, quantize, q_bits)

    logger.info(
        f"\n{'🔢' if quantize else '✅'} Conversion complete! Output directory: {output_dir}"
    )
    logger.info("\nFiles created:")
    for f in sorted(output_dir.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        logger.info(f"  {f.name}: {size_mb:.1f} MB")

    # Upload to Hugging Face if requested (and not dry run)
    if upload_repo and not dry_run:
        upload_to_hub(output_dir, upload_repo)
    elif upload_repo:
        logger.info(f"\n📁 Dry run - to upload to {upload_repo}, run without --dry-run")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Chatterbox Turbo weights to MLX format"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model ID (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for MLX weights (default: ./ChatterboxTurbo-TTS-{fp16|Nbit})",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory for downloads (default: ~/.cache/huggingface/hub)",
    )
    parser.add_argument(
        "--upload-repo",
        type=str,
        default=None,
        help="Hugging Face repo to upload to (e.g., mlx-community/chatterbox-turbo-4bit)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate all files including README but skip upload",
    )
    parser.add_argument(
        "-q",
        "--quantize",
        action="store_true",
        help="Apply 4-bit quantization to T3 backbone (reduces size significantly)",
    )
    parser.add_argument(
        "--q-bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="Quantization bits (default: 4)",
    )
    parser.add_argument(
        "--q-group-size",
        type=int,
        default=64,
        help="Quantization group size (default: 64)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Data type for weights (default: float16)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        suffix = f"{args.q_bits}bit" if args.quantize else "fp16"
        output_dir = Path(f"./ChatterboxTurbo-TTS-{suffix}")

    # Determine upload repo
    upload_repo = args.upload_repo
    if upload_repo is None and args.dry_run:
        # Default repo name for dry run
        upload_repo = f"mlx-community/{output_dir.name}"

    convert_from_source(
        model_id=args.model_id,
        output_dir=output_dir,
        cache_dir=args.cache_dir,
        quantize=args.quantize,
        q_bits=args.q_bits,
        q_group_size=args.q_group_size,
        dtype=args.dtype,
        upload_repo=upload_repo,
        dry_run=args.dry_run or args.upload_repo is None,
    )


if __name__ == "__main__":
    main()
