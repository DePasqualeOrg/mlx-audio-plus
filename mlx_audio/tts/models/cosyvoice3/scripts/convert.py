# Copyright © Anthony DePasquale

"""
Convert CosyVoice 3 weights from PyTorch to MLX format.

All weights are combined into a single model.safetensors file with prefixes:
- llm.* : Speech embedding and decoder weights
- qwen2.* : Qwen2 LLM backbone
- flow.* : Flow/DiT model
- hifigan.* : HiFi-GAN vocoder
- campplus.* : Speaker encoder

The S3TokenizerV3 is converted separately using the standalone script at
mlx_audio/codec/models/s3tokenizer/scripts/convert_v3.py.

Quantization:
- Only Qwen2 transformer layers (qwen2.model.layers.*) are quantized
- Speech embedding, flow, hifigan, and campplus remain in full precision

Usage:
    # Convert CosyVoice 3 to fp16 (downloads from HuggingFace automatically)
    python -m mlx_audio.tts.models.cosyvoice3.scripts.convert

    # Convert with 4-bit quantization (Qwen2 only)
    python -m mlx_audio.tts.models.cosyvoice3.scripts.convert --quantize

    # Convert with custom output directory
    python -m mlx_audio.tts.models.cosyvoice3.scripts.convert \\
        --output-dir ./my-cosyvoice3

    # Upload to Hugging Face
    python -m mlx_audio.tts.models.cosyvoice3.scripts.convert \\
        --upload-repo mlx-community/CosyVoice3-0.5B-fp16

Downloads weights from FunAudioLLM/Fun-CosyVoice3-0.5B-2512 by default
"""

import argparse
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Dict

import numpy as np
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"


def transpose_conv_weight(weight: np.ndarray, transpose: bool = False) -> np.ndarray:
    """Transpose convolution weight from PyTorch to MLX format."""
    if transpose:
        return np.swapaxes(np.swapaxes(weight, 0, 1), 1, 2)
    else:
        return np.swapaxes(weight, 1, 2)


def convert_qwen2_weights(pt_weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Convert Qwen2 LLM weights."""
    mlx_weights = {}
    # CosyVoice3 uses llm.model.model.* (double model prefix)
    prefix = "llm.model.model."

    for key, value in pt_weights.items():
        if not key.startswith(prefix):
            # Also check for llm.model.lm_head (skip it)
            if key.startswith("llm.model."):
                continue
            continue
        new_key = "model." + key[len(prefix) :]
        mlx_weights[new_key] = value

    return mlx_weights


def convert_speech_weights(pt_weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Convert speech embedding and decoder weights."""
    mlx_weights = {}

    for key, value in pt_weights.items():
        # CosyVoice3 has speech_embedding and llm_decoder at root level
        if key.startswith("speech_embedding"):
            mlx_weights[key] = value
        elif key.startswith("llm_decoder"):
            mlx_weights[key] = value

    return mlx_weights


def convert_flow_weights(pt_weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Convert flow model weights (including DiT)."""
    mlx_weights = {}

    for key, value in pt_weights.items():
        # CosyVoice3 flow.pt doesn't have "flow." prefix - keys start with "decoder."
        new_key = key

        # DiT time embedding
        if "decoder.estimator.time_embed." in new_key:
            new_key = new_key.replace("time_embed.time_mlp.0", "time_embed.time_mlp_0")
            new_key = new_key.replace("time_embed.time_mlp.2", "time_embed.time_mlp_2")

        # DiT input embedding conv
        if "conv_pos_embed" in new_key:
            if ".conv1.0." in new_key:
                new_key = new_key.replace(".conv1.0.", ".conv1.")
                if "weight" in new_key and len(value.shape) == 3:
                    value = transpose_conv_weight(value)
            elif ".conv2.0." in new_key:
                new_key = new_key.replace(".conv2.0.", ".conv2.")
                if "weight" in new_key and len(value.shape) == 3:
                    value = transpose_conv_weight(value)

        # DiT transformer blocks
        if "transformer_blocks." in new_key:
            new_key = new_key.replace(".to_out.0.", ".to_out_0.")
            new_key = new_key.replace(".to_out.1.", ".to_out_1.")
            new_key = new_key.replace(".ff.ff.0.0.", ".ff.ff_0_0.")
            new_key = new_key.replace(".ff.ff.1.", ".ff.ff_1.")
            new_key = new_key.replace(".ff.ff.2.", ".ff.ff_2.")

        # Pre-lookahead convolutions
        if "pre_lookahead_layer." in new_key:
            if ("conv1.weight" in new_key or "conv2.weight" in new_key) and len(
                value.shape
            ) == 3:
                value = transpose_conv_weight(value)

        mlx_weights[new_key] = value

    return mlx_weights


def convert_hifigan_weights(pt_weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Convert CausalHiFTGenerator weights."""
    mlx_weights = {}

    # Collect parametrizations (new PyTorch weight norm format)
    # foo.parametrizations.weight.original0 = g (scale)
    # foo.parametrizations.weight.original1 = v (direction)
    param_g = {}
    param_v = {}
    for k, v in pt_weights.items():
        if ".parametrizations.weight.original0" in k:
            base = k.replace(".parametrizations.weight.original0", "")
            param_g[base] = v
        elif ".parametrizations.weight.original1" in k:
            base = k.replace(".parametrizations.weight.original1", "")
            param_v[base] = v

    # Also collect old-style _g/_v weights
    old_g = {k: v for k, v in pt_weights.items() if "_g" in k}

    for key, value in pt_weights.items():
        # Skip parametrization internal keys
        if ".parametrizations.weight.original" in key:
            continue

        new_key = key

        # Old-style weight norm: combine g and v
        if "_g" in new_key:
            continue
        if "_v" in new_key:
            g_key = key.replace("_v", "_g")
            if g_key in old_g:
                g, v = old_g[g_key], value
                norm = np.sqrt(
                    np.sum(v * v, axis=tuple(range(1, len(v.shape))), keepdims=True)
                )
                value = g * v / (norm + 1e-8)
            new_key = new_key.replace("_v", "")

        # F0 predictor condnet
        if "f0_predictor.condnet." in new_key:
            match = re.match(r".*f0_predictor\.condnet\.(\d+)\.", new_key)
            if match:
                idx = int(match.group(1))
                new_key = new_key.replace(f"condnet.{idx}.", f"condnet_{idx}.")
                if (
                    "weight" in new_key
                    and idx in [0, 2, 4, 6, 8]
                    and len(value.shape) == 3
                ):
                    value = transpose_conv_weight(value)

        # Conv pre/post and upsampling
        elif any(x in new_key for x in ["conv_pre.", "conv_post.", ".ups."]):
            if "weight" in new_key and len(value.shape) == 3:
                value = transpose_conv_weight(value)

        # Source downs - direct conv weights
        elif "source_downs." in new_key:
            if "weight" in new_key and len(value.shape) == 3:
                value = transpose_conv_weight(value)

        # Source resblocks and regular resblocks (with convs sublayers)
        elif any(x in new_key for x in ["source_resblocks.", ".resblocks."]):
            if "convs" in new_key and "weight" in new_key and len(value.shape) == 3:
                value = transpose_conv_weight(value)

        mlx_weights[new_key] = value

    # Handle new-style parametrizations weight norm
    for base_key in param_v:
        if base_key in param_g:
            g = param_g[base_key]
            v = param_v[base_key]
            # Weight norm: weight = g * v / ||v||
            norm = np.sqrt(
                np.sum(v * v, axis=tuple(range(1, len(v.shape))), keepdims=True)
            )
            weight = g * v / (norm + 1e-8)

            new_key = base_key + ".weight"

            # Apply conv transposition if needed
            if len(weight.shape) == 3:
                if any(
                    x in new_key
                    for x in ["conv_pre.", "conv_post.", "ups.", "source_downs."]
                ):
                    weight = transpose_conv_weight(weight)
                elif "convs" in new_key:
                    weight = transpose_conv_weight(weight)
                elif "f0_predictor.condnet" in new_key:
                    weight = transpose_conv_weight(weight)

            # Handle condnet key renaming
            if "f0_predictor.condnet." in new_key:
                match = re.match(r".*f0_predictor\.condnet\.(\d+)\.", new_key)
                if match:
                    idx = int(match.group(1))
                    new_key = new_key.replace(f"condnet.{idx}.", f"condnet_{idx}.")

            mlx_weights[new_key] = weight

    # Add .conv. prefix for CausalConv1d wrapper layers
    # Pattern: foo.weight -> foo.conv.weight for conv layers
    final_weights = {}
    for key, value in mlx_weights.items():
        new_key = key

        # List of layer patterns that use CausalConv1d wrapper
        conv_patterns = [
            "conv_pre.",
            "conv_post.",
            "ups.",
            "source_downs.",
            ".convs1.",
            ".convs2.",
            "f0_predictor.condnet_",
        ]

        # Check if this is a weight/bias for a conv layer
        is_conv_layer = any(p in key for p in conv_patterns)
        is_weight_or_bias = key.endswith(".weight") or key.endswith(".bias")

        if is_conv_layer and is_weight_or_bias:
            # Insert .conv. before .weight or .bias
            if key.endswith(".weight"):
                new_key = key[:-7] + ".conv.weight"
            elif key.endswith(".bias"):
                new_key = key[:-5] + ".conv.bias"

        final_weights[new_key] = value

    return final_weights


def convert_campplus_weights(
    pt_weights: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Convert CAMPlus speaker encoder weights from PyTorch."""
    mlx_weights = {}
    for key, value in pt_weights.items():
        if "conv" in key and "weight" in key and len(value.shape) == 3:
            value = transpose_conv_weight(value)
        mlx_weights[key] = value
    return mlx_weights


def convert_campplus_from_chatterbox() -> Dict[str, np.ndarray]:
    """Extract and convert CAMPlus weights from Chatterbox's s3gen.safetensors.

    CosyVoice 2, CosyVoice 3, and Chatterbox use the same CAMPlus architecture
    for speaker embeddings. We extract the speaker_encoder.xvector.* weights
    from Chatterbox and convert them using CAMPPlus.sanitize().

    Returns:
        Dict of converted weights ready to save as campplus.safetensors
    """
    import mlx.core as mx
    from huggingface_hub import hf_hub_download

    from mlx_audio.codec.models.s3gen.xvector import CAMPPlus

    logger.info("Downloading Chatterbox s3gen.safetensors for CAMPlus weights...")
    s3gen_path = hf_hub_download("ResembleAI/chatterbox", "s3gen.safetensors")

    # Load s3gen weights
    s3gen_weights = dict(mx.load(s3gen_path))

    # Extract speaker_encoder.xvector.* weights (strip speaker_encoder. prefix)
    xvector_weights = {}
    for key, value in s3gen_weights.items():
        if key.startswith("speaker_encoder."):
            # Keep xvector.* prefix - CAMPPlus.sanitize() expects it
            new_key = key[len("speaker_encoder.") :]
            xvector_weights[new_key] = value

    logger.info(f"  Extracted {len(xvector_weights)} xvector weights")

    # Use CAMPPlus.sanitize() to convert weights
    campplus = CAMPPlus()
    sanitized_weights = campplus.sanitize(xvector_weights)

    # Convert to numpy for safetensors
    numpy_weights = {}
    for key, value in sanitized_weights.items():
        numpy_weights[key] = np.array(value)

    logger.info(f"  Sanitized to {len(numpy_weights)} MLX weights")
    return numpy_weights


def generate_readme(
    output_dir: Path,
    model_id: str,
    upload_repo: str,
    quantize: bool = False,
    q_bits: int = 4,
) -> None:
    """Generate README.md model card for Hugging Face."""
    from mlx_audio.version import __version__

    # Determine quantization info for the card
    quant_info = ""
    if quantize:
        quant_info = f"\n\nThis model uses **{q_bits}-bit quantization** for the Qwen2 LLM backbone, reducing memory usage while maintaining quality."

    card_text = f"""---
library_name: mlx-audio-plus
base_model:
- {model_id}
tags:
- mlx
- tts
- cosyvoice3
pipeline_tag: text-to-speech
language:
- zh
- en
- ja
- ko
- de
- fr
- ru
- it
- es
---

# {upload_repo}

This model was converted to MLX format from [{model_id}](https://huggingface.co/{model_id}) using [mlx-audio-plus](https://github.com/DePasqualeOrg/mlx-audio-plus) version **{__version__}**.{quant_info}

## Usage

```bash
pip install -U mlx-audio-plus
```

### Inference Modes

| Mode | Parameters | Description |
|------|------------|-------------|
| Cross-lingual | `ref_audio` | Zero-shot TTS (default) |
| Zero-shot | `ref_audio` + `ref_text` | Better quality with transcription |
| Instruct | `ref_audio` + `instruct_text` | Style control (e.g., "speak slowly") |
| Voice Conversion | `source_audio` + `ref_audio` | Convert audio to target voice |

### Command line

```bash
# Cross-lingual (default)
mlx_audio.tts --model {upload_repo} --text "Hello!" --ref_audio ref.wav

# Zero-shot (with transcription)
mlx_audio.tts --model {upload_repo} --text "Hello!" --ref_audio ref.wav --ref_text "Transcription of ref audio."

# Instruct (style control)
mlx_audio.tts --model {upload_repo} --text "Hello!" --ref_audio ref.wav --instruct_text "Speak slowly and calmly"

# Voice Conversion
mlx_audio.tts --model {upload_repo} --source_audio source.wav --ref_audio ref.wav
```

### Python

```python
from mlx_audio.tts.generate import generate_audio

generate_audio(
    text="Hello, this is CosyVoice 3 on MLX!",
    model="{upload_repo}",
    ref_audio="reference.wav",
    file_prefix="output",
)
```
"""
    card_path = output_dir / "README.md"
    with open(card_path, "w") as f:
        f.write(card_text)
    logger.info(f"Created {card_path}")


def quantize_qwen2_weights(
    qwen2_weights: Dict[str, np.ndarray], bits: int = 4, group_size: int = 64
) -> Dict:
    """Quantize Qwen2 LLM weights (transformer layers only)."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.models.qwen2 import Model as Qwen2Model
    from mlx_lm.models.qwen2 import ModelArgs

    logger.info(f"Quantizing Qwen2 to {bits}-bit (group_size={group_size})...")

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
    mlx_weights = {
        k: mx.array(v) for k, v in qwen2_weights.items() if k != "lm_head.weight"
    }
    qwen2_model.load_weights(list(mlx_weights.items()))
    mx.eval(qwen2_model.parameters())

    orig_weights = dict(tree_flatten(qwen2_model.parameters()))
    orig_size = sum(v.nbytes for v in orig_weights.values())
    logger.info(f"  Original size: {orig_size / 1e9:.2f} GB")

    quantized_count = [0]

    def should_quantize(path, module):
        if isinstance(module, nn.Linear) and "model.layers" in path:
            quantized_count[0] += 1
            return True
        return False

    nn.quantize(
        qwen2_model, bits=bits, group_size=group_size, class_predicate=should_quantize
    )
    mx.eval(qwen2_model.parameters())
    logger.info(f"  Quantized {quantized_count[0]} Linear layers")

    quantized_weights = dict(tree_flatten(qwen2_model.parameters()))
    new_size = sum(v.nbytes for v in quantized_weights.values())
    logger.info(
        f"  New size: {new_size / 1e9:.2f} GB ({(1 - new_size / orig_size) * 100:.1f}% reduction)"
    )

    return quantized_weights


def convert_from_source(
    model_id: str = DEFAULT_MODEL_ID,
    output_path: str = None,
    dtype: str = "float16",
    quantize: bool = False,
    q_bits: int = 4,
    q_group_size: int = 64,
    upload_repo: str = None,
    dry_run: bool = False,
) -> None:
    """Convert CosyVoice 3 weights to MLX format (single model.safetensors)."""
    import torch

    # Download from HuggingFace or use local path
    model_path = Path(model_id)
    if model_path.exists() and model_path.is_dir():
        source_path = model_path
        logger.info(f"Using local path: {source_path}")
    else:
        logger.info(f"Downloading from HuggingFace: {model_id}")
        cache_dir = snapshot_download(
            model_id,
            allow_patterns=["*.pt", "*.onnx", "CosyVoice-BlankEN/*"],
        )
        source_path = Path(cache_dir)
        logger.info(f"Downloaded to: {source_path}")

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    np_dtype = np.float16 if dtype == "float16" else np.float32
    all_weights = {}

    # LLM weights
    llm_path = source_path / "llm.pt"
    if llm_path.exists():
        logger.info("Loading LLM weights...")
        pt_llm = torch.load(llm_path, map_location="cpu", weights_only=True)
        llm_weights = {k: v.numpy().astype(np_dtype) for k, v in pt_llm.items()}

        # Speech weights (llm.* prefix)
        speech_weights = convert_speech_weights(llm_weights)
        for k, v in speech_weights.items():
            all_weights[f"llm.{k}"] = v
        logger.info(f"  Converted {len(speech_weights)} speech weights")

        # Qwen2 weights (qwen2.* prefix)
        qwen2_weights = convert_qwen2_weights(llm_weights)
        if quantize:
            qwen2_quantized = quantize_qwen2_weights(
                qwen2_weights, q_bits, q_group_size
            )
            for k, v in qwen2_quantized.items():
                all_weights[f"qwen2.{k}"] = np.array(v)
        else:
            for k, v in qwen2_weights.items():
                all_weights[f"qwen2.{k}"] = v
        logger.info(f"  Converted {len(qwen2_weights)} Qwen2 weights")

    # Flow weights
    flow_path = source_path / "flow.pt"
    if flow_path.exists():
        logger.info("Loading flow weights...")
        pt_flow = torch.load(flow_path, map_location="cpu", weights_only=True)
        flow_weights = {k: v.numpy().astype(np_dtype) for k, v in pt_flow.items()}
        mlx_flow = convert_flow_weights(flow_weights)
        for k, v in mlx_flow.items():
            all_weights[f"flow.{k}"] = v
        logger.info(f"  Converted {len(mlx_flow)} flow weights")

    # HiFi-GAN weights
    hift_path = source_path / "hift.pt"
    if hift_path.exists():
        logger.info("Loading HiFi-GAN weights...")
        pt_hift = torch.load(hift_path, map_location="cpu", weights_only=True)
        hift_weights = {k: v.numpy().astype(np_dtype) for k, v in pt_hift.items()}
        mlx_hifigan = convert_hifigan_weights(hift_weights)
        for k, v in mlx_hifigan.items():
            all_weights[f"hifigan.{k}"] = v
        logger.info(f"  Converted {len(mlx_hifigan)} HiFi-GAN weights")

    # CAMPlus weights (from Chatterbox - same architecture as CosyVoice 2/3)
    logger.info("Converting CAMPlus speaker encoder...")
    try:
        mlx_campplus = convert_campplus_from_chatterbox()
        for k, v in mlx_campplus.items():
            all_weights[f"campplus.{k}"] = v.astype(np_dtype)
        logger.info(f"  Converted {len(mlx_campplus)} CAMPlus weights")
    except Exception as e:
        logger.warning(f"Failed to convert CAMPlus: {e}")
        logger.warning("Speaker embeddings will not be available")

    # Save model.safetensors
    logger.info(f"Saving model.safetensors ({len(all_weights)} tensors)...")
    from safetensors.numpy import save_file

    save_file(all_weights, output_path / "model.safetensors")

    # Generate tokenizer.json from source tokenizer files
    # The source model may only have vocab.json + merges.txt (BPE format),
    # so we use AutoTokenizer to load and re-save in the modern tokenizer.json format
    logger.info("Converting tokenizer to tokenizer.json format...")
    tokenizer_subdirs = ["", "CosyVoice-BlankEN"]
    tokenizer_loaded = False
    for subdir in tokenizer_subdirs:
        tokenizer_path = source_path / subdir if subdir else source_path
        # Check if tokenizer files exist in this subdirectory
        has_tokenizer = (tokenizer_path / "tokenizer.json").exists() or (
            (tokenizer_path / "vocab.json").exists()
            and (tokenizer_path / "merges.txt").exists()
        )
        if has_tokenizer:
            try:
                from transformers import AutoTokenizer

                logger.info(f"  Loading tokenizer from {subdir or 'root'}...")
                tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
                tokenizer.save_pretrained(str(output_path))

                # Remove unnecessary tokenizer files (keep only tokenizer.json and tokenizer_config.json)
                unnecessary_files = [
                    "vocab.json",
                    "merges.txt",
                    "added_tokens.json",
                    "special_tokens_map.json",
                    "chat_template.jinja",
                ]
                for fname in unnecessary_files:
                    fpath = output_path / fname
                    if fpath.exists():
                        fpath.unlink()
                        logger.info(f"  Removed unnecessary file: {fname}")

                logger.info(f"  Saved tokenizer.json to {output_path}")
                tokenizer_loaded = True
                break
            except Exception as e:
                logger.warning(
                    f"  Failed to load tokenizer from {subdir or 'root'}: {e}"
                )

    if not tokenizer_loaded:
        logger.warning("No tokenizer found - text encoding will not be available")

    # Create config.json
    config = {
        "model_type": "cosyvoice3",
        "sample_rate": 24000,
        "dtype": dtype,
        "llm": {
            "llm_input_size": 896,
            "llm_output_size": 896,
            "speech_token_size": 6561,
            "extended_vocab_size": 200,
        },
        "flow": {
            "input_size": 80,
            "output_size": 80,
            "spk_embed_dim": 192,
            "vocab_size": 6561,
            "token_mel_ratio": 2,
            "pre_lookahead_len": 3,
            "dit": {
                "dim": 1024,
                "depth": 22,
                "heads": 16,
                "dim_head": 64,
                "ff_mult": 2,
                "static_chunk_size": 50,
            },
        },
        "hifigan": {
            "in_channels": 80,
            "base_channels": 512,
            "nb_harmonics": 8,
            "sampling_rate": 24000,
            "upsample_rates": [8, 5, 3],
            "upsample_kernel_sizes": [16, 11, 7],
            "istft_n_fft": 16,
            "istft_hop_len": 4,
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "source_resblock_kernel_sizes": [7, 7, 11],
            "source_resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "conv_pre_look_right": 4,
            "causal": True,
        },
    }
    if quantize:
        config["quantization"] = {
            "bits": q_bits,
            "group_size": q_group_size,
            "quantized_components": ["qwen2.model.layers"],
        }
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Generate README if upload_repo is specified
    if upload_repo:
        logger.info("Generating README.md...")
        generate_readme(
            output_dir=output_path,
            model_id=model_id,
            upload_repo=upload_repo,
            quantize=quantize,
            q_bits=q_bits,
        )

    logger.info(f"\n✅ Conversion complete! Output: {output_path}")
    logger.info("\nFiles created:")
    for f in sorted(output_path.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            logger.info(f"  {f.relative_to(output_path)}: {size_mb:.1f} MB")

    if upload_repo and not dry_run:
        from huggingface_hub import HfApi

        logger.info(f"\nUploading to {upload_repo}...")
        api = HfApi()
        api.create_repo(repo_id=upload_repo, exist_ok=True)
        api.upload_folder(
            folder_path=str(output_path), repo_id=upload_repo, repo_type="model"
        )
        logger.info(f"Upload successful! Visit https://huggingface.co/{upload_repo}")
    elif upload_repo and dry_run:
        logger.info(f"\n📁 Dry run - to upload to {upload_repo}, run without --dry-run")


def main():
    """CLI entry point for CosyVoice 3 conversion."""
    parser = argparse.ArgumentParser(description="Convert CosyVoice 3 weights to MLX")
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model ID or local path (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for MLX weights (default: ./CosyVoice3-0.5B-{fp16|Nbit})",
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
        help="Hugging Face repo to upload to (e.g., mlx-community/CosyVoice3-fp16)",
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
        help="Quantize Qwen2 LLM (4-bit by default)",
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
        suffix = "fp16" if args.dtype == "float16" else args.dtype
        if args.quantize:
            suffix = f"{args.q_bits}bit"
        output_dir = f"./CosyVoice3-0.5B-{suffix}"

    convert_from_source(
        model_id=args.model_id,
        output_path=output_dir,
        dtype=args.dtype,
        quantize=args.quantize,
        q_bits=args.q_bits,
        q_group_size=args.q_group_size,
        upload_repo=args.upload_repo,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
