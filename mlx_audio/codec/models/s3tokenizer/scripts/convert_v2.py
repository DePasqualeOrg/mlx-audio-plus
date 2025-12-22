#!/usr/bin/env python3
# Copyright © Anthony DePasquale
"""
Convert S3TokenizerV2 weights from ONNX to MLX format.

S3TokenizerV2 is a shared component used by multiple TTS models:
- Chatterbox
- Chatterbox Turbo
- CosyVoice2

The converted weights are uploaded to mlx-community/S3TokenizerV2 and
automatically downloaded when using these models with mlx-audio-plus.

Usage:
    # Convert S3TokenizerV2 to MLX format
    python mlx_audio/codec/models/s3tokenizer/scripts/convert_v2.py

    # Upload to Hugging Face
    python mlx_audio/codec/models/s3tokenizer/scripts/convert_v2.py \\
        --upload-repo mlx-community/S3TokenizerV2

Requirements:
    pip install torch safetensors huggingface_hub onnx s3tokenizer
"""

import argparse
import sys
from pathlib import Path

# Add mlx-audio-plus root to path for local imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import json
import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


def download_s3tokenizer_v2_onnx(cache_dir: Path = None) -> Path:
    """Download S3TokenizerV2 ONNX weights from Hugging Face.

    Args:
        cache_dir: Optional custom cache directory. If None, uses the default
            HF cache (~/.cache/huggingface/hub).
    """
    from huggingface_hub import hf_hub_download

    logger.info("Downloading S3TokenizerV2 ONNX from Hugging Face...")
    onnx_path = hf_hub_download(
        repo_id="FunAudioLLM/CosyVoice2-0.5B",
        filename="speech_tokenizer_v2.onnx",
        cache_dir=cache_dir,
    )
    logger.info(f"Downloaded to: {onnx_path}")
    return Path(onnx_path)


def load_onnx_weights(path: Path) -> Dict[str, np.ndarray]:
    """Load ONNX weights as numpy arrays using s3tokenizer's onnx2torch."""
    try:
        # Use s3tokenizer's conversion utility for proper key naming
        import torch
        from s3tokenizer.utils import onnx2torch

        pytorch_weights = onnx2torch(str(path), None, False)

        # Convert PyTorch tensors to numpy arrays
        weights = {}
        for key, value in pytorch_weights.items():
            if isinstance(value, torch.Tensor):
                weights[key] = value.cpu().numpy()
            else:
                weights[key] = np.array(value)

        return weights

    except ImportError:
        # Fallback: direct ONNX parsing (gives onnx:: internal names)
        logger.warning("s3tokenizer not installed, using raw ONNX parsing")
        logger.warning("This may result in incorrect weight names")
        import onnx
        from onnx import numpy_helper

        model = onnx.load(str(path))
        weights = {}

        for initializer in model.graph.initializer:
            weights[initializer.name] = numpy_helper.to_array(initializer)

        return weights


def numpy_to_mlx(weights: Dict[str, np.ndarray]) -> Dict:
    """Convert numpy arrays to MLX arrays."""
    import mlx.core as mx

    return {k: mx.array(v) for k, v in weights.items()}


def mlx_to_numpy(weights: Dict) -> Dict[str, np.ndarray]:
    """Convert MLX arrays back to numpy."""
    return {k: np.array(v) for k, v in weights.items()}


def save_mlx_safetensors(weights: Dict[str, np.ndarray], path: Path):
    """Save weights as MLX-compatible safetensors."""
    from safetensors.numpy import save_file

    # Ensure all values are numpy arrays with correct dtype
    clean_weights = {}
    for k, v in weights.items():
        if isinstance(v, np.ndarray):
            if v.dtype == np.float64:
                v = v.astype(np.float32)
            clean_weights[k] = v
        else:
            clean_weights[k] = np.array(v)

    save_file(clean_weights, path)
    logger.info(f"Saved: {path} ({len(clean_weights)} tensors)")


def generate_readme(path: Path, upload_repo: str):
    """Generate README.md model card for S3TokenizerV2 on Hugging Face."""
    from mlx_audio.version import __version__

    card_text = f"""---
library_name: mlx-audio-plus
base_model:
- FunAudioLLM/CosyVoice2-0.5B
tags:
- mlx
- speech-tokenizer
---

# {upload_repo}

S3TokenizerV2 (Supervised Semantic Speech Tokenizer) converted to MLX format from [FunAudioLLM/CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B).

This tokenizer is automatically downloaded when using Chatterbox, Chatterbox Turbo, or CosyVoice2 with [mlx-audio-plus](https://github.com/DePasqualeOrg/mlx-audio-plus) version **{__version__}**.
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


def convert_s3_tokenizer_v2(
    output_dir: Path,
    cache_dir: Path = None,
    upload_repo: str = None,
    dry_run: bool = False,
):
    """
    Convert S3TokenizerV2 weights to MLX format.

    Args:
        output_dir: Directory to save converted weights
        cache_dir: Optional custom cache directory for downloads
        upload_repo: Hugging Face repo to upload to (optional)
        dry_run: Generate files but skip upload
    """
    from mlx_audio.codec.models.s3tokenizer import S3TokenizerV2

    output_dir.mkdir(parents=True, exist_ok=True)

    # Download and convert S3Tokenizer from ONNX
    logger.info("Converting S3TokenizerV2...")
    onnx_path = download_s3tokenizer_v2_onnx(cache_dir)
    weights = load_onnx_weights(onnx_path)
    weights_mx = numpy_to_mlx(weights)

    # Use model's sanitize method for proper weight key remapping
    s3tok = S3TokenizerV2("speech_tokenizer_v2_25hz")
    weights_mx = s3tok.sanitize(weights_mx)
    weights = mlx_to_numpy(weights_mx)
    logger.info(f"  Converted {len(weights)} weights")

    # Save weights
    logger.info("\nSaving model.safetensors...")
    save_mlx_safetensors(weights, output_dir / "model.safetensors")

    # Create config.json
    logger.info("Creating config.json...")
    config = {
        "model_type": "s3_tokenizer_v2",
        "version": "2.0",
        "sample_rate": 16000,
        "token_rate": 25,
        "codebook_size": 6561,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Generate README if upload_repo is specified
    if upload_repo:
        logger.info("Generating README.md...")
        generate_readme(output_dir, upload_repo)

    logger.info(f"\n✅ S3TokenizerV2 conversion complete! Output: {output_dir}")
    logger.info("\nFiles created:")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            logger.info(f"  {f.name}: {size_mb:.1f} MB")

    # Upload if requested (and not dry run)
    if upload_repo and not dry_run:
        upload_to_hub(output_dir, upload_repo)
    elif upload_repo:
        logger.info(f"\n📁 Dry run - to upload to {upload_repo}, run without --dry-run")


def main():
    parser = argparse.ArgumentParser(
        description="Convert S3TokenizerV2 weights to MLX format"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for MLX weights (default: ./S3TokenizerV2)",
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
        help="Hugging Face repo to upload to (e.g., mlx-community/S3TokenizerV2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate all files including README but skip upload",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    output_dir = args.output_dir or Path("./S3TokenizerV2")
    upload_repo = args.upload_repo
    if upload_repo is None and args.dry_run:
        upload_repo = f"mlx-community/{output_dir.name}"

    convert_s3_tokenizer_v2(
        output_dir=output_dir,
        cache_dir=args.cache_dir,
        upload_repo=upload_repo,
        dry_run=args.dry_run or args.upload_repo is None,
    )


if __name__ == "__main__":
    main()
