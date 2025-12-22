#!/usr/bin/env python3
# Copyright © Anthony DePasquale
"""
Convert S3TokenizerV3 weights from ONNX to MLX format.

S3TokenizerV3 is used by CosyVoice3 and may be shared by future TTS models.

The converted weights are uploaded to mlx-community/S3TokenizerV3 and
automatically downloaded when using CosyVoice3 with mlx-audio-plus.

Usage:
    # Convert S3TokenizerV3 to MLX format
    python mlx_audio/codec/models/s3tokenizer/scripts/convert_v3.py

    # Upload to Hugging Face
    python mlx_audio/codec/models/s3tokenizer/scripts/convert_v3.py \\
        --upload-repo mlx-community/S3TokenizerV3

Requirements:
    pip install mlx safetensors huggingface_hub
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add mlx-audio-plus root to path for local imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)


def download_s3tokenizer_v3_onnx(cache_dir: Path = None) -> Path:
    """Download S3TokenizerV3 ONNX weights from Hugging Face.

    Args:
        cache_dir: Optional custom cache directory. If None, uses the default
            HF cache (~/.cache/huggingface/hub).
    """
    from huggingface_hub import hf_hub_download

    logger.info("Downloading S3TokenizerV3 ONNX from Hugging Face...")
    onnx_path = hf_hub_download(
        repo_id="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        filename="speech_tokenizer_v3.onnx",
        cache_dir=cache_dir,
    )
    logger.info(f"Downloaded to: {onnx_path}")
    return Path(onnx_path)


def generate_readme(path: Path, upload_repo: str):
    """Generate README.md model card for S3TokenizerV3 on Hugging Face."""
    from mlx_audio.version import __version__

    card_text = f"""---
library_name: mlx-audio-plus
base_model:
- FunAudioLLM/Fun-CosyVoice3-0.5B-2512
tags:
- mlx
- speech-tokenizer
---

# {upload_repo}

S3TokenizerV3 (Supervised Semantic Speech Tokenizer) converted to MLX format from [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512).

This tokenizer is automatically downloaded when using CosyVoice 3 with [mlx-audio-plus](https://github.com/DePasqualeOrg/mlx-audio-plus) version **{__version__}**.
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


def convert_s3_tokenizer_v3(
    output_dir: Path,
    source_path: Path = None,
    cache_dir: Path = None,
    upload_repo: str = None,
    dry_run: bool = False,
):
    """
    Convert S3TokenizerV3 weights to MLX format.

    Args:
        output_dir: Directory to save converted weights
        source_path: Path to local speech_tokenizer_v3.onnx (optional)
        cache_dir: Optional custom cache directory for downloads
        upload_repo: Hugging Face repo to upload to (optional)
        dry_run: Generate files but skip upload
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from mlx_audio.codec.models.s3tokenizer.model_v3 import S3TokenizerV3

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find ONNX file
    if source_path and (source_path / "speech_tokenizer_v3.onnx").exists():
        onnx_path = source_path / "speech_tokenizer_v3.onnx"
        logger.info(f"Using local ONNX: {onnx_path}")
    elif source_path and source_path.suffix == ".onnx":
        onnx_path = source_path
        logger.info(f"Using provided ONNX: {onnx_path}")
    else:
        onnx_path = download_s3tokenizer_v3_onnx(cache_dir)

    # Convert using the model's from_onnx method
    logger.info("Converting S3TokenizerV3 from ONNX...")
    model = S3TokenizerV3.from_onnx(str(onnx_path))

    # Extract weights
    weights = dict(tree_flatten(model.parameters()))
    logger.info(f"  Converted {len(weights)} weights")

    # Save as safetensors
    logger.info("Saving model.safetensors...")
    mx.save_safetensors(str(output_dir / "model.safetensors"), weights)

    # Create config.json
    config = {
        "model_type": "s3_tokenizer_v3",
        "n_mels": 128,
        "n_audio_ctx": 1500,
        "n_audio_state": 1280,
        "n_audio_head": 20,
        "n_audio_layer": 12,
        "n_codebook_size": 6561,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Generate README if upload_repo is specified
    if upload_repo:
        logger.info("Generating README.md...")
        generate_readme(output_dir, upload_repo)

    logger.info(f"\n✅ S3TokenizerV3 conversion complete! Output: {output_dir}")
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
        description="Convert S3TokenizerV3 weights to MLX format"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for MLX weights (default: ./S3TokenizerV3)",
    )
    parser.add_argument(
        "--source-path",
        type=Path,
        default=None,
        help="Path to local speech_tokenizer_v3.onnx file or directory containing it",
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
        help="Hugging Face repo to upload to (e.g., mlx-community/S3TokenizerV3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate all files including README but skip upload",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    output_dir = args.output_dir or Path("./S3TokenizerV3")
    upload_repo = args.upload_repo
    if upload_repo is None and args.dry_run:
        upload_repo = f"mlx-community/{output_dir.name}"

    convert_s3_tokenizer_v3(
        output_dir=output_dir,
        source_path=args.source_path,
        cache_dir=args.cache_dir,
        upload_repo=upload_repo,
        dry_run=args.dry_run or args.upload_repo is None,
    )


if __name__ == "__main__":
    main()
