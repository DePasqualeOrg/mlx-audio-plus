#!/usr/bin/env python3
"""
Convert CosyVoice2 PyTorch weights to MLX format.

Usage:
    # Convert to local directory
    python -m mlx_audio.tts.models.cosyvoice2.scripts.convert --output-dir ~/.cache/mlx_audio/cosyvoice2

    # Convert and upload to Hugging Face
    python -m mlx_audio.tts.models.cosyvoice2.scripts.convert --upload-repo mlx-community/CosyVoice2-0.5B

    # Dry run (generate files but don't upload)
    python -m mlx_audio.tts.models.cosyvoice2.scripts.convert --upload-repo mlx-community/CosyVoice2-0.5B --dry-run

This will download from FunAudioLLM/CosyVoice2-0.5B and convert to MLX format.

Requirements (for conversion only):
    pip install torch safetensors huggingface_hub

After conversion, the model only needs:
    pip install mlx mlx-lm mlx-audio-plus
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.numpy import save_file


def convert_conv_weight(weight: np.ndarray) -> np.ndarray:
    """Convert PyTorch Conv1d weight to MLX format.

    PyTorch Conv1d: (out_channels, in_channels, kernel_size)
    MLX Conv1d: (out_channels, kernel_size, in_channels)
    """
    if weight.ndim == 3:
        return np.transpose(weight, (0, 2, 1))
    return weight


def convert_conv_transpose_weight(weight: np.ndarray) -> np.ndarray:
    """Convert PyTorch ConvTranspose1d weight to MLX format.

    PyTorch ConvTranspose1d: (in_channels, out_channels, kernel_size)
    MLX ConvTranspose1d: (out_channels, kernel_size, in_channels)
    """
    if weight.ndim == 3:
        return np.transpose(weight, (1, 2, 0))
    return weight


def convert_llm_weights(pt_weights: dict) -> tuple:
    """Convert LLM (speech token embedding and decoder) weights.

    CosyVoice2's LLM has:
    - llm.model.xxx: The fine-tuned Qwen2 model weights
    - speech_embedding: Speech token embeddings
    - llm_decoder: Speech output decoder
    - llm_embedding: Special token embeddings (sos_eos, task_id)

    Returns:
        Tuple of (speech_weights, qwen2_weights)
        - speech_weights: speech_embedding, llm_decoder, llm_embedding
        - qwen2_weights: The fine-tuned Qwen2 weights for mlx_lm format
    """
    speech_weights = {}
    qwen2_weights = {}

    for key, value in pt_weights.items():
        np_value = value.cpu().numpy()

        if key.startswith("llm.model."):
            # Extract Qwen2 weights - remap from llm.model.model.xxx to model.xxx
            # PyTorch CosyVoice2 structure:
            #   llm.model.model.embed_tokens.weight (Qwen2Model.embed_tokens)
            #   llm.model.lm_head.weight (Qwen2ForCausalLM.lm_head)
            # MLX mlx_lm expects:
            #   model.embed_tokens.weight
            #   (no lm_head in safetensors, it's often tied to embed_tokens)
            new_key = key[len("llm.model.") :]  # Remove "llm.model." prefix
            qwen2_weights[new_key] = np_value
        else:
            # Handle weight conversions for non-Qwen2 weights
            if "conv" in key.lower() and "weight" in key:
                if "transpose" in key.lower() or "deconv" in key.lower():
                    np_value = convert_conv_transpose_weight(np_value)
                else:
                    np_value = convert_conv_weight(np_value)
            speech_weights[key] = np_value

    return speech_weights, qwen2_weights


def remap_decoder_key(key: str) -> str:
    """Remap PyTorch decoder keys to MLX format.

    PyTorch DownBlock/UpBlock structure: (resnet, [transformers], downsample/upsample)
    - Index 0: resnet block
    - Index 1: list of transformer blocks
    - Index 2: downsample/upsample

    MLX uses named attributes: resnet, transformer_N, downsample/upsample
    """
    import re

    # Block list indexing: down_blocks.0.xxx -> down_blocks_0.xxx
    key = re.sub(r"\.down_blocks\.(\d+)\.", r".down_blocks_\1.", key)
    key = re.sub(r"\.mid_blocks\.(\d+)\.", r".mid_blocks_\1.", key)
    key = re.sub(r"\.up_blocks\.(\d+)\.", r".up_blocks_\1.", key)

    # Down/Up block component mapping (0=resnet, 1.N=transformer_N, 2=downsample/upsample)
    # PyTorch: down_blocks_0.0.xxx -> MLX: down_blocks_0.resnet.xxx
    # PyTorch: down_blocks_0.1.N.xxx -> MLX: down_blocks_0.transformer_N.xxx
    # PyTorch: down_blocks_0.2.xxx -> MLX: down_blocks_0.downsample.xxx
    key = re.sub(r"(down_blocks_\d+)\.0\.", r"\1.resnet.", key)
    key = re.sub(r"(down_blocks_\d+)\.1\.(\d+)\.", r"\1.transformer_\2.", key)
    key = re.sub(r"(down_blocks_\d+)\.2\.", r"\1.downsample.", key)

    key = re.sub(r"(up_blocks_\d+)\.0\.", r"\1.resnet.", key)
    key = re.sub(r"(up_blocks_\d+)\.1\.(\d+)\.", r"\1.transformer_\2.", key)
    key = re.sub(r"(up_blocks_\d+)\.2\.", r"\1.upsample.", key)

    # Mid block component mapping (0=resnet, 1.N=transformer_N)
    key = re.sub(r"(mid_blocks_\d+)\.0\.", r"\1.resnet.", key)
    key = re.sub(r"(mid_blocks_\d+)\.1\.(\d+)\.", r"\1.transformer_\2.", key)

    # ResNet block internal mapping
    # PyTorch CausalBlock1D structure:
    #   block.0 = CausalConv1d
    #   block.1 = Transpose (no weights)
    #   block.2 = LayerNorm
    #   block.3 = Transpose (no weights)
    #   block.4 = Mish (no weights)
    # MLX CausalBlock1D structure:
    #   conv = CausalConv1d
    #   norm = LayerNorm
    key = re.sub(r"\.block1\.block\.0\.", r".block1.conv.conv.", key)
    key = re.sub(r"\.block1\.block\.2\.", r".block1.norm.", key)
    key = re.sub(r"\.block2\.block\.0\.", r".block2.conv.conv.", key)
    key = re.sub(r"\.block2\.block\.2\.", r".block2.norm.", key)

    # MLP mapping: mlp.1 -> mlp_linear
    key = re.sub(r"\.mlp\.1\.", r".mlp_linear.", key)

    # Transformer attention mapping
    # PyTorch: attn1.to_q/k/v -> MLX: attn.query_proj/key_proj/value_proj
    # PyTorch: attn1.to_out.0 -> MLX: attn.out_proj
    key = key.replace(".attn1.to_q.", ".attn.query_proj.")
    key = key.replace(".attn1.to_k.", ".attn.key_proj.")
    key = key.replace(".attn1.to_v.", ".attn.value_proj.")
    key = key.replace(".attn1.to_out.0.", ".attn.out_proj.")

    # FeedForward mapping
    # PyTorch: ff.net.0.proj (GEGLU) -> MLX: ff.layers.0
    # PyTorch: ff.net.2 (output) -> MLX: ff.layers.1
    key = key.replace(".ff.net.0.proj.", ".ff.layers.0.")
    key = key.replace(".ff.net.2.", ".ff.layers.1.")

    # Final block mapping - same structure as CausalBlock1D
    # block.0 = Conv, block.2 = LayerNorm
    key = re.sub(r"\.final_block\.block\.0\.", r".final_block.conv.conv.", key)
    key = re.sub(r"\.final_block\.block\.2\.", r".final_block.norm.", key)

    # Downsample/upsample conv mapping
    # For regular downsamples with .conv
    key = re.sub(r"\.downsample\.weight$", r".downsample.conv.weight", key)
    key = re.sub(r"\.downsample\.bias$", r".downsample.conv.bias", key)
    key = re.sub(r"\.upsample\.weight$", r".upsample.conv.weight", key)
    key = re.sub(r"\.upsample\.bias$", r".upsample.conv.bias", key)

    return key


def convert_flow_weights(pt_weights: dict) -> dict:
    """Convert Flow model weights (encoder + decoder)."""
    mlx_weights = {}
    import re

    for key, value in pt_weights.items():
        np_value = value.cpu().numpy()

        # Note: We keep the decoder.estimator.xxx prefix structure
        # as our MLX CausalMaskedDiffWithXvec expects this hierarchy

        # Remap LinearNoSubsampling keys
        # PyTorch: embed.out.0.xxx (Linear) -> embed.linear.xxx
        # PyTorch: embed.out.1.xxx (LayerNorm) -> embed.norm.xxx
        # PyTorch: embed.out.2 is Dropout (no weights)
        key = key.replace(".out.0.", ".linear.")
        key = key.replace(".out.1.", ".norm.")

        # Remap encoder/up_encoder module list keys
        # PyTorch: encoders.0.xxx -> encoders_0.xxx
        # PyTorch: up_encoders.0.xxx -> up_encoders_0.xxx
        key = re.sub(r"encoder\.encoders\.(\d+)\.", r"encoder.encoders_\1.", key)
        key = re.sub(r"encoder\.up_encoders\.(\d+)\.", r"encoder.up_encoders_\1.", key)

        # Remap decoder/estimator keys
        if "decoder.estimator" in key:
            key = remap_decoder_key(key)

        # Convert convolution weights AFTER key remapping
        # so we can check for the right patterns
        # Match both ".conv" and "res_conv", "final_proj" etc.
        is_conv_weight = (
            "conv" in key.lower() or "final_proj" in key
        ) and "weight" in key
        if is_conv_weight and np_value.ndim == 3:
            # NOTE: In CosyVoice2's CausalConditionalDecoder, the "upsample" layers are
            # actually CausalConv1d (regular Conv1d with causal padding), NOT ConvTranspose1d.
            # The only actual ConvTranspose in the original Matcha is in Upsample1D,
            # but CosyVoice2 replaces the final upsample with CausalConv1d.
            # So we should NOT use convert_conv_transpose_weight here.
            np_value = convert_conv_weight(np_value)

        mlx_weights[key] = np_value

    return mlx_weights


def combine_weight_norm(weights: dict, base_key: str) -> np.ndarray:
    """Combine weight normalization parameters into actual weight.

    PyTorch weight_norm stores:
    - original0: direction/scale (g)
    - original1: unnormalized weight (v)

    Combined weight = g * v / ||v||
    """
    g_key = f"{base_key}.parametrizations.weight.original0"
    v_key = f"{base_key}.parametrizations.weight.original1"

    if g_key in weights and v_key in weights:
        g = weights[g_key]
        v = weights[v_key]
        # Normalize v and scale by g
        # For Conv1d: v is (out, kernel, in), normalize over (kernel, in)
        norm = np.linalg.norm(v.reshape(v.shape[0], -1), axis=1, keepdims=True)
        norm = norm.reshape(g.shape)
        weight = g * v / (norm + 1e-8)
        return weight
    return None


def remap_hift_key(key: str) -> str:
    """Remap HiFT key from PyTorch to MLX format.

    Main mapping: condnet.N -> condnet_N (for f0_predictor.condnet)
    """
    import re

    # Map condnet.N to condnet_N
    key = re.sub(r"\.condnet\.(\d+)\.", r".condnet_\1.", key)
    return key


def convert_hift_weights(pt_weights: dict) -> dict:
    """Convert HiFT (HiFi-GAN) vocoder weights."""
    mlx_weights = {}

    # First pass: identify base keys with weight_norm
    weight_norm_bases = set()
    for key in pt_weights.keys():
        if ".parametrizations.weight.original" in key:
            base = key.rsplit(".parametrizations.weight.original", 1)[0]
            weight_norm_bases.add(base)

    # Convert weights, combining weight_norm parameters
    processed_keys = set()
    for key, value in pt_weights.items():
        # Skip if already processed as part of weight_norm
        if key in processed_keys:
            continue

        # Check if this is a weight_norm parameter
        if ".parametrizations.weight.original" in key:
            base = key.rsplit(".parametrizations.weight.original", 1)[0]
            if base in weight_norm_bases:
                # Combine weight_norm
                combined = combine_weight_norm(
                    {k: v.cpu().numpy() for k, v in pt_weights.items()}, base
                )
                if combined is not None:
                    # Mark both keys as processed
                    processed_keys.add(f"{base}.parametrizations.weight.original0")
                    processed_keys.add(f"{base}.parametrizations.weight.original1")

                    # Convert to MLX format
                    new_key = f"{base}.weight"
                    # Remap key
                    new_key = remap_hift_key(new_key)
                    if combined.ndim == 3:
                        if "ups" in base:
                            combined = convert_conv_transpose_weight(combined)
                        else:
                            combined = convert_conv_weight(combined)
                    mlx_weights[new_key] = combined
            continue

        np_value = value.cpu().numpy()

        # Convert convolution weights
        # Match conv, source_downs, source_resblocks, resblocks, f0_upsample, etc.
        is_conv_weight = "weight" in key and np_value.ndim == 3
        if is_conv_weight:
            # Transposed convs in upsampling path
            if "ups" in key:
                np_value = convert_conv_transpose_weight(np_value)
            else:
                np_value = convert_conv_weight(np_value)

        # Remap key
        key = remap_hift_key(key)
        mlx_weights[key] = np_value

    return mlx_weights


def generate_readme(output_dir: Path, model_id: str, upload_repo: str) -> None:
    """Generate README.md model card for Hugging Face."""
    from mlx_audio.version import __version__

    card_text = f"""---
library_name: mlx-audio-plus
base_model:
- {model_id}
tags:
- mlx
- tts
- cosyvoice2
pipeline_tag: text-to-speech
language:
- en
- zh
- ja
- ko
---

# {upload_repo}

This model was converted to MLX format from [{model_id}](https://huggingface.co/{model_id}) using [mlx-audio-plus](https://github.com/DePasqualeOrg/mlx-audio-plus) version **{__version__}**.

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
    text="Hello, this is CosyVoice2 on MLX!",
    model="{upload_repo}",
    ref_audio="reference.wav",
    file_prefix="output",
)
```
"""
    card_path = output_dir / "README.md"
    with open(card_path, "w") as f:
        f.write(card_text)
    print(f"   Created {card_path}")


def upload_to_hub(output_dir: Path, upload_repo: str) -> None:
    """Upload converted model to Hugging Face Hub."""
    from huggingface_hub import HfApi

    print(f"\nUploading to {upload_repo}...")
    api = HfApi()
    api.create_repo(repo_id=upload_repo, exist_ok=True)
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=upload_repo,
        repo_type="model",
    )
    print(f"Upload successful! Visit https://huggingface.co/{upload_repo}")


def quantize_qwen2_weights(
    qwen2_weights: dict,
    bits: int = 4,
    group_size: int = 64,
) -> dict:
    """
    Quantize Qwen2 LLM weights using MLX quantization.

    Only quantizes the transformer layers (model.layers.*), keeping embeddings
    and other components in full precision.

    Args:
        qwen2_weights: Dict of numpy arrays with Qwen2 weights
        bits: Quantization bits (4 or 8)
        group_size: Quantization group size (default: 64)

    Returns:
        Dict of quantized weights (mix of numpy and MLX arrays)
    """
    import mlx.nn as nn
    from mlx_lm.models.qwen2 import Model as Qwen2Model
    from mlx_lm.models.qwen2 import ModelArgs

    print(f"   Quantizing Qwen2 to {bits}-bit (group_size={group_size})...")

    # Qwen2-0.5B-Instruct config
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

    # Create model and load weights
    qwen2_model = Qwen2Model(qwen2_args)

    # Convert numpy weights to MLX
    # Filter out lm_head.weight since it's tied to embeddings in Qwen2-0.5B
    mlx_weights = {}
    for k, v in qwen2_weights.items():
        if k == "lm_head.weight":
            # Skip - tied to embeddings
            continue
        mlx_weights[k] = mx.array(v)
    qwen2_model.load_weights(list(mlx_weights.items()))
    mx.eval(qwen2_model.parameters())

    # Calculate original size
    from mlx.utils import tree_flatten

    orig_weights = dict(tree_flatten(qwen2_model.parameters()))
    orig_size = sum(v.nbytes for v in orig_weights.values())
    print(f"   Original size: {orig_size / 1e9:.2f} GB")

    # Define quantization predicate - only quantize transformer layers
    quantized_count = [0]

    def should_quantize(path, module):
        """Only quantize Linear layers in transformer blocks."""
        if isinstance(module, nn.Linear):
            # Quantize model.layers.* (transformer blocks)
            if "model.layers" in path:
                quantized_count[0] += 1
                return True
        return False

    # Apply quantization
    nn.quantize(
        qwen2_model, bits=bits, group_size=group_size, class_predicate=should_quantize
    )
    mx.eval(qwen2_model.parameters())
    print(f"   Quantized {quantized_count[0]} Linear layers")

    # Get quantized weights
    quantized_weights = dict(tree_flatten(qwen2_model.parameters()))
    new_size = sum(v.nbytes for v in quantized_weights.values())
    print(f"   New size: {new_size / 1e9:.2f} GB")
    print(f"   Reduction: {(1 - new_size / orig_size) * 100:.1f}%")

    return quantized_weights


def save_mlx_weights(weights: dict, path: Path) -> None:
    """Save MLX weights directly (preserves quantized format)."""
    mx.save_safetensors(str(path), weights, metadata={"format": "mlx"})


def convert_campplus_from_chatterbox() -> dict:
    """Extract and convert CAMPlus weights from Chatterbox's s3gen.safetensors.

    CosyVoice2 and Chatterbox use the same CAMPlus architecture for speaker
    embeddings. We extract the speaker_encoder.xvector.* weights from Chatterbox
    and convert them using CAMPPlus.sanitize().

    Returns:
        Dict of converted weights ready to save as campplus.safetensors
    """
    from mlx_audio.tts.models.chatterbox.s3gen.xvector import CAMPPlus

    print("   Downloading Chatterbox s3gen.safetensors...")
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

    print(f"   Extracted {len(xvector_weights)} xvector weights")

    # Use CAMPPlus.sanitize() to convert weights
    campplus = CAMPPlus()
    sanitized_weights = campplus.sanitize(xvector_weights)

    # Convert to numpy for safetensors
    numpy_weights = {}
    for key, value in sanitized_weights.items():
        numpy_weights[key] = np.array(value)

    print(f"   Sanitized to {len(numpy_weights)} MLX weights")
    return numpy_weights


def get_default_output_dir(quantize: bool, q_bits: int, dtype: str) -> Path:
    """Generate default output directory name based on conversion settings.

    Examples:
        - CosyVoice2-0.5B-fp16 (default float16)
        - CosyVoice2-0.5B-fp32 (float32)
        - CosyVoice2-0.5B-4bit (4-bit quantized)
        - CosyVoice2-0.5B-8bit (8-bit quantized)
    """
    base_name = "CosyVoice2-0.5B"

    if quantize:
        suffix = f"{q_bits}bit"
    elif dtype == "float32":
        suffix = "fp32"
    elif dtype == "bfloat16":
        suffix = "bf16"
    else:  # float16 is default
        suffix = "fp16"

    return Path.home() / ".cache" / "mlx_audio" / f"{base_name}-{suffix}"


def convert_from_source(
    model_id: str = "FunAudioLLM/CosyVoice2-0.5B",
    output_dir: Path = None,
    quantize: bool = False,
    q_bits: int = 4,
    q_group_size: int = 64,
    dtype: str = "float16",
    upload_repo: str = None,
    dry_run: bool = False,
) -> None:
    """
    Convert CosyVoice2 PyTorch weights to MLX format.

    This function is called by the central conversion utility when it detects
    a CosyVoice2 model, or can be called directly.

    Args:
        model_id: Hugging Face model ID (default: FunAudioLLM/CosyVoice2-0.5B)
        output_dir: Output directory for MLX weights
        quantize: Whether to quantize weights
        q_bits: Quantization bits (4 or 8)
        q_group_size: Quantization group size (default: 64)
        dtype: Data type for weights (float16, bfloat16, float32)
        upload_repo: Hugging Face repo to upload to
        dry_run: Generate files but skip upload
    """
    import shutil

    if output_dir is None:
        output_dir = get_default_output_dir(quantize, q_bits, dtype)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine target dtype
    dtype_map = {"float16": np.float16, "bfloat16": np.float16, "float32": np.float32}
    target_dtype = dtype_map.get(dtype, np.float16) if dtype else np.float16
    dtype_name = dtype or "float16"

    print(f"Converting CosyVoice2 from {model_id}")
    print(f"Output directory: {output_dir}")
    print(f"Target dtype: {dtype_name}")
    if quantize:
        print(f"Quantization: {q_bits}-bit for Qwen2 LLM (group_size={q_group_size})")

    # Download model files (or use local path)
    print("\n1. Downloading model files...")
    model_path = Path(model_id)
    if model_path.exists() and model_path.is_dir():
        # Local path provided
        cache_path = model_path
        print(f"   Using local path: {cache_path}")
    else:
        # Hugging Face repo ID
        cache_dir = snapshot_download(
            model_id,
            allow_patterns=["*.pt", "CosyVoice-BlankEN/*"],
        )
        cache_path = Path(cache_dir)
        print(f"   Downloaded to: {cache_path}")

    # Generate tokenizer files (tokenizer.json + tokenizer_config.json)
    # tokenizer.json is the "fast tokenizer" format that contains vocab + merges + config
    # Both Python transformers and swift-transformers can use this format
    tokenizer_src = cache_path / "CosyVoice-BlankEN"
    if tokenizer_src.exists():
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_src))
            # save_pretrained with legacy_format=False only generates:
            # - tokenizer.json (fast tokenizer with vocab + merges)
            # - tokenizer_config.json (tokenizer settings)
            # - special_tokens_map.json (not needed, will be deleted)
            tokenizer.save_pretrained(str(output_dir), legacy_format=False)

            # Remove files that are not needed for TTS inference
            # - special_tokens_map.json: redundant (in tokenizer.json and tokenizer_config.json)
            # - chat_template.jinja: only needed for chat models, not TTS
            for redundant_file in ["special_tokens_map.json", "chat_template.jinja"]:
                file_path = output_dir / redundant_file
                if file_path.exists():
                    file_path.unlink()

            print(
                f"   Generated tokenizer files (tokenizer.json + tokenizer_config.json)"
            )
        except Exception as e:
            print(f"   Warning: Could not generate tokenizer files: {e}")
            # Fallback: copy individual files for Python transformers compatibility
            for config_file in ["merges.txt", "tokenizer_config.json", "vocab.json"]:
                src_file = tokenizer_src / config_file
                if src_file.exists():
                    shutil.copy(src_file, output_dir / config_file)
            print(
                f"   Fallback: Copied legacy tokenizer files (vocab.json, merges.txt)"
            )
            print(f"   Note: swift-transformers requires tokenizer.json")

    # Helper to convert weights to target dtype
    def apply_dtype(weights: dict) -> dict:
        return {k: v.astype(target_dtype) for k, v in weights.items()}

    # Collect all model weights (except Qwen2) into a single dict
    # Will be saved as model.safetensors at the end
    all_weights = {}

    # Convert LLM weights
    print("\n3. Converting LLM weights...")
    llm_pt_path = cache_path / "llm.pt"
    qwen2_quantized = False
    if llm_pt_path.exists():
        llm_pt = torch.load(llm_pt_path, map_location="cpu", weights_only=True)
        speech_weights, qwen2_weights = convert_llm_weights(llm_pt)

        # Apply dtype conversion to speech weights and add to consolidated weights
        speech_weights = apply_dtype(speech_weights)
        for key, value in speech_weights.items():
            all_weights[f"llm.{key}"] = value
        print(f"   Converted LLM speech weights ({len(speech_weights)} tensors)")

        # Add Qwen2 weights to consolidated model
        if qwen2_weights:
            if quantize:
                # Quantize Qwen2 LLM weights
                qwen2_weights = apply_dtype(
                    qwen2_weights
                )  # First convert to target dtype
                quantized_weights = quantize_qwen2_weights(
                    qwen2_weights, bits=q_bits, group_size=q_group_size
                )
                # Add quantized weights with qwen2. prefix
                for key, value in quantized_weights.items():
                    # MLX quantized weights are already mx.array, convert to numpy
                    all_weights[f"qwen2.{key}"] = np.array(value)
                print(f"   Quantized Qwen2 weights ({len(quantized_weights)} tensors)")
                qwen2_quantized = True
            else:
                qwen2_weights = apply_dtype(qwen2_weights)
                for key, value in qwen2_weights.items():
                    all_weights[f"qwen2.{key}"] = value
                print(f"   Converted Qwen2 weights ({len(qwen2_weights)} tensors)")
    else:
        print(f"   Warning: {llm_pt_path} not found")

    # Convert Flow weights
    print("\n4. Converting Flow weights...")
    flow_pt_path = cache_path / "flow.pt"
    if flow_pt_path.exists():
        flow_pt = torch.load(flow_pt_path, map_location="cpu", weights_only=True)
        flow_mlx = convert_flow_weights(flow_pt)
        flow_mlx = apply_dtype(flow_mlx)
        for key, value in flow_mlx.items():
            all_weights[f"flow.{key}"] = value
        print(f"   Converted Flow weights ({len(flow_mlx)} tensors)")

        # Note: Random noise for flow matching is generated at runtime by MLX
        # This means output won't be bit-exact with PyTorch, but is still valid
    else:
        print(f"   Warning: {flow_pt_path} not found")

    # Convert HiFT weights
    print("\n5. Converting HiFT (vocoder) weights...")
    hift_pt_path = cache_path / "hift.pt"
    if hift_pt_path.exists():
        hift_pt = torch.load(hift_pt_path, map_location="cpu", weights_only=True)
        hift_mlx = convert_hift_weights(hift_pt)
        hift_mlx = apply_dtype(hift_mlx)
        for key, value in hift_mlx.items():
            all_weights[f"hift.{key}"] = value
        print(f"   Converted HiFT weights ({len(hift_mlx)} tensors)")
    else:
        print(f"   Warning: {hift_pt_path} not found")

    # Convert CAMPlus from Chatterbox (same architecture as CosyVoice2)
    print("\n6. Converting CAMPlus speaker encoder...")
    try:
        campplus_mlx = convert_campplus_from_chatterbox()
        campplus_mlx = apply_dtype(campplus_mlx)
        for key, value in campplus_mlx.items():
            all_weights[f"campplus.{key}"] = value
        print(f"   Converted CAMPlus weights ({len(campplus_mlx)} tensors)")
    except Exception as e:
        print(f"   Warning: Failed to convert CAMPlus: {e}")
        print("   Speaker embeddings will not be available")

    # Save consolidated model weights
    print("\n7. Saving consolidated model.safetensors...")
    save_file(all_weights, str(output_dir / "model.safetensors"))
    print(f"   Saved model.safetensors ({len(all_weights)} total tensors)")

    # Create a simple config.json for MLX loading
    print("\n8. Creating MLX config...")
    mlx_config = {
        "model_type": "cosyvoice2",
        "version": "0.5B",
        "sample_rate": 24000,
        "mel_channels": 80,
        "speech_token_size": 6561,
        "dtype": dtype_name,
    }
    if qwen2_quantized:
        mlx_config["quantization"] = {
            "bits": q_bits,
            "group_size": q_group_size,
            "quantized_components": ["tokenizer/model.layers"],
        }
    with open(output_dir / "config.json", "w") as f:
        json.dump(mlx_config, f, indent=2)
    print("   Created config.json")

    # Generate README if uploading
    if upload_repo:
        print("\n8. Generating README.md...")
        generate_readme(output_dir, model_id, upload_repo)

    # Print summary
    print(f"\n{'=' * 60}")
    print("Conversion complete!")
    print(f"{'=' * 60}")
    print(f"\nOutput directory: {output_dir}")
    print("\nFiles created:")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}: {size_mb:.1f} MB")
        elif f.is_dir():
            print(f"  {f.name}/")

    print("\nTo use the model:")
    if upload_repo:
        print(
            f'  python -m mlx_audio.tts.generate --model {upload_repo} --text "Hello" --ref_audio ref.wav'
        )
    else:
        print(
            f'  python -m mlx_audio.tts.generate --model {output_dir} --text "Hello" --ref_audio ref.wav'
        )

    # Upload to Hugging Face if requested
    if upload_repo and not dry_run:
        upload_to_hub(output_dir, upload_repo)
    elif upload_repo:
        print(f"\n📁 Dry run - to upload to {upload_repo}, run without --dry-run")


def main():
    """CLI entry point for CosyVoice2 conversion."""
    parser = argparse.ArgumentParser(description="Convert CosyVoice2 to MLX")
    parser.add_argument(
        "--model-id",
        default="FunAudioLLM/CosyVoice2-0.5B",
        help="Hugging Face model ID",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for MLX weights (default: ~/.cache/mlx_audio/CosyVoice2-0.5B-{fp16,fp32,4bit,8bit})",
    )
    parser.add_argument(
        "--upload-repo",
        type=str,
        default=None,
        help="Hugging Face repo to upload to (e.g., mlx-community/CosyVoice2-0.5B)",
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
        help="Quantize weights (not yet supported)",
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
        choices=["float16", "bfloat16", "float32"],
        help="Data type for weights (default: float16)",
    )
    args = parser.parse_args()

    convert_from_source(
        model_id=args.model_id,
        output_dir=args.output_dir,
        quantize=args.quantize,
        q_bits=args.q_bits,
        q_group_size=args.q_group_size,
        dtype=args.dtype,
        upload_repo=args.upload_repo,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
