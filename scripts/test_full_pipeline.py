#!/usr/bin/env python3
"""Test the full Chatterbox pipeline with the default voice."""

import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np
import scipy.io.wavfile as wav

from huggingface_hub import hf_hub_download


def main():
    print("=" * 60)
    print("Full Chatterbox Pipeline Test")
    print("=" * 60)

    # Use pre-converted weights
    ckpt_dir = Path("/tmp/chatterbox-test")
    if not (ckpt_dir / "model.safetensors").exists():
        print(f"ERROR: Converted weights not found at {ckpt_dir}")
        print("Run: PYTHONPATH=. python mlx_audio/tts/models/chatterbox/scripts/convert_chatterbox.py --output-dir /tmp/chatterbox-test")
        return
    print(f"Checkpoint directory: {ckpt_dir}")

    # Load model
    print("\nLoading model...")
    from mlx_audio.tts.models.chatterbox.chatterbox import Model

    model = Model.from_pretrained(ckpt_dir)

    # Load reference audio for voice cloning
    print("\nLoading reference audio...")
    ref_audio_path = Path(__file__).parent.parent / "reference-audio" / "anthony.wav"
    if not ref_audio_path.exists():
        print(f"Reference audio not found at {ref_audio_path}")
        print("Please provide a reference wav file in reference-audio/")
        return

    import soundfile as sf
    ref_audio, ref_sr = sf.read(ref_audio_path)
    ref_audio = mx.array(ref_audio.astype(np.float32))
    print(f"Reference audio: {len(ref_audio)/ref_sr:.2f}s at {ref_sr}Hz")

    # Generate speech
    text = "Hello world! This is a test of the Chatterbox speech synthesizer."
    print(f"\nText: {text}")
    print("\nGenerating speech...")

    import time
    start_time = time.time()

    for result in model.generate(
        text=text,
        audio_prompt=ref_audio,
        audio_prompt_sr=ref_sr,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        repetition_penalty=2.0,
        top_p=0.8,
        verbose=True,
    ):
        audio_np = np.array(result.audio)
        sample_rate = result.sample_rate

    elapsed = time.time() - start_time
    duration = len(audio_np) / sample_rate

    print(f"\nGeneration complete!")
    print(f"Audio shape: {audio_np.shape}")
    print(f"Audio duration: {duration:.2f} seconds")
    print(f"Generation time: {elapsed:.2f} seconds")
    print(f"Real-time factor: {elapsed/duration:.2f}x")
    print(f"Audio range: [{audio_np.min():.4f}, {audio_np.max():.4f}]")

    # Analyze audio quality
    # Simple SNR estimate
    noise_floor = np.percentile(np.abs(audio_np), 10)
    signal_level = np.percentile(np.abs(audio_np), 90)
    snr_estimate = 20 * np.log10(signal_level / (noise_floor + 1e-10))
    print(f"SNR estimate: {snr_estimate:.1f} dB")

    # Save audio
    output_path = Path("pipeline_test.wav")
    audio_int16 = (audio_np * 32767).astype(np.int16)
    wav.write(output_path, sample_rate, audio_int16)
    print(f"\nSaved audio to {output_path}")


if __name__ == "__main__":
    main()
