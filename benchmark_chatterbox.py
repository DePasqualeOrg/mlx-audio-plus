#!/usr/bin/env python3
"""
Chatterbox Pipeline Benchmark

Measures timing for each stage of the Chatterbox TTS pipeline to identify bottlenecks.
"""

import argparse
import tempfile
import time
import urllib.request
from pathlib import Path

import mlx.core as mx
import numpy as np

# Default reference audio URL (same as Swift benchmark)
DEFAULT_REFERENCE_AUDIO_URL = "https://archive.org/download/short_poetry_001_librivox/dead_boche_graves_sm.mp3"


def download_audio(url: str) -> str:
    """Download audio file to temp location and return path."""
    print(f"Downloading reference audio from {url}...")
    suffix = Path(url).suffix or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        urllib.request.urlretrieve(url, f.name)
        return f.name


def benchmark_chatterbox(
    model_path: str,
    reference_audio_path: str = None,
    reference_audio_url: str = None,
    text: str = "Hello, this is a test of the Chatterbox text to speech system.",
    num_runs: int = 3,
    warmup_runs: int = 1,
    seed: int = 42,
):
    """Run benchmark on Chatterbox pipeline stages."""

    print("=" * 60)
    print("Chatterbox Pipeline Benchmark (Python MLX)")
    print("=" * 60)

    # Set random seed for reproducibility
    mx.random.seed(seed)
    print(f"\nRandom seed: {seed}")

    # Import here to time model loading
    from mlx_audio.tts.models.chatterbox.chatterbox import Model

    # Load reference audio
    import soundfile as sf

    # Download if URL provided, otherwise use local path
    if reference_audio_path is None:
        audio_url = reference_audio_url or DEFAULT_REFERENCE_AUDIO_URL
        reference_audio_path = download_audio(audio_url)

    ref_wav, ref_sr = sf.read(reference_audio_path)
    if ref_wav.ndim > 1:
        ref_wav = ref_wav.mean(axis=1)  # Convert to mono
    ref_wav = mx.array(ref_wav.astype(np.float32))

    print(f"\nReference audio: {reference_audio_path}")
    print(f"  Sample rate: {ref_sr} Hz")
    print(f"  Duration: {len(ref_wav) / ref_sr:.2f}s")
    print(f"\nText: \"{text}\"")
    print(f"\nModel: {model_path}")

    # Time model loading
    print("\n" + "-" * 40)
    print("Stage 0: Model Loading")
    print("-" * 40)

    load_start = time.perf_counter()
    model = Model.from_pretrained(model_path)
    mx.eval(model.parameters())  # Ensure weights are loaded
    load_time = time.perf_counter() - load_start
    print(f"  Model load time: {load_time:.3f}s")

    # Warmup runs
    print(f"\nPerforming {warmup_runs} warmup run(s)...")
    for _ in range(warmup_runs):
        for result in model.generate(
            text=text,
            audio_prompt=ref_wav,
            audio_prompt_sr=ref_sr,
            verbose=False,
        ):
            mx.eval(result.audio)

    # Benchmark runs with detailed timing
    print(f"\nPerforming {num_runs} benchmark run(s)...")

    stage_times = {
        "prepare_conditionals": [],
        "text_tokenization": [],
        "t3_inference": [],
        "s3gen_waveform": [],
        "total": [],
    }

    for run in range(num_runs):
        print(f"\n--- Run {run + 1}/{num_runs} ---")

        # Reset seed for each run for reproducibility
        mx.random.seed(seed + run)

        total_start = time.perf_counter()

        # Stage 1: Prepare conditionals
        stage_start = time.perf_counter()
        conds = model.prepare_conditionals(ref_wav, ref_sr, exaggeration=0.1)
        mx.eval(conds.t3.speaker_emb)
        if conds.gen:
            for v in conds.gen.values():
                if isinstance(v, mx.array):
                    mx.eval(v)
        stage_times["prepare_conditionals"].append(time.perf_counter() - stage_start)

        # Stage 2: Text tokenization
        from mlx_audio.tts.models.chatterbox.chatterbox import punc_norm
        stage_start = time.perf_counter()
        normalized_text = punc_norm(text)
        text_tokens = model.tokenizer.text_to_tokens(normalized_text)
        mx.eval(text_tokens)
        stage_times["text_tokenization"].append(time.perf_counter() - stage_start)

        # Add start/end tokens
        cfg_weight = 0.5
        if cfg_weight > 0.0:
            text_tokens = mx.concatenate([text_tokens, text_tokens], axis=0)
        sot = model.t3.hp.start_text_token
        eot = model.t3.hp.stop_text_token
        sot_tokens = mx.full((text_tokens.shape[0], 1), sot, dtype=mx.int32)
        eot_tokens = mx.full((text_tokens.shape[0], 1), eot, dtype=mx.int32)
        text_tokens = mx.concatenate([sot_tokens, text_tokens, eot_tokens], axis=1)

        # Stage 3: T3 inference (text -> speech tokens)
        stage_start = time.perf_counter()
        speech_tokens = model.t3.inference(
            t3_cond=conds.t3,
            text_tokens=text_tokens,
            max_new_tokens=1000,
            temperature=0.8,
            cfg_weight=cfg_weight,
            repetition_penalty=1.2,
            min_p=0.05,
            top_p=1.0,
        )
        mx.eval(speech_tokens)
        stage_times["t3_inference"].append(time.perf_counter() - stage_start)

        num_speech_tokens = speech_tokens.shape[1]

        # Post-process tokens
        from mlx_audio.tts.models.chatterbox.chatterbox import drop_invalid_tokens, SPEECH_VOCAB_SIZE
        speech_tokens = speech_tokens[0:1]
        speech_tokens = drop_invalid_tokens(speech_tokens)
        mask = speech_tokens < SPEECH_VOCAB_SIZE
        valid_count = int(mx.sum(mask.astype(mx.int32)))
        sorted_indices = mx.argsort(-mask.astype(mx.int32))
        valid_indices = sorted_indices[:valid_count]
        speech_tokens = mx.take(speech_tokens, valid_indices)
        speech_tokens = mx.expand_dims(speech_tokens, 0)

        # Stage 4: S3Gen waveform generation
        stage_start = time.perf_counter()
        wav = model.s3gen(
            speech_tokens=speech_tokens,
            ref_dict=conds.gen,
            finalize=True,
        )
        mx.eval(wav)
        stage_times["s3gen_waveform"].append(time.perf_counter() - stage_start)

        stage_times["total"].append(time.perf_counter() - total_start)

        # Print per-run results
        audio_duration = wav.shape[-1] / model.sample_rate
        rtf = stage_times["total"][-1] / audio_duration

        print(f"  prepare_conditionals: {stage_times['prepare_conditionals'][-1]:.3f}s")
        print(f"  text_tokenization:    {stage_times['text_tokenization'][-1]:.3f}s")
        print(f"  t3_inference:         {stage_times['t3_inference'][-1]:.3f}s ({num_speech_tokens} tokens)")
        print(f"  s3gen_waveform:       {stage_times['s3gen_waveform'][-1]:.3f}s")
        print(f"  total:                {stage_times['total'][-1]:.3f}s")
        print(f"  audio_duration:       {audio_duration:.2f}s")
        print(f"  RTF:                  {rtf:.2f}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY (averaged over {} runs)".format(num_runs))
    print("=" * 60)

    for stage, times in stage_times.items():
        avg = np.mean(times)
        std = np.std(times)
        pct = (avg / np.mean(stage_times["total"])) * 100 if stage != "total" else 100
        print(f"{stage:25s}: {avg:.3f}s ± {std:.3f}s ({pct:5.1f}%)")

    # RTF summary
    audio_duration = wav.shape[-1] / model.sample_rate
    avg_total = np.mean(stage_times["total"])
    avg_rtf = avg_total / audio_duration
    print(f"\nAverage RTF: {avg_rtf:.2f}")
    print(f"Audio duration: {audio_duration:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Chatterbox TTS pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Chatterbox-TTS-4bit",
        help="Model path or HF repo",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to local reference audio file (optional, downloads default if not provided)",
    )
    parser.add_argument(
        "--audio-url",
        type=str,
        default=None,
        help="URL to download reference audio from (optional, uses default URL if not provided)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, this is a test of the Chatterbox text to speech system.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of benchmark runs",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup runs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    benchmark_chatterbox(
        model_path=args.model,
        reference_audio_path=args.audio,
        reference_audio_url=args.audio_url,
        text=args.text,
        num_runs=args.runs,
        warmup_runs=args.warmup,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
