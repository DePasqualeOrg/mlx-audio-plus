#!/usr/bin/env python3
"""
Chatterbox Detailed Pipeline Benchmark

Measures timing for each sub-component of prepare_conditionals and T3 inference.
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


def benchmark_prepare_conditionals(model, ref_wav, ref_sr, num_runs=3):
    """Detailed benchmark of prepare_conditionals components."""
    from mlx_audio.tts.models.chatterbox.chatterbox import resample_audio, S3_SR, S3GEN_SR
    from mlx_audio.tts.models.chatterbox.s3tokenizer import log_mel_spectrogram

    print("\n" + "=" * 60)
    print("PREPARE_CONDITIONALS DETAILED BENCHMARK")
    print("=" * 60)

    stage_times = {
        "resample_to_24k": [],
        "resample_24k_to_16k": [],
        "resample_to_16k_full": [],
        "mel_spectrogram_s3gen": [],
        "mel_spectrogram_t3": [],
        "s3_tokenizer_s3gen": [],
        "s3_tokenizer_t3": [],
        "s3gen_embed_ref": [],
        "voice_encoder": [],
        "total": [],
    }

    for run in range(num_runs):
        print(f"\n--- Run {run + 1}/{num_runs} ---")

        wav = ref_wav
        if wav.ndim == 2:
            wav = wav.squeeze(0)

        total_start = time.perf_counter()

        # Resample to 24kHz for S3Gen
        start = time.perf_counter()
        ref_wav_24k = resample_audio(wav, ref_sr, S3GEN_SR) if ref_sr != S3GEN_SR else wav
        ref_wav_24k = ref_wav_24k[:model.DEC_COND_LEN]
        mx.eval(ref_wav_24k)
        stage_times["resample_to_24k"].append(time.perf_counter() - start)

        # Resample 24kHz to 16kHz for S3Gen tokenization
        start = time.perf_counter()
        ref_wav_16k_from_24k = resample_audio(ref_wav_24k, S3GEN_SR, S3_SR)
        mx.eval(ref_wav_16k_from_24k)
        stage_times["resample_24k_to_16k"].append(time.perf_counter() - start)

        # Resample original to 16kHz for T3 encoder conditioning
        start = time.perf_counter()
        ref_wav_16k_full = resample_audio(wav, ref_sr, S3_SR) if ref_sr != S3_SR else wav
        ref_wav_16k = ref_wav_16k_full[:model.ENC_COND_LEN]
        mx.eval(ref_wav_16k)
        mx.eval(ref_wav_16k_full)
        stage_times["resample_to_16k_full"].append(time.perf_counter() - start)

        # S3Gen mel spectrogram
        start = time.perf_counter()
        s3gen_mel = log_mel_spectrogram(ref_wav_16k_from_24k)
        s3gen_mel = mx.expand_dims(s3gen_mel, 0)
        s3gen_mel_len = mx.array([s3gen_mel.shape[2]])
        mx.eval(s3gen_mel)
        stage_times["mel_spectrogram_s3gen"].append(time.perf_counter() - start)

        # T3 mel spectrogram
        start = time.perf_counter()
        t3_mel = log_mel_spectrogram(ref_wav_16k)
        t3_mel = mx.expand_dims(t3_mel, 0)
        t3_mel_len = mx.array([t3_mel.shape[2]])
        mx.eval(t3_mel)
        stage_times["mel_spectrogram_t3"].append(time.perf_counter() - start)

        # S3Gen tokenization
        start = time.perf_counter()
        s3gen_tokens, s3gen_token_lens = model.s3_tokenizer(s3gen_mel, s3gen_mel_len)
        mx.eval(s3gen_tokens)
        stage_times["s3_tokenizer_s3gen"].append(time.perf_counter() - start)

        # T3 tokenization
        start = time.perf_counter()
        t3_tokens, t3_token_lens = model.s3_tokenizer(t3_mel, t3_mel_len)
        mx.eval(t3_tokens)
        stage_times["s3_tokenizer_t3"].append(time.perf_counter() - start)

        # S3Gen embed_ref
        start = time.perf_counter()
        s3gen_ref_dict = model.s3gen.embed_ref(
            ref_wav=mx.expand_dims(ref_wav_24k, 0),
            ref_sr=S3GEN_SR,
            ref_speech_tokens=s3gen_tokens,
            ref_speech_token_lens=s3gen_token_lens,
        )
        for v in s3gen_ref_dict.values():
            if isinstance(v, mx.array):
                mx.eval(v)
        stage_times["s3gen_embed_ref"].append(time.perf_counter() - start)

        # Voice encoder
        start = time.perf_counter()
        ve_embed = model.ve.embeds_from_wavs([ref_wav_16k_full], sample_rate=S3_SR)
        ve_embed = mx.mean(ve_embed, axis=0, keepdims=True)
        mx.eval(ve_embed)
        stage_times["voice_encoder"].append(time.perf_counter() - start)

        stage_times["total"].append(time.perf_counter() - total_start)

        # Print per-run results
        for stage, times in stage_times.items():
            if times:
                print(f"  {stage:25s}: {times[-1]:.4f}s")

    # Print summary
    print("\n" + "-" * 40)
    print("SUMMARY (averaged)")
    print("-" * 40)
    total_avg = np.mean(stage_times["total"])
    for stage, times in stage_times.items():
        avg = np.mean(times)
        std = np.std(times)
        pct = (avg / total_avg) * 100 if stage != "total" else 100
        print(f"{stage:25s}: {avg:.4f}s ± {std:.4f}s ({pct:5.1f}%)")


def benchmark_t3_inference(model, ref_wav, ref_sr, text, num_runs=3, seed=42):
    """Detailed benchmark of T3 inference components."""
    from mlx_audio.tts.models.chatterbox.chatterbox import punc_norm

    print("\n" + "=" * 60)
    print("T3 INFERENCE DETAILED BENCHMARK")
    print("=" * 60)

    # Prepare conditionals once
    conds = model.prepare_conditionals(ref_wav, ref_sr, exaggeration=0.1)

    # Prepare text tokens
    normalized_text = punc_norm(text)
    text_tokens = model.tokenizer.text_to_tokens(normalized_text)

    cfg_weight = 0.5
    if cfg_weight > 0.0:
        text_tokens = mx.concatenate([text_tokens, text_tokens], axis=0)

    sot = model.t3.hp.start_text_token
    eot = model.t3.hp.stop_text_token
    sot_tokens = mx.full((text_tokens.shape[0], 1), sot, dtype=mx.int32)
    eot_tokens = mx.full((text_tokens.shape[0], 1), eot, dtype=mx.int32)
    text_tokens = mx.concatenate([sot_tokens, text_tokens, eot_tokens], axis=1)

    stage_times = {
        "prepare_conditioning": [],
        "text_embedding": [],
        "initial_forward": [],
        "generation_loop": [],
        "tokens_per_second": [],
        "total": [],
    }

    for run in range(num_runs):
        print(f"\n--- Run {run + 1}/{num_runs} ---")
        mx.random.seed(seed + run)

        total_start = time.perf_counter()

        # Prepare conditioning
        start = time.perf_counter()
        cond_emb = model.t3.prepare_conditioning(conds.t3)
        mx.eval(cond_emb)
        stage_times["prepare_conditioning"].append(time.perf_counter() - start)

        # Text embedding
        start = time.perf_counter()
        text_embeddings = model.t3.text_emb(text_tokens)
        if cfg_weight > 0.0:
            text_embeddings = mx.concatenate([
                text_embeddings[0:1],
                mx.zeros_like(text_embeddings[0:1]),
            ], axis=0)
        if model.t3.hp.input_pos_emb == "learned":
            text_embeddings = text_embeddings + model.t3.text_pos_emb(text_tokens)
        mx.eval(text_embeddings)
        stage_times["text_embedding"].append(time.perf_counter() - start)

        # Initial forward pass
        start = time.perf_counter()
        bos_token = mx.array([[model.t3.hp.start_speech_token]], dtype=mx.int32)
        bos_embed = model.t3.speech_emb(bos_token)
        bos_embed = bos_embed + model.t3.speech_pos_emb.get_fixed_embedding(0)
        if cfg_weight > 0.0:
            bos_embed = mx.concatenate([bos_embed, bos_embed], axis=0)

        if cond_emb.shape[0] != text_embeddings.shape[0]:
            cond_emb = mx.broadcast_to(
                cond_emb,
                [text_embeddings.shape[0]] + list(cond_emb.shape[1:]),
            )

        input_embeddings = mx.concatenate([cond_emb, text_embeddings, bos_embed], axis=1)

        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(model.t3.tfmr)
        hidden = model.t3.tfmr.model(inputs=None, cache=cache, input_embeddings=input_embeddings)
        mx.eval(hidden)
        stage_times["initial_forward"].append(time.perf_counter() - start)

        # Generation loop
        start = time.perf_counter()
        generated_ids = [model.t3.hp.start_speech_token]
        max_new_tokens = 200  # Limit for benchmark

        for step in range(max_new_tokens):
            logits = model.t3.speech_head(hidden[:, -1:, :])
            logits = logits.squeeze(1)

            if cfg_weight > 0.0 and logits.shape[0] > 1:
                cond_logits = logits[0:1]
                uncond_logits = logits[1:2]
                logits = cond_logits + cfg_weight * (cond_logits - uncond_logits)
            else:
                logits = logits[0:1]

            # Sample (simplified for benchmark)
            probs = mx.softmax(logits / 0.8, axis=-1)
            next_token = mx.random.categorical(probs)
            next_token_id = int(next_token[0])

            if next_token_id == model.t3.hp.stop_speech_token:
                generated_ids.append(next_token_id)
                break

            generated_ids.append(next_token_id)

            # Next step
            next_embed = model.t3.speech_emb(next_token.reshape(1, 1))
            next_embed = next_embed + model.t3.speech_pos_emb.get_fixed_embedding(step + 1)
            if cfg_weight > 0.0:
                next_embed = mx.concatenate([next_embed, next_embed], axis=0)

            hidden = model.t3.tfmr.model(inputs=None, cache=cache, input_embeddings=next_embed)
            mx.eval(hidden)

        gen_time = time.perf_counter() - start
        stage_times["generation_loop"].append(gen_time)
        stage_times["tokens_per_second"].append(len(generated_ids) / gen_time)

        stage_times["total"].append(time.perf_counter() - total_start)

        # Print per-run results
        print(f"  prepare_conditioning:   {stage_times['prepare_conditioning'][-1]:.4f}s")
        print(f"  text_embedding:         {stage_times['text_embedding'][-1]:.4f}s")
        print(f"  initial_forward:        {stage_times['initial_forward'][-1]:.4f}s")
        print(f"  generation_loop:        {stage_times['generation_loop'][-1]:.4f}s ({len(generated_ids)} tokens)")
        print(f"  tokens_per_second:      {stage_times['tokens_per_second'][-1]:.1f} tok/s")
        print(f"  total:                  {stage_times['total'][-1]:.4f}s")

    # Print summary
    print("\n" + "-" * 40)
    print("SUMMARY (averaged)")
    print("-" * 40)
    for stage, times in stage_times.items():
        avg = np.mean(times)
        std = np.std(times)
        print(f"{stage:25s}: {avg:.4f} ± {std:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Detailed Chatterbox benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Chatterbox-TTS-4bit",
        help="Model path or HF repo",
    )
    parser.add_argument(
        "--audio-url",
        type=str,
        default=None,
        help="URL to download reference audio from",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Chatterbox Detailed Benchmark (Python MLX)")
    print("=" * 60)

    mx.random.seed(args.seed)

    # Import and load model
    from mlx_audio.tts.models.chatterbox.chatterbox import Model
    import soundfile as sf

    # Load reference audio
    audio_url = args.audio_url or DEFAULT_REFERENCE_AUDIO_URL
    audio_path = download_audio(audio_url)
    ref_wav, ref_sr = sf.read(audio_path)
    if ref_wav.ndim > 1:
        ref_wav = ref_wav.mean(axis=1)
    ref_wav = mx.array(ref_wav.astype(np.float32))

    print(f"\nReference audio: {audio_path}")
    print(f"  Sample rate: {ref_sr} Hz")
    print(f"  Duration: {len(ref_wav) / ref_sr:.2f}s")

    # Load model
    print(f"\nLoading model: {args.model}")
    model = Model.from_pretrained(args.model)
    mx.eval(model.parameters())

    # Run benchmarks
    benchmark_prepare_conditionals(model, ref_wav, ref_sr, num_runs=args.runs)
    benchmark_t3_inference(model, ref_wav, ref_sr, args.text, num_runs=args.runs, seed=args.seed)


if __name__ == "__main__":
    main()
