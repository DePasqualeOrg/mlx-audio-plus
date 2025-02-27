import sys
import os
import soundfile as sf
import argparse
import json


from utils import load_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="prince-canuma/Kokoro-82M", help="Path or repo id of the model")
    parser.add_argument("--text", type=str, default='The sky above the port', help="Text to generate")
    parser.add_argument("--voice", type=str, default="af_heart", help="Voice name")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed of the audio")
    parser.add_argument("--lang_code", type=str, default="a", help="Language code")
    parser.add_argument("--file_prefix", type=str, default="audio", help="Output file name prefix")
    parser.add_argument("--verbose", action="store_false", help="Print verbose output")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        model = load_model(model_path=args.model)
        print(
            f"Model: {args.model}\n"
            f"Text: {args.text}\n"
            f"Voice: {args.voice}\n"
            f"Speed: {args.speed}x\n"
            f"Language: {args.lang_code}"
        )
        print("==========")
        results = model.generate(text=args.text, voice=args.voice, speed=args.speed, lang_code=args.lang_code, verbose=True)
        print(f"\033[92mAudio generated successfully, saving to\033[0m {args.file_prefix}!")


        for i, result in enumerate(results):

            sf.write(f"{args.file_prefix}_{i:03d}.wav", result.audio, 24000)

            if args.verbose:
                print("==========")
                print(f"Duration:              {result.audio_duration}")
                print(f"Samples/sec:           {result.audio_samples['samples-per-sec']:.1f}")
                print(f"Prompt:                {result.token_count} tokens, {result.prompt['tokens-per-sec']:.1f} tokens-per-sec")
                print(f"Audio:                 {result.audio_samples['samples']} samples, {result.audio_samples['samples-per-sec']:.1f} samples-per-sec")
                print(f"Real-time factor:      {result.real_time_factor:.2f}x")
                print(f"Processing time:       {result.processing_time_seconds:.2f}s")
                print(f"Peak memory usage:     {result.peak_memory_usage:.2f}GB")

    except ImportError as e:
        print(f"Import error: {e}")
        print("This might be due to incorrect Python path. Check your project structure.")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
