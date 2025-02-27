import sys
import os
import soundfile as sf
import argparse
import json


from utils import load_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="prince-canuma/Kokoro-82M", help="Path or repo id of the model")
    parser.add_argument("--text", type=str, default="Hello world", help="Text to generate")
    parser.add_argument("--voice", type=str, default="af_heart", help="Voice name")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed of the audio")
    parser.add_argument("--lang_code", type=str, default="a", help="Language code")
    parser.add_argument("--file_name", type=str, default="audio.wav", help="Output file name")
    parser.add_argument("--verbose", action="store_false", help="Print verbose output")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        model = load_model(model_path=args.model)
        print("Generating audio with the following parameters:")
        print("--------------------------------")
        print(f"    Text: {args.text}")
        print(f"    Voice: {args.voice}")
        print(f"    Speed: {args.speed}x")
        print(f"    Language: {args.lang_code}")
        print("--------------------------------")
        results = model.generate(text=args.text, voice=args.voice, speed=args.speed, lang_code=args.lang_code, verbose=True)
        print("\033[92mAudio generated successfully!\033[0m")
        print(f"File name: {args.file_name}")
        for _, samples, audio in results:
            sf.write(args.file_name, audio, 24000)
    except ImportError as e:
        print(f"Import error: {e}")
        print("This might be due to incorrect Python path. Check your project structure.")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
