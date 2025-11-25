# MLX Audio TTS examples for macOS

This is an example of using MLX Audio to run local TTS models.

Steps to run:

 - Open project in Xcode
 - Change project signing in "Signing and Capabilities" project settings
 - Run the App

All models now download their weights automatically from Hugging Face Hub on first use. No manual file downloads required!

eSpeak NG phoneme data is provided via the espeak-ng-spm Swift Package and compiled automatically at runtime.

# Kokoro

**Model weights are downloaded automatically from [prince-canuma/Kokoro-82M](https://huggingface.co/prince-canuma/Kokoro-82M)**

Implemented and working. Based on [Kokoro TTS for iOS](https://github.com/mlalma/kokoro-ios). All credit to mlalma for that work!

Uses MLX Swift and eSpeak NG. M1 chip or better is required.


# Orpheus

**Model weights are downloaded automatically from:**
- [mlx-community/orpheus-3b-0.1-ft-4bit](https://huggingface.co/mlx-community/orpheus-3b-0.1-ft-4bit)
- [mlx-community/snac_24khz](https://huggingface.co/mlx-community/snac_24khz)

Currently runs quite slow due to MLX-Swift not letting us compile layers with caching. On an M1 we see a 0.1x processing speed so be patient!

The full Orpheus functionality is implemented including:
 - Voices: tara, leah, jess, leo, dan, mia, zac, zoe
 - Expressions: <laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>

# Marvis

Marvis is an advanced conversational TTS model with streaming support. It uses the Marvis architecture combined with Mimi vocoder for high-quality speech synthesis.

Features:
 - Streaming audio generation for real-time TTS
 - Two conversational voices: conversational_a and conversational_b
 - Downloads model weights automatically on first use from Hugging Face
 - Optimized for Apple Silicon with MLX framework

The model runs at 24kHz sample rate and provides natural-sounding conversational speech.
