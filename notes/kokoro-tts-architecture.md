# How Kokoro TTS Works

A beginner-friendly guide to understanding the Kokoro Text-to-Speech model in Swift.

## Table of Contents

1. [Prerequisites: What You Need to Know](#prerequisites-what-you-need-to-know)
2. [The Fundamental Challenge of TTS](#the-fundamental-challenge-of-tts)
3. [Background Concepts](#background-concepts)
4. [The Kokoro Pipeline](#the-kokoro-pipeline)
5. [Stage-by-Stage Breakdown](#stage-by-stage-breakdown)
6. [Architecture Diagram](#architecture-diagram)
7. [Key Files Reference](#key-files-reference)
8. [Technical Specifications](#technical-specifications)

---

## Prerequisites: What You Need to Know

This guide assumes you understand:
- Basic neural network concepts (layers, embeddings, forward passes)
- How LLM tokenization works (text → token IDs)
- What transformers/BERT-style encoders do (contextual embeddings)

---

## The Fundamental Challenge of TTS

### Why TTS is Harder Than It Looks

At first glance, TTS seems straightforward: convert text to sound. But consider these challenges:

**1. Text doesn't tell you how long sounds should be**

The word "hello" has 5 letters, but:
- How long should the "h" sound last? 50ms? 100ms?
- Should "llo" be drawn out ("hellooooo") or quick?
- The text doesn't say—the model must figure this out.

**2. Text doesn't tell you the pitch or emotion**

"I'm fine" can mean very different things:
- Said with rising pitch: a genuine response
- Said with flat pitch: sarcasm or sadness
- The text is identical, but the audio is completely different.

**3. Text isn't sounds**

The letter "c" makes different sounds in "cat" vs "city". The letters "ough" sound different in "through", "though", "rough", and "cough". Text is a lossy representation of speech.

**4. Audio is continuous, text is discrete**

An LLM outputs discrete tokens one at a time. But audio is a continuous waveform—tens of thousands of amplitude values per second that must flow smoothly together.

### The LLM vs TTS Comparison

| Aspect | LLM | TTS |
|--------|-----|-----|
| Input | Tokens (discrete) | Text (discrete) |
| Output | Next token (discrete) | Audio waveform (continuous) |
| Timing | Not relevant | Critical—must predict duration |
| Output size | 1 token at a time | Thousands of samples per phoneme |
| Prosody | N/A | Must predict pitch, energy, rhythm |

---

## Background Concepts

Before diving into Kokoro's architecture, let's understand the building blocks.

### What is Audio?

Audio is a sequence of numbers representing air pressure over time. When you see a "waveform," you're looking at these pressure values plotted against time.

```
Amplitude
    ^
    |    /\      /\      /\
    |   /  \    /  \    /  \
----+--/----\--/----\--/----\--→ Time
    | /      \/      \/      \
    |/
```

Key concepts:
- **Sample rate**: How many numbers per second (Kokoro uses 24,000 Hz = 24,000 samples/second)
- **Amplitude**: The value at each sample point (typically -1.0 to 1.0)
- **1 second of audio** = 24,000 float values in Kokoro

### What are Phonemes?

**Phonemes are the basic units of speech sound**—not letters, but actual sounds.

English has about 44 phonemes, even though it has only 26 letters. This is because:
- One letter can make multiple sounds: "c" in "cat" (/k/) vs "city" (/s/)
- Multiple letters can make one sound: "sh" in "ship" (/ʃ/)
- Some sounds have no dedicated letter: the "zh" in "measure" (/ʒ/)

**Example phonemization:**

| Word | Letters | Phonemes (IPA) | Phoneme count |
|------|---------|----------------|---------------|
| "hello" | h-e-l-l-o | /h-ə-l-oʊ/ | 4 |
| "through" | t-h-r-o-u-g-h | /θ-r-uː/ | 3 |
| "cat" | c-a-t | /k-æ-t/ | 3 |

**Why use phonemes?**
- They're unambiguous—each phoneme maps to exactly one sound
- They're language-independent at the acoustic level
- The model learns to generate sounds, not interpret spelling rules

### What is a Spectrogram?

A spectrogram is a way to visualize audio that's more useful for neural networks than raw waveforms.

**Raw waveform**: amplitude over time (hard for neural networks to work with directly)

**Spectrogram**: breaks audio into frequency components over time

```
Frequency
    ^
    |  ████      ████
    |  ████████  ████████
    |  ████████████████████
    +------------------------→ Time
```

Think of it like a music equalizer visualization, but captured as data. Each column is a moment in time, and the values show how much energy is present at each frequency.

**Mel spectrogram**: A spectrogram using the "mel scale," which matches how humans perceive pitch (we're more sensitive to differences in low frequencies than high frequencies).

### What is a Vocoder?

A **vocoder** converts spectrograms back into audio waveforms.

```
Mel Spectrogram → [Vocoder] → Audio Waveform
  (2D: freq × time)           (1D: amplitude × time)
```

This is non-trivial because:
- The spectrogram has lost phase information (when each frequency started)
- You need to generate ~256× more data points (going from ~100 mel bins to 24,000 samples/sec)
- The output must sound natural, not robotic or glitchy

Modern neural vocoders (like HiFi-GAN used in Kokoro) learn to do this conversion with high quality.

### What is STFT?

**STFT (Short-Time Fourier Transform)** is the mathematical operation that converts audio to spectrograms.

- **Forward STFT**: Audio waveform → Spectrogram
- **Inverse STFT**: Spectrogram → Audio waveform

The "short-time" part means we analyze small windows of audio (e.g., 1024 samples) at a time, sliding the window forward, to see how frequencies change over time.

### What is F0 (Fundamental Frequency)?

**F0 is the pitch of your voice**—how high or low it sounds.

- Measured in Hertz (Hz)
- Male voices: typically 85-180 Hz
- Female voices: typically 165-255 Hz

When you ask a question, your F0 typically rises at the end ("Are you coming?↗"). When you make a statement, it falls ("I'm coming.↘"). This pitch contour is crucial for natural-sounding speech.

**F0 contour example:**

```
Pitch
  ^
  |        /
  |   ____/     "Are you coming?"
  |  /
  +------------------→ Time

Pitch
  ^
  |  ____
  |      \      "I'm coming."
  |       \___
  +------------------→ Time
```

### What is Prosody?

**Prosody** encompasses all the "musical" aspects of speech:

- **Pitch (F0)**: How high or low
- **Duration**: How long each sound lasts
- **Energy/Loudness**: How loud or soft
- **Rhythm**: The pattern of stressed and unstressed syllables

Prosody is what distinguishes a robot reading text from a human speaking. It conveys emotion, emphasis, and meaning beyond the words themselves.

### What is AdaIN (Adaptive Instance Normalization)?

**AdaIN is a technique for applying "style" to neural network features.**

Originally from image style transfer (making a photo look like a Van Gogh painting), it works by:

1. Normalizing the content features (removing their original statistics)
2. Applying new statistics (mean and variance) from a style embedding

```
content_features → Normalize → Scale & Shift by style → styled_features
```

In Kokoro, AdaIN is used to apply a voice's characteristics to the speech features. The same text encoded differently will sound like different speakers.

### What is an Alignment Matrix?

The **alignment problem** is: how do we match input phonemes to output audio frames?

If we have 10 phonemes and need to generate 500 audio frames, which frames correspond to which phonemes?

An **alignment matrix** is a mapping that says:
- Phoneme 1 → frames 1-40
- Phoneme 2 → frames 41-120
- Phoneme 3 → frames 121-150
- ... and so on

```
Phonemes:    |  h  |   ə   |  l  | oʊ  |
             ↓     ↓       ↓     ↓
Frames:      [====][======][===][=====]
             40    80+     30   50 frames
```

In Kokoro, the duration predictor outputs how many frames each phoneme should get, and then an alignment matrix is constructed to expand the phoneme representations to match.

---

## The Kokoro Pipeline

Now that you understand the building blocks, here's the overall flow:

```
"Hello world"
      ↓
[Text Normalization] → "Hello world"
      ↓
[Phonemization] → /həˈloʊ wɜːld/
      ↓
[Token IDs] → [14, 52, 23, 8, 45, ...]
      ↓
[BERT Encoder] → Contextual embeddings (understands meaning)
      ↓
[Duration Predictor] → [40, 80, 30, 50, ...] frames per phoneme
      ↓
[Prosody Predictor] → F0 curve, energy curve
      ↓
[Alignment/Expansion] → Expand phoneme features to frame length
      ↓
[Decoder + Vocoder] → Audio waveform
      ↓
🔊 24,000 samples per second
```

---

## Stage-by-Stage Breakdown

### Stage 1: Text Processing

**Purpose**: Convert messy real-world text into clean phoneme sequences.

**Files**: `KokoroTokenizer.swift`, `SentenceTokenizer.swift`

#### Step 1a: Sentence Splitting

Long text is split into sentences because:
- The model has a maximum sequence length (510 tokens)
- Processing sentence-by-sentence allows streaming
- Different languages may need different handling

#### Step 1b: Text Normalization

Real text contains things that aren't directly speakable:

| Input | Normalized |
|-------|------------|
| `$100` | "one hundred dollars" |
| `3:30` | "three thirty" |
| `Dr.` | "Doctor" |
| `5-10` | "five to ten" |
| `2.5` | "two point five" |

The tokenizer has extensive rules (749 lines!) to handle these cases.

#### Step 1c: Phonemization

Text is converted to phonemes using:
1. **eSpeak NG**: An open-source speech synthesizer that knows pronunciation rules
2. **Lexicon fallback**: Pre-built dictionaries for common words (for higher quality)

Output includes stress markers:
- `ˈ` (primary stress): The emphasized syllable
- `ˌ` (secondary stress): A lesser emphasis

Example: "hello" → `həˈloʊ` (stress on "lo")

#### Step 1d: Token Conversion

Phonemes are converted to integer IDs, just like LLM tokenization:

```
/h/ → 14
/ə/ → 52
/ˈ/ → 3   (stress marker)
/l/ → 23
/oʊ/ → 8
```

### Stage 2: Text Encoding

**Purpose**: Create rich, contextual representations of the phoneme sequence.

**Files**: `CustomAlbert.swift`, `TextEncoder.swift`

#### Why Two Encoders?

Kokoro uses two parallel encoders because they serve different purposes:

**BERT-style encoder (`CustomAlbert`):**
- Transformer architecture (self-attention)
- Understands linguistic context and relationships
- Output used for duration and prosody prediction
- Knows that a question should have rising pitch, etc.

**Acoustic encoder (`TextEncoder`):**
- CNN + LSTM architecture
- Produces features optimized for sound generation
- Captures local acoustic patterns
- Output goes directly to the decoder

This separation lets the model learn both high-level linguistic understanding and low-level acoustic patterns.

### Stage 3: Duration Prediction

**Purpose**: Decide how long each phoneme should last.

**Files**: `DurationEncoder.swift`, `KokoroTTS.swift`

**The Problem:**

Given "hello" with 4 phonemes, how do we generate, say, 0.5 seconds of audio (12,000 samples)?

The model must predict:
- /h/: 50ms (1,200 samples)
- /ə/: 100ms (2,400 samples)
- /l/: 80ms (1,920 samples)
- /oʊ/: 270ms (6,480 samples)

**How It Works:**

1. **DurationEncoder**: LSTM layers process BERT output + voice style
2. **PredictorLSTM**: Outputs a duration value for each phoneme
3. **Alignment Matrix**: Constructed from durations to expand features

The **speed parameter** multiplies all durations—speed=2.0 halves durations for faster speech.

**Why This Matters:**

Without duration prediction, the model wouldn't know how to "stretch" 4 phonemes into thousands of audio samples. This is the key difference from LLMs, which output one token at a time.

### Stage 4: Prosody Prediction

**Purpose**: Make speech sound natural with appropriate pitch and energy.

**File**: `ProsodyPredictor.swift`

**What It Predicts:**

1. **F0 (pitch) contour**: A curve showing pitch over time
2. **N (noise energy)**: Breathiness and voicing characteristics

**Why It's Needed:**

The same sentence with different prosody conveys different meanings:

| Prosody Pattern | Meaning |
|-----------------|---------|
| "You're LEAVING?" (rising F0) | Surprise/question |
| "You're leaving." (falling F0) | Statement/acceptance |
| "YOU'RE leaving?" (stress on "you're") | Emphasis on who |
| "You're leaving." (flat F0) | Robotic/sad |

**Architecture:**

```
BERT output + Voice style
         ↓
    Shared LSTM
    ↙        ↘
F0 Predictor   N Predictor
(AdaIN blocks) (AdaIN blocks)
    ↓              ↓
Pitch curve   Energy curve
```

The AdaIN blocks let the voice embedding influence the prosody style—different voices have different typical pitch ranges and speaking patterns.

### Stage 5: Voice Embeddings

**Purpose**: Define the characteristics of each speaker's voice.

**File**: `VoiceLoader.swift`

**What Are Voice Embeddings?**

Each voice is a learned vector of 256 numbers that captures:
- Average pitch range
- Speaking rhythm patterns
- Voice quality (breathy, clear, etc.)
- Accent characteristics

These embeddings are **pre-computed** and stored as JSON files. During inference, the embedding is loaded and used to condition various parts of the model.

**How They're Used:**

The embedding is split into two parts:
- **First 128 dims**: Style for the decoder (affects sound quality)
- **Last 128 dims**: Style for prosody (affects pitch and rhythm)

These are injected into the model via AdaIN layers, essentially telling the model "make this sound like voice X."

**Available Voices:**

60+ voices across languages:
- `af_*`: African-accented female voices
- `am_*`: American male voices
- `bf_*`/`bm_*`: British voices
- `jf_*`/`jm_*`: Japanese voices
- etc.

### Stage 6: Decoder and Vocoder

**Purpose**: Convert acoustic features into actual audio waveforms.

**Files**: `Decoder.swift`, `Generator.swift`

This is where all the pieces come together.

#### Step 6a: Decoder

The decoder integrates:
- Encoded text features (from TextEncoder)
- F0 predictions (pitch curve)
- N predictions (energy curve)
- Voice style (via AdaIN)

It uses a series of **AdaIN Residual Blocks** that progressively transform the features while applying voice style at each step.

```
Text features + F0 + N + Voice style
              ↓
      [AdaIN ResBlock 1]
              ↓
      [AdaIN ResBlock 2]
              ↓
      [AdaIN ResBlock 3]
              ↓
      [AdaIN ResBlock 4]
              ↓
     Decoded features
```

#### Step 6b: Generator (HiFi-GAN Vocoder)

The generator converts decoded features into raw audio. This is based on **HiFi-GAN**, a state-of-the-art neural vocoder.

**Key Components:**

1. **Harmonic-Noise Source (SourceModuleHnNSF)**:
   - Uses F0 to generate harmonic overtones (the pitched part of voice)
   - Generates noise component (the breathy/unvoiced part)
   - This gives the vocoder a "starting point" that already has the right pitch

2. **Upsampling Blocks**:
   - Progressively increase temporal resolution
   - Go from ~100 frames/second to 24,000 samples/second
   - Use transposed convolutions (opposite of pooling)

3. **Residual Blocks**:
   - Refine audio quality at each resolution
   - Multiple kernel sizes capture different frequency patterns

4. **Inverse STFT**:
   - Final conversion from spectral representation to waveform
   - Reconstructs the phase information that was lost

**The Output:**

24,000 float values per second of audio, ready to be:
- Played through speakers (`AudioSamplePlayer`)
- Saved to a WAV file (`AudioFileWriter`)

---

## Architecture Diagram

```
                    ┌─────────────────────────────────────┐
                    │     INPUT: Text + Voice ID + Speed  │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  SentenceTokenizer          │
                    │  (Split by language)        │
                    └──────────────┬──────────────┘
                                   │ (per sentence)
                    ┌──────────────▼──────────────┐
                    │  KokoroTokenizer            │
                    │  (Text → Phonemes)          │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  PhonemeTokenizer           │
                    │  (Phonemes → Token IDs)     │
                    └──────────────┬──────────────┘
         ┌──────────────────────────┼──────────────────────────┐
         │                          │                          │
    ┌────▼─────┐            ┌──────▼──────┐          ┌────────▼─────┐
    │ CustomA- │            │ Duration    │          │ Voice        │
    │ lbert    │            │ Encoder     │          │ Embeddings   │
    │ (BERT)   │            └──────┬──────┘          └────────┬─────┘
    └────┬─────┘                   │                         │
         │                         │                         │
    ┌────▼─────────┐      ┌────────▼──────┐        ┌────────▼──────┐
    │ Text         │      │ Predictor     │        │ Split style   │
    │ Encoder      │      │ LSTM + Proj   │        │ (128+128)     │
    │ (CNN+LSTM)   │      │ (durations)   │        │               │
    └────┬─────────┘      └────────┬──────┘        └────────┬──────┘
         │                         │                        │
         │                ┌────────▼──────┐                │
         │                │ Alignment     │                │
         │                │ Matrix        │                │
         │                └────────┬──────┘                │
         │                         │                       │
         └─────────────┬───────────┼───────────────────────┘
                       │           │
         ┌─────────────▼─┐  ┌──────▼──────┐
         │ Prosody       │  │ Decoder     │
         │ Predictor     │  │             │
         │ (F0, N pred)  │  │ (AdaIN Res) │
         └─────────────┬─┘  └──────┬──────┘
                       │           │
                       └─────┬─────┘
                             │
                       ┌─────▼─────────┐
                       │ Generator     │
                       │ (HiFi-GAN)    │
                       │ - HN Source   │
                       │ - Upsample    │
                       │ - ResBlocks   │
                       │ - Inverse STFT│
                       └─────┬─────────┘
                             │
                       ┌─────▼──────────┐
                       │ Audio Samples  │
                       │ (24kHz float)  │
                       └────────────────┘
```

---

## Summary: LLM vs TTS Mental Model

| Concept | LLM | TTS (Kokoro) |
|---------|-----|--------------|
| Input | Token IDs | Token IDs (phonemes) |
| Output | Next token | Thousands of audio samples |
| Core challenge | Predict likely next token | Predict duration, pitch, and generate audio |
| Timing | Irrelevant | Critical (duration prediction) |
| Style/voice | Via prompting | Via embedding injection (AdaIN) |
| Decoder | Vocabulary projection | Neural vocoder |

**The unique TTS components you won't find in LLMs:**
1. **Phonemization**: Converting spelling to sounds
2. **Duration prediction**: How long each sound lasts
3. **Prosody prediction**: Pitch (F0) and energy curves
4. **Alignment/expansion**: Stretching phonemes to audio length
5. **Neural vocoder**: Generating actual waveforms

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `KokoroEngine.swift` | Main public API - orchestrates entire pipeline |
| `KokoroTTS.swift` | Core TTS actor - model initialization, text-to-audio conversion |
| `KokoroWeightLoader.swift` | Downloads weights from Hugging Face Hub |
| `VoiceLoader.swift` | Loads pre-computed voice embeddings from bundled JSON files |
| `KokoroTokenizer.swift` | Text preprocessing and phonemization (749 lines) |
| `SentenceTokenizer.swift` | Language-aware sentence splitting |
| `CustomAlbert.swift` | BERT-based text encoder |
| `TextEncoder.swift` | CNN + LSTM acoustic feature encoder |
| `DurationEncoder.swift` | LSTM-based duration prediction |
| `ProsodyPredictor.swift` | Predicts F0 (pitch) and N (noise) |
| `Decoder.swift` | Pre-processes mel features with F0/N integration |
| `Generator.swift` | HiFi-GAN-style vocoder - generates waveforms |
| `MLXSTFT.swift` | STFT analysis/synthesis for vocoder |
| `SourceModuleHnNSF.swift` | Harmonic-Noise source generation |

---

## Technical Specifications

- **Maximum sequence length**: 510 tokens
- **Sample rate**: 24,000 Hz
- **Model size**: Kokoro-82M from Hugging Face Hub
- **Voice diversity**: 60+ unique voices across multiple languages
- **Supported languages**: English (US/GB), Spanish, French, Italian, Portuguese, Chinese, Japanese, Hindi
- **Speed control**: Float multiplier on predicted durations
- **Streaming**: AsyncThrowingStream for real-time audio generation
