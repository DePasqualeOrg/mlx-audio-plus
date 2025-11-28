# MLX Audio Swift Library Architecture

A guide to understanding how the Swift TTS library is structured to support three fundamentally different model types.

## Table of Contents

1. [Overview](#overview)
2. [The Challenge](#the-challenge)
3. [Architecture Layers](#architecture-layers)
4. [The TTSEngine Protocol](#the-ttsengine-protocol)
5. [Engine Adaptors](#engine-adaptors)
6. [Shared Components](#shared-components)
7. [Model Isolation](#model-isolation)
8. [Design Patterns Used](#design-patterns-used)
9. [Architecture Diagram](#architecture-diagram)
10. [Adding a New Model](#adding-a-new-model)
11. [Design Tradeoffs](#design-tradeoffs)

---

## Overview

The MLX Audio Swift library supports three TTS models with fundamentally different architectures:

| Model | Paradigm | Key Components |
|-------|----------|----------------|
| **Kokoro** | Classical TTS | BERT + Duration/Prosody + HiFi-GAN |
| **Orpheus** | Generative LLM | Llama-3B + SNAC codec |
| **Marvis** | Multimodal LLM | Dual Llama + Mimi codec |

Despite these differences, the library provides a **unified API** that lets applications switch between models seamlessly.

---

## The Challenge

How do you create a common interface for three models that:

1. **Process text differently**: Phonemes vs BPE tokens vs reference audio
2. **Generate audio differently**: Deterministic vs autoregressive vs iterative codebooks
3. **Have different capabilities**: Speed control, streaming, expressions, voice cloning
4. **Use different audio representations**: Mel spectrograms vs SNAC codes vs Mimi codebooks

The solution is a **layered architecture** with careful abstraction boundaries.

---

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                      App Layer                               │
│         EngineManager, AppState, SwiftUI Views              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Protocol Layer                             │
│              TTSEngine, StreamingTTSEngine                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Engine Layer                               │
│        KokoroEngine, OrpheusEngine, MarvisEngine            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Model Layer                                │
│           KokoroTTS, OrpheusTTS, MarvisTTS                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Shared Layer                               │
│      AudioSamplePlayer, Voice, TTSError, TTSProvider        │
└─────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

| Layer | Responsibility |
|-------|----------------|
| **App** | User interaction, state management, UI binding |
| **Protocol** | Define the common interface all engines implement |
| **Engine** | Adapt model-specific code to the common protocol |
| **Model** | Actual TTS implementation (neural networks, tokenizers) |
| **Shared** | Utilities used by all engines (audio playback, error types) |

---

## The TTSEngine Protocol

The core abstraction is the `TTSEngine` protocol:

```swift
@MainActor
public protocol TTSEngine: Observable {
    // MARK: - Identity
    var provider: TTSProvider { get }

    // MARK: - State
    var isLoaded: Bool { get }
    var isGenerating: Bool { get }
    var isPlaying: Bool { get }
    var lastGeneratedAudioURL: URL? { get }
    var generationTime: TimeInterval { get }

    // MARK: - Voice Management
    var availableVoices: [Voice] { get }
    var selectedVoiceID: String { get set }

    // MARK: - Lifecycle
    func load(progressHandler: (@Sendable (Progress) -> Void)?) async throws
    func cleanup() async throws

    // MARK: - Generation
    func generate(text: String, speed: Float) async throws -> AudioResult

    // MARK: - Playback
    func play() async throws
    func stop() async
}
```

### Design Decisions

**Why `@MainActor`?**

All state properties are UI-bound. Using `@MainActor` ensures thread safety without manual synchronization.

**Why `Observable`?**

SwiftUI integration. Views can observe `isLoaded`, `isGenerating`, etc. and update automatically.

**Why no `associatedtype`?**

Allows using `any TTSEngine` as an existential type:

```swift
// This works because there's no associatedtype
var currentEngine: any TTSEngine
```

With `associatedtype`, you'd need generics everywhere, complicating the API.

**Why `String` for voice IDs?**

Each model has its own voice enum:
- Kokoro: `TTSVoice.afHeart`
- Orpheus: `OrpheusVoice.tara`
- Marvis: `MarvisTTS.Voice.conversational_a`

Using `String` as the common currency avoids type gymnastics.

### Optional Streaming Protocol

Streaming is only supported by Marvis, so it's a separate protocol:

```swift
@MainActor
public protocol StreamingTTSEngine: TTSEngine {
    func generateStreaming(text: String, speed: Float) -> AsyncThrowingStream<AudioChunk, Error>
}
```

This keeps the base protocol clean. Code that needs streaming can check:

```swift
if let streamingEngine = engine as? StreamingTTSEngine {
    for try await chunk in streamingEngine.generateStreaming(text: text, speed: 1.0) {
        // Handle chunk
    }
}
```

---

## Engine Adaptors

Each engine is an **adaptor** that wraps model-specific code:

### KokoroEngine (299 lines)

```swift
@Observable
@MainActor
public final class KokoroEngine: TTSEngine {
    // Protocol state
    public private(set) var isLoaded = false
    public private(set) var isGenerating = false
    public private(set) var isPlaying = false

    // Model-specific
    private var tts: KokoroTTS?
    private var audioPlayer: AudioSamplePlayer?
    private var lastGeneratedSamples: [Float] = []

    // Voice resolution (String ID → model-specific enum)
    private func resolveVoice(_ id: String) -> TTSVoice? {
        TTSVoice.allCases.first { $0.id == id }
    }

    public func generate(text: String, speed: Float) async throws -> AudioResult {
        guard let tts else { throw TTSError.modelNotLoaded }
        isGenerating = true
        defer { isGenerating = false }

        let voice = resolveVoice(selectedVoiceID) ?? .afHeart
        let samples = try await tts.generate(text: text, voice: voice, speed: speed)
        lastGeneratedSamples = samples
        return .samples(samples)
    }
}
```

### OrpheusEngine (216 lines)

Simplest engine. Adds model-specific properties:

```swift
@Observable
@MainActor
public final class OrpheusEngine: TTSEngine {
    // Model-specific settings (not in protocol)
    public var temperature: Float = 0.6
    public var topP: Float = 0.8

    public func generate(text: String, speed: Float) async throws -> AudioResult {
        // Note: speed parameter is ignored (documented)
        let samples = try await tts.generate(
            text: text,
            voice: voice,
            temperature: temperature,
            topP: topP
        )
        return .samples(samples)
    }
}
```

### MarvisEngine (406 lines)

Most complex. Implements both protocols:

```swift
@Observable
@MainActor
public final class MarvisEngine: TTSEngine, StreamingTTSEngine {
    // Model-specific settings
    public var modelVariant: MarvisModelVariant = .v0_2_250m
    public var qualityLevel: MarvisQualityLevel = .medium
    public var streamingInterval: TimeInterval = 0.5

    // Uses actor for serialization
    private var session: MarvisTTSSession?

    // Streaming implementation
    public func generateStreaming(text: String, speed: Float) -> AsyncThrowingStream<AudioChunk, Error> {
        AsyncThrowingStream { continuation in
            Task {
                for try await chunk in session.generateStreaming(...) {
                    continuation.yield(chunk)
                }
                continuation.finish()
            }
        }
    }
}
```

### Adaptor Pattern Summary

| Engine | Lines | Special Features | Wraps |
|--------|-------|------------------|-------|
| Kokoro | 299 | Speed control, chunk streaming | `KokoroTTS` actor |
| Orpheus | 216 | Temperature, topP | `OrpheusTTS` actor |
| Marvis | 406 | Streaming, quality levels, model variants | `MarvisTTSSession` actor |

---

## Shared Components

### Audio Playback

```
MLXAudio/Audio/
├── AudioSamplePlayer.swift    # Play raw samples via AVAudioEngine
├── AudioFileWriter.swift      # Save to WAV/CAF files
├── AudioFilePlayer.swift      # Play from file
└── AudioSessionManager.swift  # Manage AVAudioSession
```

`AudioSamplePlayer` is used by all three engines:

```swift
public class AudioSamplePlayer {
    private let sampleRate: Double  // 24000 for all models
    private let engine: AVAudioEngine

    public func play(samples: [Float]) async {
        // Queue samples to audio engine
    }

    public func enqueue(samples: [Float]) {
        // For streaming: add to buffer without blocking
    }
}
```

### Data Models

**TTSProvider** - Source of truth for model capabilities:

```swift
public enum TTSProvider: String, CaseIterable {
    case kokoro
    case orpheus
    case marvis

    public var displayName: String { ... }

    public var supportsStreaming: Bool {
        self == .marvis
    }

    public var supportsSpeed: Bool {
        self == .kokoro
    }

    public var supportsExpressions: Bool {
        self == .orpheus
    }

    public var defaultVoiceID: String { ... }
    public var availableVoiceIDs: [String] { ... }
}
```

**Voice** - Unified voice representation:

```swift
public struct Voice: Identifiable, Hashable {
    public let id: String
    public let provider: TTSProvider
    public let displayName: String
    public let languageCode: String?

    // Factory methods for each model type
    public static func fromKokoroID(_ id: String) -> Voice { ... }
    public static func fromOrpheusID(_ id: String) -> Voice { ... }
    public static func fromMarvisID(_ id: String) -> Voice { ... }
}
```

**TTSError** - Unified error handling:

```swift
public enum TTSError: LocalizedError {
    case modelNotLoaded
    case generationFailed(String)
    case invalidVoice(String)
    case audioPlaybackFailed(String)
    case downloadFailed(String)
    case streamingNotSupported

    public var errorDescription: String? { ... }
}
```

**AudioResult** - Generation output:

```swift
public enum AudioResult {
    case samples([Float])
    case file(URL)

    public var sampleCount: Int { ... }
    public var duration: TimeInterval { ... }
}
```

---

## Model Isolation

Each model has a dedicated directory with **complete isolation**:

```
TTS/
├── Kokoro/                      # 40 files
│   ├── KokoroEngine.swift       # Adaptor (protocol conformance)
│   ├── Albert/                  # BERT-style encoder
│   │   ├── CustomAlbert.swift
│   │   ├── AlbertAttention.swift
│   │   └── ...
│   ├── BuildingBlocks/          # Neural network components
│   │   ├── AdaIN1d.swift
│   │   ├── LSTM.swift
│   │   └── ...
│   ├── Decoder/                 # HiFi-GAN vocoder
│   │   ├── Decoder.swift
│   │   ├── Generator.swift
│   │   └── ...
│   ├── TextProcessing/          # Phoneme tokenization
│   │   ├── KokoroTokenizer.swift
│   │   ├── ESpeakNGEngine.swift
│   │   └── ...
│   ├── TTSEngine/               # Core pipeline
│   │   ├── KokoroTTS.swift
│   │   ├── DurationEncoder.swift
│   │   └── ...
│   └── Resources/               # Voice embeddings (JSON)
│
├── Orpheus/                     # 25 files
│   ├── OrpheusEngine.swift      # Adaptor
│   ├── BuildingBlocks/          # Transformer components
│   │   ├── TransformerBlock.swift
│   │   ├── RoPE.swift
│   │   └── ...
│   ├── SNAC/                    # Audio codec
│   │   ├── SNACDecoder.swift
│   │   └── ...
│   ├── TextProcessing/          # BPE tokenizer
│   │   ├── OrpheusTokenizer.swift
│   │   └── ...
│   └── TTSEngine/
│       ├── OrpheusTTS.swift
│       └── ...
│
└── Marvis/                      # 30 files
    ├── MarvisEngine.swift       # Adaptor
    ├── Mimi/                    # Audio codec
    │   ├── MimiCodec.swift
    │   ├── SeaNetEncoder.swift
    │   └── ...
    ├── Models/                  # Llama architecture
    │   ├── MarvisModel.swift
    │   ├── MarvisDecoder.swift
    │   └── ...
    └── Audio/                   # Marvis-specific audio
        └── ...
```

### Why Complete Isolation?

The models share almost nothing at the implementation level:

| Component | Kokoro | Orpheus | Marvis |
|-----------|--------|---------|--------|
| Text processing | eSpeak phonemes | BPE tokens | HF tokenizer |
| Encoder | BERT (CustomAlbert) | Llama embedding | Llama embedding |
| Core model | Duration + Prosody predictors | Single Llama-3B | Backbone + Decoder Llamas |
| Audio codec | HiFi-GAN vocoder | SNAC (7-layer) | Mimi (multi-codebook) |

Trying to share code between them would create artificial coupling and make each model harder to understand.

---

## Design Patterns Used

### 1. Adaptor Pattern

Each `*Engine` class adapts model-specific code to the `TTSEngine` protocol:

```swift
// Model-specific code (doesn't know about TTSEngine)
actor KokoroTTS {
    func generate(text: String, voice: TTSVoice, speed: Float) async throws -> [Float]
}

// Adaptor (implements TTSEngine, wraps KokoroTTS)
class KokoroEngine: TTSEngine {
    private var tts: KokoroTTS?

    func generate(text: String, speed: Float) async throws -> AudioResult {
        let voice = resolveVoice(selectedVoiceID)
        let samples = try await tts!.generate(text: text, voice: voice, speed: speed)
        return .samples(samples)
    }
}
```

### 2. Factory Pattern

`EngineManager` creates engines based on provider:

```swift
final class EngineManager {
    private(set) var currentEngine: any TTSEngine

    private static func createEngine(for provider: TTSProvider) -> any TTSEngine {
        switch provider {
        case .kokoro: return KokoroEngine()
        case .orpheus: return OrpheusEngine()
        case .marvis: return MarvisEngine()
        }
    }

    func selectProvider(_ provider: TTSProvider) async {
        try? await currentEngine.cleanup()
        currentEngine = Self.createEngine(for: provider)
    }
}
```

### 3. Actor Pattern

All model implementations use Swift actors for thread safety:

```swift
actor KokoroTTS {
    private var model: KokoroModel?

    func generate(...) async throws -> [Float] {
        // Actor isolation ensures single-threaded access to model
    }
}
```

Marvis adds an extra layer with `MarvisTTSSession` to serialize streaming operations.

### 4. Protocol Extension for Defaults

Common behavior could be added via protocol extensions:

```swift
extension TTSEngine {
    var canStream: Bool {
        self is StreamingTTSEngine
    }
}
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         App Layer                                │
│  ┌─────────────────┐         ┌─────────────────┐                │
│  │  EngineManager  │────────▶│    AppState     │                │
│  │                 │         │  (@Observable)  │                │
│  │ - createEngine()│         │                 │                │
│  │ - selectProvider│         │ - text          │                │
│  │ - loadEngine()  │         │ - speed         │                │
│  │ - generate()    │         │ - isGenerating  │                │
│  └────────┬────────┘         └─────────────────┘                │
│           │                                                      │
│           │ holds: any TTSEngine                                │
└───────────┼──────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Protocol Layer                              │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    TTSEngine                                │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │ provider: TTSProvider                                 │  │ │
│  │  │ isLoaded, isGenerating, isPlaying: Bool              │  │ │
│  │  │ availableVoices: [Voice]                             │  │ │
│  │  │ selectedVoiceID: String                              │  │ │
│  │  │ load(), generate(), play(), stop(), cleanup()        │  │ │
│  │  └──────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              △                                   │
│                              │ extends                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              StreamingTTSEngine (optional)                  │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │ generateStreaming() -> AsyncThrowingStream           │  │ │
│  │  └──────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
            │
            │ conforms to
            ▼
┌──────────────────────────────────────────────────────────────────┐
│                       Engine Layer                                │
│                                                                   │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐  │
│  │  KokoroEngine    │ │  OrpheusEngine   │ │  MarvisEngine    │  │
│  │  ──────────────  │ │  ──────────────  │ │  ──────────────  │  │
│  │  TTSEngine ✓     │ │  TTSEngine ✓     │ │  TTSEngine ✓     │  │
│  │                  │ │                  │ │  Streaming ✓     │  │
│  │  ──────────────  │ │  ──────────────  │ │  ──────────────  │  │
│  │  (speed works)   │ │  + temperature   │ │  + qualityLevel  │  │
│  │                  │ │  + topP          │ │  + modelVariant  │  │
│  │                  │ │                  │ │  + streamInterval│  │
│  └────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘  │
│           │                    │                    │            │
└───────────┼────────────────────┼────────────────────┼────────────┘
            │ wraps              │ wraps              │ wraps
            ▼                    ▼                    ▼
┌──────────────────────────────────────────────────────────────────┐
│                        Model Layer                                │
│                                                                   │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐  │
│  │   KokoroTTS      │ │   OrpheusTTS     │ │   MarvisTTS      │  │
│  │   (actor)        │ │   (actor)        │ │   (actor)        │  │
│  │                  │ │                  │ │                  │  │
│  │ ┌──────────────┐ │ │ ┌──────────────┐ │ │ ┌──────────────┐ │  │
│  │ │CustomAlbert  │ │ │ │ OrpheusModel │ │ │ │ MarvisModel  │ │  │
│  │ │(BERT encoder)│ │ │ │ (Llama-3B)   │ │ │ │ (Backbone)   │ │  │
│  │ └──────────────┘ │ │ └──────────────┘ │ │ └──────────────┘ │  │
│  │ ┌──────────────┐ │ │ ┌──────────────┐ │ │ ┌──────────────┐ │  │
│  │ │DurationEnc   │ │ │ │ SNACDecoder  │ │ │ │MarvisDecoder │ │  │
│  │ │ProsodyPred   │ │ │ │ (audio codec)│ │ │ │ (Llama)      │ │  │
│  │ └──────────────┘ │ │ └──────────────┘ │ │ └──────────────┘ │  │
│  │ ┌──────────────┐ │ │ ┌──────────────┐ │ │ ┌──────────────┐ │  │
│  │ │Generator     │ │ │ │OrpheusToken- │ │ │ │ MimiCodec    │ │  │
│  │ │(HiFi-GAN)    │ │ │ │izer (BPE)    │ │ │ │ (audio codec)│ │  │
│  │ └──────────────┘ │ │ └──────────────┘ │ │ └──────────────┘ │  │
│  │ ┌──────────────┐ │ │                  │ │                  │  │
│  │ │KokoroToken-  │ │ │                  │ │                  │  │
│  │ │izer(phoneme)│ │ │                  │ │                  │  │
│  │ └──────────────┘ │ │                  │ │                  │  │
│  └──────────────────┘ └──────────────────┘ └──────────────────┘  │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
            │
            │ uses
            ▼
┌───────────────────────────────────────────────────────────────────┐
│                        Shared Layer                                │
│                                                                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │AudioSamplePlayer│  │ AudioFileWriter │  │AudioSessionMgr  │    │
│  │                 │  │                 │  │                 │    │
│  │ - play()        │  │ - write()       │  │ - configure()   │    │
│  │ - enqueue()     │  │ - save()        │  │                 │    │
│  │ - stop()        │  │                 │  │                 │    │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘    │
│                                                                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │   TTSProvider   │  │     Voice       │  │   AudioResult   │    │
│  │                 │  │                 │  │                 │    │
│  │ - kokoro        │  │ - id: String    │  │ - samples([])   │    │
│  │ - orpheus       │  │ - displayName   │  │ - file(URL)     │    │
│  │ - marvis        │  │ - languageCode  │  │                 │    │
│  │                 │  │                 │  │                 │    │
│  │ + supportsSpeed │  │ + fromKokoroID()│  │                 │    │
│  │ + supportsStream│  │ + fromOrpheusID │  │                 │    │
│  │ + defaultVoice  │  │ + fromMarvisID()│  │                 │    │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘    │
│                                                                    │
│  ┌─────────────────┐  ┌─────────────────┐                         │
│  │    TTSError     │  │   Constants     │                         │
│  │                 │  │                 │                         │
│  │ - modelNotLoaded│  │ - sampleRate    │                         │
│  │ - genFailed     │  │ - maxTokens     │                         │
│  │ - invalidVoice  │  │ - speedRange    │                         │
│  └─────────────────┘  └─────────────────┘                         │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Adding a New Model

To add a fourth TTS model (e.g., "NewModel"), follow this pattern:

### 1. Create Model Directory

```
TTS/NewModel/
├── NewModelEngine.swift      # Adaptor
├── TTSEngine/
│   └── NewModelTTS.swift     # Core implementation (actor)
├── BuildingBlocks/           # Model-specific components
├── TextProcessing/           # Model-specific tokenizer
└── AudioCodec/               # If needed
```

### 2. Implement the Core TTS Actor

```swift
actor NewModelTTS {
    private var model: NewModelModel?

    func load(progressHandler: @Sendable (Progress) -> Void) async throws {
        // Download weights, initialize model
    }

    func generate(text: String, voice: NewModelVoice) async throws -> [Float] {
        // Model-specific generation
    }
}
```

### 3. Create the Engine Adaptor

```swift
@Observable
@MainActor
public final class NewModelEngine: TTSEngine {
    public let provider: TTSProvider = .newModel

    public private(set) var isLoaded = false
    public private(set) var isGenerating = false
    public private(set) var isPlaying = false
    public private(set) var lastGeneratedAudioURL: URL?
    public private(set) var generationTime: TimeInterval = 0

    public var availableVoices: [Voice] {
        NewModelVoice.allCases.map { Voice.fromNewModelID($0.rawValue) }
    }

    public var selectedVoiceID: String = "default"

    private var tts: NewModelTTS?
    private var audioPlayer: AudioSamplePlayer?
    private var lastGeneratedSamples: [Float] = []

    public func load(progressHandler: (@Sendable (Progress) -> Void)?) async throws {
        tts = NewModelTTS()
        try await tts!.load(progressHandler: progressHandler ?? { _ in })
        audioPlayer = AudioSamplePlayer(sampleRate: Constants.sampleRate)
        isLoaded = true
    }

    public func generate(text: String, speed: Float) async throws -> AudioResult {
        guard let tts else { throw TTSError.modelNotLoaded }
        isGenerating = true
        defer { isGenerating = false }

        let start = Date()
        let voice = resolveVoice(selectedVoiceID)
        let samples = try await tts.generate(text: text, voice: voice)
        generationTime = Date().timeIntervalSince(start)
        lastGeneratedSamples = samples
        return .samples(samples)
    }

    public func play() async throws {
        guard let audioPlayer else { throw TTSError.modelNotLoaded }
        guard !lastGeneratedSamples.isEmpty else { return }
        isPlaying = true
        await audioPlayer.play(samples: lastGeneratedSamples)
        isPlaying = false
    }

    public func stop() async {
        await audioPlayer?.stop()
        isPlaying = false
    }

    public func cleanup() async throws {
        await stop()
        tts = nil
        audioPlayer = nil
        isLoaded = false
    }

    private func resolveVoice(_ id: String) -> NewModelVoice {
        NewModelVoice(rawValue: id) ?? .default
    }
}
```

### 4. Update Shared Components

**TTSProvider.swift:**

```swift
public enum TTSProvider: String, CaseIterable {
    case kokoro
    case orpheus
    case marvis
    case newModel  // Add new case

    public var supportsStreaming: Bool {
        switch self {
        case .marvis, .newModel: return true  // If applicable
        default: return false
        }
    }
    // ... update other properties
}
```

**Voice.swift:**

```swift
public static func fromNewModelID(_ id: String) -> Voice {
    Voice(
        id: id,
        provider: .newModel,
        displayName: formatNewModelVoiceName(id),
        languageCode: inferLanguageCode(id)
    )
}
```

**EngineManager.swift:**

```swift
private static func createEngine(for provider: TTSProvider) -> any TTSEngine {
    switch provider {
    case .kokoro: return KokoroEngine()
    case .orpheus: return OrpheusEngine()
    case .marvis: return MarvisEngine()
    case .newModel: return NewModelEngine()  // Add new case
    }
}
```

---

## Design Tradeoffs

### What Works Well

| Decision | Benefit |
|----------|---------|
| Protocol without `associatedtype` | Can use `any TTSEngine` existential |
| Separate `StreamingTTSEngine` | Base protocol stays clean |
| Model-specific properties on engines | Protocol doesn't bloat |
| Complete model isolation | Each model is self-contained |
| Shared audio utilities | No duplication of playback code |
| `TTSProvider` feature flags | UI can adapt to model capabilities |
| Actor-based TTS implementations | Thread safety without manual locks |

### Acceptable Compromises

| Compromise | Why It's OK |
|------------|-------------|
| `speed` parameter ignored by Orpheus/Marvis | Documented, UI can hide slider |
| Downcasting needed for model-specific settings | Keeps protocol minimal |
| ~40 lines duplicated per engine | Not worth abstracting |
| Voice ID is `String` not enum | Flexibility across model types |

### Potential Improvements

| Improvement | When to Consider |
|-------------|------------------|
| Base class for common engine code | If adding 4th+ model |
| Shared weight loading utilities | If Hub code diverges |
| Capability protocol instead of flags | If capabilities become complex |
| Voice type with associated values | If voice features diverge more |

---

## Summary

The MLX Audio Swift architecture successfully supports three fundamentally different TTS models through:

1. **A minimal, practical protocol** that captures common operations
2. **Engine adaptors** that bridge model-specific code to the protocol
3. **Complete model isolation** where implementations differ
4. **Shared utilities** where functionality is truly common
5. **Feature flags** that let the UI adapt to model capabilities

The architecture prioritizes **pragmatism over purity**—it doesn't try to abstract away differences that are fundamental to each model's design, but it does provide enough commonality for a unified user experience.
