//
//  MarvisEngine.swift
//  MLXAudio
//
//  Marvis TTS engine conforming to TTSEngine and StreamingTTSEngine protocols.
//  Wraps the existing MarvisTTS implementation.
//

import Foundation
import MLX

/// Marvis TTS engine - advanced conversational TTS with streaming support
@Observable
@MainActor
public final class MarvisEngine: TTSEngine, StreamingTTSEngine { // Warning: Conformance of 'MarvisEngine' to protocol 'StreamingTTSEngine' crosses into main actor-isolated code and can cause data races; this is an error in the Swift 6 language mode
    // MARK: - TTSEngine Protocol Properties

    public let provider: TTSProvider = .marvis
    public private(set) var isLoaded: Bool = false
    public private(set) var isGenerating: Bool = false
    public private(set) var isPlaying: Bool = false
    public private(set) var lastGeneratedAudioURL: URL?
    public private(set) var generationTime: TimeInterval = 0

    public var availableVoices: [Voice] {
        TTSProvider.marvis.availableVoices
    }

    public var selectedVoiceID: String = TTSProvider.marvis.defaultVoiceID

    // MARK: - Marvis-Specific Properties

    /// Model variant to use
    public var modelVariant: MarvisTTS.ModelVariant = .default

    /// Quality level (affects codebook count)
    public var qualityLevel: MarvisTTS.QualityLevel = .maximum

    /// Streaming interval in seconds
    public var streamingInterval: Double = TTSConstants.Timing.defaultStreamingInterval

    /// Whether playback is enabled during streaming generation
    /// Note: For non-streaming, playback is handled by the AudioEngine in play()
    public var playbackEnabled: Bool = false

    // MARK: - Private Properties

    @ObservationIgnored private var marvisSession: MarvisTTS?
    @ObservationIgnored private var audioPlayer: AudioSamplePlayer?
    @ObservationIgnored private var generationTask: Task<Void, Never>?
    @ObservationIgnored private var lastGeneratedSamples: [Float] = []
    @ObservationIgnored private var lastModelVariant: MarvisTTS.ModelVariant?

    // MARK: - Initialization

    public init() {
        Log.tts.debug("MarvisEngine initialized")
    }

    deinit {
        generationTask?.cancel()
    }

    // MARK: - TTSEngine Protocol Methods

    public func load(progressHandler: ((Progress) -> Void)?) async throws {
        // Check if we need to reload due to model variant change
        if isLoaded && lastModelVariant == modelVariant {
            Log.tts.debug("MarvisEngine already loaded with same model variant")
            return
        }

        // Clean up existing session if switching variants
        if isLoaded && lastModelVariant != modelVariant {
            Log.model.info("Model variant changed, reloading...")
            try await cleanup()
        }

        Log.model.info("Loading Marvis TTS model (\(self.modelVariant.displayName))...")

        do {
            // Resolve voice for MarvisTTS
            let voice = resolveVoice(selectedVoiceID) ?? .conversationalA

            marvisSession = try await MarvisTTS(
                voice: voice,
                model: modelVariant.repoId,
                progressHandler: progressHandler ?? { _ in },
                playbackEnabled: playbackEnabled
            )

            // Create audio player for external playback control
            audioPlayer = AudioSamplePlayer(sampleRate: TTSConstants.Audio.marvisSampleRate)

            lastModelVariant = modelVariant
            isLoaded = true
            Log.model.info("Marvis TTS model loaded successfully")
        } catch {
            Log.model.error("Failed to load Marvis model: \(error.localizedDescription)")
            throw TTSError.modelLoadFailed(underlying: error)
        }
    }

    public func generate(text: String, speed: Float) async throws -> AudioResult {
        // Note: Marvis doesn't support speed adjustment, parameter is ignored
        guard isLoaded, let marvisSession = marvisSession else {
            throw TTSError.modelNotLoaded
        }

        let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedText.isEmpty else {
            throw TTSError.invalidArgument("Text cannot be empty")
        }

        generationTask?.cancel()
        isGenerating = true
        generationTime = 0
        lastGeneratedSamples = []

        do {
            let result = try await marvisSession.generate(
                for: trimmedText,
                quality: qualityLevel
            )

            generationTime = result.processingTime
            lastGeneratedSamples = result.audio
            isGenerating = false

            Log.tts.timing("Marvis generation", duration: generationTime)
            Log.tts.rtf("Marvis", rtf: result.realTimeFactor)

            // Save to file
            do {
                let fileURL = try AudioFileWriter.save(
                    samples: result.audio,
                    sampleRate: result.sampleRate,
                    filename: TTSConstants.FileNames.marvisOutput.replacingOccurrences(of: ".wav", with: "")
                )
                lastGeneratedAudioURL = fileURL
            } catch {
                Log.audio.error("Failed to save audio file: \(error.localizedDescription)")
            }

            return .samples(
                data: result.audio,
                sampleRate: result.sampleRate,
                processingTime: result.processingTime
            )

        } catch {
            isGenerating = false
            Log.tts.error("Marvis generation failed: \(error.localizedDescription)")
            throw TTSError.generationFailed(underlying: error)
        }
    }

    public func play() async throws {
        guard let audioPlayer = audioPlayer else {
            throw TTSError.modelNotLoaded
        }

        guard !lastGeneratedSamples.isEmpty else {
            Log.audio.warning("No audio to play")
            return
        }

        isPlaying = true
        await audioPlayer.play(samples: lastGeneratedSamples)
        isPlaying = false
    }

    public func stop() async {
        generationTask?.cancel()
        generationTask = nil
        isGenerating = false

        marvisSession?.stopPlayback()

        await audioPlayer?.stop()
        isPlaying = false

        Log.tts.debug("MarvisEngine stopped")
    }

    public func cleanup() async throws {
        await stop()

        try? marvisSession?.cleanupMemory()
        marvisSession = nil
        audioPlayer = nil
        lastGeneratedSamples.removeAll()
        lastModelVariant = nil
        isLoaded = false

        Log.tts.debug("MarvisEngine cleaned up")
    }

    // MARK: - StreamingTTSEngine Protocol

    public func generateStreaming(text: String, speed: Float) -> AsyncThrowingStream<AudioChunk, Error> {
        AsyncThrowingStream { continuation in
            Task { @MainActor in
                guard self.isLoaded, let marvisSession = self.marvisSession else {
                    continuation.finish(throwing: TTSError.modelNotLoaded)
                    return
                }

                guard let audioPlayer = self.audioPlayer else {
                    continuation.finish(throwing: TTSError.modelNotLoaded)
                    return
                }

                let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmedText.isEmpty else {
                    continuation.finish(throwing: TTSError.invalidArgument("Text cannot be empty"))
                    return
                }

                self.isGenerating = true
                self.isPlaying = true
                self.generationTime = 0
                self.lastGeneratedSamples = []

                // Reset audio player before streaming
                await audioPlayer.stop()

                var allSamples: [Float] = []
                var isFirst = true

                do {
                    let stream = marvisSession.stream(
                        trimmedText,
                        quality: self.qualityLevel,
                        interval: self.streamingInterval
                    )

                    for try await result in stream {
                        if isFirst {
                            self.generationTime = result.processingTime
                            isFirst = false
                        }

                        allSamples.append(contentsOf: result.audio)

                        // Play chunk through AudioSamplePlayer
                        audioPlayer.enqueue(samples: result.audio)

                        let chunk = AudioChunk(
                            samples: result.audio,
                            sampleRate: result.sampleRate,
                            isLast: false,
                            processingTime: result.processingTime
                        )
                        continuation.yield(chunk)
                    }

                    // Final chunk
                    self.lastGeneratedSamples = allSamples
                    self.isGenerating = false

                    // Wait for playback to complete
                    await audioPlayer.awaitCompletion()
                    self.isPlaying = false

                    // Save to file
                    if !allSamples.isEmpty {
                        do {
                            let fileURL = try AudioFileWriter.save(
                                samples: allSamples,
                                sampleRate: TTSConstants.Audio.marvisSampleRate,
                                filename: TTSConstants.FileNames.marvisOutput.replacingOccurrences(of: ".wav", with: "")
                            )
                            self.lastGeneratedAudioURL = fileURL
                        } catch {
                            Log.audio.error("Failed to save audio file: \(error.localizedDescription)")
                        }
                    }

                    continuation.finish()

                } catch {
                    self.isGenerating = false
                    self.isPlaying = false
                    Log.tts.error("Marvis streaming failed: \(error.localizedDescription)")
                    continuation.finish(throwing: TTSError.generationFailed(underlying: error))
                }
            }
        }
    }

    // MARK: - Extended API

    /// Generate and immediately play audio
    public func say(_ text: String) async throws {
        _ = try await generate(text: text, speed: 1.0)
        // Playback is handled by MarvisTTS when playbackEnabled is true
    }

    /// Generate with streaming playback
    public func sayStreaming(_ text: String) async throws {
        for try await _ in generateStreaming(text: text, speed: 1.0) {
            // Chunks are played automatically by MarvisTTS
        }
    }

    // MARK: - Private Helpers

    private func resolveVoice(_ voiceID: String) -> MarvisTTS.Voice? {
        MarvisTTS.Voice(rawValue: voiceID)
    }
}

// MARK: - Voice Helpers

extension MarvisEngine {
    /// Get the Voice object for the currently selected voice
    var selectedVoice: Voice? {
        availableVoices.first { $0.id == selectedVoiceID }
    }

    /// Select a voice by Voice object
    func selectVoice(_ voice: Voice) {
        selectedVoiceID = voice.id
    }
}

// MARK: - Quality Level Helpers

extension MarvisEngine {
    /// Available quality levels
    static let qualityLevels = MarvisTTS.QualityLevel.allCases

    /// Description for each quality level
    func qualityDescription(for level: MarvisTTS.QualityLevel) -> String {
        switch level {
        case .low:
            return "\(level.codebookCount) codebooks - Fastest, lower quality"
        case .medium:
            return "\(level.codebookCount) codebooks - Balanced"
        case .high:
            return "\(level.codebookCount) codebooks - Slower, better quality"
        case .maximum:
            return "\(level.codebookCount) codebooks - Slowest, best quality"
        }
    }
}
