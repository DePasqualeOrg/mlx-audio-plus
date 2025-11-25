//
//  AudioFilePlayer.swift
//  MLXAudio
//
//  Plays audio files using AVAudioPlayer.
//  Used by views for replaying saved audio with progress tracking.
//

import AVFoundation
import Foundation

/// Plays audio files with progress tracking
@Observable
class AudioFilePlayer: NSObject {
    // MARK: - Public Properties

    /// Whether audio is currently playing
    var isPlaying: Bool = false

    /// Current playback position in seconds
    var currentTime: TimeInterval = 0

    /// Total duration of the loaded audio in seconds
    var duration: TimeInterval = 0

    /// URL of the currently loaded audio file
    var currentAudioURL: URL?

    // MARK: - Private Properties

    @ObservationIgnored private var player: AVAudioPlayer?
    @ObservationIgnored private var timer: Timer?

    // MARK: - Initialization

    override init() {
        super.init()
    }

    deinit {
        stop()
    }

    // MARK: - Playback Control

    /// Load an audio file for playback
    /// - Parameter url: URL of the audio file to load
    func loadAudio(from url: URL) {
        do {
            // Stop any existing playback
            stop()

            // Create new player
            player = try AVAudioPlayer(contentsOf: url)
            player?.delegate = self
            player?.prepareToPlay()

            // Update state
            currentAudioURL = url
            duration = player?.duration ?? 0
            currentTime = 0

            Log.audio.debug("Loaded audio: \(url.lastPathComponent), duration: \(self.duration)s")
        } catch {
            Log.audio.error("Failed to load audio: \(error.localizedDescription)")
            currentAudioURL = nil
            duration = 0
            currentTime = 0
        }
    }

    /// Start or resume playback
    func play() {
        guard let player = player else { return }

        player.play()
        isPlaying = true
        startTimer()

        Log.audio.debug("Playback started")
    }

    /// Pause playback
    func pause() {
        player?.pause()
        isPlaying = false
        stopTimer()

        Log.audio.debug("Playback paused")
    }

    /// Toggle between play and pause
    func togglePlayPause() {
        if isPlaying {
            pause()
        } else {
            play()
        }
    }

    /// Stop playback and reset to beginning
    func stop() {
        player?.stop()
        isPlaying = false
        stopTimer()
        currentTime = 0

        Log.audio.debug("Playback stopped")
    }

    /// Seek to a specific time
    /// - Parameter time: Target time in seconds
    func seek(to time: TimeInterval) {
        guard let player = player else { return }
        player.currentTime = max(0, min(time, duration))
        currentTime = player.currentTime
    }

    // MARK: - Timer Management

    private func startTimer() {
        stopTimer()
        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            guard let self = self, let player = self.player else { return }
            self.currentTime = player.currentTime
        }
        timer?.tolerance = 0.05
    }

    private func stopTimer() {
        timer?.invalidate()
        timer = nil
    }
}

// MARK: - AVAudioPlayerDelegate

extension AudioFilePlayer: AVAudioPlayerDelegate {
    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        isPlaying = false
        stopTimer()
        currentTime = 0

        Log.audio.debug("Playback finished (success: \(flag))")
    }

    func audioPlayerDecodeErrorDidOccur(_ player: AVAudioPlayer, error: Error?) {
        Log.audio.error("Audio decode error: \(error?.localizedDescription ?? "unknown")")
        isPlaying = false
        stopTimer()
    }
}
