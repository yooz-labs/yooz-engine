// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

// MARK: - STT Model Protocol

/// Protocol defining the interface for all speech-to-text model backends
/// Implementations: ParakeetModelAdapter, FastConformerModelAdapter
public protocol STTModel: AnyObject, Sendable {
    /// The language this model instance is configured for
    var language: STTLanguage { get }

    /// The model family (architecture type)
    var modelFamily: ModelFamily { get }

    /// Preprocessing configuration for audio
    var preprocessConfig: PreprocessConfig { get }

    /// Transcribe audio samples to text
    /// - Parameter audio: Audio samples at 16kHz
    /// - Returns: Transcription result with aligned tokens
    func transcribe(_ audio: [Float]) -> TranscriptionResult

    /// Create a streaming session for real-time transcription
    /// - Returns: A streaming session that can process audio incrementally
    func createStreamingSession() -> STTStreamingSession
}

// MARK: - Streaming Session Protocol

/// Protocol for streaming transcription sessions
/// Each model family provides its own implementation
public protocol STTStreamingSession: AnyObject, Sendable {
    /// Add audio samples and get current transcription state
    /// - Parameter samples: Audio samples at 16kHz
    /// - Returns: Current transcription result (finalized + draft)
    func addAudio(samples: [Float]) -> ParakeetResult

    /// Finalize the session and get complete transcription
    /// - Returns: Final transcription result
    func finalize() -> ParakeetResult

    /// Finalize with full timestamp information
    /// - Returns: Transcription result with aligned tokens
    func finalizeWithTimestamps() -> TranscriptionResult

    /// Reset the session state for a new recognition pass
    func reset()

    /// Full reset clearing all state including accumulated context
    func fullReset()
}

public extension STTStreamingSession {
    /// Default: fullReset is the same as reset
    func fullReset() {
        reset()
    }
}

// MARK: - Model Loading Error

/// Errors that can occur during model loading
public enum STTModelError: Error, LocalizedError {
    case modelNotFound(String)
    case configurationError(String)
    case weightLoadingError(String)
    case unsupportedLanguage(STTLanguage)
    case modelNotImplemented(ModelFamily)

    public var errorDescription: String? {
        switch self {
        case let .modelNotFound(path):
            "Model not found at: \(path)"
        case let .configurationError(message):
            "Configuration error: \(message)"
        case let .weightLoadingError(message):
            "Failed to load weights: \(message)"
        case let .unsupportedLanguage(language):
            "Language not supported: \(language.displayName)"
        case let .modelNotImplemented(family):
            "Model family not yet implemented: \(family.rawValue)"
        }
    }
}
