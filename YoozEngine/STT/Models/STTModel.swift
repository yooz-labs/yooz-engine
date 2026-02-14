// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import MLX

// MARK: - STT Model Protocol

/// Protocol defining the interface for all speech-to-text models
/// All model implementations (Parakeet, FastConformer, SenseVoice) must conform
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

    /// Reset the session state
    func reset()
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
        case .modelNotFound(let path):
            return "Model not found at: \(path)"
        case .configurationError(let message):
            return "Configuration error: \(message)"
        case .weightLoadingError(let message):
            return "Failed to load weights: \(message)"
        case .unsupportedLanguage(let language):
            return "Language not supported: \(language.displayName)"
        case .modelNotImplemented(let family):
            return "Model family not yet implemented: \(family.rawValue)"
        }
    }
}

// MARK: - Model Loading Protocol

/// Protocol for model loaders that can instantiate STT models
public protocol STTModelLoader {

    /// Model family this loader handles
    static var modelFamily: ModelFamily { get }

    /// Load a model for the given language from a directory
    /// - Parameters:
    ///   - directory: Directory containing model files (config.json, model.safetensors)
    ///   - language: Target language
    ///   - dtype: Data type for weights (default: bfloat16)
    /// - Returns: Loaded model instance
    static func load(
        from directory: URL,
        language: STTLanguage,
        dtype: DType
    ) throws -> any STTModel
}
