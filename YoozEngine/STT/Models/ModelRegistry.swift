// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import MLX

// MARK: - Model Registry

/// Central registry for managing STT models
/// Only one model is loaded at a time to conserve memory
/// Switching languages may require unloading the current model
public actor ModelRegistry {

    // MARK: - Singleton

    public static let shared = ModelRegistry()

    // MARK: - State

    /// Currently loaded model (if any)
    private var loadedModel: (any STTModel)?

    /// Language of the currently loaded model
    private var loadedLanguage: STTLanguage?

    /// Model directory URL (typically from app bundle)
    private var modelDirectory: URL?

    /// Data type for model weights
    private var dtype: DType = .bfloat16

    // MARK: - Initialization

    private init() {}

    // MARK: - Configuration

    /// Configure the registry with model directory and options
    /// - Parameters:
    ///   - modelDirectory: Directory containing model files
    ///   - dtype: Data type for weights (default: bfloat16)
    public func configure(modelDirectory: URL, dtype: DType = .bfloat16) {
        self.modelDirectory = modelDirectory
        self.dtype = dtype
    }

    // MARK: - Model Access

    /// Get or load a model for the specified language
    /// If a different model family is needed, the current model is unloaded first
    /// - Parameter language: Target language
    /// - Returns: Loaded model ready for transcription
    /// - Throws: STTModelError if loading fails
    public func getModel(for language: STTLanguage) async throws -> any STTModel {
        // Check if we need the same model family
        if let model = loadedModel,
           let loadedLang = loadedLanguage,
           loadedLang.modelIdentifier == language.modelIdentifier {
            // Same model family, just return current model
            // Note: For true multi-language models, we'd update the language here
            return model
        }

        // Different model needed, unload current
        await unloadCurrentModel()

        // Load new model
        let model = try await loadModel(for: language)
        loadedModel = model
        loadedLanguage = language

        return model
    }

    /// Get the currently loaded model without loading a new one
    /// - Returns: Current model or nil if none loaded
    public func currentModel() -> (any STTModel)? {
        loadedModel
    }

    /// Get the currently loaded language
    /// - Returns: Current language or nil if no model loaded
    public func currentLanguage() -> STTLanguage? {
        loadedLanguage
    }

    /// Check if a model is loaded for the given language
    public func isLoaded(for language: STTLanguage) -> Bool {
        loadedLanguage?.modelIdentifier == language.modelIdentifier
    }

    /// Check if any model is loaded
    public var hasLoadedModel: Bool {
        loadedModel != nil
    }

    // MARK: - Model Lifecycle

    /// Unload the current model and free resources
    public func unloadCurrentModel() async {
        loadedModel = nil
        loadedLanguage = nil

        // Clear GPU cache to free memory
        GPU.clearCache()
    }

    /// Preload a model for a language (useful for app startup)
    /// - Parameter language: Language to preload
    /// - Throws: STTModelError if loading fails
    public func preload(language: STTLanguage) async throws {
        _ = try await getModel(for: language)
    }

    // MARK: - Private Methods

    /// Load a model for the specified language
    private func loadModel(for language: STTLanguage) async throws -> any STTModel {
        guard let modelDir = modelDirectory else {
            throw STTModelError.modelNotFound("Model directory not configured")
        }

        // Check if language is implemented
        guard language.isImplemented else {
            throw STTModelError.modelNotImplemented(language.modelFamily)
        }

        // Load based on model family
        switch language.modelFamily {
        case .parakeetTDT:
            return try await loadParakeetModel(from: modelDir, language: language)

        case .fastConformer:
            return try await loadFastConformerModel(from: modelDir, language: language)

        case .cjk:
            throw STTModelError.modelNotImplemented(.cjk)
        }
    }

    /// Load Parakeet TDT model
    private func loadParakeetModel(from directory: URL, language: STTLanguage) async throws -> any STTModel {
        // For now, use the model directory directly
        // In future, we might have language-specific subdirectories
        let model = try ParakeetModel.fromDirectory(directory, dtype: dtype)

        // Wrap in adapter to conform to STTModel protocol
        return ParakeetModelAdapter(model: model, language: language)
    }

    /// Load FastConformer Hybrid model (Arabic/Persian)
    private func loadFastConformerModel(from directory: URL, language: STTLanguage) async throws -> any STTModel {
        // FastConformer models are in language-specific subdirectories
        // e.g., /models/fastconformer-ar/ or /models/fastconformer-fa/
        let modelSubdir = directory.appendingPathComponent(language.modelIdentifier)

        // Check if subdirectory exists, otherwise try base directory
        let modelPath: URL
        if FileManager.default.fileExists(atPath: modelSubdir.appendingPathComponent("config.json").path) {
            modelPath = modelSubdir
        } else {
            // Fallback to base directory (for testing or single-model setups)
            let baseConfigPath = directory.appendingPathComponent("config.json").path
            guard FileManager.default.fileExists(atPath: baseConfigPath) else {
                throw STTModelError.modelNotFound(
                    "FastConformer model for \(language.displayName) not found at \(modelSubdir.path) or \(directory.path)"
                )
            }

            #if DEBUG
            print("[ModelRegistry] Warning: Language-specific model not found at \(modelSubdir.path), falling back to base directory")
            #endif

            modelPath = directory
        }

        let model = try FastConformerModel.fromDirectory(modelPath, dtype: dtype)

        // Wrap in adapter to conform to STTModel protocol
        return FastConformerModelAdapter(model: model, language: language)
    }
}

// MARK: - Parakeet Model Adapter

/// Adapter to make ParakeetModel conform to STTModel protocol
final class ParakeetModelAdapter: STTModel, @unchecked Sendable {

    let underlyingModel: ParakeetModel
    public let language: STTLanguage

    init(model: ParakeetModel, language: STTLanguage) {
        self.underlyingModel = model
        self.language = language
    }

    public var modelFamily: ModelFamily {
        .parakeetTDT
    }

    public var preprocessConfig: PreprocessConfig {
        underlyingModel.config.preprocessor
    }

    public func transcribe(_ audio: [Float]) -> TranscriptionResult {
        underlyingModel.transcribe(audio)
    }

    public func createStreamingSession() -> STTStreamingSession {
        let transcriber = StreamingTranscriber(model: underlyingModel)
        return StreamingSessionAdapter(transcriber: transcriber)
    }
}

// MARK: - Streaming Session Adapter

/// Adapter to make StreamingTranscriber conform to STTStreamingSession
final class StreamingSessionAdapter: STTStreamingSession, @unchecked Sendable {

    private let transcriber: StreamingTranscriber

    init(transcriber: StreamingTranscriber) {
        self.transcriber = transcriber
    }

    public func addAudio(samples: [Float]) -> ParakeetResult {
        transcriber.addAudio(samples: samples)
    }

    public func finalize() -> ParakeetResult {
        transcriber.finalize()
    }

    public func finalizeWithTimestamps() -> TranscriptionResult {
        transcriber.finalizeWithTimestamps()
    }

    public func reset() {
        transcriber.reset()
    }
}

// MARK: - FastConformer Model Adapter

/// Adapter to make FastConformerModel conform to STTModel protocol
final class FastConformerModelAdapter: STTModel, @unchecked Sendable {

    let underlyingModel: FastConformerModel
    public let language: STTLanguage

    init(model: FastConformerModel, language: STTLanguage) {
        self.underlyingModel = model
        self.language = language
    }

    public var modelFamily: ModelFamily {
        .fastConformer
    }

    public var preprocessConfig: PreprocessConfig {
        underlyingModel.config.preprocessor
    }

    public func transcribe(_ audio: [Float]) -> TranscriptionResult {
        underlyingModel.transcribe(audio)
    }

    public func createStreamingSession() -> STTStreamingSession {
        let transcriber = FastConformerStreamingTranscriber(model: underlyingModel)
        return FastConformerStreamingSessionAdapter(transcriber: transcriber)
    }
}

// MARK: - FastConformer Streaming Transcriber

/// Streaming transcriber for FastConformer models
/// Similar to StreamingTranscriber but uses RNNT decoding
///
/// For long recordings (>30s), use periodic finalization to prevent memory issues:
/// 1. Call `addAudio()` to accumulate and get draft results
/// 2. When you have a natural break (silence, punctuation), call `finalizeChunk()`
/// 3. This commits the current transcription and resets for the next chunk
final class FastConformerStreamingTranscriber: @unchecked Sendable {
    private let model: FastConformerModel
    private var audioBuffer: [Float] = []
    private var encoderCache: [ConformerCache]
    private var previousEncoderOutput: MLXArray?
    private var lastTokens: [AlignedToken] = []
    private var finalizedText: String = ""
    private let minChunkSamples: Int

    /// Threshold for auto-finalization warning (~30 seconds)
    /// Beyond this, users should call finalizeChunk() to prevent memory issues
    private let warnThresholdFrames: Int = 375

    /// Hard limit for encoder frames (~60 seconds)
    /// At this point, we auto-finalize to prevent memory exhaustion
    private let maxEncoderOutputFrames: Int = 750

    init(model: FastConformerModel) {
        self.model = model
        self.encoderCache = model.createEncoderCaches()

        // Calculate minimum chunk size for encoder processing
        let subsamplingFactor = model.config.encoder.subsamplingFactor
        let hopLength = model.config.preprocessor.hopLength
        self.minChunkSamples = subsamplingFactor * hopLength * 8  // ~8 encoder frames
    }

    func addAudio(samples: [Float]) -> ParakeetResult {
        audioBuffer.append(contentsOf: samples)

        // Only process if we have enough samples
        guard audioBuffer.count >= minChunkSamples else {
            return ParakeetResult(
                text: finalizedText,
                finalized: finalizedText,
                draft: ""
            )
        }

        // Check if we need to auto-finalize to prevent memory issues
        if let prev = previousEncoderOutput, prev.dim(1) >= maxEncoderOutputFrames {
            #if DEBUG
            print("[FastConformerStreaming] Auto-finalizing chunk at \(maxEncoderOutputFrames) frames to prevent memory issues")
            #endif
            _ = finalizeChunk()
        }

        // Transcribe with streaming
        let (tokens, newEncoderOutput) = model.transcribeStreaming(
            audioBuffer,
            encoderCache: encoderCache,
            previousEncoderOutput: previousEncoderOutput
        )

        // Update encoder output cache
        if let prev = previousEncoderOutput {
            previousEncoderOutput = concatenated([prev, newEncoderOutput], axis: 1)

            #if DEBUG
            // Warn if approaching memory limit
            if previousEncoderOutput!.dim(1) >= warnThresholdFrames &&
               previousEncoderOutput!.dim(1) < warnThresholdFrames + 50 {
                print("[FastConformerStreaming] Warning: Approaching memory limit. Consider calling finalizeChunk() at natural breaks.")
            }
            #endif
        } else {
            previousEncoderOutput = newEncoderOutput
        }

        lastTokens = tokens

        // Build result
        let fullText = tokens.map { $0.text }.joined()

        // For streaming, treat all text as draft until finalized
        return ParakeetResult(
            text: finalizedText + fullText,
            finalized: finalizedText,
            draft: fullText
        )
    }

    /// Finalize the current chunk and prepare for the next one
    /// Call this at natural breaks (silence, sentence boundaries) for long recordings
    /// Returns the finalized text for this chunk
    func finalizeChunk() -> String {
        // Decode current buffer
        if !audioBuffer.isEmpty && previousEncoderOutput != nil {
            let (tokens, _) = model.transcribeStreaming(
                audioBuffer,
                encoderCache: encoderCache,
                previousEncoderOutput: previousEncoderOutput
            )
            lastTokens = tokens
        }

        let chunkText = lastTokens.map { $0.text }.joined()
        finalizedText += chunkText

        // Reset for next chunk
        audioBuffer = []
        encoderCache = model.createEncoderCaches()
        previousEncoderOutput = nil
        lastTokens = []

        return chunkText
    }

    func finalize() -> ParakeetResult {
        // Process any remaining audio
        if !audioBuffer.isEmpty {
            let (tokens, _) = model.transcribeStreaming(
                audioBuffer,
                encoderCache: encoderCache,
                previousEncoderOutput: previousEncoderOutput
            )
            lastTokens = tokens
        }

        let fullText = lastTokens.map { $0.text }.joined()
        finalizedText = fullText

        return ParakeetResult(
            text: fullText,
            finalized: fullText,
            draft: ""
        )
    }

    func finalizeWithTimestamps() -> TranscriptionResult {
        _ = finalize()
        return TranscriptionResult(tokens: lastTokens)
    }

    func reset() {
        audioBuffer = []
        encoderCache = model.createEncoderCaches()
        previousEncoderOutput = nil
        lastTokens = []
        finalizedText = ""
    }
}

// MARK: - FastConformer Streaming Session Adapter

/// Adapter to make FastConformerStreamingTranscriber conform to STTStreamingSession
final class FastConformerStreamingSessionAdapter: STTStreamingSession, @unchecked Sendable {

    private let transcriber: FastConformerStreamingTranscriber

    init(transcriber: FastConformerStreamingTranscriber) {
        self.transcriber = transcriber
    }

    public func addAudio(samples: [Float]) -> ParakeetResult {
        transcriber.addAudio(samples: samples)
    }

    public func finalize() -> ParakeetResult {
        transcriber.finalize()
    }

    public func finalizeWithTimestamps() -> TranscriptionResult {
        transcriber.finalizeWithTimestamps()
    }

    public func reset() {
        transcriber.reset()
    }
}
