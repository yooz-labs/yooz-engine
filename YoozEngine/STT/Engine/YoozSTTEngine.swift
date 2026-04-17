// Copyright 2026 Yooz Labs. All rights reserved.

import Combine
import EngineCore
import Foundation
import MLX

/// Result from streaming transcription
public struct ParakeetResult: Equatable, Sendable {
    /// Full text (finalized + draft)
    public let text: String
    /// Confirmed text that won't change
    public let finalized: String
    /// Tentative text that may change with more audio
    public let draft: String

    public var isEmpty: Bool {
        text.trimmingCharacters(in: .whitespaces).isEmpty
    }

    public static let empty = ParakeetResult(text: "", finalized: "", draft: "")

    public init(text: String, finalized: String, draft: String) {
        self.text = text
        self.finalized = finalized
        self.draft = draft
    }
}

/// Native Swift STT Engine with multi-language support
/// Supports Parakeet TDT (English/European), FastConformer (Arabic/Persian/Hebrew), and CJK models
/// Drop-in replacement for ParakeetMLXManager (Python bridge)
public final class YoozSTTEngine: ObservableObject, @unchecked Sendable {

    // MARK: - Singleton

    public static let shared = YoozSTTEngine()

    // MARK: - Published Properties

    @Published public private(set) var isReady = false
    @Published public private(set) var isStreaming = false
    @Published public private(set) var currentResult = ParakeetResult.empty
    @Published public private(set) var lastError: String?
    @Published public private(set) var currentLanguage: STTLanguage = .english

    // MARK: - Callbacks

    /// Callback for transcription updates
    public var onTranscription: ((ParakeetResult) -> Void)?

    /// Callback for errors
    public var onError: ((String) -> Void)?

    // MARK: - Private Properties

    private var model: ParakeetModel?
    private var streamingContext: StreamingTranscriber?
    private let sampleRate: Int = 16000

    /// Lock for thread safety
    private let lock = NSLock()

    // MARK: - Initialization

    private init() {}

    // MARK: - Model Management

    /// Load the model for a specific language
    /// - Parameter language: The target language (defaults to English)
    /// - Throws: YoozSTTError if model loading fails
    public func start(language: STTLanguage = .english) async throws {
        NSLog("YoozSTTEngine: start(language: %@) called", language.rawValue)

        // Check if already loaded with same language
        if model != nil && currentLanguage == language {
            NSLog("YoozSTTEngine: Already started with language %@", language.rawValue)
            return
        }

        // If switching languages, stop first
        if model != nil && currentLanguage != language {
            NSLog("YoozSTTEngine: Switching language from %@ to %@", currentLanguage.rawValue, language.rawValue)
            stop()
        }

        NSLog("YoozSTTEngine: Loading model for %@...", language.displayName)

        do {
            // Get model directory from app bundle
            let modelDir = try getModelDirectory()
            NSLog("YoozSTTEngine: Got model directory: %@", modelDir.path)

            // Check if language is implemented
            guard language.isImplemented else {
                throw YoozSTTError.languageNotSupported(language)
            }

            // Load model
            NSLog("YoozSTTEngine: Calling ParakeetModel.fromDirectory...")
            let loadedModel = try ParakeetModel.fromDirectory(modelDir, dtype: .bfloat16)

            lock.lock()
            self.model = loadedModel
            lock.unlock()

            await MainActor.run {
                self.isReady = true
                self.currentLanguage = language
            }

            NSLog("YoozSTTEngine: Model loaded successfully for %@!", language.displayName)
        } catch let error as YoozSTTError {
            let errorMsg = error.localizedDescription ?? "Unknown error"
            NSLog("YoozSTTEngine: ERROR - %@", errorMsg)
            await MainActor.run {
                self.lastError = errorMsg
            }
            onError?(errorMsg)
            throw error
        } catch {
            let errorMsg = "Failed to load model: \(error.localizedDescription)"
            NSLog("YoozSTTEngine: ERROR - %@", errorMsg)
            await MainActor.run {
                self.lastError = errorMsg
            }
            onError?(errorMsg)
            throw YoozSTTError.modelLoadFailed(error.localizedDescription)
        }
    }

    /// Switch to a different language
    /// If the new language uses the same model family, this is fast.
    /// If it requires a different model, the current one is unloaded first.
    /// - Parameter language: The target language
    /// - Throws: YoozSTTError if model loading fails
    public func setLanguage(_ language: STTLanguage) async throws {
        guard language != currentLanguage else {
            NSLog("YoozSTTEngine: Already using language %@", language.rawValue)
            return
        }

        // Stop any active stream
        if isStreaming {
            _ = await stopStream()
        }

        // Check if we can reuse current model
        if currentLanguage.modelIdentifier == language.modelIdentifier {
            // Same model family, just update language
            await MainActor.run {
                self.currentLanguage = language
            }
            NSLog("YoozSTTEngine: Switched to %@ (same model)", language.displayName)
            return
        }

        // Need to load a different model
        try await start(language: language)
    }

    /// Get list of available (implemented) languages
    public var availableLanguages: [STTLanguage] {
        STTLanguage.implemented
    }

    /// Unload the model and free resources
    public func stop() {
        lock.lock()
        defer { lock.unlock() }

        if isStreaming {
            streamingContext = nil
        }

        model = nil

        // Update state synchronously for immediate effect
        if Thread.isMainThread {
            self.isReady = false
            self.isStreaming = false
            self.currentResult = .empty
        } else {
            DispatchQueue.main.sync {
                self.isReady = false
                self.isStreaming = false
                self.currentResult = .empty
            }
        }

        print("YoozSTTEngine: Stopped")
    }

    // MARK: - Streaming Control

    /// Start streaming transcription session
    public func startStream(mode: AudioMode = .normal) {
        lock.lock()
        defer { lock.unlock() }

        guard let model = model else {
            let errorMsg = "Cannot start stream: model not loaded. Call start() first."
            NSLog("YoozSTTEngine: %@", errorMsg)
            DispatchQueue.main.async {
                self.lastError = errorMsg
            }
            onError?(errorMsg)
            return
        }

        guard streamingContext == nil else {
            print("YoozSTTEngine: Already streaming")
            return
        }

        // Create config with mode-specific settings
        // audioMode affects: preemphasis (0.99 whispered vs 0.97 normal) and spectral tilt compensation
        var config = model.config.preprocessor
        config.audioMode = mode

        streamingContext = StreamingTranscriber(
            model: model,
            preprocessConfig: config
        )

        // Update state synchronously for immediate effect
        if Thread.isMainThread {
            self.isStreaming = true
            self.currentResult = .empty
        } else {
            DispatchQueue.main.sync {
                self.isStreaming = true
                self.currentResult = .empty
            }
        }

        print("YoozSTTEngine: Stream started in \(mode.rawValue) mode")
    }

    /// Stop streaming and get final result (async - processes remaining audio)
    public func stopStream() async -> ParakeetResult {
        lock.lock()
        let context = streamingContext
        streamingContext = nil
        lock.unlock()

        guard let context = context else {
            return currentResult
        }

        // Finalize transcription
        let result = context.finalize()

        await MainActor.run {
            self.isStreaming = false
            self.currentResult = result
        }

        onTranscription?(result)
        print("YoozSTTEngine: Stream stopped, final text: '\(result.text.prefix(50))...'")

        // Clear GPU cache to avoid memory accumulation between streams
        GPU.clearCache()

        return result
    }

    /// Stop streaming synchronously (returns immediately)
    public func stopStreamSync() -> ParakeetResult {
        lock.lock()
        let context = streamingContext
        streamingContext = nil
        lock.unlock()

        let result = context?.finalize() ?? currentResult

        DispatchQueue.main.async {
            self.isStreaming = false
            self.currentResult = result
        }

        return result
    }

    /// Reset the stream (start fresh, clears all accumulated context)
    public func resetStream() {
        lock.lock()
        let context = streamingContext
        streamingContext = nil
        lock.unlock()

        // Full reset clears accumulated context
        context?.reset()

        DispatchQueue.main.async {
            self.isStreaming = false
            self.currentResult = .empty
        }
    }

    /// Add audio samples to the transcriber
    /// - Parameter samples: Float32 audio samples at 16kHz
    public func addAudio(samples: [Float]) {
        lock.lock()

        guard model != nil else {
            lock.unlock()
            let errorMsg = "Cannot add audio: model not loaded. Call start() first."
            NSLog("YoozSTTEngine: %@", errorMsg)
            DispatchQueue.main.async {
                self.lastError = errorMsg
            }
            onError?(errorMsg)
            return
        }

        // Auto-start stream if needed
        if streamingContext == nil {
            NSLog("YoozSTTEngine: Auto-starting stream")
            streamingContext = StreamingTranscriber(model: model!)
            DispatchQueue.main.async {
                self.isStreaming = true
            }
        }

        let context = streamingContext!
        lock.unlock()

        // Add audio and get result
        NSLog("YoozSTTEngine: Adding %d samples", samples.count)
        let result = context.addAudio(samples: samples)
        NSLog("YoozSTTEngine: Result text='%@' finalized='%@'", result.text, result.finalized)

        DispatchQueue.main.async {
            self.currentResult = result
        }

        onTranscription?(result)
    }

    // MARK: - Status

    public var isRunning: Bool {
        return model != nil
    }

    // MARK: - Batch Transcription

    /// Transcribe audio in batch mode, independent from any active streaming session
    ///
    /// Use this for accurate transcription of complete audio segments while streaming
    /// continues for real-time preview. The batch transcription runs in parallel without
    /// affecting the streaming session state.
    ///
    /// - Parameter samples: Audio samples at 16kHz
    /// - Returns: Transcription result with finalized text
    public func batchTranscribe(samples: [Float], mode: AudioMode = .normal) async -> ParakeetResult {
        // Get model reference synchronously (before any async work)
        guard let transcriber = createBatchTranscriber(mode: mode) else {
            print("YoozSTTEngine: Cannot batch transcribe - model not loaded")
            return .empty
        }

        // Process all audio at once
        _ = transcriber.addAudio(samples: samples)

        // Finalize to get complete transcription
        let result = transcriber.finalize()

        print("YoozSTTEngine: Batch transcription complete (\(mode.rawValue)): '\(result.text.prefix(50))...'")
        return result
    }

    /// Transcribe audio with word-level timestamps
    ///
    /// Use this for accurate transcription with timing information. Each token includes
    /// start time and duration in seconds. Useful for:
    /// - Overlapping context transcription (filter tokens by timestamp)
    /// - Word-level highlighting
    /// - Subtitle generation
    ///
    /// - Parameter samples: Audio samples at 16kHz
    /// - Returns: TranscriptionResult with aligned tokens containing timestamps
    /// - Throws: `YoozSTTError.notReady` when no model is loaded. The text-only
    ///   `batchTranscribe` path preserves legacy empty-return behaviour, but the
    ///   aligned path (engine#34) refuses to silently impersonate a silent
    ///   transcription — callers get an explicit error instead of a 200 OK with
    ///   empty tokens that would be indistinguishable from genuine silence.
    public func batchTranscribeAligned(samples: [Float], mode: AudioMode = .normal) async throws -> TranscriptionResult {
        guard let transcriber = createBatchTranscriber(mode: mode) else {
            print("YoozSTTEngine: Cannot batch transcribe - model not loaded")
            throw YoozSTTError.notReady
        }

        // Process all audio at once
        _ = transcriber.addAudio(samples: samples)

        // Finalize with timestamps
        let result = transcriber.finalizeWithTimestamps()

        print("YoozSTTEngine: Batch transcription with timestamps complete (\(mode.rawValue)): \(result.tokens.count) tokens")
        return result
    }

    /// Create an independent transcriber for batch or per-connection streaming.
    ///
    /// Used by the API server to create per-WebSocket streaming sessions
    /// without interfering with the singleton streaming state.
    public func createBatchTranscriber(mode: AudioMode = .normal) -> StreamingTranscriber? {
        lock.lock()
        defer { lock.unlock() }

        guard let model = model else {
            return nil
        }

        // Create config with mode-specific settings
        // audioMode affects: preemphasis (0.99 whispered vs 0.97 normal) and spectral tilt compensation
        var config = model.config.preprocessor
        config.audioMode = mode

        return StreamingTranscriber(
            model: model,
            preprocessConfig: config
        )
    }

    // MARK: - Private Methods

    private func getModelDirectory() throws -> URL {
        NSLog("YoozSTTEngine: getModelDirectory called")

        // 1. Check EngineConfig.modelsDirectory (Application Support)
        let engineModelsDir = EngineConfig.modelsDirectory
        let engineConfigPath = engineModelsDir.appendingPathComponent("config.json").path
        NSLog("YoozSTTEngine: Checking engine models dir: %@", engineConfigPath)

        if FileManager.default.fileExists(atPath: engineConfigPath) {
            NSLog("YoozSTTEngine: Found model at engine models directory")
            return engineModelsDir
        }

        // 2. Check app bundle Resources/Models/
        guard let resourcePath = Bundle.main.resourcePath else {
            NSLog("YoozSTTEngine: ERROR - resourcePath is nil")
            throw YoozSTTError.modelNotFound("Bundle.main.resourcePath is nil")
        }

        let resourceDir = URL(fileURLWithPath: resourcePath)
        let modelsDir = resourceDir.appendingPathComponent("Models")
        let modelsConfigPath = modelsDir.appendingPathComponent("config.json").path
        NSLog("YoozSTTEngine: Checking %@", modelsConfigPath)

        if FileManager.default.fileExists(atPath: modelsConfigPath) {
            NSLog("YoozSTTEngine: Found model at Resources/Models/")
            return modelsDir
        }

        // 3. Check Resources/ directly (xcodegen may flatten structure)
        let directConfigPath = resourceDir.appendingPathComponent("config.json").path
        NSLog("YoozSTTEngine: Checking %@", directConfigPath)

        if FileManager.default.fileExists(atPath: directConfigPath) {
            NSLog("YoozSTTEngine: Found model at Resources/")
            return resourceDir
        }

        // Model not found
        NSLog("YoozSTTEngine: ERROR - Model not found!")
        throw YoozSTTError.modelNotFound(
            "STT model not found. Checked: \(engineModelsDir.path), \(resourceDir.path)"
        )
    }
}

// MARK: - Errors

public enum YoozSTTError: Error, LocalizedError {
    case modelNotFound(String)
    case modelLoadFailed(String)
    case notReady
    case streamError(String)
    case languageNotSupported(STTLanguage)

    public var errorDescription: String? {
        switch self {
        case let .modelNotFound(message):
            "Model not found: \(message)"
        case let .modelLoadFailed(reason):
            "Failed to load model: \(reason)"
        case .notReady:
            "YoozSTTEngine is not ready. Call start() first."
        case let .streamError(reason):
            "Streaming error: \(reason)"
        case let .languageNotSupported(language):
            "Language not yet supported: \(language.displayName). Only \(STTLanguage.implemented.map(\.displayName).joined(separator: ", ")) are currently available."
        }
    }
}
