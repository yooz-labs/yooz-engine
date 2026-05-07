// Copyright 2026 Yooz Labs. All rights reserved.

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

    // MARK: - Version

    /// STT component version (independent of EngineConfig.version).
    public static let version = "0.6.6"

    // MARK: - Singleton

    public static let shared = YoozSTTEngine()

    // MARK: - Published Properties

    @Published public private(set) var isReady = false
    @Published public private(set) var isStreaming = false
    @Published public private(set) var currentResult = ParakeetResult.empty
    @Published public private(set) var lastError: String?
    @Published public private(set) var currentLanguage: STTLanguage = .english

    /// Fraction-completed [0.0, 1.0] for an in-progress HF model
    /// download. Resets to 0 on `start(language:)` and ticks up to 1.0
    /// as files stream in. Stays at 1.0 once the load completes
    /// (cached snapshots jump straight to 1.0). UI clients poll via
    /// the `progress` field of `/v1/stt/status`.
    @Published public private(set) var downloadProgress: Double = 0

    /// Active STT backend. Default is `.parakeet`; switchable via
    /// `setBackend(_:)` (HTTP `POST /v1/stt/engine`) or the
    /// `YOOZ_STT_BACKEND` env var honored by `EngineConfig`.
    @Published public private(set) var currentBackend: STTBackendID = .parakeet

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

    private init() {
        // Honor the env-driven default at construction time.
        self.currentBackend = EngineConfig.sttBackend
    }

    // MARK: - Backend Selection

    /// Switch the active STT backend. Unloads any model held by the
    /// previous backend so the next `start(language:)` /
    /// `batchTranscribe(...)` call starts cleanly.
    ///
    /// This call is light-weight: it does NOT trigger a model fetch
    /// or pipeline load. Heavy work happens lazily on the next
    /// transcribe call, matching the existing Parakeet flow.
    public func setBackend(_ backend: STTBackendID) async {
        guard backend != currentBackend else { return }
        NSLog(
            "YoozSTTEngine: switching backend %@ -> %@",
            currentBackend.rawValue, backend.rawValue
        )

        // Drop Parakeet/FastConformer state.
        stop()

        // Drop Qwen3 state if it was loaded.
        await Qwen3ASRBackend.shared.unload()

        await MainActor.run {
            self.currentBackend = backend
            self.isReady = false
        }
    }

    /// Whether the currently-selected backend has a model loaded and
    /// ready to transcribe. Differs from `isReady` only for the Qwen3
    /// backend, which keeps its load state inside the actor.
    public func isCurrentBackendLoaded() async -> Bool {
        switch currentBackend {
        case .parakeet, .fastConformer, .appleSTT:
            return isRunning
        case .qwen3ASRPreview:
            return await Qwen3ASRBackend.shared.isLoaded
        }
    }

    // MARK: - Model Management

    /// Load the model for a specific language.
    /// - Parameters:
    ///   - language: The target language (defaults to English).
    ///   - allowFetch: When `true` (default), the engine pulls the
    ///     model from Hugging Face if no local snapshot is staged
    ///     under `EngineConfig.modelsDirectory` or the bundle. When
    ///     `false`, the load fails with `modelNotFound` rather than
    ///     hitting the network. Mirrors the wire shape `/v1/stt/load`
    ///     already exposes for the Qwen3 backend.
    /// - Throws: `YoozSTTError` if model loading fails.
    public func start(
        language: STTLanguage = .english,
        allowFetch: Bool = true
    ) async throws {
        NSLog("YoozSTTEngine: start(language: %@) called", language.rawValue)

        // Backend-specific dispatch: the Qwen3 backend lives entirely
        // inside its actor and does not touch `model` / `currentLanguage`.
        if currentBackend == .qwen3ASRPreview {
            try await startQwen3Backend(language: language)
            return
        }

        try await loadParakeetModel(language: language, allowFetch: allowFetch)
    }

    /// Load the Parakeet/FastConformer model for `language` without
    /// touching `currentBackend`. Used by the auto-fallback adapter so
    /// the hook can run a single Parakeet transcription without
    /// permanently flipping the engine's backend selection — the
    /// user's preview pick is honored at next process start (or after
    /// a manual `POST /v1/stt/engine`).
    public func loadParakeetModel(
        language: STTLanguage,
        allowFetch: Bool = true
    ) async throws {
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

        // Reset progress at the start of every load so a stale 1.0
        // from a prior run does not mislead clients polling
        // /v1/stt/status during the early phase of a switch.
        await MainActor.run { self.downloadProgress = 0 }

        do {
            // Check if language is implemented
            guard language.isImplemented else {
                throw YoozSTTError.languageNotSupported(language)
            }

            // Resolve the model directory. Order:
            //   1. EngineConfig.modelsDirectory (legacy bundled drop)
            //   2. App bundle Resources/Models/ (and Resources/ flat)
            //   3. HF auto-download via swift-transformers Hub (gated
            //      by `allowFetch`; cached snapshots bypass the
            //      network and resolve in O(1)).
            let modelDir = try await getModelDirectory(
                for: language,
                allowFetch: allowFetch
            )
            NSLog("YoozSTTEngine: Got model directory: %@", modelDir.path)

            // Load model
            NSLog("YoozSTTEngine: Calling ParakeetModel.fromDirectory...")
            let loadedModel = try ParakeetModel.fromDirectory(modelDir, dtype: .bfloat16)

            lock.lock()
            self.model = loadedModel
            lock.unlock()

            await MainActor.run {
                self.isReady = true
                self.currentLanguage = language
                self.downloadProgress = 1
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
    public func batchTranscribeAligned(samples: [Float], mode: AudioMode = .normal) async -> TranscriptionResult {
        guard let transcriber = createBatchTranscriber(mode: mode) else {
            print("YoozSTTEngine: Cannot batch transcribe - model not loaded")
            return TranscriptionResult(tokens: [])
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
    func createBatchTranscriber(mode: AudioMode = .normal) -> StreamingTranscriber? {
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

    // MARK: - Qwen3 Backend

    /// Load the Qwen3-ASR pipeline for the given language. The
    /// pipeline auto-detects language at runtime, so the language
    /// parameter is mostly informational (recorded as
    /// `currentLanguage`). Requires the model directory to be ready
    /// on disk; callers run `Qwen3ASRModelFetcher` first if needed.
    private func startQwen3Backend(language: STTLanguage) async throws {
        guard
            STTBackendID.qwen3ASRPreview.supportedLanguages.contains(language)
        else {
            throw YoozSTTError.languageNotSupported(language)
        }

        let modelDir = Qwen3ASRModelFetcher.defaultModelDir
        guard FileManager.default.fileExists(
            atPath: modelDir.appendingPathComponent("config.json").path
        ) else {
            throw YoozSTTError.modelNotFound(
                "Qwen3-ASR model directory not ready at "
                + "\(modelDir.path) — run model fetch first"
            )
        }

        do {
            try await Qwen3ASRBackend.shared.ensureLoaded(modelDir: modelDir)
            await MainActor.run {
                self.isReady = true
                self.currentLanguage = language
            }
            NSLog(
                "YoozSTTEngine: Qwen3-ASR ready (language=%@)",
                language.rawValue
            )
        } catch {
            let msg = "Failed to load Qwen3-ASR pipeline: \(error)"
            NSLog("YoozSTTEngine: ERROR - %@", msg)
            await MainActor.run {
                self.lastError = msg
            }
            onError?(msg)
            throw YoozSTTError.modelLoadFailed(msg)
        }
    }

    /// Run a Qwen3-ASR batch transcription. Returns a `ParakeetResult`
    /// so callers (the existing HTTP layer) don't have to branch on
    /// the backend type. Errors are SWALLOWED — the swallow is
    /// historical (preserved for callers that don't care about
    /// distinguishing failure modes); new code should call
    /// `batchTranscribeQwen3Throwing(...)` and map the typed error.
    @available(
        *, deprecated,
        message: "Errors are silently swallowed as `.empty`; SDK consumers see blank text on every failure mode. Prefer batchTranscribeQwen3Throwing(samples:language:) and map the typed Qwen3ASRError."
    )
    public func batchTranscribeQwen3(
        samples: [Float],
        language: STTLanguage
    ) async -> ParakeetResult {
        do {
            return try await batchTranscribeQwen3Throwing(
                samples: samples,
                language: language
            )
        } catch {
            NSLog(
                "YoozSTTEngine: Qwen3 batch transcribe failed: %@",
                String(describing: error)
            )
            return .empty
        }
    }

    /// Run a Qwen3-ASR batch transcription, propagating the typed
    /// `Qwen3ASRError` (or any other underlying error) so the
    /// HTTP layer can map cases to the right status code.
    public func batchTranscribeQwen3Throwing(
        samples: [Float],
        language: STTLanguage
    ) async throws -> ParakeetResult {
        // `STTLanguage.qwen3LanguageHint` is the single source of
        // truth for this mapping, shared with
        // `Qwen3ASRPreviewBackendAdapter`.
        let result = try await Qwen3ASRBackend.shared.transcribe(
            pcm: samples,
            language: language.qwen3LanguageHint
        )
        return ParakeetResult(
            text: result.text,
            finalized: result.text,
            draft: ""
        )
    }

    // MARK: - Private Methods

    private func getModelDirectory(
        for language: STTLanguage,
        allowFetch: Bool
    ) async throws -> URL {
        NSLog("YoozSTTEngine: getModelDirectory(for: %@) called", language.rawValue)

        // 1. EngineConfig.modelsDirectory (legacy bundled drop). We
        //    treat the presence of `config.json` as the readiness
        //    sentinel; partial drops still fall through to step 2/3.
        let engineModelsDir = EngineConfig.modelsDirectory
        let engineConfigPath = engineModelsDir.appendingPathComponent("config.json").path
        NSLog("YoozSTTEngine: Checking engine models dir: %@", engineConfigPath)

        if FileManager.default.fileExists(atPath: engineConfigPath) {
            NSLog("YoozSTTEngine: Found model at engine models directory")
            return engineModelsDir
        }

        // 2. App bundle Resources/Models/ — used by older yooz-whisper
        //    drops that vendor the model alongside the binary. Phase 5
        //    thin-client builds skip this and rely on step 3.
        if let resourcePath = Bundle.main.resourcePath {
            let resourceDir = URL(fileURLWithPath: resourcePath)
            let modelsDir = resourceDir.appendingPathComponent("Models")
            let modelsConfigPath = modelsDir.appendingPathComponent("config.json").path
            NSLog("YoozSTTEngine: Checking %@", modelsConfigPath)

            if FileManager.default.fileExists(atPath: modelsConfigPath) {
                NSLog("YoozSTTEngine: Found model at Resources/Models/")
                return modelsDir
            }

            // 3. Resources/ directly (xcodegen may flatten structure).
            let directConfigPath = resourceDir.appendingPathComponent("config.json").path
            NSLog("YoozSTTEngine: Checking %@", directConfigPath)

            if FileManager.default.fileExists(atPath: directConfigPath) {
                NSLog("YoozSTTEngine: Found model at Resources/")
                return resourceDir
            }
        } else {
            NSLog("YoozSTTEngine: Bundle.main.resourcePath is nil; skipping bundle lookup")
        }

        // 4. Hugging Face auto-download. Cached snapshots resolve in
        //    O(1); first-run downloads stream into the swift-huggingface
        //    HubCache root. The progress closure is invoked off-actor
        //    by HubApi; marshal to MainActor before touching the
        //    `@Published` property.
        guard language.huggingFaceID != nil else {
            NSLog(
                "YoozSTTEngine: No HF mirror wired for %@ (family %@)",
                language.rawValue, language.modelFamily.rawValue
            )
            throw YoozSTTError.modelNotFound(
                "STT model for \(language.displayName) is not on disk and no Hugging Face mirror is wired for the \(language.modelFamily.rawValue) family yet (issue #41)."
            )
        }

        let alreadyCached = STTModelHFDownloader.isCached(for: language)
        if !alreadyCached, !allowFetch {
            throw YoozSTTError.modelNotFound(
                "STT model for \(language.displayName) is not on disk and allow_fetch=false; bring the model directory into place first or call /v1/stt/load with allow_fetch=true."
            )
        }

        NSLog(
            "YoozSTTEngine: Resolving HF snapshot for %@ (cached=%@)",
            language.rawValue, alreadyCached ? "true" : "false"
        )

        do {
            return try await STTModelHFDownloader.snapshot(for: language) { [weak self] fraction in
                guard let self else { return }
                Task { @MainActor in
                    self.downloadProgress = fraction
                }
            }
        } catch let error as STTHFDownloadError {
            throw YoozSTTError.modelNotFound(
                error.errorDescription ?? "HF download failed"
            )
        } catch {
            throw YoozSTTError.modelLoadFailed(
                "HF download failed: \(error.localizedDescription)"
            )
        }
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
