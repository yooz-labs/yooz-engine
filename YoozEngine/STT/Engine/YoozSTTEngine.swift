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

    /// Fraction-completed [0.0, 1.0] for an in-progress HF model
    /// download. Resets to 0 on `start(language:)` and on failure;
    /// reaches 1.0 on successful load. UI clients poll via the
    /// `progress` field of `/v1/stt/status`.
    ///
    /// For backends whose download is owned by an actor outside
    /// `YoozSTTEngine` (qwen3ASRPreview routes its fetch through
    /// `Qwen3ASRModelFetcher` from the `/v1/stt/load` handler), the
    /// API server calls `setDownloadProgress(_:)` as it drains the
    /// fetcher's progress stream so `/v1/stt/status` can surface a
    /// non-nil fraction for the whisper banner (engine#144).
    @Published public private(set) var downloadProgress: Double = 0

    /// Active STT backend. Default is `.parakeet`; switchable via
    /// `setBackend(_:)` (HTTP `POST /v1/stt/engine`) or the
    /// `YOOZ_STT_BACKEND` env var honored by `EngineConfig`.
    @Published public private(set) var currentBackend: STTBackendID = .parakeet

    /// Lifecycle state for the in-flight load (engine#125). Drives
    /// the `state` field on `/v1/stt/status`. Transitions:
    ///   .idle → .loading (enqueueLoad starts)
    ///   .loading → .ready (load succeeded)
    ///   .loading → .failed (load threw)
    ///   .loading → .idle (cancellation)
    ///   .failed | .ready → .loading (next enqueueLoad)
    ///   any → .idle (stop())
    @Published public private(set) var loadState: LoadState = .idle

    /// Human-readable error from the last failed load. Cleared on
    /// the next `enqueueLoad` (which transitions to .loading) and on
    /// `stop()`. Surfaced via `/v1/stt/status.lastError` when
    /// `state == .failed`.
    @Published public private(set) var lastLoadError: String?

    // MARK: - Callbacks

    /// Callback for transcription updates
    public var onTranscription: ((ParakeetResult) -> Void)?

    /// Callback for errors
    public var onError: ((String) -> Void)?

    // MARK: - Private Properties

    private var model: ParakeetModel?
    private var streamingContext: StreamingTranscriber?
    private let sampleRate: Int = 16000

    /// In-flight load handle (engine#125). When non-nil, indicates a
    /// load Task is running; a subsequent `enqueueLoad(language:)`
    /// for the same language returns this same handle so concurrent
    /// callers share one underlying load (idempotent dedup). Mutated
    /// only on MainActor.
    private var inFlightLoadTask: Task<Void, Error>?
    private var inFlightLoadLanguage: STTLanguage?

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
            // Check if language is implemented before we touch the model
            // cache — no point downloading 700 MB to find out we can't use
            // it yet.
            guard language.isImplemented else {
                throw YoozSTTError.languageNotSupported(language)
            }

            // Resolve (or download on first run) the HF snapshot for
            // this language. Cached snapshots stay in
            // ~/.cache/huggingface/hub per HubClient defaults; missing
            // snapshots download on the first call when allowFetch is
            // true. Any failure surfaces as YoozSTTError.modelLoadFailed
            // below.
            let modelDir = try await getModelDirectory(
                for: language,
                allowFetch: allowFetch
            )
            NSLog("YoozSTTEngine: Got model directory: %@", modelDir.path)


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
        } catch is CancellationError {
            // Task cancellation is not a failure — the user / caller
            // requested it. Surface it cleanly so awaiting tasks see
            // the cooperative cancellation contract; do not record
            // `lastError` so the menu-bar UI doesn't render a red
            // banner for a deliberate stop.
            await MainActor.run { self.downloadProgress = 0 }
            throw CancellationError()
        } catch {
            // Reset progress so a polling client doesn't see a
            // stalled mid-download fraction after a failed load.
            let message = error.localizedDescription
            NSLog("YoozSTTEngine: ERROR - %@", message)
            await MainActor.run {
                self.lastError = message
                self.downloadProgress = 0
            }
            onError?(message)
            throw error
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

    /// External setter used by backends whose download is owned by an
    /// actor outside this class (today: qwen3ASRPreview, fetched by
    /// `Qwen3ASRModelFetcher` and driven from the `/v1/stt/load`
    /// route). Clamps to `[0, 1]` so a stale or malformed event can't
    /// push the published value out of range. The route handler calls
    /// this as it drains the fetcher's progress stream; the value
    /// then surfaces via `/v1/stt/status.progress` for the consumer
    /// banner (engine#144).
    public func setDownloadProgress(_ fraction: Double) async {
        let clamped = min(max(fraction, 0), 1)
        await MainActor.run { self.downloadProgress = clamped }
    }

    // MARK: - Async load (engine#125)

    /// Enqueue a load on a background Task, return a handle the
    /// caller can either await (`?wait=true`) or drop (HTTP 202
    /// fire-and-forget). Idempotent: a second call with the same
    /// language while a load is in flight returns the existing
    /// handle so concurrent callers share one underlying load. A
    /// call with a different language while loading cancels the
    /// previous Task and starts a fresh one (last-write-wins so the
    /// picker's most recent selection always drives behavior).
    ///
    /// `body` is the work the Task performs — typically a closure
    /// that wraps `start(language:allowFetch:)` plus any backend-
    /// specific prep (e.g. `Qwen3ASRModelFetcher.download`). The
    /// engine takes ownership of the Task and manages state
    /// transitions on the @Published `loadState` / `lastLoadError`
    /// properties so `/v1/stt/status` consumers see them
    /// immediately.
    ///
    /// Cancellation: caller-side `Task.cancel()` propagates through
    /// `body` cooperatively; the cancelled state transitions to
    /// `.idle` (not `.failed`) so a UI driving on this state can
    /// distinguish user-initiated cancel from a real failure.
    @MainActor
    @discardableResult
    public func enqueueLoad(
        language: STTLanguage,
        body: @Sendable @escaping () async throws -> Void
    ) -> Task<Void, Error> {
        // Idempotent dedup: same language, same in-flight Task.
        if loadState == .loading,
           inFlightLoadLanguage == language,
           let existing = inFlightLoadTask
        {
            return existing
        }

        // Different language while loading → cancel the prior load
        // before starting the new one. The cancelled Task throws
        // CancellationError to any awaiters; the catch path below
        // settles state back to .idle.
        if loadState == .loading {
            inFlightLoadTask?.cancel()
        }

        loadState = .loading
        inFlightLoadLanguage = language
        lastLoadError = nil

        let task = Task<Void, Error> { [weak self] in
            do {
                try await body()
                await MainActor.run {
                    guard let self = self else { return }
                    self.loadState = .ready
                    self.inFlightLoadTask = nil
                    self.inFlightLoadLanguage = nil
                }
            } catch is CancellationError {
                await MainActor.run {
                    guard let self = self else { return }
                    self.loadState = .idle
                    // Reset downloadProgress so a stalled mid-
                    // download fraction does not linger in
                    // /v1/stt/status after the cancel (the
                    // route's `running ? nil : ...` filter only
                    // covers the loaded case; this closes the
                    // cancelled-load window).
                    self.downloadProgress = 0
                    self.inFlightLoadTask = nil
                    self.inFlightLoadLanguage = nil
                }
                throw CancellationError()
            } catch {
                let message = error.localizedDescription
                await MainActor.run {
                    guard let self = self else { return }
                    self.loadState = .failed
                    self.lastLoadError = message
                    // Same reset as the cancellation arm — a
                    // partial download fraction must not lie about
                    // an in-flight load after the failure has been
                    // observed on /v1/stt/status.lastError.
                    self.downloadProgress = 0
                    self.inFlightLoadTask = nil
                    self.inFlightLoadLanguage = nil
                }
                throw error
            }
        }

        inFlightLoadTask = task
        return task
    }

    /// Unload the model and free resources
    public func stop() {
        lock.lock()
        defer { lock.unlock() }

        if isStreaming {
            streamingContext = nil
        }

        model = nil

        // Snapshot the in-flight load handle under the lock so the
        // cancel happens off the MainActor hop below. The Task's
        // CancellationError catch transitions loadState back to
        // .idle via its own MainActor.run, which is independent of
        // the state reset we do here.
        //
        // Reset downloadProgress alongside loadState so a stopped
        // (but not yet restarted) engine doesn't keep reporting
        // `progress: 1.0` to /v1/stt/status — the route's
        // `loaded ? nil : ...` filter (engine#145) only covers the
        // loaded case; this closes the stop()-then-poll window
        // that would otherwise surface "Downloading... 100%".
        let inFlight = inFlightLoadTask

        // Update state synchronously for immediate effect
        if Thread.isMainThread {
            self.isReady = false
            self.isStreaming = false
            self.currentResult = .empty
            self.downloadProgress = 0
            self.loadState = .idle
            self.lastLoadError = nil
            self.inFlightLoadTask = nil
            self.inFlightLoadLanguage = nil
        } else {
            DispatchQueue.main.sync {
                self.isReady = false
                self.isStreaming = false
                self.currentResult = .empty
                self.downloadProgress = 0
                self.loadState = .idle
                self.lastLoadError = nil
                self.inFlightLoadTask = nil
                self.inFlightLoadLanguage = nil
            }
        }

        // Cancel after clearing the state pointer so the Task's
        // cancellation catch path can't race back through the
        // engine and re-set inFlightLoadTask to nil (already done).
        inFlight?.cancel()

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
            partialEmissionInterval: EngineConfig.streamingPartialIntervalSec,
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
            streamingContext = StreamingTranscriber(
                model: model!,
                partialEmissionInterval: EngineConfig.streamingPartialIntervalSec
            )
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
            partialEmissionInterval: EngineConfig.streamingPartialIntervalSec,
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

    /// Resolve the on-disk directory for the active STT model.
    ///
    /// Step 1-3: probe the EngineConfig models directory + the app
    /// bundle's Resources/Models/ (developer setups that pre-populated
    /// weights). Step 4 falls through to the HuggingFace HubClient
    /// snapshot path. `allowFetch == false` forces offline-only — a
    /// cold cache surfaces as `HubCacheError.cachedPathResolutionFailed`,
    /// which `APIServer.mapSTTLoadError` demuxes into the
    /// `model_not_cached` wire code.
    private func getModelDirectory(
        for language: STTLanguage,
        allowFetch: Bool
    ) async throws -> URL {
        NSLog(
            "YoozSTTEngine: getModelDirectory(for: %@, allowFetch: %@)",
            language.rawValue, allowFetch ? "true" : "false"
        )

        // Step 1-3: probe legacy on-disk locations. `config.json` is
        // the readiness sentinel; a partial drop falls through to the
        // HF cache lookup below.
        let engineModelsDir = EngineConfig.modelsDirectory
        if FileManager.default.fileExists(
            atPath: engineModelsDir.appendingPathComponent("config.json").path
        ) {
            return engineModelsDir
        }

        if let resourcePath = Bundle.main.resourcePath {
            let resourceDir = URL(fileURLWithPath: resourcePath)
            for candidate in [
                resourceDir.appendingPathComponent("Models"),
                resourceDir
            ] {
                if FileManager.default.fileExists(
                    atPath: candidate.appendingPathComponent("config.json").path
                ) {
                    return candidate
                }
            }
        }

        // Step 4: HF cache. `localFilesOnly: !allowFetch` lets the
        // downstream `HubClient` itself enforce offline mode by
        // throwing `HubCacheError.cachedPathResolutionFailed` on a
        // cache miss. That error propagates verbatim so
        // `APIServer.mapSTTLoadError` can demux it into the
        // `model_not_cached` wire code instead of getting flattened
        // into a generic 500.
        try Task.checkCancellation()
        return try await STTModelHFDownloader.snapshot(
            for: language,
            localFilesOnly: !allowFetch,
            progress: { fraction in
                Task { @MainActor in
                    self.downloadProgress = fraction
                }
            }
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
