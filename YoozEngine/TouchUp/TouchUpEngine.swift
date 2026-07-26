// TouchUpEngine.swift
// LLMModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore
import Foundation
import os.log

private let logger = Logger(subsystem: "live.yooz.engine", category: "TouchUpEngine")

/// Main entry point for AI touch-up processing.
///
/// The TouchUpEngine manages LLM models and provides smart routing
/// for transcription cleanup. It uses up to three backends:
/// - **Yooz-Light** (KD Qwen3.5-0.8B, `YoozLabs/Yooz-Light-v3-Qwen3.5-0.8B`,
///   ~605 MB 6-bit): Fast proofreading, ~300ms latency
/// - **Yooz-Quality** (KD Qwen3.5-4B, `YoozLabs/Yooz-Quality-v3-Qwen3.5-4B`,
///   ~3.2 GB 6-bit): Higher quality rewriting, ~1s latency
/// - **Apple Intelligence** (Foundation Models 3B): macOS 26+, structured generation
public actor TouchUpEngine {

    // MARK: - Singleton

    /// Production singleton. Explicitly wired to `ModelSelectionStore.shared`
    /// (the real, disk-backed instance under `EngineConfig.stateDirectory`)
    /// so the persisted-selection contract (engine#226) is only real for
    /// this instance — see `init(selectionStore:)`.
    public static let shared = TouchUpEngine(selectionStore: .shared)

    // MARK: - Properties

    /// Module key `ModelSelectionStore` / `EngineEventBus` use for
    /// TouchUp's persisted selection + events. Stable — renaming changes
    /// the persisted file's existing key (stranding a user's prior
    /// selection) and breaks any subscriber filtering `EngineEvent.module`.
    static let selectionStoreModule = "touchup"

    /// Engine-owned persisted-selection store (engine#226). Defaults to a
    /// throwaway, per-instance temp-file-backed store — NOT
    /// `ModelSelectionStore.shared` — so a plain `TouchUpEngine()` (the
    /// documented test-fresh-instance pattern above) never touches a
    /// developer's real Application Support directory and never leaks
    /// state between test instances. `.shared` overrides this to the real
    /// production store.
    private let selectionStore: ModelSelectionStore

    /// Memoized one-shot restore (see `restorePersistedSelectionIfNeeded()`).
    /// A `Task` handle, not a `Bool` flag: with a flag set before the store
    /// read's suspension point, a second concurrent first-touch would pass
    /// the guard mid-restore and observe the compiled-in default — exactly
    /// the window the restore exists to close (PR #239 review). Every
    /// caller instead awaits this single shared task, so no reader can
    /// return before the restore has fully settled.
    private var restoreTask: Task<Void, Never>?

    /// In-flight background-preload dispatches from `setActiveModelAsync`
    /// AND `requestDownload` (engine#288, generalized in engine#306), keyed
    /// by model id rather than `TouchUpModelSelection`: `requestDownload`
    /// takes an `LLMModelType`, which has no selection counterpart for a
    /// catalogue-only model (e.g. `yooz-instruct-4b`), so the id string is
    /// the only key both dispatch paths share. `TouchUpModelSelection.rawValue`
    /// and `LLMModelType.rawValue` agree for every id both types can name,
    /// so a picker switch to a model and an explicit download of the same
    /// model still collapse onto one dispatch (the cross-path dedupe PR #290
    /// relied on). A switch never cancels another dispatch's download — it
    /// runs to completion in the background — and same-id re-requests
    /// dedupe onto the existing handle. Entries remove themselves on
    /// completion via `clearBackgroundPreload`.
    private var backgroundPreloadTasks: [String: Task<Void, Never>] = [:]

    /// Per-tier MLX backend, created on demand (engine#303 — catalogue-backed
    /// `LLMModelType` replaces the fixed `lightModel`/`qualityModel` pair
    /// this dictionary generalizes). Never pruned on `unload(_:)`: an entry
    /// staying present with `isLoaded == false` is the existing "wrapper
    /// stays, weights free" contract (see `evictModelsExcept`'s doc).
    private var backends: [LLMModelType: MLXLLMBackend] = [:]

    /// The Apple Intelligence backend (Foundation Models, macOS 26+)
    private var foundationModelsBackend: FoundationModelsBackend?

    /// Bundle identifier for model loading
    private let bundleIdentifier: String

    /// Per-tier lifecycle state (engine#125). Drives the `state`
    /// field on `/v1/llm/status` for the picker's active model.
    /// Transitions per tier:
    ///   .idle → .loading (enqueueLoad starts)
    ///   .loading → .ready (preloadModel succeeded)
    ///   .loading → .failed (preloadModel threw)
    ///   .loading → .idle (cancellation)
    ///   .failed | .ready → .loading (next enqueueLoad)
    private var loadStates: [LLMModelType: LoadState] = [:]

    /// Per-tier last-load error. Cleared on the next `enqueueLoad`
    /// (which transitions to .loading). Surfaced via
    /// `/v1/llm/status.lastError` when the active tier's state is
    /// `.failed`.
    private var lastLoadErrors: [LLMModelType: String] = [:]

    /// In-flight load handles (engine#125). When non-nil for a tier,
    /// a subsequent `enqueueLoad(_:)` for that tier returns this
    /// same handle so concurrent callers share one underlying load
    /// (idempotent dedup).
    private var inFlightLoadTasks: [LLMModelType: Task<Void, Error>] = [:]

    /// Backing storage for `activeModel`. Defaults to `.yoozLight` so
    /// callers that never `setActiveModel` see the pre-picker behaviour.
    private var _activeModel: TouchUpModelSelection = .yoozLight

    /// Currently active TouchUp model. Drives `/v1/touchup` routing
    /// via `processWithActiveModel(...)` and is reflected back to
    /// clients through the picker API (`GET /v1/touchup/models`).
    ///
    /// A computed `async` property (engine#226), not a plain stored one, so
    /// EVERY read — including one that bypasses `availableModels()` /
    /// `setActiveModel(Async)` / `processWithActiveModel` — restores the
    /// persisted selection first. Several existing call sites read
    /// `.activeModel` directly without calling any of those first (e.g.
    /// `ModelManagementEndpoints`'s delete-guard, `/v1/llm/status`); those
    /// would otherwise silently observe the compiled-in `.yoozLight`
    /// default instead of a real persisted selection until some OTHER
    /// route happened to run first. External call-site syntax is
    /// unchanged (`await engine.activeModel` already required `await` —
    /// actor property access always does), so this closes the gap with no
    /// external API break.
    public var activeModel: TouchUpModelSelection {
        get async {
            await restorePersistedSelectionIfNeeded()
            return _activeModel
        }
    }

    /// Whether the engine has been preloaded
    public private(set) var isPreloaded: Bool = false

    /// User-preferred LLM for touch-up. Routing inside `process()` remains
    /// mode-based (light-fast path, quality when replacements warrant it);
    /// this property exists so thin clients can round-trip a dropdown
    /// selection through the server. Held for the engine process lifetime
    /// only — clients that need cross-session persistence must cache
    /// their own selection and re-apply via `POST /v1/llm/model` on
    /// reconnect. Wire contract: `LLMModelType.rawValue` surfaced by
    /// `GET /v1/llm/models.current` and set by `POST /v1/llm/model`.
    public private(set) var preferredModel: LLMModelType = .yoozLight

    /// Whether the light model is loaded. Reads the backend dictionary
    /// directly (not `backend(for:)`, which create-on-demand) — an
    /// uninstantiated tier reports `false` rather than being materialized
    /// just to answer this check.
    public var isLightModelLoaded: Bool {
        get async {
            guard let model = backends[.yoozLight] else { return false }
            return await model.isLoaded
        }
    }

    /// Whether the quality model is loaded. Same not-create-on-demand
    /// contract as `isLightModelLoaded`.
    public var isQualityModelLoaded: Bool {
        get async {
            guard let model = backends[.yoozQuality] else { return false }
            return await model.isLoaded
        }
    }

    /// Whether Apple Intelligence is available and loaded
    public var isFoundationModelsLoaded: Bool {
        get async {
            guard let backend = foundationModelsBackend else { return false }
            return await backend.isLoaded
        }
    }

    /// Active HuggingFace download fraction for the given LLM tier, in
    /// [0.0, 1.0]. Returns 0 before a load starts, ticks up while
    /// `MLXLLMBackend.load()` streams the snapshot, and stays at 1.0
    /// after a successful load until `unload` resets it. Returns nil
    /// when the backend instance for that tier hasn't been instantiated
    /// yet (e.g. quality has never been requested). Used by
    /// `/v1/llm/status` and the consumer-side progress banner.
    public func downloadProgress(for modelType: LLMModelType) async -> Double? {
        guard let model = backends[modelType] else { return nil }
        return await model.downloadProgress
    }

    // MARK: - Backend access (engine#303)

    /// Get-or-create the backend for `modelType`. Every catalogued model
    /// resolves here identically — there is no per-tier special case left,
    /// the point of the catalogue-backed refactor.
    private func backend(for modelType: LLMModelType) -> MLXLLMBackend {
        if let existing = backends[modelType] { return existing }
        let created = MLXLLMBackend.create(for: modelType, bundleIdentifier: bundleIdentifier)
        backends[modelType] = created
        return created
    }

    /// Ensure `modelType`'s weights are resident. Idempotent — delegates to
    /// the backend's own `isLoaded` guard, so a redundant call is a cheap
    /// no-op rather than a second load attempt.
    private func ensureLoaded(_ modelType: LLMModelType) async throws {
        let model = backend(for: modelType)
        guard await !model.isLoaded else { return }
        try await model.load()
        logger.info("\(modelType.rawValue, privacy: .public) model loaded on demand")
    }

    // MARK: - Initialization

    /// Internal init for `.shared` plus `@testable` access from
    /// `LLMModule` consumers. Production code must go through `.shared` so
    /// the singleton contract holds; tests that need a fresh instance
    /// construct one directly under `@testable import LLMModule`.
    ///
    /// `selectionStore`'s default constructs a fresh, isolated instance
    /// PER CALL (a new temp-file path each time) — every plain
    /// `TouchUpEngine()` test instance gets its own throwaway persistence,
    /// never the shared production file. See the `selectionStore` property
    /// doc for why this matters.
    init(
        bundleIdentifier: String = "live.yooz.engine",
        selectionStore: ModelSelectionStore = ModelSelectionStore(
            fileURL: FileManager.default.temporaryDirectory
                .appendingPathComponent("touchup-selection-\(UUID().uuidString).json")
        )
    ) {
        self.bundleIdentifier = bundleIdentifier
        self.selectionStore = selectionStore
    }

    // MARK: - Lifecycle

    /// Preload models for immediate use.
    ///
    /// This loads the light model (Yooz-Light) which is embedded in the app bundle.
    /// The quality model (Yooz-Quality) is loaded on-demand when needed.
    /// Apple Intelligence is loaded if available (macOS 26+).
    public func preload(loadQuality: Bool = false) async throws {
        logger.info("Preloading TouchUpEngine...")

        let light = backend(for: .yoozLight)
        try await light.load()
        logger.info("Yooz-Light model loaded")

        if loadQuality {
            let quality = backend(for: .yoozQuality)
            try await quality.load()
            logger.info("Yooz-Quality model loaded")
        }

        // Try to load Apple Intelligence if available
        await loadFoundationModelsIfAvailable()

        isPreloaded = true
        logger.info("TouchUpEngine preloaded successfully")
    }

    /// Try to load the Foundation Models backend if available on this system.
    private func loadFoundationModelsIfAvailable() async {
        let backend = FoundationModelsBackend()
        guard backend.isAvailable() else {
            logger.info("Apple Intelligence not available on this system")
            return
        }

        do {
            try await backend.load()
            foundationModelsBackend = backend
            logger.info("Apple Intelligence backend loaded")
        } catch {
            logger.warning("Failed to load Apple Intelligence: \(error.localizedDescription)")
        }
    }

    /// Per-recording-session reset (engine issue #114). Drops cached LLM
    /// state on every backend the engine owns so the next recording starts
    /// cold. Idempotent — fans out even to backends whose models aren't
    /// loaded yet (their `resetForNewSession()` is a cheap no-op then). Does
    /// NOT unload weights; this is a per-recording boundary, not a teardown.
    ///
    /// `foundationModelsBackend` already creates a fresh `LanguageModelSession`
    /// for each call (see comment on the property), so it has no per-session
    /// state to drop here.
    public func resetForNewSession() async {
        for (_, model) in backends {
            await model.resetForNewSession()
        }
    }

    /// Unload all models from memory.
    public func unload() async {
        // Cancel any in-flight loads (engine#125) so a download
        // racing with the unload doesn't end up reading from freed
        // backend state. The Task's CancellationError catch resets
        // the per-tier state to .idle.
        for (_, task) in inFlightLoadTasks {
            task.cancel()
        }
        inFlightLoadTasks.removeAll()
        loadStates.removeAll()
        lastLoadErrors.removeAll()

        for (_, model) in backends {
            await model.unload()
        }
        if let fm = foundationModelsBackend {
            await fm.unload()
        }
        isPreloaded = false
        logger.info("TouchUpEngine unloaded")
    }

    /// Unload a single model's weights from memory. Used by whisper's
    /// AI tab when the user switches to "touch-up off" or picks a
    /// different model — reclaims GPU memory without tearing down the
    /// whole engine. Idempotent: unloading an already-unloaded model
    /// is a no-op.
    public func unload(_ modelType: LLMModelType) async {
        // Cancel an in-flight load for this tier before tearing
        // down the backend; mirrors `unload()` above.
        if let task = inFlightLoadTasks.removeValue(forKey: modelType) {
            task.cancel()
        }
        loadStates[modelType] = .idle
        lastLoadErrors[modelType] = nil

        if let model = backends[modelType] {
            await model.unload()
        }
    }

    /// Drop the cached prompt-KV state for one LLM tier — or, when
    /// `modelType` is nil, every currently-loaded tier — WITHOUT unloading
    /// its weights (engine#299). This is the LLM-scoped twin of
    /// `resetForNewSession()`: that method already does exactly this per
    /// backend, but only reachable via `/v1/session/begin`, which fans the
    /// reset out to every `SessionResettable` module (STT included) as a
    /// per-recording boundary. A consumer that wants to reclaim the
    /// retained-KV memory delta (measured ~1.5 GB after sustained
    /// proofreading traffic) without evicting another module's state, and
    /// without paying a full `unload(_:)` + cold-reload round trip, needs a
    /// lever scoped to the LLM tiers alone — this is that lever.
    ///
    /// Skips any tier that is not currently resident: clearing an
    /// already-empty cache is a no-op, never an error, so an idle-policy
    /// caller can call this unconditionally without checking load state
    /// first.
    ///
    /// Caveat on the "not resident implies nothing to clear" assumption: an
    /// `unload()` that lands while a `generate()` is suspended can be followed
    /// by that generate's tail repopulating the prompt cache, leaving a
    /// backend that reports `isLoaded == false` while holding cache state.
    /// Such a tier is skipped here. That race predates this method (see the
    /// follow-up issue on `MLXLLMBackend.generate()`); it is called out rather
    /// than silently assumed away.
    ///
    /// - Returns: The tiers whose cache was actually dropped, in
    ///   `LLMModelType.allCases` order.
    @discardableResult
    public func clearCache(_ modelType: LLMModelType? = nil) async -> [LLMModelType] {
        let tiers = modelType.map { [$0] } ?? LLMModelType.allCases
        var cleared: [LLMModelType] = []
        for tier in tiers {
            guard let model = backends[tier] else { continue }
            // One hop, not check-then-act: see `clearSessionIfLoaded`.
            if await model.clearSessionIfLoaded() { cleared.append(tier) }
        }
        return cleared
    }

    /// Strict single-resident policy: unload every tier except `keep`.
    ///
    /// Called after a successful `setActiveModel` switch so the engine never
    /// holds more than one LLM/TouchUp model resident at a time. This is the
    /// in-process residency invariant: RAM tracks the *active* model, not the
    /// union of every tier the user has tried this session. Previously
    /// `setActiveModel` only ever loaded the new tier, so Light + Quality +
    /// Apple could all stay resident (~2 GB stranded). FoundationModels is
    /// OS-resident and cheap to drop; the MLX tiers free their weights via
    /// `unload(_:)`, which also returns their Metal buffers to the OS.
    ///
    /// The invariant is expressed at the *loaded-weights* level, not object
    /// existence: `unload(.yoozLight)` frees the backend's weights but leaves
    /// its `backends[.yoozLight]` entry in place (next use lazy-reloads it).
    /// So "evicted" means `isLightModelLoaded == false`, not the entry
    /// being removed from `backends`.
    private func evictModelsExcept(_ keep: TouchUpModelSelection) async {
        var evicted: [TouchUpModelSelection] = []
        // NEVER evict a tier whose load is still in flight (engine#288):
        // `unload(_:)` cancels the in-flight load task, which is what used
        // to destroy a mid-download tier the moment the user switched to a
        // faster one. A `.loading` tier holds no settled weights yet, so
        // skipping it costs no memory; its own dispatch's superseded-
        // completion path frees the memory copy if the user has moved on.
        if keep != .yoozLight, loadStates[.yoozLight] != .loading {
            await unload(.yoozLight)
            evicted.append(.yoozLight)
        }
        if keep != .yoozQuality, loadStates[.yoozQuality] != .loading {
            await unload(.yoozQuality)
            evicted.append(.yoozQuality)
        }
        if keep != .foundationModels, let fm = foundationModelsBackend {
            await fm.unload()
            foundationModelsBackend = nil
            evicted.append(.foundationModels)
        }
        let evictedList = evicted.map(\.rawValue).joined(separator: ", ")
        logger.info(
            "TouchUp single-resident: kept \(keep.rawValue, privacy: .public), evicted [\(evictedList, privacy: .public)]"
        )

        // Publish each evicted tier's post-eviction lifecycle state
        // (engine#226; PR #239 review). Without these, a subscriber that
        // learned tier A was `.loaded` before an A→B switch keeps rendering
        // it `.loaded` forever — `residencyChanged` alone names no
        // per-model state, so `EngineStateStore` can't correct the row.
        // Re-derive from `availableModels()` rather than assuming `.cached`:
        // an evicted MLX tier that was never downloaded reports
        // `.available`, and FoundationModels reports per its OS-availability
        // convention.
        let rows = await availableModels()
        for selection in evicted {
            guard let row = rows.first(where: { $0.id == selection.rawValue }) else { continue }
            await publishLoadStateChanged(selection.rawValue, state: row.loadState)
        }
    }

    /// Record the user-preferred LLM. Does not load weights — call
    /// `preloadModel(_:)` (or the /v1/llm/preload route) to warm
    /// the model after switching.
    public func setPreferredModel(_ modelType: LLMModelType) {
        preferredModel = modelType
        logger.info("TouchUpEngine preferredModel set to \(modelType.rawValue, privacy: .public)")
    }

    // MARK: - Async load (engine#125)

    /// Read the lifecycle state for a tier. Returns `.idle` for any
    /// tier that hasn't been touched yet.
    public func loadState(for modelType: LLMModelType) -> LoadState {
        return loadStates[modelType] ?? .idle
    }

    /// Read the last error captured for a tier. `nil` unless that
    /// tier's `loadState == .failed`.
    public func lastLoadError(for modelType: LLMModelType) -> String? {
        return lastLoadErrors[modelType]
    }

    /// Enqueue a load on a background Task; return a handle the
    /// caller can either await (`?wait=true`) or drop (HTTP 202
    /// fire-and-forget). Idempotent: a second call for the same
    /// tier while a load is in flight returns the existing handle so
    /// concurrent callers share one underlying load. Loads for
    /// different tiers run concurrently (a quality preload doesn't
    /// block a light preload).
    ///
    /// Cancellation: caller-side `Task.cancel()` propagates through
    /// `preloadModel` cooperatively; cancelled state transitions
    /// back to `.idle` (not `.failed`) so UI can distinguish a
    /// user-initiated cancel from a real failure.
    @discardableResult
    public func enqueueLoad(
        _ modelType: LLMModelType
    ) -> Task<Void, Error> {
        // Idempotent dedup: same tier, return existing handle.
        if loadStates[modelType] == .loading,
           let existing = inFlightLoadTasks[modelType]
        {
            return existing
        }

        loadStates[modelType] = .loading
        lastLoadErrors[modelType] = nil

        // Capture the task reference up-front so the settle hop can
        // check identity (not just state) before mutating. Without
        // identity comparison, an `unload` + immediate `enqueueLoad`
        // race lets the old task's settle clobber the new task's
        // state — see the review finding on engine#125.
        var taskHandle: Task<Void, Error>?
        // `self` is the actor singleton — guaranteed alive for the
        // process lifetime. The weak capture pattern is dropped per
        // the silent-failure review (would silently strand state at
        // .loading if self ever went away mid-load).
        let task = Task<Void, Error> {
            do {
                try await self.preloadModel(modelType)
                await self.markLoadSettled(
                    modelType, state: .ready, error: nil, owner: taskHandle
                )
            } catch is CancellationError {
                await self.markLoadSettled(
                    modelType, state: .idle, error: nil, owner: taskHandle
                )
                throw CancellationError()
            } catch {
                let message = error.localizedDescription
                await self.markLoadSettled(
                    modelType, state: .failed, error: message, owner: taskHandle
                )
                throw error
            }
        }
        taskHandle = task

        inFlightLoadTasks[modelType] = task
        return task
    }

    /// Settle the post-load state from the Task's completion handler.
    /// No-op when the in-flight task for `modelType` is no longer
    /// `owner` — covers the race where `unload(modelType)` clears
    /// state (and a new `enqueueLoad` may have started another Task)
    /// while the old Task's settle hop was still queued behind the
    /// actor. Without the identity check, an old task's settlement
    /// could overwrite a new task's `.loading` state, leaving the
    /// new load silently stranded.
    private func markLoadSettled(
        _ modelType: LLMModelType,
        state: LoadState,
        error: String?,
        owner: Task<Void, Error>?
    ) {
        guard let owner, inFlightLoadTasks[modelType] == owner else {
            return
        }
        loadStates[modelType] = state
        lastLoadErrors[modelType] = error
        inFlightLoadTasks[modelType] = nil
    }

    /// Ensure a specific catalogued model's weights are resident.
    /// Idempotent; triggers an HF download on first use for whichever
    /// model is named. Invoked by `POST /v1/llm/preload`.
    public func preloadModel(_ modelType: LLMModelType) async throws {
        try await ensureLoaded(modelType)
    }

    // MARK: - Raw LLM Generation

    /// Generate text using a specified model.
    /// Used by the `/v1/llm/generate` endpoint.
    ///
    /// - Parameters:
    ///   - prompt: The user prompt
    ///   - systemPrompt: System prompt for the model
    ///   - modelType: Which model to use (defaults to light)
    ///   - workloadClass: GPU admission class (engine#228). Defaults to
    ///     `.background` — matches the issue's classification of raw
    ///     `/v1/llm/generate` calls as throughput work. Callers with a
    ///     genuinely latency-sensitive one-off generation can override.
    /// - Returns: Generated text
    public func generate(
        prompt: String,
        systemPrompt: String,
        modelType: LLMModelType = .yoozLight,
        workloadClass: MLXWorkloadClass = .background
    ) async throws -> String {
        try await ensureLoaded(modelType)
        let model = backend(for: modelType)
        // No proofreading salvage on the raw path: this serves
        // `/v1/llm/generate`, whose callers want the model's actual output
        // (engine#312). `process()` below keeps it, where it belongs.
        return try await model.generate(
            prompt: prompt, systemPrompt: systemPrompt, workloadClass: workloadClass,
            postProcess: false
        )
    }

    /// Generate text using Apple Intelligence (Foundation Models).
    /// Used when the caller specifically wants the Apple backend.
    ///
    /// - Parameters:
    ///   - prompt: The user prompt
    ///   - systemPrompt: Optional system prompt
    /// - Returns: Generated text
    public func generateWithFoundationModels(
        prompt: String,
        systemPrompt: String? = nil
    ) async throws -> String {
        guard let backend = foundationModelsBackend, await backend.isLoaded else {
            throw LLMError.notAvailable("Apple Intelligence not available or not loaded")
        }
        return try await backend.generate(prompt: prompt, systemPrompt: systemPrompt)
    }

    // MARK: - TouchUp Processing

    /// Process text with mode-aware prompt selection and smart two-model routing.
    ///
    /// - Parameters:
    ///   - text: Transcribed text (with replacements already applied)
    ///   - mode: Processing mode controlling prompt and model selection
    ///   - replacements: List of (original, replacement) tuples to validate
    ///   - workloadClass: GPU admission class (engine#228). Defaults to
    ///     `.background` — TouchUp generation is throughput work per the
    ///     issue's classification.
    ///   - contextVocabulary: Optional dictation vocabulary hint
    ///     (engine#280 Phase 4). Sanitized, deduped, and capped
    ///     (`cappedContextVocabulary`) and, when
    ///     `EngineConfig.touchUpContextEnabled` is on, appended to the
    ///     selected prompt via `withContext`. Nil is a no-op.
    ///   - contextAppName: Optional frontmost-app display name for the
    ///     same eval-gated injection, sanitized and length-capped
    ///     (`sanitizedContextAppName`). Nil is a no-op.
    /// - Returns: ProcessResult with cleaned text and metadata
    public func process(
        text: String,
        mode: TouchUpMode,
        replacements: [(original: String, replacement: String)] = [],
        workloadClass: MLXWorkloadClass = .background,
        contextVocabulary: [String]? = nil,
        contextAppName: String? = nil
    ) async -> TouchUpProcessor.ProcessResult {
        let replacementStructs = replacements.map {
            TouchUpProcessor.Replacement(original: $0.original, replacement: $0.replacement)
        }
        let cappedVocabulary = Self.cappedContextVocabulary(contextVocabulary)

        // Mode off: no LLM cleanup is requested — return regex-only (voice
        // commands) without loading any model. Must precede the lazy-load below
        // so "off" never triggers a multi-hundred-MB download.
        if mode == .off {
            return TouchUpProcessor.processRegexOnly(text: text, replacements: replacementStructs)
        }

        // Lazy-load the light model on first use, mirroring `generate()`. The
        // in-process path (epic #192) has no eager-loader — `bootstrap()` only
        // registers actors — so without this the model would never load and
        // every cleanup would silently downgrade to regex-only passthrough.
        // Only fall back to regex-only if creation/load genuinely fails.
        let light = backend(for: .yoozLight)
        if await !light.isLoaded {
            do { try await light.load() }
            catch { logger.error("Light model load failed: \(error.localizedDescription)") }
        }
        guard await light.isLoaded else {
            logger.warning("Light model unavailable, using regex-only processing")
            return TouchUpProcessor.processRegexOnly(text: text, replacements: replacementStructs)
        }

        // If we have replacements, ensure quality model is loaded
        var qualityLoadError: String?
        if !replacements.isEmpty {
            do {
                try await ensureLoaded(.yoozQuality)
            } catch {
                qualityLoadError = error.localizedDescription
                logger.error("Failed to load quality model: \(error.localizedDescription)")
            }
        }

        // Check if quality model is available and loaded
        let qualityAvailable: Bool
        if let quality = backends[.yoozQuality] {
            qualityAvailable = await quality.isLoaded
        } else {
            qualityAvailable = false
        }


        // Select the proofread prompt based on mode and available model
        let proofreadPrompt = Self.withContext(
            prompt: selectPrompt(for: mode, qualityAvailable: qualityAvailable),
            vocabulary: cappedVocabulary,
            appName: contextAppName
        )

        // Route to appropriate processing
        if replacements.isEmpty || !qualityAvailable {
            var result = await TouchUpProcessor.process(
                text: text,
                replacements: [],
                lightModel: light,
                qualityModel: light,
                proofreadPrompt: proofreadPrompt,
                workloadClass: workloadClass
            )
            // If quality was needed but failed to load, report degraded service
            if !replacements.isEmpty, let loadError = qualityLoadError {
                result = TouchUpProcessor.ProcessResult(
                    text: result.text,
                    keepDecisions: result.keepDecisions,
                    modelUsed: result.modelUsed,
                    latencyMs: result.latencyMs,
                    fallbackReason: "Quality model unavailable: \(loadError)"
                )
            }
            return result
        } else if let quality = backends[.yoozQuality] {
            return await TouchUpProcessor.process(
                text: text,
                replacements: replacementStructs,
                lightModel: light,
                qualityModel: quality,
                proofreadPrompt: proofreadPrompt,
                workloadClass: workloadClass
            )
        } else {
            // Should not reach here since qualityAvailable was true,
            // but fall back to light model to be safe
            return await TouchUpProcessor.process(
                text: text,
                replacements: [],
                lightModel: light,
                qualityModel: light,
                proofreadPrompt: proofreadPrompt,
                workloadClass: workloadClass
            )
        }
    }

    /// Process text using Apple Intelligence backend directly.
    /// Falls back to MLX models if Foundation Models unavailable.
    ///
    /// - Parameters:
    ///   - contextVocabulary: Optional dictation vocabulary hint
    ///     (engine#280 Phase 4). Only reaches the model when this method
    ///     falls back to `process(...)` (MLX Light) — the successful Apple
    ///     Intelligence branch below composes `systemPrompt` directly from
    ///     `YoozPrompts.appleStandard`/`appleFull`, not `selectPrompt`, so it
    ///     stays context-free by design (disclosed scope: MLX Light/Quality
    ///     get context, Apple Intelligence does not).
    ///   - contextAppName: Same scope note as `contextVocabulary`.
    public func processWithFoundationModels(
        text: String,
        mode: TouchUpMode,
        contextVocabulary: [String]? = nil,
        contextAppName: String? = nil
    ) async -> TouchUpProcessor.ProcessResult {
        let startTime = CFAbsoluteTimeGetCurrent()

        // Lazy-load Apple Intelligence on first use (the in-process path has no
        // eager-loader). `load()` just opens a LanguageModelSession — the system
        // model is resident, no download — so this is cheap and idempotent.
        if foundationModelsBackend == nil {
            await loadFoundationModelsIfAvailable()
        }
        guard let backend = foundationModelsBackend, await backend.isLoaded else {
            logger.warning("Foundation Models not available, falling back to MLX")
            var result = await process(
                text: text, mode: mode,
                contextVocabulary: contextVocabulary, contextAppName: contextAppName
            )
            result = TouchUpProcessor.ProcessResult(
                text: result.text,
                keepDecisions: result.keepDecisions,
                modelUsed: result.modelUsed,
                latencyMs: result.latencyMs,
                fallbackReason: "Foundation Models not available, used MLX"
            )
            return result
        }

        let systemPrompt: String
        switch mode {
        case .off:
            return processRegexOnly(text: text)
        case .light, .standard:
            systemPrompt = YoozPrompts.appleStandard
        case .full:
            systemPrompt = YoozPrompts.appleFull
        }

        do {
            let result = try await backend.generate(prompt: text, systemPrompt: systemPrompt)
            let latencyMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
            return TouchUpProcessor.ProcessResult(
                text: result,
                keepDecisions: [],
                modelUsed: .foundationModels,
                latencyMs: latencyMs,
                fallbackReason: nil
            )
        } catch {
            logger.error("Foundation Models generation failed: \(error.localizedDescription)")
            // Apply voice commands as minimal processing before returning
            let processed = TouchUpProcessor.applyCommands(text)
            let latencyMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
            return TouchUpProcessor.ProcessResult(
                text: processed,
                keepDecisions: [],
                modelUsed: .fallbackRegex,
                latencyMs: latencyMs,
                fallbackReason: "Foundation Models failed: \(error.localizedDescription)"
            )
        }
    }

    /// Process text with regex only (no LLM).
    public nonisolated func processRegexOnly(
        text: String,
        replacements: [(original: String, replacement: String)] = []
    ) -> TouchUpProcessor.ProcessResult {
        let replacementStructs = replacements.map {
            TouchUpProcessor.Replacement(original: $0.original, replacement: $0.replacement)
        }
        return TouchUpProcessor.processRegexOnly(text: text, replacements: replacementStructs)
    }

    // MARK: - Prompt Selection

    /// Select the appropriate proofread prompt based on mode and model availability.
    private func selectPrompt(
        for mode: TouchUpMode,
        qualityAvailable: Bool
    ) -> String {
        switch mode {
        case .off:
            // Should not reach here; off mode skips LLM entirely
            return TouchUpPrompts.proofread
        case .light:
            return YoozPrompts.lightStandard
        case .standard:
            if qualityAvailable {
                return YoozPrompts.qualityStandard
            } else {
                return YoozPrompts.lightStandard
            }
        case .full:
            if qualityAvailable {
                return YoozPrompts.qualityFull
            } else {
                return YoozPrompts.lightFull
            }
        }
    }

    // MARK: - Context injection (engine#280 Phase 4, eval-gated)

    /// Server-side defensive cap on the NUMBER of vocabulary terms attached
    /// to a TouchUp request (engine#280 / whisper#317): regardless of how
    /// many terms a caller sends, at most this many are honored downstream.
    /// Applied on receipt in `process(...)`/`processWithActiveModel(...)`
    /// so neither call path (loopback `APIServer` nor in-process
    /// `InProcessTransport`) can forget to enforce it. See also
    /// `touchUpContextTermCharacterCap`/`touchUpContextAppNameCharacterCap`
    /// for the SIZE bounds (review item 1) — this cap alone does not bound
    /// how long any individual term or the app name can be.
    public static let touchUpContextVocabularyCap = 30

    /// Per-term character cap (engine#280 review item 1). A term longer
    /// than this is DROPPED entirely rather than truncated: truncating
    /// mid-word could turn a real term into a different, misleading word
    /// (e.g. a 200-char product name truncated to "Ac" reads as noise, or
    /// worse, a real different term), whereas dropping it just means that
    /// one term never made it into the prompt.
    public static let touchUpContextTermCharacterCap = 100

    /// Character cap for the app name (engine#280 review item 1). Unlike
    /// vocabulary terms, there is only ever one app name, so an over-cap
    /// value is TRUNCATED rather than dropped — losing the feature
    /// entirely for an app with a long display name is a worse outcome
    /// than a truncated one.
    public static let touchUpContextAppNameCharacterCap = 60

    /// Shared control-character/whitespace sanitizer (engine#280 review
    /// item 2), used by both vocabulary terms and the app name: collapses
    /// embedded newlines, tabs, and any other control character — plus
    /// runs of plain whitespace — down to a single space, then trims the
    /// ends. Without this, a caller-supplied term or app name containing a
    /// raw newline could corrupt the composed prompt's line structure
    /// (e.g. injecting a fake extra "Text will be pasted into:" line).
    static func sanitizedContextText(_ text: String) -> String {
        let breakCharacters = CharacterSet.controlCharacters.union(.whitespacesAndNewlines)
        return text.components(separatedBy: breakCharacters)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    /// Sanitize one vocabulary term: the shared whitespace/control-character
    /// cleanup above, plus dropping commas (review item 2) — the composed
    /// prompt joins terms with `", "`, so an un-dropped comma inside a term
    /// like "Acme, Inc." would read ambiguously as two separate terms in
    /// that list. Commas become spaces (not deleted outright) so both
    /// "Acme,Inc" and "Acme, Inc." land as the same "Acme Inc." rather than
    /// gluing words together.
    static func sanitizedContextTerm(_ term: String) -> String {
        sanitizedContextText(term.replacingOccurrences(of: ",", with: " "))
    }

    /// Sanitize the app name: the shared whitespace/control-character
    /// cleanup only — commas are legitimate in an app name ("Acme, Inc." is
    /// unambiguous on its own, unlike inside a comma-joined list) — plus
    /// the character cap (review item 1). Returns `nil` for a `nil` or
    /// blank-after-sanitizing input.
    static func sanitizedContextAppName(_ appName: String?) -> String? {
        guard let appName else { return nil }
        let sanitized = sanitizedContextText(appName)
        guard !sanitized.isEmpty else { return nil }
        guard sanitized.count > touchUpContextAppNameCharacterCap else { return sanitized }
        return String(sanitized.prefix(touchUpContextAppNameCharacterCap))
    }

    /// Receipt-time pipeline for `contextVocabulary` (engine#280 review item
    /// 2): sanitize -> drop terms that are empty after sanitizing OR exceed
    /// `touchUpContextTermCharacterCap` (dropped whole, never truncated —
    /// see that constant's doc) -> dedupe case-insensitively (first
    /// occurrence wins) -> cap to `touchUpContextVocabularyCap`. The ORDER
    /// matters: capping before filtering would let blank, over-length, or
    /// duplicate terms consume cap slots that a real term further down the
    /// caller's list needed. A pure, synchronous transform — testable
    /// without a loaded model.
    static func cappedContextVocabulary(_ vocabulary: [String]?) -> [String]? {
        guard let vocabulary else { return nil }
        var seenLowercased = Set<String>()
        var result: [String] = []
        for term in vocabulary {
            let sanitized = sanitizedContextTerm(term)
            guard !sanitized.isEmpty, sanitized.count <= touchUpContextTermCharacterCap else { continue }
            guard seenLowercased.insert(sanitized.lowercased()).inserted else { continue }
            result.append(sanitized)
            if result.count == touchUpContextVocabularyCap { break }
        }
        return result
    }

    /// Append a compact context block to `prompt` when injection is enabled
    /// (`EngineConfig.touchUpContextEnabled`) and at least one of
    /// `vocabulary`/`appName` is non-empty after sanitizing. Applied to
    /// `selectPrompt`'s RETURN VALUE, never to the `YoozPrompts` constants
    /// themselves — `YoozPromptsParityTests` locks those verbatim against
    /// the fine-tuned weights' training data, so this composition step must
    /// stay strictly downstream of them. Disabling the flag, or omitting
    /// both fields, reproduces `prompt` byte-for-byte. Sanitizes each term
    /// and the app name itself (via the shared helpers above) so this
    /// function is safe even when called directly with un-sanitized input
    /// (as the pure-function tests do) — the count cap (30 terms) is
    /// `cappedContextVocabulary`'s job upstream, not re-applied here.
    static func withContext(
        prompt: String,
        vocabulary: [String]?,
        appName: String?
    ) -> String {
        guard EngineConfig.touchUpContextEnabled else {
            // One-line breadcrumb (engine#280 review item 5), counts only —
            // no term/app-name CONTENT in the log — so a future "why isn't
            // context showing up in the prompt" investigation doesn't need
            // a source dive to learn the flag is simply off.
            if vocabulary != nil || appName != nil {
                logger.debug(
                    "TouchUp context received but injection disabled (vocabularyCount=\(vocabulary?.count ?? 0), appNameAttached=\(appName != nil))"
                )
            }
            return prompt
        }
        let terms = (vocabulary ?? [])
            .map(Self.sanitizedContextTerm)
            .filter { !$0.isEmpty && $0.count <= touchUpContextTermCharacterCap }
        let sanitizedAppName = Self.sanitizedContextAppName(appName)
        var lines: [String] = []
        if !terms.isEmpty {
            lines.append("Known terms the speaker may use: \(terms.joined(separator: ", ")).")
        }
        if let sanitizedAppName {
            lines.append("Text will be pasted into: \(sanitizedAppName).")
        }
        guard !lines.isEmpty else { return prompt }
        return prompt + "\n\n" + lines.joined(separator: "\n")
    }

    // MARK: - Model Info

    /// Whether `modelType`'s snapshot is on disk in the HF cache. Probes a
    /// throwaway backend when the tier hasn't been instantiated yet, so
    /// checking cache state never persists a backend instance just to look.
    private func isCached(_ modelType: LLMModelType) async -> Bool {
        if let existing = backends[modelType] {
            return await existing.isModelCached
        }
        let temp = MLXLLMBackend.create(for: modelType, bundleIdentifier: bundleIdentifier)
        return await temp.isModelCached
    }

    /// Whether the Light model snapshot is on disk in the HF cache.
    /// Both tiers download from HF on first use (PR #93 / issue #77),
    /// so neither is "always cached".
    public var isLightModelCached: Bool {
        get async { await isCached(.yoozLight) }
    }

    /// Whether the Quality model snapshot is on disk in the HF cache.
    public var isQualityModelCached: Bool {
        get async { await isCached(.yoozQuality) }
    }

    // MARK: - Picker API

    /// Snapshot of every TouchUp model the engine knows about,
    /// with lifecycle state + active flag. Drives the picker UI in
    /// consumer apps via `GET /v1/touchup/models`.
    ///
    /// FoundationModels availability is OS-version gated by
    /// `FoundationModelsBackend.isAvailable()`; the user-opt-in
    /// state cannot be read without actually attempting a load, so
    /// the picker may show `.available` for a model that fails to
    /// load — the route handler maps that to 501 `model_unavailable`
    /// when `setModel(.foundationModels)` is called with `preload: true`.
    /// MLX tiers are always available — they download on first use.
    ///
    /// Postcondition: exactly one row has `isActive == true`. The
    /// `precondition(...)` below catches a future drift if a new
    /// case is added to `TouchUpModelSelection` without updating
    /// this method's row list.
    public func availableModels() async -> [TouchUpModelInfo] {
        await restorePersistedSelectionIfNeeded()
        let lightLoaded = await isLightModelLoaded
        let qualityLoaded = await isQualityModelLoaded
        let lightCached = await isLightModelCached
        let qualityCached = await isQualityModelCached
        let fmLoaded = await isFoundationModelsLoaded
        let fmAvailable = FoundationModelsBackend().isAvailable()

        let models: [TouchUpModelInfo] = [
            row(for: .yoozLight, loadState: loadState(
                isAvailable: true, isCached: lightCached, isLoaded: lightLoaded
            )),
            row(for: .yoozQuality, loadState: loadState(
                isAvailable: true, isCached: qualityCached, isLoaded: qualityLoaded
            )),
            row(for: .foundationModels, loadState: loadState(
                // FoundationModels has no on-disk artifact the engine
                // controls; treat "available" as equivalent to
                // "cached" so the picker UX never claims a download
                // step for an OS-provided backend.
                isAvailable: fmAvailable, isCached: fmAvailable, isLoaded: fmLoaded
            ))
        ]

        // Canonical-pattern invariant: exactly one active row. Drift
        // here cascades into every consumer app's picker UX.
        precondition(
            models.filter(\.isActive).count == 1,
            "TouchUp picker invariant: expected exactly one active model row"
        )
        return models
    }

    /// Build a single picker row from a selection + resolved load
    /// state. Centralises the mapping so a new selection case is a
    /// one-line addition (plus the `availableModels()` enumeration).
    ///
    /// Reads `_activeModel` (the private backing field), not the public
    /// `activeModel` accessor: every caller of `row(for:loadState:)` is
    /// `availableModels()`, which already ran
    /// `restorePersistedSelectionIfNeeded()` before calling this, so
    /// re-checking here would be redundant — and `row` stays a cheap
    /// synchronous helper instead of needing `async` just to read a value
    /// that's already current.
    private func row(
        for selection: TouchUpModelSelection,
        loadState: ModelLoadState
    ) -> TouchUpModelInfo {
        TouchUpModelInfo(
            id: selection.rawValue,
            displayName: selection.displayName,
            description: selection.description,
            tier: selection.tier,
            sizeBytes: selection.estimatedSize,
            loadState: loadState,
            isActive: _activeModel == selection
        )
    }

    /// Resolve the four-state lifecycle from the legacy three-flag
    /// pattern so backends keep their existing `isLoaded` / `isCached`
    /// surfaces. Total ordering is `unavailable < available < cached
    /// < loaded`; higher state implies lower (loaded ⇒ cached ⇒
    /// available).
    private func loadState(
        isAvailable: Bool,
        isCached: Bool,
        isLoaded: Bool
    ) -> ModelLoadState {
        if !isAvailable { return .unavailable }
        if isLoaded { return .loaded }
        if isCached { return .cached }
        return .available
    }

    /// Build a `TouchUpModelInfo` row for any catalogued `LLMModelType`,
    /// not just the three TouchUp picker selections (engine#306) —
    /// `availableModels()`'s `row(for:loadState:)` only ever names its
    /// fixed three, so a generate-only catalogue model (e.g.
    /// `yooz-instruct-4b`) needs its own row builder to be addressable by
    /// `requestDownload`/`cancelDownload`. `displayName` / `description` /
    /// `sizeBytes` are catalogue-sourced and identical to what the picker
    /// already reports for `.yoozLight`/`.yoozQuality` (both draw from
    /// `LLMModelCatalog` underneath, see `TouchUpModelSelection`'s
    /// matching values), so this is a strict generalization, not a
    /// divergent source of truth. `tier` and `isActive` fall back to the
    /// non-picker defaults (`.unknown`, `false`) for a model with no
    /// `TouchUpModelSelection` counterpart, which is never the TouchUp
    /// picker's active model by construction. MLX tiers are always
    /// `isAvailable: true` — they download on first use, mirroring
    /// `availableModels()`'s convention for the two picker tiers.
    private func catalogRow(for modelType: LLMModelType) async -> TouchUpModelInfo {
        let selection = TouchUpModelSelection(rawValue: modelType.rawValue)
        let active = await activeModel
        let isLoaded = await backends[modelType]?.isLoaded ?? false
        let cached = await isCached(modelType)
        return TouchUpModelInfo(
            id: modelType.rawValue,
            displayName: modelType.displayName,
            description: modelType.description,
            tier: selection?.tier ?? .unknown,
            sizeBytes: modelType.estimatedSize,
            loadState: loadState(isAvailable: true, isCached: cached, isLoaded: isLoaded),
            isActive: selection.map { active == $0 } ?? false
        )
    }

    /// Set the active model and (optionally) preload it. Returns the
    /// info row for the new active model so the caller does not need
    /// a follow-up `availableModels()` round-trip.
    ///
    /// `preload: true` (default) is the recommended path — it makes
    /// a picker change one-shot, so the next `/v1/touchup` call does
    /// not pay a cold-start. With `preload: false`, the call only
    /// updates `activeModel`; the next `process(...)` call may
    /// silently fall back to MLX if the user picks
    /// `.foundationModels` and the backend is not loaded. The route
    /// handler exposes `preload` via the request body but defaults
    /// it to `true` so SDK consumers do not have to think about this.
    ///
    /// Throws `LLMError.notAvailable` if the caller picks
    /// `.foundationModels` on a system without Apple Intelligence,
    /// or any error from the underlying load path otherwise.
    @discardableResult
    public func setActiveModel(
        _ selection: TouchUpModelSelection,
        preload: Bool = true
    ) async throws -> TouchUpModelInfo {
        await restorePersistedSelectionIfNeeded()
        switch selection {
        case .yoozLight:
            if preload {
                // Route through the cancellable `enqueueLoad` state machine so
                // the load is bounded (no unbounded `loadModelContainer` await
                // that hangs the picker) and observable via `loadState(for:)` /
                // `/v1/llm/status`. `enqueueLoad(.yoozLight)` runs
                // `preloadModel(.yoozLight)`, equivalent to the prior inline load
                // under normal conditions (it throws rather than silently no-ops
                // if `createLight` ever returns nil).
                try await awaitLoadTask(
                    enqueueLoad(.yoozLight),
                    deadlineSeconds: EngineConfig.modelLoadDeadlineSeconds
                )
            }
        case .yoozQuality:
            if preload {
                try await awaitLoadTask(
                    enqueueLoad(.yoozQuality),
                    deadlineSeconds: EngineConfig.modelLoadDeadlineSeconds
                )
            }
        case .foundationModels:
            // Validate availability up-front so a non-26 host never
            // sees `activeModel == .foundationModels` followed by a
            // silent MLX fallback on the next call. This is the path
            // the route handler maps to 501 `model_unavailable`.
            let backend = foundationModelsBackend ?? FoundationModelsBackend()
            guard backend.isAvailable() else {
                throw LLMError.notAvailable(
                    "Apple Intelligence is not available on this system. Requires macOS 26+ and an opted-in user."
                )
            }
            if preload, await !backend.isLoaded {
                try await backend.load()
            }
            foundationModelsBackend = backend
        }

        await recordActiveModel(selection)

        // Strict single-resident: now that the newly-selected tier is loaded
        // and active, drop every other tier so RAM tracks only the active
        // model. Done after the load above so a failed switch never leaves the
        // engine with nothing resident.
        await evictModelsExcept(selection)
        await EngineEventBus.shared.publish(EngineEvent(
            kind: .residencyChanged, module: Self.selectionStoreModule, modelId: selection.rawValue
        ))

        // `availableModels()` always emits exactly one row per
        // selection (precondition'd above). The `first(where:)` is
        // total here; `LLMError.notLoaded` was a dead throw and is
        // intentionally absent.
        let models = await availableModels()
        return models.first(where: { $0.isActive })!
    }

    // MARK: - Persisted selection + events (engine#226)

    /// Restore the persisted active-model selection on first touch of this
    /// instance. Idempotent — runs at most once. Deliberately NOT run from
    /// `init` (an actor's synchronous `init` cannot `await` the store's
    /// file read); instead invoked from the `activeModel` getter itself,
    /// so EVERY read restores — direct property access,
    /// `availableModels()`, `setActiveModel(Async)`, and
    /// `processWithActiveModel` alike — with no way for a caller to
    /// observe the compiled-in `.yoozLight` default when a persisted
    /// selection exists.
    ///
    /// Memoized as a shared `Task` (not a Bool once-flag): a flag set
    /// before the store read's suspension point let a second concurrent
    /// first-touch pass the guard mid-restore and read the unrestored
    /// default (PR #239 review). With the task handle, every concurrent
    /// caller awaits the SAME restore and none returns before it settles.
    /// Mutation paths (`setActiveModel(Async)`) also await this before
    /// writing, so a slow restore can never clobber a fresher selection.
    ///
    /// Restoring only updates `_activeModel` — it never preloads weights.
    /// The existing lazy-load contract (`process()` loads the light model
    /// on first use; `processWithActiveModel` loads whichever tier is
    /// active on first use) is unchanged; this just makes sure THAT lazy
    /// load targets the right tier after a restart.
    private func restorePersistedSelectionIfNeeded() async {
        if let existing = restoreTask {
            await existing.value
            return
        }
        let task = Task { await self.performPersistedSelectionRestore() }
        restoreTask = task
        await task.value
    }

    private func performPersistedSelectionRestore() async {
        guard let storedId = await selectionStore.activeId(for: Self.selectionStoreModule) else {
            return
        }
        guard let selection = TouchUpModelSelection(rawValue: storedId) else {
            // A persisted id from an older build (e.g. the pre-#282
            // `yooz-*-v2` wire ids) no longer parses. Falling back to the
            // compiled-in default is correct, but doing it silently would
            // hide why a user's tier changed after an upgrade — fail loudly
            // instead (PR #239 review precedent, PR #283 review).
            logger.error(
                "TouchUp persisted selection \(storedId, privacy: .public) has no TouchUpModelSelection; keeping default \(self._activeModel.rawValue, privacy: .public)"
            )
            return
        }
        _activeModel = selection
        logger.info(
            "TouchUp active model restored from persistence: \(selection.rawValue, privacy: .public)"
        )
    }

    /// Update `activeModel`, persist it (engine#226), and publish
    /// `modelChanged`. Shared by both the blocking `setActiveModel` and the
    /// non-blocking `setActiveModelAsync` so the two entry points can never
    /// diverge on the persistence/eventing contract.
    private func recordActiveModel(_ selection: TouchUpModelSelection) async {
        _activeModel = selection
        logger.info("TouchUp active model set to \(selection.rawValue, privacy: .public)")
        await selectionStore.setActiveId(selection.rawValue, for: Self.selectionStoreModule)
        await EngineEventBus.shared.publish(EngineEvent(
            kind: .modelChanged, module: Self.selectionStoreModule, modelId: selection.rawValue
        ))
    }

    /// Non-blocking counterpart to `setActiveModel(_:preload:)`
    /// (engine#226): records + persists the selection and returns
    /// immediately — the preload (when requested) runs on a detached
    /// background `Task` whose progress and terminal state are observable
    /// only through `EngineEventBus` (`loadStateChanged` /
    /// `downloadProgress` / `residencyChanged`) or a follow-up
    /// `GET /v1/touchup/models` poll, never through this call's return
    /// value — which reflects state at the MOMENT of the call, before the
    /// background load has had a chance to run.
    ///
    /// Foundation Models has no download to bound — its availability check
    /// is a synchronous OS-capability probe, so it still runs inline here,
    /// matching `setActiveModel`'s contract: an unsupported system throws
    /// `LLMError.notAvailable` from THIS call (mapped to the route's 501
    /// `model_unavailable`), not silently from the background task where
    /// no caller could observe it.
    @discardableResult
    public func setActiveModelAsync(
        _ selection: TouchUpModelSelection,
        preload: Bool = true
    ) async throws -> TouchUpModelInfo {
        await restorePersistedSelectionIfNeeded()

        if selection == .foundationModels {
            let backend = foundationModelsBackend ?? FoundationModelsBackend()
            guard backend.isAvailable() else {
                throw LLMError.notAvailable(
                    "Apple Intelligence is not available on this system. Requires macOS 26+ and an opted-in user."
                )
            }
        }

        await recordActiveModel(selection)
        let rows = await availableModels()
        let row = rows.first(where: { $0.isActive })!

        guard preload else { return row }

        // Per-tier dispatches (engine#288): a switch must NEVER cancel
        // another tier's in-flight dispatch — its download continues in the
        // background and its progress watcher stays alive for the real
        // duration. Same-tier re-requests dedupe onto the running dispatch
        // (enqueueLoad already coalesces the underlying load). The
        // superseded-completion guard inside the dispatch handles the
        // "user moved on before I finished" case.
        if backgroundPreloadTasks[selection.rawValue] == nil {
            backgroundPreloadTasks[selection.rawValue] = Task {
                await self.preloadActiveSelectionInBackground(selection)
                await self.clearBackgroundPreload(selection.rawValue)
            }
        }
        return row
    }

    /// Remove a finished dispatch's handle (actor-isolated bookkeeping for
    /// `backgroundPreloadTasks`; called by the dispatch itself on the way
    /// out so a later re-selection or re-download can start a fresh
    /// dispatch).
    private func clearBackgroundPreload(_ modelId: String) {
        backgroundPreloadTasks[modelId] = nil
    }

    /// Explicit download, decoupled from the active selection (engine#288
    /// slice 2; generalized from the 3-case picker onto the full LLM
    /// catalogue in engine#306): fetch `modelType`'s weights without
    /// switching models. Reuses the per-model background-load dispatch
    /// (`downloadModel`) — if `modelType` is NOT the TouchUp picker's
    /// active selection when the load settles, the superseded-completion
    /// path frees the memory copy and the model lands `.cached` on disk;
    /// if it IS active, this is an ordinary preload. A catalogue model
    /// with no `TouchUpModelSelection` counterpart (e.g. `yooz-instruct-4b`)
    /// is never the active selection, so it always takes the
    /// superseded-completion path. Dedupes onto any dispatch already
    /// running for the model id, including a picker-switch dispatch for
    /// the same id from `setActiveModelAsync`. Progress/outcome arrive via
    /// events. Returns the model's current picker row (nil for an id with
    /// no row — structurally unreachable, `catalogRow(for:)` is total for
    /// every `LLMModelType`).
    public func requestDownload(
        _ modelType: LLMModelType
    ) async -> TouchUpModelInfo? {
        let row = await catalogRow(for: modelType)
        // Nothing to fetch (PR #290 review): `.cached` is complete on
        // disk and `.loaded` is resident — dispatching anyway would
        // materialize weights into GPU memory only for the superseded-
        // completion path to evict them again (real wasted work, and a
        // residency blip for the actively generating tier).
        if row.loadState == .cached || row.loadState == .loaded {
            return row
        }
        if backgroundPreloadTasks[modelType.rawValue] == nil {
            backgroundPreloadTasks[modelType.rawValue] = Task {
                await self.downloadModel(modelType)
                await self.clearBackgroundPreload(modelType.rawValue)
            }
        }
        return row
    }

    /// Cancel an in-flight download for `modelType` (engine#288 slice 2;
    /// generalized in engine#306). Only acts when the model is actually
    /// `.loading` — cancelling a loaded/settled model is a no-op (never
    /// unloads a resident model). `unload(_:)` cancels the in-flight load
    /// task; the model's dispatch observes the cancellation and publishes
    /// the settled `.available` row state itself (cooperative-cancel path,
    /// no error toast).
    public func cancelDownload(
        _ modelType: LLMModelType
    ) async -> TouchUpModelInfo? {
        if loadState(for: modelType) == .loading {
            await unload(modelType)
            // Also cancel + clear the OUTER dispatch handle (PR #290
            // review): `unload` only cancels the inner load task, and
            // cancellation unwinds cooperatively — until the old dispatch
            // exits and self-clears, a rapid re-download/re-select of the
            // same model would dedupe onto the dying dispatch and silently
            // no-op. Clearing here guarantees the next request always
            // starts fresh.
            backgroundPreloadTasks[modelType.rawValue]?.cancel()
            backgroundPreloadTasks[modelType.rawValue] = nil
        }
        return await catalogRow(for: modelType)
    }

    /// Background load + single-resident eviction for
    /// `setActiveModelAsync`. Runs disconnected from the HTTP/in-process
    /// request that triggered it — every observable outcome goes through
    /// `EngineEventBus`, never a return value.
    private func preloadActiveSelectionInBackground(_ selection: TouchUpModelSelection) async {
        switch selection {
        case .foundationModels:
            // Availability was already validated synchronously in
            // `setActiveModelAsync`; opening a `LanguageModelSession` is
            // cheap (no download), so this still completes quickly, but
            // stays off the request path for uniformity with the MLX tiers.
            do {
                let backend = foundationModelsBackend ?? FoundationModelsBackend()
                if await !backend.isLoaded {
                    try await backend.load()
                }
                foundationModelsBackend = backend
                await publishLoadStateChanged(selection.rawValue, state: .loaded)
            } catch {
                await publishLoadStateChanged(
                    selection.rawValue, state: .available, message: error.localizedDescription
                )
                return
            }
            // Only reached on a successful load, and only while this
            // dispatch is still the CURRENT one: `activeModel == selection`
            // (a superseded late completion must not evict the model the
            // user has since switched to). `LLMModelType(rawValue:
            // "foundation-models")` is structurally nil (not a catalogue
            // entry), so unlike the MLX case below there is no settled row
            // to unload/republish on the superseded branch here — Apple
            // Intelligence has no weights to evict in the first place.
            guard await activeModel == selection, !Task.isCancelled else { return }
            await evictModelsExcept(selection)
            await EngineEventBus.shared.publish(EngineEvent(
                kind: .residencyChanged, module: Self.selectionStoreModule, modelId: selection.rawValue
            ))
        case .yoozLight, .yoozQuality:
            guard let modelType = LLMModelType(rawValue: selection.rawValue) else {
                // Structurally unreachable today (the two MLX selections'
                // raw values match LLMModelType's 1:1), but the enums are
                // deliberately separate and expected to be able to drift
                // (see TouchUpModelSelection's doc). If they ever do, a
                // silent return here would strand the picker in "loading"
                // forever with no diagnostic — fail loudly on both channels
                // instead (PR #239 review).
                logger.error(
                    "TouchUp background preload: no LLMModelType for selection \(selection.rawValue, privacy: .public)"
                )
                await publishLoadStateChanged(
                    selection.rawValue, state: .available,
                    message: "internal: no LLM backend maps to '\(selection.rawValue)'"
                )
                return
            }
            await downloadModel(modelType)
        }
    }

    /// Background load for one catalogued LLM model (engine#306): the
    /// shared core of the picker-switch dispatch
    /// (`preloadActiveSelectionInBackground`'s MLX case) and the
    /// explicit-download dispatch (`requestDownload`), which generalized
    /// this from `TouchUpModelSelection` onto the full `LLMModelType`
    /// catalogue. Runs disconnected from the request that triggered it;
    /// every observable outcome goes through `EngineEventBus`, keyed by
    /// `modelType.rawValue`.
    ///
    /// On success, evicts every OTHER tier ONLY while this dispatch is
    /// still the CURRENT one: the TouchUp picker's active selection still
    /// names this model (a superseded model's late completion must not
    /// evict the model the user has since switched to). Superseded
    /// completion (engine#288): free the MEMORY copy but keep the
    /// downloaded weights on disk. A catalogue model with no
    /// `TouchUpModelSelection` counterpart (e.g. `yooz-instruct-4b`) can
    /// never be the active selection, so it always takes the superseded
    /// branch and lands `.cached` on disk rather than staying resident —
    /// the "an explicit download never makes a non-picker model resident"
    /// contract engine#306 requires.
    private func downloadModel(_ modelType: LLMModelType) async {
        let task = enqueueLoad(modelType)
        let progressWatcher = Task { await self.watchDownloadProgress(modelType) }
        defer { progressWatcher.cancel() }
        do {
            // Bounded (PR #239 review): `Task.value` on the unstructured
            // load handle ignores the awaiting task's cancellation, so
            // without a deadline a wedged download would pin this task +
            // its progress watcher forever. Same deadline the blocking
            // `setActiveModel` path applies.
            try await awaitLoadTask(
                task, deadlineSeconds: EngineConfig.modelLoadDeadlineSeconds
            )
            await publishLoadStateChanged(modelType.rawValue, state: .loaded)
        } catch {
            // A cooperative cancellation (e.g. a rapid picker
            // double-switch cancelling this model's in-flight load via
            // `unload`) is not a user-facing failure — omit `message`
            // so the picker UI doesn't render a spurious error toast.
            let message = error is CancellationError ? nil : error.localizedDescription
            await publishLoadStateChanged(modelType.rawValue, state: .available, message: message)
            return
        }

        let activeSelection = await activeModel
        guard activeSelection.rawValue == modelType.rawValue, !Task.isCancelled else {
            await unload(modelType)
            let row = await catalogRow(for: modelType)
            await publishLoadStateChanged(modelType.rawValue, state: row.loadState)
            return
        }
        await evictModelsExcept(activeSelection)
        await EngineEventBus.shared.publish(EngineEvent(
            kind: .residencyChanged, module: Self.selectionStoreModule, modelId: modelType.rawValue
        ))
    }

    /// Poll `downloadProgress(for:)` while a background load is in flight
    /// and publish throttled `downloadProgress` events. Cancelled by the
    /// caller once the load's terminal outcome is known. Polling (rather
    /// than a push callback from `MLXLLMBackend`) keeps this change
    /// contained to `TouchUpEngine` — `MLXLLMBackend` already exposes the
    /// live fraction via `downloadProgress(for:)` (used by `/v1/llm/status`
    /// today), so no new callback plumbing is needed.
    private func watchDownloadProgress(_ modelType: LLMModelType) async {
        var lastPublished: Double = -1
        var ticksSincePublish = 0
        while !Task.isCancelled {
            // Reads the backend's live KVO-fed fraction (engine#292). The
            // byte-on-disk alternative was measured and rejected: this
            // downloader stages the one multi-GB weights file outside the hub
            // repo and moves it in at the end, so on-disk bytes sit flat at
            // ~0.6% for the entire transfer and then jump.
            // Stop the moment the tier stops loading (PR #293 review):
            // `markLoadSettled` flips `loadStates` from inside the load task,
            // strictly BEFORE this watcher's `defer { cancel() }` runs (that
            // waits on `awaitLoadTask` plus a `publishLoadStateChanged`
            // await). A failed/cancelled load also leaves the backend's
            // fraction at its last partial value, so without this guard the
            // watcher could wake in that window and publish "still
            // downloading" for a tier whose failed/available
            // `loadStateChanged` is already in flight. Mirrors the same gate
            // `inFlightDownloadFractions()` applies below.
            guard loadStates[modelType] == .loading else { return }
            let fraction = await downloadProgress(for: modelType) ?? 0
            // Publish on a 0.5% move OR at least every ~2s (7 ticks) while a
            // fetch is in progress (engine#292). The old rule published only
            // on a >=2% move, so with a per-file source the single 0.58% step
            // never cleared it and exactly ONE frame escaped per download —
            // the 0%-forever bar.
            //
            // The ~2s arm is a deliberate KEEP-ALIVE, not a progress claim:
            // measured, `completedUnitCount` is flat for the whole multi-GB
            // file (see `MLXLLMBackend.load`'s progressHandler comment), so
            // repeating the same fraction is how a consumer distinguishes
            // "still downloading" from "engine went away". A consumer must
            // render an unchanging fraction as indeterminate rather than as
            // a precise percentage — an identical value arriving every 2s is
            // the signal for that, and the terminal truth stays
            // `loadStateChanged`.
            let moved = fraction - lastPublished >= 0.005
            // DO NOT "optimize" this into dedupe-by-value: republishing an
            // UNCHANGED fraction is the intended keep-alive, and flat
            // stretches are the normal case (per-file granularity), so
            // suppressing them reintroduces the frozen-bar bug this fixes.
            let overdue = fraction > 0 && ticksSincePublish >= 7
            if fraction > 0, fraction < 1, moved || overdue {
                lastPublished = fraction
                ticksSincePublish = 0
                await EngineEventBus.shared.publish(EngineEvent(
                    kind: .downloadProgress, module: Self.selectionStoreModule,
                    modelId: modelType.rawValue, progress: fraction
                ))
            } else {
                ticksSincePublish += 1
            }
            try? await Task.sleep(for: .milliseconds(300))
        }
    }

    /// In-flight download fraction per wire id, for the `/v1/state` snapshot
    /// (engine#292). Only tiers whose `loadState` is `.loading` report a
    /// value — a settled tier's backend still holds its final fraction, and
    /// reporting that would render a permanent "download in progress" row.
    /// Cheap: an in-memory read per MLX tier, no filesystem work.
    public func inFlightDownloadFractions() async -> [String: Double] {
        var result: [String: Double] = [:]
        for modelType in LLMModelType.allCases where loadStates[modelType] == .loading {
            guard let fraction = await downloadProgress(for: modelType) else { continue }
            if fraction > 0, fraction < 1 {
                result[modelType.rawValue] = fraction
            }
        }
        return result
    }

    private func publishLoadStateChanged(
        _ modelId: String, state: ModelLoadState, message: String? = nil
    ) async {
        await EngineEventBus.shared.publish(EngineEvent(
            kind: .loadStateChanged, module: Self.selectionStoreModule,
            modelId: modelId, loadState: state, message: message
        ))
    }

    /// Process text through the currently active model. Used by the
    /// `/v1/touchup` route; preserves the existing `mode` semantics
    /// (regex-only vs LLM, prompt strength) while letting the picker
    /// override which backend handles the LLM call.
    ///
    /// For `.yoozQuality`, the dispatch must NOT delegate to
    /// `process(...)` because that method auto-routes to the light
    /// model when `replacements.isEmpty`. The picker-aware path
    /// runs inference directly through the loaded quality backend
    /// so a user-picked Quality is honored regardless of whether
    /// replacements are present.
    public func processWithActiveModel(
        text: String,
        mode: TouchUpMode,
        replacements: [(original: String, replacement: String)] = [],
        workloadClass: MLXWorkloadClass = .background,
        contextVocabulary: [String]? = nil,
        contextAppName: String? = nil
    ) async -> TouchUpProcessor.ProcessResult {
        await restorePersistedSelectionIfNeeded()
        switch await activeModel {
        case .foundationModels:
            return await processWithFoundationModels(
                text: text, mode: mode,
                contextVocabulary: contextVocabulary, contextAppName: contextAppName
            )
        case .yoozLight:
            return await process(
                text: text, mode: mode, replacements: replacements, workloadClass: workloadClass,
                contextVocabulary: contextVocabulary, contextAppName: contextAppName
            )
        case .yoozQuality:
            // Force the quality backend on both routing slots so the
            // user's pick is honored even when no replacements are
            // present (the legacy `process(...)` auto-routing only
            // uses quality when replacements force it).
            do {
                try await ensureLoaded(.yoozQuality)
            } catch {
                // Quality load failed — fall back to the legacy MLX
                // path. The fallback uses the light model and
                // surfaces the load failure as a warning string.
                return await process(
                    text: text, mode: mode, replacements: replacements, workloadClass: workloadClass,
                    contextVocabulary: contextVocabulary, contextAppName: contextAppName
                )
            }
            guard let quality = backends[.yoozQuality], await quality.isLoaded else {
                return await process(
                    text: text, mode: mode, replacements: replacements, workloadClass: workloadClass,
                    contextVocabulary: contextVocabulary, contextAppName: contextAppName
                )
            }
            let proofreadPrompt = Self.withContext(
                prompt: selectPrompt(for: mode, qualityAvailable: true),
                vocabulary: Self.cappedContextVocabulary(contextVocabulary),
                appName: contextAppName
            )
            let replacementStructs = replacements.map {
                TouchUpProcessor.Replacement(
                    original: $0.original, replacement: $0.replacement
                )
            }
            let result = await TouchUpProcessor.process(
                text: text,
                replacements: replacementStructs,
                lightModel: quality,
                qualityModel: quality,
                proofreadPrompt: proofreadPrompt,
                workloadClass: workloadClass
            )
            // Relabel: `TouchUpProcessor.process` hard-codes the
            // `modelUsed` field based on which routing slot it took,
            // not which backend the slot held. The picker explicitly
            // ran inference through the quality backend, so the
            // user-visible report should say so.
            return TouchUpProcessor.ProcessResult(
                text: result.text,
                keepDecisions: result.keepDecisions,
                modelUsed: result.modelUsed == .light ? .quality : result.modelUsed,
                latencyMs: result.latencyMs,
                fallbackReason: result.fallbackReason
            )
        }
    }

    /// Get model info for display: one entry per catalogued model
    /// (engine#303), not just the two TouchUp tiers — `APIServer`'s
    /// `GET /v1/llm/models` enumerates every servable model this way.
    public func getModelInfo() async -> [LLMModelInfo] {
        var result: [LLMModelInfo] = []
        for modelType in LLMModelType.allCases {
            let isLoaded = await backends[modelType]?.isLoaded ?? false
            let isCached = await isCached(modelType)
            result.append(LLMModelInfo(type: modelType, isLoaded: isLoaded, isCached: isCached))
        }
        return result
    }
}

// MARK: - LLM Model Info

/// Information about a model's status (renamed from ModelInfo to avoid collision with APITypes.ModelInfo)
public struct LLMModelInfo: Sendable {
    public let type: LLMModelType
    public let isLoaded: Bool
    public let isCached: Bool

    public init(type: LLMModelType, isLoaded: Bool, isCached: Bool) {
        self.type = type
        self.isLoaded = isLoaded
        self.isCached = isCached
    }
}
