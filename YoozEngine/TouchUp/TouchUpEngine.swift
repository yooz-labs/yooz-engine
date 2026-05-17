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
/// - **Yooz-Light** (Qwen2.5-0.5B): Fast proofreading, ~200ms latency
/// - **Yooz-Quality** (Qwen3-1.7B): Higher quality proofreading, ~490ms latency
/// - **Apple Intelligence** (Foundation Models 3B): macOS 26+, structured generation
public actor TouchUpEngine {

    // MARK: - Singleton

    public static let shared = TouchUpEngine()

    // MARK: - Properties

    /// The light model backend (Yooz-Light, Qwen2.5-0.5B)
    private var lightModel: MLXLLMBackend?

    /// The quality model backend (Yooz-Quality, Qwen3-1.7B)
    private var qualityModel: MLXLLMBackend?

    /// The Apple Intelligence backend (Foundation Models, macOS 26+)
    private var foundationModelsBackend: FoundationModelsBackend?

    /// Bundle identifier for model loading
    private let bundleIdentifier: String

    /// Currently active TouchUp model. Drives `/v1/touchup` routing
    /// via `processWithActiveModel(...)` and is reflected back to
    /// clients through the picker API (`GET /v1/touchup/models`).
    /// Defaults to `.yoozLight` so callers that never `setActiveModel`
    /// see the pre-picker behaviour.
    public private(set) var activeModel: TouchUpModelSelection = .yoozLight

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

    /// Whether the light model is loaded
    public var isLightModelLoaded: Bool {
        get async {
            guard let model = lightModel else { return false }
            return await model.isLoaded
        }
    }

    /// Whether the quality model is loaded
    public var isQualityModelLoaded: Bool {
        get async {
            guard let model = qualityModel else { return false }
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

    // MARK: - Initialization

    /// Private to preserve the `.shared` singleton contract used across the
    /// engine. Tests that need to inspect a fresh instance should do so via
    /// the shared actor.
    private init(bundleIdentifier: String = "live.yooz.engine") {
        self.bundleIdentifier = bundleIdentifier
    }

    // MARK: - Lifecycle

    /// Preload models for immediate use.
    ///
    /// This loads the light model (Yooz-Light) which is embedded in the app bundle.
    /// The quality model (Yooz-Quality) is loaded on-demand when needed.
    /// Apple Intelligence is loaded if available (macOS 26+).
    public func preload(loadQuality: Bool = false) async throws {
        logger.info("Preloading TouchUpEngine...")

        if lightModel == nil {
            lightModel = MLXLLMBackend.createLight(bundleIdentifier: bundleIdentifier)
        }

        if let light = lightModel {
            try await light.load()
            logger.info("Yooz-Light model loaded")
        }

        if loadQuality {
            if qualityModel == nil {
                qualityModel = MLXLLMBackend.createQuality(bundleIdentifier: bundleIdentifier)
            }
            if let quality = qualityModel {
                try await quality.load()
                logger.info("Yooz-Quality model loaded")
            }
        }

        // Try to load Apple Intelligence if available
        await loadFoundationModelsIfAvailable()

        isPreloaded = true
        logger.info("TouchUpEngine preloaded successfully")
    }

    /// Ensure the quality model is loaded.
    /// Downloads from GHCR if not cached.
    public func loadQualityModel() async throws {
        if qualityModel == nil {
            qualityModel = MLXLLMBackend.createQuality(bundleIdentifier: bundleIdentifier)
        }

        guard let quality = qualityModel else { return }

        if await !quality.isLoaded {
            try await quality.load()
            logger.info("Yooz-Quality model loaded on-demand")
        }
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
        if let light = lightModel {
            await light.resetForNewSession()
        }
        if let quality = qualityModel {
            await quality.resetForNewSession()
        }
    }

    /// Unload all models from memory.
    public func unload() async {
        if let light = lightModel {
            await light.unload()
        }
        if let quality = qualityModel {
            await quality.unload()
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
        switch modelType {
        case .yoozLight:
            if let light = lightModel {
                await light.unload()
            }
        case .yoozQuality:
            if let quality = qualityModel {
                await quality.unload()
            }
        }
    }

    /// Record the user-preferred LLM. Does not load weights — call
    /// `preloadModel(_:)` (or the /v1/llm/preload route) to warm
    /// the model after switching.
    public func setPreferredModel(_ modelType: LLMModelType) {
        preferredModel = modelType
        logger.info("TouchUpEngine preferredModel set to \(modelType.rawValue, privacy: .public)")
    }

    /// Ensure a specific model's weights are resident. Idempotent;
    /// loads the light model in-place (it is embedded in the app
    /// bundle) and triggers a GHCR download for the quality model on
    /// first use. Invoked by `POST /v1/llm/preload`.
    public func preloadModel(_ modelType: LLMModelType) async throws {
        switch modelType {
        case .yoozLight:
            if lightModel == nil {
                lightModel = MLXLLMBackend.createLight(bundleIdentifier: bundleIdentifier)
            }
            guard let light = lightModel else {
                throw LLMError.notLoaded
            }
            if await !light.isLoaded {
                try await light.load()
                logger.info("Yooz-Light model preloaded on demand")
            }
        case .yoozQuality:
            try await loadQualityModel()
        }
    }

    // MARK: - Raw LLM Generation

    /// Generate text using a specified model.
    /// Used by the `/v1/llm/generate` endpoint.
    ///
    /// - Parameters:
    ///   - prompt: The user prompt
    ///   - systemPrompt: System prompt for the model
    ///   - modelType: Which model to use (defaults to light)
    ///   - kvCompression: Optional per-request KV cache compression mode.
    ///     When `nil`, the cached model uses `EngineConfig.kvCompression`
    ///     (default `.off`). When non-nil and different from the cached
    ///     model's mode, a temporary backend is constructed for this call
    ///     so the cached model's prompt-cache state is preserved.
    /// - Returns: Generated text
    public func generate(
        prompt: String,
        systemPrompt: String,
        modelType: LLMModelType = .yoozLight,
        kvCompression: KVCompressionMode? = nil
    ) async throws -> String {
        // Per-request kvCompression override path: build a fresh backend
        // with the requested mode rather than mutating the cached one.
        // This is safe because backends are cheap to construct (the model
        // weights live in `ModelContainer` which is loaded on `load()`).
        if let override = kvCompression,
           override != EngineConfig.kvCompression {
            let backend = MLXLLMBackend.create(
                for: modelType,
                bundleIdentifier: bundleIdentifier,
                kvCompression: override
            )
            try await backend.load()
            defer { Task { await backend.unload() } }
            return try await backend.generate(prompt: prompt, systemPrompt: systemPrompt)
        }

        let model: MLXLLMBackend

        switch modelType {
        case .yoozLight:
            if lightModel == nil {
                lightModel = MLXLLMBackend.createLight(bundleIdentifier: bundleIdentifier)
            }
            guard let light = lightModel else {
                throw LLMError.notLoaded
            }
            if await !light.isLoaded {
                try await light.load()
            }
            model = light

        case .yoozQuality:
            try await loadQualityModel()
            guard let quality = qualityModel else {
                throw LLMError.notLoaded
            }
            model = quality
        }

        return try await model.generate(prompt: prompt, systemPrompt: systemPrompt)
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
    /// - Returns: ProcessResult with cleaned text and metadata
    public func process(
        text: String,
        mode: TouchUpMode,
        replacements: [(original: String, replacement: String)] = []
    ) async -> TouchUpProcessor.ProcessResult {
        let replacementStructs = replacements.map {
            TouchUpProcessor.Replacement(original: $0.original, replacement: $0.replacement)
        }

        // Ensure light model is loaded
        guard let light = lightModel, await light.isLoaded else {
            logger.warning("Light model not loaded, using regex-only processing")
            return TouchUpProcessor.processRegexOnly(text: text, replacements: replacementStructs)
        }

        // If we have replacements, ensure quality model is loaded
        var qualityLoadError: String?
        if !replacements.isEmpty {
            do {
                try await loadQualityModel()
            } catch {
                qualityLoadError = error.localizedDescription
                logger.error("Failed to load quality model: \(error.localizedDescription)")
            }
        }

        // Check if quality model is available and loaded
        let qualityAvailable: Bool
        if let quality = qualityModel {
            qualityAvailable = await quality.isLoaded
        } else {
            qualityAvailable = false
        }


        // Select the proofread prompt based on mode and available model
        let proofreadPrompt = selectPrompt(
            for: mode,
            qualityAvailable: qualityAvailable
        )

        // Route to appropriate processing
        if replacements.isEmpty || !qualityAvailable {
            var result = await TouchUpProcessor.process(
                text: text,
                replacements: [],
                lightModel: light,
                qualityModel: light,
                proofreadPrompt: proofreadPrompt
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
        } else if let quality = qualityModel {
            return await TouchUpProcessor.process(
                text: text,
                replacements: replacementStructs,
                lightModel: light,
                qualityModel: quality,
                proofreadPrompt: proofreadPrompt
            )
        } else {
            // Should not reach here since qualityAvailable was true,
            // but fall back to light model to be safe
            return await TouchUpProcessor.process(
                text: text,
                replacements: [],
                lightModel: light,
                qualityModel: light,
                proofreadPrompt: proofreadPrompt
            )
        }
    }

    /// Process text using Apple Intelligence backend directly.
    /// Falls back to MLX models if Foundation Models unavailable.
    public func processWithFoundationModels(
        text: String,
        mode: TouchUpMode
    ) async -> TouchUpProcessor.ProcessResult {
        let startTime = CFAbsoluteTimeGetCurrent()

        guard let backend = foundationModelsBackend, await backend.isLoaded else {
            logger.warning("Foundation Models not available, falling back to MLX")
            var result = await process(text: text, mode: mode)
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

    // MARK: - Model Info

    /// Whether the Light model snapshot is on disk in the HF cache.
    /// Both tiers download from HF on first use (PR #93 / issue #77),
    /// so neither is "always cached".
    public var isLightModelCached: Bool {
        get async {
            guard let light = lightModel else {
                let temp = MLXLLMBackend.createLight(bundleIdentifier: bundleIdentifier)
                return await temp.isModelCached
            }
            return await light.isModelCached
        }
    }

    /// Whether the Quality model snapshot is on disk in the HF cache.
    public var isQualityModelCached: Bool {
        get async {
            guard let quality = qualityModel else {
                let temp = MLXLLMBackend.createQuality(bundleIdentifier: bundleIdentifier)
                return await temp.isModelCached
            }
            return await quality.isModelCached
        }
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
            isActive: activeModel == selection
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
        switch selection {
        case .yoozLight:
            if preload {
                if lightModel == nil {
                    lightModel = MLXLLMBackend.createLight(bundleIdentifier: bundleIdentifier)
                }
                if let light = lightModel, await !light.isLoaded {
                    try await light.load()
                }
            }
        case .yoozQuality:
            if preload {
                try await loadQualityModel()
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

        activeModel = selection
        logger.info("TouchUp active model set to \(selection.rawValue, privacy: .public)")

        // `availableModels()` always emits exactly one row per
        // selection (precondition'd above). The `first(where:)` is
        // total here; `LLMError.notLoaded` was a dead throw and is
        // intentionally absent.
        let models = await availableModels()
        return models.first(where: { $0.isActive })!
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
        replacements: [(original: String, replacement: String)] = []
    ) async -> TouchUpProcessor.ProcessResult {
        switch activeModel {
        case .foundationModels:
            return await processWithFoundationModels(text: text, mode: mode)
        case .yoozLight:
            return await process(text: text, mode: mode, replacements: replacements)
        case .yoozQuality:
            // Force the quality backend on both routing slots so the
            // user's pick is honored even when no replacements are
            // present (the legacy `process(...)` auto-routing only
            // uses quality when replacements force it).
            do {
                try await loadQualityModel()
            } catch {
                // Quality load failed — fall back to the legacy MLX
                // path. The fallback uses the light model and
                // surfaces the load failure as a warning string.
                return await process(text: text, mode: mode, replacements: replacements)
            }
            guard let quality = qualityModel, await quality.isLoaded else {
                return await process(text: text, mode: mode, replacements: replacements)
            }
            let proofreadPrompt = selectPrompt(for: mode, qualityAvailable: true)
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
                proofreadPrompt: proofreadPrompt
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

    /// Get model info for display
    public func getModelInfo() async -> (light: LLMModelInfo, quality: LLMModelInfo) {
        let lightLoaded = await isLightModelLoaded
        let qualityLoaded = await isQualityModelLoaded
        let lightCached = await isLightModelCached
        let qualityCached = await isQualityModelCached

        return (
            light: LLMModelInfo(
                type: .yoozLight,
                isLoaded: lightLoaded,
                isCached: lightCached
            ),
            quality: LLMModelInfo(
                type: .yoozQuality,
                isLoaded: qualityLoaded,
                isCached: qualityCached
            )
        )
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
