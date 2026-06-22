// InfiniteEngine.swift
// InfiniteModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore
import Foundation
import os.log

private let infiniteEngineLogger = Logger(
    subsystem: "live.yooz.engine",
    category: "InfiniteEngine"
)

public enum InfiniteError: Error, LocalizedError, Sendable, Equatable {
    case invalidModel(String)
    case modelUnavailable(String)
    case modelSetFailed(String)
    case sessionNotFound(String)
    case invalidSessionInput(String)
    case sessionLimitExceeded(Int)
    case generationUnavailable(String)
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .invalidModel(let id):
            return "Unknown Infinite model: \(id)"
        case .modelUnavailable(let id):
            return "Infinite model is unavailable on this system: \(id)"
        case .modelSetFailed(let reason):
            return "Failed to set Infinite model: \(reason)"
        case .sessionNotFound(let id):
            return "Unknown Infinite session: \(id)"
        case .invalidSessionInput(let reason):
            return "Invalid Infinite session input: \(reason)"
        case .sessionLimitExceeded(let limit):
            return "Infinite session limit exceeded: \(limit)"
        case .generationUnavailable(let reason):
            return "Infinite generation is unavailable: \(reason)"
        case .generationFailed(let reason):
            return "Infinite generation failed: \(reason)"
        }
    }
}

public actor InfiniteEngine {
    public static let shared = InfiniteEngine()

    public nonisolated static let maxActiveSessions = 16
    public nonisolated static var cleanupPolicy: String {
        "explicit_delete_or_process_exit;max_active_sessions=\(maxActiveSessions)"
    }

    public private(set) var activeModel: InfiniteModelSelection = .gemma4E4B1M
    /// The model whose weights are actually resident. Assigned by the
    /// inference backend once a model is fully loaded (Phase 7); until then it
    /// stays `nil`, so `isLoaded` is `false` and `status().state` never reaches
    /// `ready` by design.
    private var loadedModel: InfiniteModelSelection?
    private var preparedBackend: InfiniteBackendHandle?
    /// The real MLX backend, lazily loaded on first generate for a
    /// Swift-runtime-supported model (Phase 7).
    private var loadedBackend: MLXInfiniteBackend?
    /// In-flight loads keyed by selection, so concurrent first-generate calls
    /// for the same model await one load rather than each loading the (multi-GB)
    /// model — the actor releases isolation across the `await`, so without this
    /// two callers could both see `loadedBackend == nil` and double-load. Keyed
    /// (not a single slot) because more than one model is now
    /// `swiftRuntimeSupported` (Qwen + Gemma4 26B): a single slot let a
    /// concurrent load of a *different* model clobber it and trigger duplicate
    /// multi-GB loads.
    private var loadTasks: [InfiniteModelSelection: Task<MLXInfiniteBackend, Error>] = [:]
    private var lastLoadError: String?
    private var sessions: [String: SessionRecord] = [:]
    private let backendAdapter: any InfiniteBackendAdapter

    init(backendAdapter: any InfiniteBackendAdapter = CatalogInfiniteBackendAdapter()) {
        self.backendAdapter = backendAdapter
    }

    /// Restore the engine to its freshly-initialized state: no active sessions,
    /// no prepared backend, no remembered load error, and the default active
    /// model. The process-wide `shared` actor accumulates state across calls
    /// (sessions, a preloaded `preparedBackend`), so tests that exercise the
    /// served routes against `shared` must call this in both `setUp` and
    /// `tearDown` to stay independent of execution order.
    public func reset() {
        sessions.removeAll()
        preparedBackend = nil
        loadedBackend = nil
        for task in loadTasks.values { task.cancel() }
        loadTasks.removeAll()
        loadedModel = nil
        lastLoadError = nil
        activeModel = .gemma4E4B1M
    }

    /// Load (or reuse) the MLX backend for `selection`, serializing concurrent
    /// first-generate calls for the same model through one keyed in-flight task
    /// so the (multi-GB) weights load at most once.
    ///
    /// In-flight loads are keyed by selection because more than one model is now
    /// `swiftRuntimeSupported` (`qwen35B1M` + Gemma4 26B): a single in-flight
    /// slot let a concurrent load of a *different* model clobber it and trigger
    /// duplicate loads. Only one model stays resident in `loadedBackend`; a load
    /// for a different selection evicts the previous one (last write wins).
    private func loadBackend(
        for selection: InfiniteModelSelection
    ) async throws -> MLXInfiniteBackend {
        if let loaded = loadedBackend, loaded.selection == selection {
            return loaded
        }
        if let inFlight = loadTasks[selection] {
            return try await inFlight.value
        }
        let task = Task { try await MLXInfiniteBackend.load(selection.descriptor) }
        loadTasks[selection] = task
        defer { loadTasks[selection] = nil }
        do {
            let backend = try await task.value
            loadedBackend = backend
            loadedModel = selection
            lastLoadError = nil
            return backend
        } catch is CancellationError {
            // `reset()` cancels in-flight loads; surface cooperative cancellation
            // as-is rather than mislabeling it a model-load failure (which would
            // also disagree with the CancellationError a concurrent joiner sees).
            throw CancellationError()
        } catch let error as InfiniteError {
            lastLoadError = error.localizedDescription
            throw error
        } catch {
            lastLoadError = error.localizedDescription
            throw InfiniteError.modelSetFailed(error.localizedDescription)
        }
    }

    public var isLoaded: Bool {
        loadedModel == activeModel
    }

    public func availableModels() -> [InfiniteModelInfo] {
        let models = InfiniteModelSelection.allCases.map { info(for: $0) }
        precondition(
            models.filter(\.isActive).count == 1,
            "Infinite picker must expose exactly one active row"
        )
        return models
    }

    public func setActiveModel(
        _ selection: InfiniteModelSelection,
        preload: Bool
    ) async throws -> InfiniteModelInfo {
        guard isModelSelectable(selection) else {
            throw InfiniteError.modelUnavailable(selection.rawValue)
        }

        let nextPreparedBackend: InfiniteBackendHandle?
        if preload {
            do {
                nextPreparedBackend = try await backendAdapter.prepare(selection.descriptor)
            } catch {
                lastLoadError = error.localizedDescription
                throw InfiniteError.modelSetFailed(error.localizedDescription)
            }
        } else {
            nextPreparedBackend = preparedBackend?.selection == selection ? preparedBackend : nil
        }

        activeModel = selection
        preparedBackend = nextPreparedBackend
        lastLoadError = nil

        return info(for: selection)
    }

    public func status() -> InfiniteStatus {
        InfiniteStatus(
            loaded: isLoaded,
            modelId: activeModel.rawValue,
            progress: nil,
            state: state,
            activeSessions: sessions.count,
            maxContextTokens: activeModel.maxContextTokens,
            ramTier: activeModel.ramTier,
            backendKind: activeModel.backendKind,
            cleanupPolicy: Self.cleanupPolicy,
            resources: resourceMetrics(for: activeModel),
            lastError: lastLoadError
        )
    }

    public func listSessions() -> [InfiniteSessionInfo] {
        sessions.values
            .sorted { $0.createdAt < $1.createdAt }
            .map(sessionInfo)
    }

    public func session(id: String) throws -> InfiniteSessionInfo {
        try sessionInfo(for: id)
    }

    public func createSession(
        request: InfiniteCreateSessionRequest
    ) throws -> InfiniteSessionInfo {
        guard sessions.count < Self.maxActiveSessions else {
            throw InfiniteError.sessionLimitExceeded(Self.maxActiveSessions)
        }
        let selection = try sessionSelection(modelId: request.modelId)
        guard isModelSelectable(selection) else {
            throw InfiniteError.modelUnavailable(selection.rawValue)
        }
        let now = Self.timestamp()
        let id = UUID().uuidString
        let record = SessionRecord(
            id: id,
            selection: selection,
            label: request.label,
            createdAt: now,
            updatedAt: now
        )
        sessions[id] = record
        return sessionInfo(record)
    }

    public func append(
        sessionID: String,
        request: InfiniteAppendSessionRequest
    ) throws -> InfiniteAppendSessionResponse {
        // Session existence is checked first so a bad id maps to 404
        // session_not_found rather than 400 invalid_session_input (matches
        // generate()'s ordering).
        guard var record = sessions[sessionID] else {
            throw InfiniteError.sessionNotFound(sessionID)
        }
        guard !request.text.isEmpty else {
            throw InfiniteError.invalidSessionInput("append text must not be empty")
        }
        record.inputCharacters += request.text.count
        record.estimatedInputTokens += Self.estimatedTokens(for: request.text)
        record.accumulatedText += request.text
        record.updatedAt = Self.timestamp()
        sessions[sessionID] = record
        let contextWindow = record.selection.maxContextTokens
        if record.estimatedInputTokens > contextWindow {
            let detail = "~\(record.estimatedInputTokens) tokens vs \(contextWindow)-token native window for \(record.selection.rawValue)"
            infiniteEngineLogger.warning(
                "Infinite session \(sessionID, privacy: .public) over context: \(detail, privacy: .public); excess truncated at prefill (#180)."
            )
        }
        return InfiniteAppendSessionResponse(
            session: sessionInfo(record),
            appendedCharacters: request.text.count,
            estimatedAppendedTokens: Self.estimatedTokens(for: request.text)
        )
    }

    public func generate(
        sessionID: String,
        request: InfiniteGenerateSessionRequest
    ) async throws -> InfiniteGenerateSessionResponse {
        guard let record = sessions[sessionID] else {
            throw InfiniteError.sessionNotFound(sessionID)
        }
        if let maxTokens = request.maxTokens, maxTokens <= 0 {
            throw InfiniteError.invalidSessionInput("maxTokens must be greater than zero")
        }
        let selection = record.selection
        guard isModelSelectable(selection) else {
            throw InfiniteError.modelUnavailable(selection.rawValue)
        }
        // All three MLX rows run (Qwen #184, Gemma4 26B #184, Gemma4 E4B #186);
        // only retrieval mode has no MLX backend wired. Fail clearly here rather
        // than advertise a capability we can't run.
        guard selection.swiftRuntimeSupported else {
            throw InfiniteError.generationUnavailable(
                "model \(selection.rawValue) (\(selection.backendKind)) is not yet runnable by the Swift MLX runtime; the retrieval backend is not yet wired"
            )
        }

        // Lazily load the real backend on first generate for this model.
        let backend = try await loadBackend(for: selection)

        // Map any MLX/tokenizer runtime fault to a typed error so the route
        // returns a structured `generation_failed` 500, not a bare 500 the SDK
        // can only see as a transport error. Typed InfiniteErrors (e.g. the
        // native-window bound) pass through unchanged.
        let result: InfiniteGenerationResult
        do {
            result = try await backend.generate(
                context: record.accumulatedText,
                prompt: request.prompt ?? "",
                maxTokens: request.maxTokens ?? 256,
                nativeContextTokens: selection.nativeContextTokens,
                // Production sampling is fixed at 0.7; the `temperature` arg
                // exists so the parity test can request greedy (0.0) decoding.
                // `InfiniteGenerateSessionRequest` has no temperature field yet.
                temperature: 0.7
            )
        } catch let error as InfiniteError {
            throw error
        } catch {
            throw InfiniteError.generationFailed(error.localizedDescription)
        }

        if var updated = sessions[sessionID] {
            updated.updatedAt = Self.timestamp()
            sessions[sessionID] = updated
        }

        let base = resourceMetrics(for: selection)
        let metrics = InfiniteResourceMetrics(
            physicalMemoryBytes: base.physicalMemoryBytes,
            wiredMemoryLimitBytes: base.wiredMemoryLimitBytes,
            requiredRAMTier: base.requiredRAMTier,
            peakMemoryBytes: nil,
            prefillTokensPerSecond: nil,
            decodeTokensPerSecond: result.decodeTokensPerSecond,
            draftAcceptanceRate: nil
        )
        return InfiniteGenerateSessionResponse(
            sessionId: sessionID,
            text: result.text,
            finishReason: result.finishReason,
            resources: metrics
        )
    }

    public func checkpoint(
        sessionID: String,
        request: InfiniteCheckpointSessionRequest
    ) throws -> InfiniteCheckpointSessionResponse {
        guard var record = sessions[sessionID] else {
            throw InfiniteError.sessionNotFound(sessionID)
        }
        let checkpoint = InfiniteSessionCheckpoint(
            id: UUID().uuidString,
            label: request.label,
            createdAt: Self.timestamp(),
            inputCharacters: record.inputCharacters,
            estimatedInputTokens: record.estimatedInputTokens,
            resources: resourceMetrics(for: record.selection)
        )
        record.checkpoints.append(checkpoint)
        record.updatedAt = checkpoint.createdAt
        sessions[sessionID] = record
        return InfiniteCheckpointSessionResponse(
            session: sessionInfo(record),
            checkpoint: checkpoint
        )
    }

    public func deleteSession(id: String) throws -> InfiniteDeleteSessionResponse {
        guard sessions.removeValue(forKey: id) != nil else {
            throw InfiniteError.sessionNotFound(id)
        }
        return InfiniteDeleteSessionResponse(sessionId: id, deleted: true)
    }

    public func resetForRecordingBoundary() {
        // The engine-wide /v1/session/begin boundary is per recording.
        // Infinite long-context sessions are durable engine resources
        // and are cleaned up only through explicit /v1/infinite/sessions
        // lifecycle calls or process exit.
    }

    private func info(for selection: InfiniteModelSelection) -> InfiniteModelInfo {
        InfiniteModelInfo(
            id: selection.rawValue,
            displayName: selection.displayName,
            description: selection.description,
            tier: selection.tier,
            sizeBytes: selection.sizeBytes,
            loadState: loadState(for: selection),
            isActive: selection == activeModel,
            maxContextTokens: selection.maxContextTokens,
            nativeContextTokens: selection.nativeContextTokens,
            ramTier: selection.ramTier,
            backendKind: selection.backendKind,
            adapterKind: selection.adapterKind,
            huggingFaceID: selection.huggingFaceID,
            revision: selection.revision,
            requiresAppleSilicon: true,
            evidenceRef: selection.evidenceRef
        )
    }

    private func loadState(for selection: InfiniteModelSelection) -> ModelLoadState {
        guard isModelSelectable(selection) else { return .unavailable }
        if loadedModel == selection {
            return .loaded
        }
        if InfiniteCacheProbe.isCached(selection.descriptor) {
            return .cached
        }
        return .available
    }

    private func isModelSelectable(_ selection: InfiniteModelSelection) -> Bool {
        #if arch(arm64)
        return InfiniteRAMTier.current.supports(required: selection.requiredRAMTier)
        #else
        return false
        #endif
    }

    private var state: String {
        if isLoaded {
            return "ready"
        }
        if preparedBackend?.selection == activeModel {
            return "adapter_ready"
        }
        return "idle"
    }

    private func sessionSelection(modelId: String?) throws -> InfiniteModelSelection {
        guard let modelId, !modelId.isEmpty else {
            return activeModel
        }
        guard let selection = InfiniteModelSelection(rawValue: modelId) else {
            throw InfiniteError.invalidModel(modelId)
        }
        return selection
    }

    private func sessionInfo(for id: String) throws -> InfiniteSessionInfo {
        guard let record = sessions[id] else {
            throw InfiniteError.sessionNotFound(id)
        }
        return sessionInfo(record)
    }

    private func sessionInfo(_ record: SessionRecord) -> InfiniteSessionInfo {
        InfiniteSessionInfo(
            id: record.id,
            modelId: record.selection.rawValue,
            label: record.label,
            state: "open",
            createdAt: record.createdAt,
            updatedAt: record.updatedAt,
            contextWindowTokens: record.selection.maxContextTokens,
            inputCharacters: record.inputCharacters,
            estimatedInputTokens: record.estimatedInputTokens,
            checkpointCount: record.checkpoints.count,
            cleanupPolicy: Self.cleanupPolicy,
            resources: resourceMetrics(for: record.selection)
        )
    }

    private func resourceMetrics(
        for selection: InfiniteModelSelection
    ) -> InfiniteResourceMetrics {
        InfiniteResourceMetrics(
            physicalMemoryBytes: Self.int64Clamping(ProcessInfo.processInfo.physicalMemory),
            wiredMemoryLimitBytes: selection.requiredRAMTier.minimumPhysicalMemoryBytes,
            requiredRAMTier: selection.ramTier,
            peakMemoryBytes: nil,
            prefillTokensPerSecond: nil,
            decodeTokensPerSecond: nil,
            draftAcceptanceRate: nil
        )
    }

    /// Shared formatter. `ISO8601DateFormatter.string(from:)` is thread-safe
    /// and the engine is an actor, so a single instance is reused instead of
    /// allocating + configuring one per session call (mirrors APIServer).
    private static let isoTimestampFormatter = ISO8601DateFormatter()

    private static func timestamp() -> String {
        isoTimestampFormatter.string(from: Date())
    }

    private static func estimatedTokens(for text: String) -> Int {
        max(1, Int((Double(text.count) / 4.0).rounded(.up)))
    }

    /// Saturates (does not wrap) to `Int64.max`. Physical memory is never
    /// large enough to fire on real hardware; the clamp is defensive.
    private static func int64Clamping(_ value: UInt64) -> Int64 {
        if value > UInt64(Int64.max) {
            return Int64.max
        }
        return Int64(value)
    }
}

private struct SessionRecord: Sendable {
    let id: String
    let selection: InfiniteModelSelection
    let label: String?
    let createdAt: String
    var updatedAt: String
    var inputCharacters: Int = 0
    var estimatedInputTokens: Int = 0
    var checkpoints: [InfiniteSessionCheckpoint] = []
    /// Raw appended context, fed to the backend at generate time. (Phase 7
    /// rebuilds the prefill per call; per-session KV reuse is a #182 follow-up.)
    var accumulatedText: String = ""
}
