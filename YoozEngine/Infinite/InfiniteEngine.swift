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
    /// `session` has no checkpoint matching `checkpoint` (an explicit id
    /// that doesn't exist, or no checkpoint at all when one was implied,
    /// e.g. resuming/forking a session that was never checkpointed).
    case checkpointNotFound(session: String, checkpoint: String)
    /// A checkpoint's on-disk contents failed
    /// `InfiniteSessionStore.verify(manifest:against:session:checkpoint:)`
    /// at resume time (schema/model/tokenizer mismatch, or a tampered
    /// `tokens.bin`). Carries the `SessionIntegrityError.description`.
    case checkpointIntegrity(String)
    /// `session` is currently `.generating` (a generate is in flight);
    /// concurrent append/generate/checkpoint/resume/fork/delete calls for
    /// the same session must wait rather than race the backend actor.
    case sessionBusy(String)

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
        case .checkpointNotFound(let session, let checkpoint):
            return "Unknown Infinite checkpoint '\(checkpoint)' for session '\(session)'"
        case .checkpointIntegrity(let reason):
            return "Infinite checkpoint integrity check failed: \(reason)"
        case .sessionBusy(let id):
            return "Infinite session '\(id)' is busy (a generate is in flight)"
        }
    }
}

public actor InfiniteEngine {
    public static let shared = InfiniteEngine()

    public nonisolated static let maxActiveSessions = 16
    /// Most sessions allowed live (KV cache resident) at once (engine#266).
    /// Opening/resuming a session beyond this budget auto-parks the
    /// least-recently-touched other hot session first — see
    /// `enforceHotBudget`. Independent of `maxActiveSessions`: a session can
    /// be tracked (counted against the 16 cap) while parked, contributing
    /// nothing to RAM.
    static let maxHotSessions = 2
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
    /// Durable checkpoint storage (engine#266). Reads `YOOZ_INFINITE_SESSIONS_DIR`
    /// at init time — same env-override convention as `InfiniteSessionStoreTests`
    /// — so tests can point a fresh `InfiniteEngine()` at an isolated temp root.
    private let store: InfiniteSessionStore

    init(
        backendAdapter: any InfiniteBackendAdapter = CatalogInfiniteBackendAdapter(),
        store: InfiniteSessionStore = InfiniteSessionStore()
    ) {
        self.backendAdapter = backendAdapter
        self.store = store
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
            // Backend eviction parking (engine#266): the previous backend
            // (if any, and if it's actually a different model) is about to
            // be orphaned by the reassignment below. Checkpoint + release
            // every session still hot under it first, so their KV state
            // survives on disk instead of silently vanishing — narrowing
            // the "last write wins" gap `deleteSession`'s doc comment used
            // to describe as unconditional.
            if let previous = loadedBackend, previous.selection != selection {
                await parkHotSessions(usingSelection: previous.selection, backend: previous)
            }
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
    ) async throws -> InfiniteAppendSessionResponse {
        // Session existence is checked first so a bad id maps to 404
        // session_not_found rather than 400 invalid_session_input (matches
        // generate()'s ordering).
        let (record0, wasParked) = try beginBusyOperation(sessionID: sessionID)
        var record = record0
        do {
            guard !request.text.isEmpty else {
                throw InfiniteError.invalidSessionInput("append text must not be empty")
            }
            let selection = record.selection
            guard isModelSelectable(selection) else {
                throw InfiniteError.modelUnavailable(selection.rawValue)
            }
            try requireSwiftRuntimeSupport(selection)

            // Loads the real backend lazily — appendTokens does real GPU work
            // (chunked prefill onto the session's durable KV cache), unlike the
            // pre-#265 bookkeeping-only append.
            let backend = try await loadBackend(for: selection)
            record = try await ensureLiveBackendState(record: record, wasParked: wasParked, backend: backend)

            let outcome: SessionAppendOutcome
            do {
                outcome = try await backend.appendTokens(id: sessionID, text: request.text)
            } catch is CancellationError {
                infiniteEngineLogger.debug(
                    "Infinite append cancelled for session \(sessionID, privacy: .public)"
                )
                throw CancellationError()
            } catch let error as InfiniteError {
                throw error
            } catch {
                infiniteEngineLogger.error(
                    "Infinite append failed for session \(sessionID, privacy: .public): \(error.localizedDescription, privacy: .public)"
                )
                throw InfiniteError.generationFailed(error.localizedDescription)
            }

            record.inputCharacters += request.text.count
            record.tokenCount = outcome.totalTokenCount
            record.updatedAt = Self.timestamp()
            record.state = .open
            sessions[sessionID] = record
            let contextWindow = record.selection.maxContextTokens
            if record.tokenCount > contextWindow {
                let detail = "\(record.tokenCount) tokens vs \(contextWindow)-token native window for \(record.selection.rawValue)"
                infiniteEngineLogger.warning(
                    "Infinite session \(sessionID, privacy: .public) over context: \(detail, privacy: .public); excess truncated at prefill (#180)."
                )
            }
            return InfiniteAppendSessionResponse(
                session: sessionInfo(record),
                appendedCharacters: request.text.count,
                estimatedAppendedTokens: outcome.appendedTokenCount
            )
        } catch {
            endBusyOperation(sessionID: sessionID, finalState: record.hasLiveBackendState ? .open : .parked)
            throw error
        }
    }

    public func generate(
        sessionID: String,
        request: InfiniteGenerateSessionRequest
    ) async throws -> InfiniteGenerateSessionResponse {
        let (record0, wasParked) = try beginBusyOperation(sessionID: sessionID)
        var record = record0
        do {
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
            try requireSwiftRuntimeSupport(selection)

            // Lazily load the real backend on first generate for this model.
            let backend = try await loadBackend(for: selection)
            record = try await ensureLiveBackendState(record: record, wasParked: wasParked, backend: backend)

            // Map any MLX/tokenizer runtime fault to a typed error so the route
            // returns a structured `generation_failed` 500, not a bare 500 the SDK
            // can only see as a transport error. Typed InfiniteErrors (e.g. the
            // native-window bound) pass through unchanged.
            let outcome: SessionGenerateOutcome
            do {
                outcome = try await backend.generateSession(
                    id: sessionID,
                    prompt: request.prompt ?? "",
                    maxTokens: request.maxTokens ?? 256,
                    // Production sampling defaults to 0.7; `temperature` lets a
                    // caller (the live parity test) request greedy (0.0) decoding.
                    temperature: request.temperature ?? 0.7
                )
            } catch let error as InfiniteError {
                infiniteEngineLogger.error(
                    "Infinite generation failed for session \(sessionID, privacy: .public): \(error.localizedDescription, privacy: .public)"
                )
                throw error
            } catch is CancellationError {
                // Task cancellation (client disconnect, caller teardown — the
                // admission gate's `checkpoint()` is the newest cancellation
                // source, engine#228) is not a model fault: rethrow it typed so
                // it never masquerades as `generation_failed` in logs or on the
                // wire. Debug-level on purpose.
                infiniteEngineLogger.debug(
                    "Infinite generation cancelled for session \(sessionID, privacy: .public)"
                )
                throw CancellationError()
            } catch {
                // Log here — this used to be the engine's only completely silent
                // failure path (the HTTP response body was the sole trace of a
                // real MLX/Metal fault during Infinite generation).
                infiniteEngineLogger.error(
                    "Infinite generation failed for session \(sessionID, privacy: .public): \(error.localizedDescription, privacy: .public)"
                )
                throw InfiniteError.generationFailed(error.localizedDescription)
            }

            record.tokenCount = outcome.totalTokenCount
            record.updatedAt = Self.timestamp()
            record.state = .open
            sessions[sessionID] = record

            let base = resourceMetrics(for: selection)
            let metrics = InfiniteResourceMetrics(
                physicalMemoryBytes: base.physicalMemoryBytes,
                wiredMemoryLimitBytes: base.wiredMemoryLimitBytes,
                requiredRAMTier: base.requiredRAMTier,
                peakMemoryBytes: nil,
                prefillTokensPerSecond: outcome.prefillTokensPerSecond,
                decodeTokensPerSecond: outcome.decodeTokensPerSecond,
                draftAcceptanceRate: nil
            )
            return InfiniteGenerateSessionResponse(
                sessionId: sessionID,
                text: outcome.text,
                finishReason: outcome.finishReason,
                resources: metrics
            )
        } catch {
            endBusyOperation(sessionID: sessionID, finalState: record.hasLiveBackendState ? .open : .parked)
            throw error
        }
    }

    /// Checkpoints a session's live KV cache to disk (`InfiniteSessionStore`),
    /// pinned with the model identity, tokenizer hash, and token record a
    /// later `resume`/`fork` verifies against. `request.park == true` also
    /// releases the session's live state afterward (engine#266).
    public func checkpoint(
        sessionID: String,
        request: InfiniteCheckpointSessionRequest
    ) async throws -> InfiniteCheckpointSessionResponse {
        let (record0, wasParked) = try beginBusyOperation(sessionID: sessionID)
        var record = record0
        do {
            let selection = record.selection
            guard isModelSelectable(selection) else {
                throw InfiniteError.modelUnavailable(selection.rawValue)
            }
            try requireSwiftRuntimeSupport(selection)

            let backend = try await loadBackend(for: selection)
            record = try await ensureLiveBackendState(record: record, wasParked: wasParked, backend: backend)

            let parentCheckpointId = record.lastCheckpointId
            let outcome = try await performCheckpoint(
                sessionID: sessionID,
                record: record,
                backend: backend,
                label: request.label,
                park: request.park ?? false
            )
            return InfiniteCheckpointSessionResponse(
                session: sessionInfo(outcome.record),
                checkpoint: outcome.checkpoint,
                sizeBytes: outcome.result.sizeBytes,
                tokenCount: outcome.result.tokenRecord.count,
                durationSeconds: outcome.result.durationSeconds,
                parentCheckpointId: parentCheckpointId
            )
        } catch {
            endBusyOperation(sessionID: sessionID, finalState: record.hasLiveBackendState ? .open : .parked)
            throw error
        }
    }

    /// Resumes a session from a checkpoint (engine#266): integrity-verifies
    /// the checkpoint against the model actually loaded, loads its KV cache,
    /// and reinstalls it as the session's live state. A no-op success when
    /// the session is already `.open` and no explicit `checkpointId` was
    /// requested. Works even if `sessionID` was never touched by this
    /// process (rehydrates a minimal record from the checkpoint's own
    /// manifest) — a checkpoint is meant to survive process exit, not just
    /// RAM eviction, so a fresh engine process must still be able to resume it.
    public func resume(
        sessionID: String,
        request: InfiniteResumeSessionRequest
    ) async throws -> InfiniteSessionInfo {
        let alreadyTracked = sessions[sessionID] != nil
        guard let existing = try lookupOrRehydrate(sessionID: sessionID) else {
            throw InfiniteError.sessionNotFound(sessionID)
        }
        guard existing.state != .generating else {
            throw InfiniteError.sessionBusy(sessionID)
        }
        if !alreadyTracked {
            guard sessions.count < Self.maxActiveSessions else {
                throw InfiniteError.sessionLimitExceeded(Self.maxActiveSessions)
            }
        }
        sessions[sessionID] = existing

        if existing.state == .open, request.checkpointId == nil {
            return sessionInfo(existing)
        }

        var record = existing
        let priorState = record.state
        record.state = .generating
        sessions[sessionID] = record
        do {
            guard isModelSelectable(record.selection) else {
                throw InfiniteError.modelUnavailable(record.selection.rawValue)
            }
            try requireSwiftRuntimeSupport(record.selection)

            // Resolve + read the checkpoint before paying for a model load,
            // so an unknown checkpoint id 404s fast.
            let (checkpointId, manifest) = try resolveCheckpointForResume(
                sessionID: sessionID, requested: request.checkpointId
            )
            try await enforceHotBudget(admitting: sessionID)
            let backend = try await loadBackend(for: record.selection)
            record = try await installResumedCheckpoint(
                record: record, backend: backend, checkpointId: checkpointId, manifest: manifest
            )
            record.state = .open
            sessions[sessionID] = record
            return sessionInfo(record)
        } catch {
            record.state = priorState
            sessions[sessionID] = record
            throw error
        }
    }

    /// Forks a checkpoint into a brand-new, independent session
    /// (engine#266): clones the on-disk checkpoint (`InfiniteSessionStore.fork`)
    /// under a fresh session id and does NOT auto-resume it (the new
    /// session starts `.parked`). Respects the 16-session cap. A hot,
    /// never-checkpointed source session takes one implicit checkpoint first.
    public func fork(
        sessionID: String,
        request: InfiniteForkSessionRequest
    ) async throws -> InfiniteSessionInfo {
        let trackedSource = sessions[sessionID]
        if let trackedSource {
            guard trackedSource.state != .generating else {
                throw InfiniteError.sessionBusy(sessionID)
            }
        } else {
            guard try store.latestCheckpoint(session: sessionID) != nil else {
                throw InfiniteError.sessionNotFound(sessionID)
            }
        }
        guard sessions.count < Self.maxActiveSessions else {
            throw InfiniteError.sessionLimitExceeded(Self.maxActiveSessions)
        }

        let checkpointId = try await resolveForkCheckpointId(
            sessionID: sessionID, requested: request.checkpointId, trackedSource: trackedSource
        )

        let newSessionID = UUID().uuidString
        let newCheckpointID = UUID().uuidString
        do {
            try store.fork(
                session: sessionID,
                checkpoint: checkpointId,
                intoSession: newSessionID,
                newCheckpointId: newCheckpointID,
                label: request.label
            )
        } catch is InfiniteSessionStoreError {
            throw InfiniteError.checkpointNotFound(session: sessionID, checkpoint: checkpointId)
        }

        let forkedManifest = try store.readManifest(session: newSessionID, checkpoint: newCheckpointID)
        let sourceSelection: InfiniteModelSelection
        if let trackedSource {
            sourceSelection = trackedSource.selection
        } else if let selection = InfiniteModelSelection(rawValue: forkedManifest.model.selectionId) {
            sourceSelection = selection
        } else {
            throw InfiniteError.invalidModel(forkedManifest.model.selectionId)
        }

        let now = Self.timestamp()
        let checkpointInfo = InfiniteSessionCheckpoint(
            id: newCheckpointID,
            label: request.label,
            createdAt: now,
            // Character-count bookkeeping lives only on the in-memory
            // SessionRecord, not the manifest, so a forked (or rehydrated)
            // checkpoint's own char count is unknown; 0 is an honest gap,
            // not a wrong answer.
            inputCharacters: 0,
            estimatedInputTokens: forkedManifest.tokenCount,
            resources: resourceMetrics(for: sourceSelection)
        )
        let newRecord = SessionRecord(
            id: newSessionID,
            selection: sourceSelection,
            label: request.label,
            createdAt: now,
            updatedAt: now,
            tokenCount: forkedManifest.tokenCount,
            checkpoints: [checkpointInfo],
            state: .parked,
            lastCheckpointId: newCheckpointID,
            hasLiveBackendState: false
        )
        sessions[newSessionID] = newRecord
        return sessionInfo(newRecord)
    }

    public func deleteSession(id: String) async throws -> InfiniteDeleteSessionResponse {
        guard let record = sessions[id] else {
            throw InfiniteError.sessionNotFound(id)
        }
        guard record.state != .generating else {
            throw InfiniteError.sessionBusy(id)
        }
        sessions.removeValue(forKey: id)
        // Release the backend's live KV cache for this session, but only
        // when the resident backend still matches the session's own model
        // — if a different model has since been loaded, backend-eviction
        // parking (`parkHotSessions`) already checkpointed + released it.
        if let backend = loadedBackend, loadedModel == record.selection {
            await backend.releaseSession(id: id)
        }
        try store.deleteSession(id)
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

    /// Shared by `append` and `generate`: both do real backend work now
    /// (engine#265), so both need the same "can the Swift MLX runtime
    /// actually run this model" guard before calling `loadBackend`.
    private func requireSwiftRuntimeSupport(_ selection: InfiniteModelSelection) throws {
        guard selection.swiftRuntimeSupported else {
            throw InfiniteError.generationUnavailable(
                "model \(selection.rawValue) (\(selection.backendKind)) is not yet runnable " +
                    "by the Swift MLX runtime; the retrieval backend is not yet wired"
            )
        }
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
            state: record.state.rawValue,
            createdAt: record.createdAt,
            updatedAt: record.updatedAt,
            contextWindowTokens: record.selection.maxContextTokens,
            inputCharacters: record.inputCharacters,
            estimatedInputTokens: record.tokenCount,
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

    /// Saturates (does not wrap) to `Int64.max`. Physical memory is never
    /// large enough to fire on real hardware; the clamp is defensive.
    private static func int64Clamping(_ value: UInt64) -> Int64 {
        if value > UInt64(Int64.max) {
            return Int64.max
        }
        return Int64(value)
    }

    // MARK: - Busy guard (engine#266)

    /// Existence + busy-guard + "claim exclusive access for the duration of
    /// this call" — shared by `append`/`generate`/`checkpoint`. Marks the
    /// session `.generating` immediately (before any `await`) so a second
    /// concurrent call for the same session — actor reentrancy during this
    /// call's own awaits, e.g. a client racing two requests — observes
    /// `sessionBusy` instead of interleaving with the backend actor's own
    /// in-flight work. Callers MUST call `endBusyOperation` on every exit path.
    private func beginBusyOperation(sessionID: String) throws -> (record: SessionRecord, wasParked: Bool) {
        guard var record = sessions[sessionID] else {
            throw InfiniteError.sessionNotFound(sessionID)
        }
        guard record.state != .generating else {
            throw InfiniteError.sessionBusy(sessionID)
        }
        let wasParked = record.state == .parked
        record.state = .generating
        sessions[sessionID] = record
        return (record, wasParked)
    }

    /// Restores `sessionID`'s state after `beginBusyOperation`, on both the
    /// success and failure paths (`finalState` is the caller's choice either
    /// way — success sets `.open`/`.parked` deliberately; failure passes
    /// whatever the pre-operation state effectively was, so a failed call
    /// never leaves a session stuck `.generating`). A no-op if the session
    /// was deleted mid-operation.
    private func endBusyOperation(sessionID: String, finalState: SessionState) {
        guard var record = sessions[sessionID] else { return }
        record.state = finalState
        sessions[sessionID] = record
    }

    // MARK: - Hot-session lifecycle (engine#266)

    /// Ensures `record`'s session has real backend-resident state before the
    /// caller does its own work: resumes from the latest checkpoint if it
    /// was parked (enforcing the hot-session budget first), or opens a
    /// fresh — possibly empty — backend session the first time it is ever
    /// touched (a session checkpointed before its first append/generate is
    /// trivial but must still produce a loadable `cache.safetensors`).
    /// Returns `record` unchanged if it already has live backend state.
    private func ensureLiveBackendState(
        record: SessionRecord,
        wasParked: Bool,
        backend: MLXInfiniteBackend
    ) async throws -> SessionRecord {
        if wasParked {
            let (checkpointId, manifest) = try resolveCheckpointForResume(
                sessionID: record.id, requested: nil
            )
            try await enforceHotBudget(admitting: record.id)
            return try await installResumedCheckpoint(
                record: record, backend: backend, checkpointId: checkpointId, manifest: manifest
            )
        }
        guard !record.hasLiveBackendState else { return record }
        try await enforceHotBudget(admitting: record.id)
        await backend.openSession(id: record.id)
        var updated = record
        updated.hasLiveBackendState = true
        return updated
    }

    /// Enforces `maxHotSessions` (engine#266): if `id` is about to become
    /// (or remain) hot alongside `maxHotSessions` other non-parked sessions,
    /// checkpoints + releases the least-recently-touched OTHER `.open`
    /// session first (by `updatedAt`). Never picks a `.generating` session —
    /// evicting mid-flight would race the backend actor's own in-flight
    /// call for it. Throws `sessionBusy` if every other hot session is
    /// currently generating (nothing safe to evict right now).
    private func enforceHotBudget(admitting id: String) async throws {
        let hotOthers = sessions.values.filter { $0.id != id && $0.state != .parked }
        guard hotOthers.count >= Self.maxHotSessions else { return }
        let evictable = hotOthers
            .filter { $0.state == .open }
            .sorted { $0.updatedAt < $1.updatedAt }
        guard let victim = evictable.first else {
            throw InfiniteError.sessionBusy(id)
        }
        let backend = try await loadBackend(for: victim.selection)
        _ = try await performCheckpoint(
            sessionID: victim.id, record: victim, backend: backend, label: nil, park: true
        )
    }

    /// Backend eviction parking (engine#266): checkpoints + releases every
    /// session still `.open` under `selection` before its backend instance
    /// (`backend`) is orphaned by `loadBackend` switching to a different
    /// model. Skips `.generating` sessions — one mid-flight on `backend`
    /// holds its own reference to it directly (captured before the swap),
    /// so it keeps running to completion against the now-orphaned instance;
    /// parking it here would race that in-flight call. That is a narrow,
    /// documented residual gap (not solved by this phase): a session
    /// evicted-from-hot-tracking while genuinely mid-generate loses its
    /// state once the orphaned backend deallocates, exactly as before this
    /// phase's "last write wins" eviction. Best-effort: a single session's
    /// checkpoint failure is logged and does not block parking the rest or
    /// throw into the caller's own unrelated model load.
    private func parkHotSessions(usingSelection selection: InfiniteModelSelection, backend: MLXInfiniteBackend) async {
        let victims = sessions.values.filter { $0.selection == selection && $0.state == .open }
        for victim in victims {
            do {
                _ = try await performCheckpoint(
                    sessionID: victim.id, record: victim, backend: backend, label: nil, park: true
                )
            } catch {
                let detail = "session \(victim.id): \(error.localizedDescription)"
                infiniteEngineLogger.error(
                    "Infinite backend-eviction checkpoint failed for \(detail, privacy: .public)"
                )
            }
        }
    }

    // MARK: - Checkpoint disk I/O (engine#266)

    /// Result of `performCheckpoint`: the wire DTO, the raw disk-write
    /// facts (`CheckpointWriteResult`), and the session record updated to
    /// reflect the new checkpoint.
    private struct CheckpointOutcome {
        let checkpoint: InfiniteSessionCheckpoint
        let result: CheckpointWriteResult
        let record: SessionRecord
    }

    /// Writes a checkpoint (`cache.safetensors` + `tokens.bin` +
    /// `manifest.json` via the MLX backend + `InfiniteSessionStore`) and
    /// applies the result to `record`'s bookkeeping, optionally parking it.
    /// Shared by the public `checkpoint` route, hot-budget eviction,
    /// backend-eviction parking, and fork's implicit pre-checkpoint.
    @discardableResult
    private func performCheckpoint(
        sessionID: String,
        record: SessionRecord,
        backend: MLXInfiniteBackend,
        label: String?,
        park: Bool
    ) async throws -> CheckpointOutcome {
        let result = try await writeCheckpoint(sessionID: sessionID, record: record, backend: backend, label: label)
        let checkpointInfo = InfiniteSessionCheckpoint(
            id: result.checkpointId,
            label: label,
            createdAt: Self.timestamp(),
            inputCharacters: record.inputCharacters,
            estimatedInputTokens: result.tokenRecord.count,
            resources: resourceMetrics(for: record.selection)
        )
        var updated = record
        updated.checkpoints.append(checkpointInfo)
        updated.lastCheckpointId = result.checkpointId
        updated.tokenCount = result.tokenRecord.count
        updated.updatedAt = Self.timestamp()
        updated.state = park ? .parked : .open
        sessions[sessionID] = updated
        if park {
            await backend.releaseSession(id: sessionID)
        }
        return CheckpointOutcome(checkpoint: checkpointInfo, result: result, record: updated)
    }

    /// Branches + persists session `sessionID`'s live KV cache to a fresh
    /// checkpoint directory (`InfiniteSessionStore.checkpointDirectory`),
    /// pinning the model identity, tokenizer hash, and token record a later
    /// `resume`/`fork` verifies against. Pure disk + backend I/O — does not
    /// touch `sessions[sessionID]`; `performCheckpoint` owns that.
    private func writeCheckpoint(
        sessionID: String,
        record: SessionRecord,
        backend: MLXInfiniteBackend,
        label: String?
    ) async throws -> CheckpointWriteResult {
        let checkpointId = UUID().uuidString
        let checkpointDir = try store.checkpointDirectory(session: sessionID, checkpoint: checkpointId)
        let cacheURL = checkpointDir.appendingPathComponent("cache.safetensors", isDirectory: false)

        let start = Date.timeIntervalSinceReferenceDate
        let snapshot = try await backend.checkpointSession(id: sessionID, cacheURL: cacheURL)
        let durationSeconds = Date.timeIntervalSinceReferenceDate - start
        let tokenizerHash = try await backend.tokenizerHash()

        try store.writeTokens(snapshot.tokenRecord, session: sessionID, checkpoint: checkpointId)
        let tokensData = try InfiniteSessionStore.encodeTokens(snapshot.tokenRecord)
        let manifest = InfiniteSessionManifest(
            model: ModelIdentity(
                selectionId: record.selection.rawValue,
                repoId: record.selection.huggingFaceID ?? "",
                revision: record.selection.revision ?? ""
            ),
            tokenizerHash: tokenizerHash,
            tokenCount: snapshot.tokenRecord.count,
            pendingTokenId: snapshot.pendingToken,
            tokenIdsSHA256: InfiniteSessionStore.sha256Hex(tokensData),
            cacheConfig: SessionKnobs(),
            parentCheckpointId: record.lastCheckpointId,
            label: label
        )
        try store.writeManifest(manifest, session: sessionID, checkpoint: checkpointId)

        let attrs = try? FileManager.default.attributesOfItem(atPath: cacheURL.path)
        let sizeBytes = (attrs?[.size] as? NSNumber)?.int64Value ?? 0

        return CheckpointWriteResult(
            checkpointId: checkpointId,
            tokenRecord: snapshot.tokenRecord,
            pendingToken: snapshot.pendingToken,
            sizeBytes: sizeBytes,
            durationSeconds: durationSeconds
        )
    }

    // MARK: - Resume disk I/O (engine#266)

    /// Resolves which checkpoint a resume (explicit or auto-rehydrate)
    /// should use — an explicit id, or the session's latest — and reads its
    /// manifest. Pure disk I/O, no backend/MLX work, so an unknown
    /// checkpoint id 404s before paying for a model load.
    private func resolveCheckpointForResume(
        sessionID: String,
        requested: String?
    ) throws -> (id: String, manifest: InfiniteSessionManifest) {
        let checkpointId: String
        if let requested {
            checkpointId = requested
        } else if let latest = try store.latestCheckpoint(session: sessionID)?.id {
            checkpointId = latest
        } else {
            throw InfiniteError.checkpointNotFound(session: sessionID, checkpoint: "<none>")
        }
        do {
            let manifest = try store.readManifest(session: sessionID, checkpoint: checkpointId)
            return (checkpointId, manifest)
        } catch {
            throw InfiniteError.checkpointNotFound(session: sessionID, checkpoint: checkpointId)
        }
    }

    /// Integrity-verifies (`InfiniteSessionStore.verify`) the resolved
    /// checkpoint against `backend`'s live facts, loads its KV cache, and
    /// installs it. Returns `record` updated to reflect the resumed state;
    /// does not touch `.state` — callers own that.
    private func installResumedCheckpoint(
        record: SessionRecord,
        backend: MLXInfiniteBackend,
        checkpointId: String,
        manifest: InfiniteSessionManifest
    ) async throws -> SessionRecord {
        let liveFacts = LiveSessionFacts(
            schemaVersion: InfiniteSessionManifest.currentSchemaVersion,
            repoId: record.selection.huggingFaceID ?? "",
            revision: record.selection.revision ?? "",
            tokenizerHash: try await backend.tokenizerHash()
        )
        do {
            try store.verify(
                manifest: manifest, against: liveFacts, session: record.id, checkpoint: checkpointId
            )
        } catch let error as SessionIntegrityError {
            throw InfiniteError.checkpointIntegrity(error.description)
        } catch {
            // `verify` also re-reads tokens.bin from disk to recompute its
            // sha256 — a missing/unreadable file surfaces as a plain
            // CocoaError/POSIXError, not a `SessionIntegrityError`. Still an
            // integrity failure from the caller's point of view.
            throw InfiniteError.checkpointIntegrity(
                "failed to verify checkpoint \(checkpointId): \(error.localizedDescription)"
            )
        }

        let tokenRecord = try store.readTokens(session: record.id, checkpoint: checkpointId)
        let cacheURL = try store.checkpointDirectory(session: record.id, checkpoint: checkpointId)
            .appendingPathComponent("cache.safetensors", isDirectory: false)
        try await backend.resumeSession(
            id: record.id,
            cacheURL: cacheURL,
            tokenRecord: tokenRecord,
            pendingToken: manifest.pendingTokenId
        )

        var updated = record
        updated.tokenCount = tokenRecord.count
        updated.lastCheckpointId = checkpointId
        updated.hasLiveBackendState = true
        updated.updatedAt = Self.timestamp()
        return updated
    }

    /// Looks up `sessionID` in the in-memory `sessions` dict; if absent (the
    /// engine process restarted since this session was last active — the
    /// `SessionRecord` bookkeeping row is in-memory only, but a checkpoint
    /// durably survives on disk, per engine#266's core promise that a
    /// session survives process exit, not just RAM eviction), reconstructs a
    /// minimal `.parked` record from its most recent on-disk checkpoint.
    /// Returns `nil` only when there is truly no record and no checkpoint at
    /// all for `sessionID`.
    private func lookupOrRehydrate(sessionID: String) throws -> SessionRecord? {
        if let record = sessions[sessionID] { return record }
        guard let (checkpointId, manifest) = try store.latestCheckpoint(session: sessionID) else {
            return nil
        }
        guard let selection = InfiniteModelSelection(rawValue: manifest.model.selectionId) else {
            return nil
        }
        let diskCheckpoints = try store.listCheckpoints(session: sessionID)
        let now = Self.timestamp()
        return SessionRecord(
            id: sessionID,
            selection: selection,
            label: manifest.label,
            createdAt: Self.isoTimestampFormatter.string(from: diskCheckpoints.first?.manifest.createdAt ?? manifest.createdAt),
            updatedAt: now,
            tokenCount: manifest.tokenCount,
            checkpoints: diskCheckpoints.map { entry in
                InfiniteSessionCheckpoint(
                    id: entry.id,
                    label: entry.manifest.label,
                    createdAt: Self.isoTimestampFormatter.string(from: entry.manifest.createdAt),
                    // Character-count bookkeeping is in-memory only (see
                    // `fork`'s identical caveat) — unknown after rehydration.
                    inputCharacters: 0,
                    estimatedInputTokens: entry.manifest.tokenCount,
                    resources: resourceMetrics(for: selection)
                )
            },
            state: .parked,
            lastCheckpointId: checkpointId,
            hasLiveBackendState: false
        )
    }

    /// Resolves which checkpoint `fork` should clone: an explicit id, the
    /// source's latest on disk, or — when neither exists and the source is
    /// hot (`.open`) — an implicit checkpoint taken on the spot (a
    /// never-checkpointed but live session should still be forkable).
    private func resolveForkCheckpointId(
        sessionID: String,
        requested: String?,
        trackedSource: SessionRecord?
    ) async throws -> String {
        if let requested {
            return requested
        }
        if let latest = try store.latestCheckpoint(session: sessionID)?.id {
            return latest
        }
        guard let trackedSource, trackedSource.state == .open else {
            throw InfiniteError.checkpointNotFound(session: sessionID, checkpoint: "<none>")
        }

        let (record0, wasParked) = try beginBusyOperation(sessionID: sessionID)
        var record = record0
        do {
            let selection = record.selection
            guard isModelSelectable(selection) else {
                throw InfiniteError.modelUnavailable(selection.rawValue)
            }
            try requireSwiftRuntimeSupport(selection)
            let backend = try await loadBackend(for: selection)
            record = try await ensureLiveBackendState(record: record, wasParked: wasParked, backend: backend)
            let outcome = try await performCheckpoint(
                sessionID: sessionID, record: record, backend: backend, label: nil, park: false
            )
            return outcome.result.checkpointId
        } catch {
            endBusyOperation(sessionID: sessionID, finalState: record.hasLiveBackendState ? .open : .parked)
            throw error
        }
    }

    // MARK: - Test-only escape hatches (engine#266 busy-guard unit gate)

    /// Directly marks `id`'s session `.generating`, so unit tests (no model
    /// needed) can exercise the busy-guard 409 paths without a real
    /// in-flight generate. Internal, not public API.
    func setGeneratingForTesting(id: String) {
        guard var record = sessions[id] else { return }
        record.state = .generating
        sessions[id] = record
    }

    /// Directly marks `id`'s session `.parked` with no backend/disk work, so
    /// unit tests can exercise resume/fork paths against a parked record
    /// without a real checkpoint round trip. Internal, not public API.
    func setParkedForTesting(id: String) {
        guard var record = sessions[id] else { return }
        record.state = .parked
        sessions[id] = record
    }

    /// Installs `backend` directly as the resident backend for `selection`,
    /// bypassing `loadBackend`'s real `MLXInfiniteBackend.load
    /// (selection.descriptor)` call. Lets a live test drive `InfiniteEngine`'s
    /// real append/generate/checkpoint/resume/fork orchestration against a
    /// small pinned test model loaded under a cosmetic `selection` label —
    /// mirrors `MLXInfiniteBackend`'s own test comment ("the selection label
    /// is cosmetic ... never stored in InfiniteEngine.loadedBackend") except
    /// here it deliberately IS, so the engine's real RAM-tier gating
    /// (`isModelSelectable`) still runs against a widely-runnable label
    /// (e.g. `.gemma4E4B1M`, reduced tier) while the actual downloaded
    /// weights are the tiny pinned test repo. Internal, not public API.
    func installBackendForTesting(_ backend: MLXInfiniteBackend, as selection: InfiniteModelSelection) {
        loadedBackend = backend
        loadedModel = selection
    }
}

/// Lifecycle state of one Infinite session (engine#266). Wire value is the
/// lowercased case name (`SessionState.rawValue`) via `InfiniteSessionInfo.state`.
private enum SessionState: String, Sendable {
    /// Backend-resident (or never yet touched) and idle — a normal target
    /// for append/generate/checkpoint.
    case open
    /// Checkpointed to disk with its live KV cache released from RAM.
    /// `append`/`generate`/`checkpoint` transparently resume it first.
    case parked
    /// A generate/append/checkpoint/resume call is in flight for this
    /// session; any other op on it throws `sessionBusy`.
    case generating
}

/// Sendable facts a checkpoint write hands back to `InfiniteEngine` so it
/// can update `SessionRecord` bookkeeping and build the wire response
/// without re-deriving them.
private struct CheckpointWriteResult {
    let checkpointId: String
    let tokenRecord: [Int]
    let pendingToken: Int?
    let sizeBytes: Int64
    let durationSeconds: Double
}

private struct SessionRecord: Sendable {
    let id: String
    let selection: InfiniteModelSelection
    let label: String?
    let createdAt: String
    var updatedAt: String
    /// Running total of appended characters. Cheap bookkeeping, kept
    /// independent of the backend (unlike `tokenCount`, this never needs a
    /// model loaded to report).
    var inputCharacters: Int = 0
    /// Real token count from the backend's live session (`tokenRecord.count`
    /// — durable cache tokens plus the one pending token, if any). `0`
    /// until the first `append`/`generate` call opens a backend session;
    /// the wire field name stays `estimatedInputTokens` for API stability
    /// even though the value is now exact, not estimated (engine#265).
    var tokenCount: Int = 0
    var checkpoints: [InfiniteSessionCheckpoint] = []
    /// engine#266 lifecycle state; see `SessionState`.
    var state: SessionState = .open
    /// Most recent checkpoint written for this session, if any — the
    /// default `resume`/`fork` target and the manifest `parentCheckpointId`
    /// of the next checkpoint.
    var lastCheckpointId: String?
    /// Whether this session has ever actually held live backend state
    /// (`MLXInfiniteBackend.openSession`/a successful resume) — a freshly
    /// created, never-touched session is `.open` but NOT hot in the RAM
    /// sense, so it must not count against `maxHotSessions` or force an
    /// eviction (see `ensureLiveBackendState`/`enforceHotBudget`).
    var hasLiveBackendState: Bool = false
}
