// EngineStateWireTypes.swift
// YoozEngineWire
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Discriminates the frames pushed over `/v1/events` (engine#226). One
/// channel carries every module's model-selection lifecycle, replacing the
/// per-app reconciliation + polling stacks each consumer previously had to
/// build for itself (see the engine#226 issue body for the whisper-side
/// cost this is paying down).
public enum EngineEventKind: String, Codable, Sendable, CaseIterable {
    /// The active model for a module changed (`setModel`/persisted-restore).
    /// Fires synchronously with the picker route's response — a client
    /// that only reads the HTTP response already has this; the event
    /// exists so every OTHER subscriber (a second window, a menu-bar
    /// status item) sees the change too.
    case modelChanged
    /// A model's lifecycle state changed (`unavailable < available <
    /// cached < loaded`), including the transition when a load fails: the
    /// state falls back to `available` and `message` carries the error.
    case loadStateChanged
    /// Download progress ticked for a model whose weights are actively
    /// being fetched. `progress` is a fraction in `[0.0, 1.0)`; a
    /// `loadStateChanged` frame at `.loaded` follows once the fetch and
    /// subsequent in-memory load both complete.
    case downloadProgress
    /// A module's resident-model set changed (e.g. the single-resident
    /// eviction that runs after a successful model switch).
    case residencyChanged
    /// Forward-compat fallback: an event kind this SDK build does not
    /// know. Never emitted by the engine; produced only by the tolerant
    /// decode below when a newer engine ships a kind this build predates.
    case unknown

    /// Tolerant decode: any unknown raw value maps to `.unknown` instead
    /// of throwing — the `ModelTier`/`ModelLoadState` convention, chosen
    /// deliberately over `GPUWorkloadClass`-style strict decode because
    /// the rationales differ by channel direction. A strict request body
    /// protects explicit CALLER intent (an unknown value is the caller's
    /// typo; a 400 lets it retry). A push frame has no caller and no
    /// retry: strict decode on an older SDK would fail the whole
    /// `EngineEvent` decode for every frame of a newer engine's fifth
    /// kind, silently dropping `module`/`ts` along with it. Tolerant
    /// decode keeps the frame; subscribers ignore `.unknown` via
    /// `EngineEvent.payload`.
    public init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        self = EngineEventKind(rawValue: raw) ?? .unknown
    }
}

/// One frame pushed to `/v1/events` subscribers — WebSocket on loopback,
/// `AsyncStream<EngineEvent>` via `EngineTransport.openEvents()` in-process
/// (engine#226). Single wire shape for every event kind; fields not
/// meaningful for a given `kind` are nil (e.g. `progress` is only set on
/// `.downloadProgress`).
public struct EngineEvent: Codable, Sendable, Equatable {
    public let kind: EngineEventKind
    /// Module the event concerns, e.g. `"touchup"`. Matches
    /// `EngineModuleSnapshot.module` so a subscriber can route the event to
    /// the right slice of its local state.
    public let module: String
    /// The model/backend id the event concerns, when applicable.
    public let modelId: String?
    /// New lifecycle state, set on `.loadStateChanged`.
    public let loadState: ModelLoadState?
    /// Download fraction in `[0.0, 1.0)`, set on `.downloadProgress`.
    public let progress: Double?
    /// Human-readable detail — populated on a failed load
    /// (`.loadStateChanged` falling back to `.available`) so a subscriber
    /// can surface why without a follow-up call.
    public let message: String?
    /// ISO-8601 timestamp the engine generated this event at. A plain
    /// string (not `Date`) so the wire shape is immune to
    /// `JSONEncoder`/`JSONDecoder` date-strategy drift between the engine
    /// and SDK — matches `SessionBeginResponse.ts`.
    public let ts: String

    public init(
        kind: EngineEventKind,
        module: String,
        modelId: String? = nil,
        loadState: ModelLoadState? = nil,
        progress: Double? = nil,
        message: String? = nil,
        ts: String = ISO8601DateFormatter().string(from: Date())
    ) {
        self.kind = kind
        self.module = module
        self.modelId = modelId
        self.loadState = loadState
        self.progress = progress
        self.message = message
        self.ts = ts
    }
}

/// Typed view of an `EngineEvent`'s kind-dependent fields (engine#226,
/// PR #239 review). The wire shape stays a flat struct (JSON has no native
/// sum type, and every sibling DTO in this target is flat), but consumers
/// should match on `event.payload` rather than hand-rolling per-kind
/// optional unwraps — the compiler then enforces exhaustive handling and a
/// frame whose required fields are missing lands in `.unrecognized` in one
/// place instead of being silently dropped by an ad hoc `guard let`.
public enum EngineEventPayload: Sendable, Equatable {
    case modelChanged(modelId: String)
    case loadStateChanged(modelId: String, loadState: ModelLoadState, message: String?)
    case downloadProgress(modelId: String, progress: Double)
    case residencyChanged(modelId: String)
    /// The event kind is unknown to this SDK build (`EngineEventKind.unknown`,
    /// tolerant decode of a newer engine's kind) or a required field for a
    /// known kind is missing (malformed frame). Subscribers ignore it;
    /// `EngineEvent.module` / `.ts` remain readable for diagnostics.
    case unrecognized
}

extension EngineEvent {
    /// Resolve the kind-dependent optionals into the typed payload. See
    /// `EngineEventPayload`.
    public var payload: EngineEventPayload {
        switch kind {
        case .modelChanged:
            guard let modelId else { return .unrecognized }
            return .modelChanged(modelId: modelId)
        case .loadStateChanged:
            guard let modelId, let loadState else { return .unrecognized }
            return .loadStateChanged(modelId: modelId, loadState: loadState, message: message)
        case .downloadProgress:
            guard let modelId, let progress else { return .unrecognized }
            return .downloadProgress(modelId: modelId, progress: progress)
        case .residencyChanged:
            guard let modelId else { return .unrecognized }
            return .residencyChanged(modelId: modelId)
        case .unknown:
            return .unrecognized
        }
    }
}

// MARK: - GET /v1/state

/// One row of a `GET /v1/state` module snapshot: the seven canonical
/// `ModelInfo` fields every module picker publishes (AGENTS.md "Module
/// model picker pattern"). Module-specific extension fields (e.g. STT's
/// `supportsStreaming`) are intentionally dropped here — `/v1/state` is the
/// module-agnostic snapshot `EngineStateStore` renders generically; a
/// picker UI that needs the extensions still calls that module's own
/// `GET /v1/<module>/models`.
public struct EngineModelSnapshotRow: Codable, Sendable, Equatable {
    public let id: String
    public let displayName: String
    public let description: String
    public let tier: ModelTier
    public let sizeBytes: Int64?
    public let loadState: ModelLoadState
    public let isActive: Bool

    public init(
        id: String,
        displayName: String,
        description: String,
        tier: ModelTier,
        sizeBytes: Int64?,
        loadState: ModelLoadState,
        isActive: Bool
    ) {
        self.id = id
        self.displayName = displayName
        self.description = description
        self.tier = tier
        self.sizeBytes = sizeBytes
        self.loadState = loadState
        self.isActive = isActive
    }
}

/// One module's slice of `GET /v1/state`: its picker catalog + active id,
/// tagged with the module name so a client walking
/// `EngineStateSnapshot.modules` can route each entry to the right local
/// slice (matches `EngineEvent.module`).
public struct EngineModuleSnapshot: Codable, Sendable, Equatable {
    public let module: String
    public let models: [EngineModelSnapshotRow]
    public let activeId: String

    public init(module: String, models: [EngineModelSnapshotRow], activeId: String) {
        self.module = module
        self.models = models
        self.activeId = activeId
    }
}

/// Response body for `GET /v1/state` (engine#226): every module's picker
/// catalog + active id in one call, so a consumer can render a picker UI
/// before the first `/v1/events` frame arrives. A module with no picker
/// (or not bundled in this build variant) is simply absent from `modules`.
public struct EngineStateSnapshot: Codable, Sendable, Equatable {
    public let modules: [EngineModuleSnapshot]

    public init(modules: [EngineModuleSnapshot]) {
        self.modules = modules
    }
}
