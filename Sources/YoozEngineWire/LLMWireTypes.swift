// LLMWireTypes.swift
// YoozEngineWire
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Request body for `POST /v1/llm/generate`.
///
/// Canonical keys only (`systemPrompt` camelCase). The legacy snake_case
/// `system_prompt` spelling some pre-SDK callers still post is a
/// loopback-server-only decode concern, handled by the
/// `LegacyLLMGenerateRequest` shim in `YoozEngine/Server/APITypes.swift` —
/// the same pattern as `LegacySTTSetBackendRequest` — so this shared type
/// stays free of transport-specific compat baggage.
public struct LLMGenerateRequest: Codable, Sendable, Equatable {
    public let prompt: String
    public let model: String?
    public let systemPrompt: String?
    /// Optional GPU-admission override (engine#228). Nil (the default)
    /// lets the engine classify this call as `.background` — see
    /// `GPUWorkloadClass`. Deliberately strict on decode: an unknown value
    /// rejects the request rather than silently downgrading (see the
    /// `GPUWorkloadClass` doc).
    public let workloadClass: GPUWorkloadClass?

    public init(
        prompt: String,
        model: String? = nil,
        systemPrompt: String? = nil,
        workloadClass: GPUWorkloadClass? = nil
    ) {
        self.prompt = prompt
        self.model = model
        self.systemPrompt = systemPrompt
        self.workloadClass = workloadClass
    }
}

public struct LLMGenerateResponse: Codable, Sendable, Equatable {
    public let text: String
    public let model: String
    public let tokensGenerated: Int?
    public let processingTimeMs: Int?

    public init(
        text: String,
        model: String,
        tokensGenerated: Int? = nil,
        processingTimeMs: Int? = nil
    ) {
        self.text = text
        self.model = model
        self.tokensGenerated = tokensGenerated
        self.processingTimeMs = processingTimeMs
    }
}

// MARK: - Model management

/// Coarse capability label for a catalogued LLM model (engine#303): a
/// TouchUp proofreading head rewrites/corrects text it is handed, while a
/// general model is suited to classification, structured output, or other
/// non-proofreading instruct use. Lets `GET /v1/llm/models` consumers (e.g.
/// remi's auto-approve classifier) pick sensibly instead of guessing from
/// `displayName`, and lets a picker UI group by capability.
public enum LLMModelPurpose: String, Codable, Sendable, CaseIterable, Equatable {
    case proofread
    case general
}

/// Describes one LLM model known to the engine. Returned by
/// `GET /v1/llm/models` and consumed by thin-client UI (e.g. whisper's
/// "Touch-up Model" dropdown).
///
/// The field set is intentionally forward-compatible: `sizeBytes`,
/// `latencyHintMs`, and `purpose` are optional so future backends (Apple
/// Intelligence, remote) can omit them without breaking decoders. `id` is
/// the stable wire value (`LLMModelType.rawValue` on the server side, e.g.
/// `"yooz-light-v3"`); `displayName` is user-facing ("Yooz-Light").
public struct LLMModelInfo: Codable, Sendable, Equatable {
    public let id: String
    public let displayName: String
    public let sizeBytes: Int64?
    public let loaded: Bool
    public let latencyHintMs: Int?
    public let purpose: LLMModelPurpose?

    public init(
        id: String,
        displayName: String,
        sizeBytes: Int64? = nil,
        loaded: Bool = false,
        latencyHintMs: Int? = nil,
        purpose: LLMModelPurpose? = nil
    ) {
        self.id = id
        self.displayName = displayName
        self.sizeBytes = sizeBytes
        self.loaded = loaded
        self.latencyHintMs = latencyHintMs
        self.purpose = purpose
    }
}

/// Response body for `GET /v1/llm/models`. `current` carries the id of
/// the engine's preferred model — held for the engine process lifetime;
/// clients that need cross-session persistence must cache the value
/// themselves and re-apply via `setModel(_:)` after reconnect.
/// `available` is the full catalogue with per-model load state.
public struct LLMModelsResponse: Codable, Sendable, Equatable {
    public let current: String
    public let available: [LLMModelInfo]

    public init(current: String, available: [LLMModelInfo]) {
        self.current = current
        self.available = available
    }
}

/// Request body for `POST /v1/llm/model`, `/v1/llm/preload`, and
/// `/v1/llm/unload`. Wire contract: a single JSON object with a `model`
/// key holding the model id (e.g. `"yooz-light-v3"`).
public struct LLMModelSelection: Codable, Sendable, Equatable {
    public let model: String

    public init(model: String) {
        self.model = model
    }
}

/// Request body for `POST /v1/llm/clear-cache` (engine#299). Unlike
/// `LLMModelSelection.model`, this `model` is OPTIONAL: naming a tier clears
/// only that tier's prompt-KV cache, and omitting it (or posting an empty
/// body — see `APIServer`'s route for why both mean the same thing) clears
/// every currently-loaded tier. This is the LLM-scoped middle lever between
/// `/v1/session/begin` (fans a KV-cache reset out to every module, STT
/// included — too broad for a memory-reclaim call) and `/v1/llm/unload`
/// (frees the cache AND the weights — too destructive when the caller only
/// wants the ~GB-scale retained-KV delta back).
public struct LLMClearCacheRequest: Codable, Sendable, Equatable {
    public let model: String?

    public init(model: String? = nil) {
        self.model = model
    }
}

/// Response body for `POST /v1/llm/clear-cache`. `cleared` carries the wire
/// ids of the tiers that actually had a cache to drop. A tier that was not
/// loaded (or, when `model` was omitted, no tiers loaded at all) is a
/// success no-op and is simply absent from the list — never an error, since
/// clearing an already-empty cache trivially succeeds.
public struct LLMClearCacheResponse: Codable, Sendable, Equatable {
    public let cleared: [String]

    public init(cleared: [String]) {
        self.cleared = cleared
    }
}

/// Response body for `GET /v1/llm/status`. Shape parity with `STTStatus` so
/// consumer apps can template a single progress-banner view-model over both
/// endpoints.
public struct LLMStatus: Codable, Sendable, Equatable {
    public let loaded: Bool
    /// Wire id of the preferred LLM model (e.g. `"yooz-light-v3"`).
    public let modelId: String?
    /// Fraction-completed [0.0, 1.0] for an in-progress HF model
    /// download. `nil` when no download is in flight.
    public let progress: Double?
    /// Lifecycle state for the active LLM tier (engine#125). `nil`
    /// on pre-#125 server builds — consumers MAY infer state from
    /// `loaded` + `progress` when nil.
    public let state: LoadState?
    /// Human-readable error from the last failed load. `nil` unless
    /// `state == .failed`.
    public let lastError: String?

    public init(
        loaded: Bool,
        modelId: String?,
        progress: Double?,
        state: LoadState? = nil,
        lastError: String? = nil
    ) {
        self.loaded = loaded
        self.modelId = modelId
        self.progress = progress
        self.state = state
        self.lastError = lastError
    }
}
