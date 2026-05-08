import Foundation

public struct LLMGenerateRequest: Codable, Sendable {
    public let prompt: String
    public let model: String?
    public let systemPrompt: String?

    public init(
        prompt: String,
        model: String? = nil,
        systemPrompt: String? = nil
    ) {
        self.prompt = prompt
        self.model = model
        self.systemPrompt = systemPrompt
    }
}

public struct LLMGenerateResponse: Codable, Sendable {
    public let text: String
    public let model: String
    public let tokensGenerated: Int?
    public let processingTimeMs: Int?
}

// MARK: - Model management

/// Describes one LLM model known to the engine. Returned by
/// `GET /v1/llm/models` and consumed by thin-client UI (e.g. whisper's
/// "Touch-up Model" dropdown).
///
/// The field set is intentionally forward-compatible: `sizeBytes` and
/// `latencyHintMs` are optional so future backends (Apple Intelligence,
/// remote) can omit them without breaking decoders. `id` is the stable
/// wire value (`LLMModelType.rawValue` on the server side, e.g.
/// `"yooz-light-v3"`); `displayName` is user-facing ("Yooz-Light").
public struct LLMModelInfo: Codable, Sendable, Equatable {
    public let id: String
    public let displayName: String
    public let sizeBytes: Int64?
    public let loaded: Bool
    public let latencyHintMs: Int?

    public init(
        id: String,
        displayName: String,
        sizeBytes: Int64? = nil,
        loaded: Bool = false,
        latencyHintMs: Int? = nil
    ) {
        self.id = id
        self.displayName = displayName
        self.sizeBytes = sizeBytes
        self.loaded = loaded
        self.latencyHintMs = latencyHintMs
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
