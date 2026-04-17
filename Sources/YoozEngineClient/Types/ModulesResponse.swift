import Foundation

/// Response body for `GET /v1/modules`.
///
/// Mirrors `EngineCore.ModulesResponse` for SDK consumers that can't import
/// the server module. The server emits this with `.sortedKeys` for
/// deterministic output; `Codable` decoding on the client side is
/// order-agnostic so no encoder configuration is required.
///
/// See `Sources/EngineCore/ModulesResponse.swift` for authoritative docs on
/// the wire format and the unified-versioning scheme locked in A1.
public struct ModulesResponse: Codable, Sendable, Equatable {
    /// Engine semantic version (`EngineConfig.version`). Identical to each
    /// manifest's `version` under the unified-versioning scheme.
    public let engineVersion: String

    /// Which build flavor this process is (`full`, `whisper`, ...).
    public let buildVariant: String

    /// Every registered module, sorted by `name` by the server.
    public let modules: [ModuleManifest]

    public init(engineVersion: String, buildVariant: String, modules: [ModuleManifest]) {
        self.engineVersion = engineVersion
        self.buildVariant = buildVariant
        self.modules = modules
    }
}

/// Per-module entry inside `ModulesResponse.modules`.
///
/// `version` is the engine version (unified scheme). `loaded` and `error`
/// come from the server's `AIModule.healthCheck()`. `detail` carries
/// module-specific key/value pairs (e.g. `{"rules_total": "1560"}` for
/// grammar, `{"model": "silero-vad-unified-v6.0.0"}` for VAD).
public struct ModuleManifest: Codable, Sendable, Equatable {
    public let name: String
    public let version: String
    public let loaded: Bool
    public let error: String?
    public let detail: [String: String]

    public init(
        name: String,
        version: String,
        loaded: Bool,
        error: String?,
        detail: [String: String]
    ) {
        self.name = name
        self.version = version
        self.loaded = loaded
        self.error = error
        self.detail = detail
    }
}
