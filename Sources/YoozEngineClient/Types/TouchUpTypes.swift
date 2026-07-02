import Foundation

public enum TouchUpMode: String, Codable, Sendable {
    case off
    case light
    case standard
    case full
}

public struct TouchUpRequest: Codable, Sendable {
    public let text: String
    public let mode: TouchUpMode
    public let language: String?
    /// Optional GPU-admission override (engine#228). Nil (the default)
    /// lets the engine classify this call as `.background` — see
    /// `GPUWorkloadClass`.
    public let workloadClass: GPUWorkloadClass?

    public init(
        text: String,
        mode: TouchUpMode,
        language: String? = nil,
        workloadClass: GPUWorkloadClass? = nil
    ) {
        self.text = text
        self.mode = mode
        self.language = language
        self.workloadClass = workloadClass
    }
}

public struct TouchUpResponse: Codable, Sendable {
    public let result: String
    public let mode: TouchUpMode
    public let processingTimeMs: Int?
    public let modelUsed: String?
    public let warnings: [String]?

    public init(
        result: String,
        mode: TouchUpMode,
        processingTimeMs: Int? = nil,
        modelUsed: String? = nil,
        warnings: [String]? = nil
    ) {
        self.result = result
        self.mode = mode
        self.processingTimeMs = processingTimeMs
        self.modelUsed = modelUsed
        self.warnings = warnings
    }
}

// MARK: - Picker (canonical module-picker pattern)
//
// Mirror of the engine's `TouchUpModelInfo` / `TouchUpModelsResponse`
// wire shapes. SDK-side copies (instead of importing the engine
// target) keep the SDK self-contained — consumers only depend on
// the SDK module. `TouchUpModelInfoBoundaryTests` encodes one side
// and decodes the other to catch drift.

/// Coarse class for a model in any picker (TouchUp / STT / TTS /
/// future). Module-neutral name — earlier `TouchUpModelTier` was
/// renamed in #99 because the second adopter (STT) made the
/// TouchUp prefix nonsensical when STT backends carry a `tier`
/// field. The old name is retained as a typealias for one
/// release to ease consumer migration.
///
/// `unknown` is the forward-compat fallback: an SDK consumer
/// running against a newer engine that ships a fifth tier sees
/// `.unknown` rather than failing to decode.
public enum ModelTier: String, Codable, Sendable, CaseIterable {
    case light
    case quality
    case premium
    case unknown

    /// Tolerant decode: any unknown raw value maps to `.unknown`
    /// instead of throwing. Forward compat for picker UIs that
    /// poll a newer engine than the SDK was built against.
    public init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        self = ModelTier(rawValue: raw) ?? .unknown
    }
}

/// Back-compat typealias for the TouchUp-prefixed name shipped in
/// #97. Remove once consumer apps (whisper, notes, voice) are
/// known to be on the new name (target: v0.7.x of the engine SDK).
public typealias TouchUpModelTier = ModelTier

/// Lifecycle state of a single picker row. Replaces three boolean
/// flags so the `loaded ⇒ cached ⇒ available` invariant is encoded
/// in the type system (illegal combinations like
/// "loaded but not cached" become unrepresentable). Module-neutral
/// name — see `ModelTier` for the rename rationale.
public enum ModelLoadState: String, Codable, Sendable, CaseIterable {
    /// Picker UI greys this out — not selectable on this system.
    case unavailable
    /// Selectable but first use will download.
    case available
    /// Weights present in cache; first use loads from disk.
    case cached
    /// Resident in memory; calls hit the model immediately.
    case loaded

    /// Tolerant decode: any unknown raw value maps to
    /// `.unavailable` (the safest fallback — picker greys it out).
    public init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        self = ModelLoadState(rawValue: raw) ?? .unavailable
    }
}

/// Back-compat typealias. See `TouchUpModelTier` for removal plan.
public typealias TouchUpModelLoadState = ModelLoadState

/// One model in the TouchUp picker. Snapshot at the time of the
/// `availableModels()` call; re-fetch after `setModel(_:preload:)`
/// to learn the new active id and any cache/load changes the
/// preload triggered.
public struct TouchUpModelInfo: Codable, Sendable, Equatable {
    /// Stable wire id (e.g. `"yooz-light-v2"`).
    public let id: String
    /// Picker-visible name (e.g. "Yooz-Light").
    public let displayName: String
    /// One-line subtitle for picker UX (latency hint etc.).
    public let description: String
    /// Coarse class for badge / sort UX.
    public let tier: ModelTier
    /// Approximate on-disk size after first-run download. `nil` for
    /// OS-provided backends (`.premium` tier).
    public let sizeBytes: Int64?
    /// Lifecycle state. Encodes the `loaded ⇒ cached ⇒ available`
    /// invariant in a single field.
    public let loadState: ModelLoadState
    /// Whether `/v1/touchup` currently routes through this model.
    /// The engine guarantees exactly one row per response has
    /// `isActive == true`.
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

/// Response for `availableModels()`. `activeId` is the id of the
/// entry where `isActive == true`.
public struct TouchUpModelsResponse: Codable, Sendable {
    public let models: [TouchUpModelInfo]
    public let activeId: String

    public init(models: [TouchUpModelInfo], activeId: String) {
        self.models = models
        self.activeId = activeId
    }
}

/// Request body for `setModel(_:preload:)`. `preload` defaults to
/// `true` server-side; consumers rarely need to construct this
/// directly (use `TouchUpClient.setModel(id:preload:)`).
public struct TouchUpSetModelRequest: Codable, Sendable {
    public let id: String
    public let preload: Bool?

    public init(id: String, preload: Bool? = nil) {
        self.id = id
        self.preload = preload
    }
}
