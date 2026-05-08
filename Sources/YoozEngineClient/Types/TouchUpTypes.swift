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

    public init(text: String, mode: TouchUpMode, language: String? = nil) {
        self.text = text
        self.mode = mode
        self.language = language
    }
}

public struct TouchUpResponse: Codable, Sendable {
    public let result: String
    public let mode: TouchUpMode
    public let processingTimeMs: Int?
    public let modelUsed: String?
    public let warnings: [String]?
}

// MARK: - Picker (canonical module-picker pattern)
//
// Mirror of the engine's `TouchUpModelInfo` / `TouchUpModelsResponse`
// wire shapes. SDK-side copies (instead of importing the engine
// target) keep the SDK self-contained and Swift Concurrency-safe
// — consumers only depend on the SDK module.

/// One model in the TouchUp picker. Snapshot at the time of the
/// `availableModels()` call; re-fetch after `setModel(_:preload:)`
/// to learn the new active id and any cache/load changes the
/// preload triggered.
public struct TouchUpModelInfo: Codable, Sendable, Equatable {
    /// Stable wire id (e.g. `"yooz-light-v3"`).
    public let id: String
    /// Picker-visible name (e.g. "Yooz-Light").
    public let displayName: String
    /// One-line subtitle for picker UX (latency hint etc.).
    public let description: String
    /// Coarse tier label (`light` / `quality` / `premium`). UI
    /// renders Pro badges or sort hints off this.
    public let tier: String
    /// Approximate on-disk size after first-run download. `nil` for
    /// OS-provided backends (Apple Intelligence).
    public let sizeBytes: Int64?
    /// Whether this option is selectable on this system.
    public let isAvailable: Bool
    /// Whether the weights are already on disk (no download needed).
    public let isCached: Bool
    /// Whether the model is currently resident in memory.
    public let isLoaded: Bool
    /// Whether `/v1/touchup` currently routes through this model.
    public let isActive: Bool

    public init(
        id: String,
        displayName: String,
        description: String,
        tier: String,
        sizeBytes: Int64?,
        isAvailable: Bool,
        isCached: Bool,
        isLoaded: Bool,
        isActive: Bool
    ) {
        self.id = id
        self.displayName = displayName
        self.description = description
        self.tier = tier
        self.sizeBytes = sizeBytes
        self.isAvailable = isAvailable
        self.isCached = isCached
        self.isLoaded = isLoaded
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
