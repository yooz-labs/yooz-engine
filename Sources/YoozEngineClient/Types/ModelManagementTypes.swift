import Foundation

/// One row in the model-management inventory (`GET /v1/models`).
///
/// Unlike the per-module picker rows (`TouchUpModelInfo`, `STTBackendInfo`),
/// this is a disk-hygiene view: `sizeBytes` is the **real on-disk footprint**
/// the engine measured (not a static estimate), and `deletable` says whether the
/// app may offer a Delete button for it. The app's "Manage Models" tab renders
/// these directly.
public struct ManagedModelInfo: Codable, Sendable, Equatable {
    /// Stable delete handle: an LLM model id (e.g. `yooz-quality-v2`) or the
    /// HuggingFace hub directory name (`models--<ns>--<repo>`) for a swept model.
    public let id: String
    /// Owning module: `llm`, `stt`, etc. Drives grouping/labels in the UI.
    public let module: String
    public let displayName: String
    /// Reclaimable on-disk bytes (hub blobs + any models-directory copy; the
    /// read-only app-bundle copy is not counted).
    public let sizeBytes: Int64
    /// Available to load without a download (on disk or bundled).
    public let cached: Bool
    /// Currently resident in memory.
    public let loaded: Bool
    /// The module's active model — never offered for deletion.
    public let isActive: Bool
    /// Whether the app may delete it (has a reclaimable footprint and isn't
    /// active).
    public let deletable: Bool

    public init(
        id: String,
        module: String,
        displayName: String,
        sizeBytes: Int64,
        cached: Bool,
        loaded: Bool,
        isActive: Bool,
        deletable: Bool
    ) {
        self.id = id
        self.module = module
        self.displayName = displayName
        self.sizeBytes = sizeBytes
        self.cached = cached
        self.loaded = loaded
        self.isActive = isActive
        self.deletable = deletable
    }
}

/// Response for `GET /v1/models` — the full model-management inventory.
public struct ManagedModelsResponse: Codable, Sendable, Equatable {
    public let models: [ManagedModelInfo]

    public init(models: [ManagedModelInfo]) {
        self.models = models
    }
}

/// Response for `DELETE /v1/models/:id`.
public struct DeleteModelResult: Codable, Sendable, Equatable {
    public let id: String
    public let reclaimedBytes: Int64

    public init(id: String, reclaimedBytes: Int64) {
        self.id = id
        self.reclaimedBytes = reclaimedBytes
    }
}

/// Response for `POST /v1/models/cleanup` — the one-shot disk-hygiene migration.
public struct ModelCleanupResult: Codable, Sendable, Equatable {
    public let totalReclaimedBytes: Int64
    /// Bytes reclaimed per hub repo directory name.
    public let perRepo: [String: Int64]

    public init(totalReclaimedBytes: Int64, perRepo: [String: Int64]) {
        self.totalReclaimedBytes = totalReclaimedBytes
        self.perRepo = perRepo
    }
}
