// TouchUpPickerTypes.swift
// LLMModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore
import Foundation

/// Engine-side picker row for the canonical TouchUp model picker.
///
/// Lives in LLMModule (the producer) so consumers within the engine
/// (`APIServer`) can reference the type without LLMModule depending
/// on Hummingbird. The wire-side `ResponseCodable` conformance is
/// added via extension in `YoozEngine/Server/APITypes.swift`. The
/// SDK-side mirror lives in `YoozEngineClient/Types/TouchUpTypes.swift`
/// and is kept in sync via the `TouchUpModelInfoBoundaryTests`
/// encode/decode round-trip.
public struct TouchUpModelInfo: Codable, Sendable, Equatable {
    /// Stable wire id (e.g. `yooz-light-v2`). Matches
    /// `TouchUpModelSelection.rawValue`.
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
    /// invariant in a single field rather than three loose booleans.
    public let loadState: ModelLoadState
    /// Whether `/v1/touchup` currently routes through this model.
    /// Exactly one row per response has `isActive == true` (pinned
    /// by `availableModels()`'s precondition).
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

/// Response for `GET /v1/touchup/models`. `activeId` is the id of
/// the entry where `isActive == true` — surfaced separately so a
/// client that only cares about the current selection does not
/// have to scan the array.
public struct TouchUpModelsResponse: Codable, Sendable {
    public let models: [TouchUpModelInfo]
    public let activeId: String

    public init(models: [TouchUpModelInfo], activeId: String) {
        self.models = models
        self.activeId = activeId
    }
}

/// Request body for `POST /v1/touchup/model`. `preload` defaults
/// to `true` server-side so a one-shot picker change is enough to
/// avoid a cold-start on the next `/v1/touchup` call.
public struct TouchUpSetModelRequest: Codable, Sendable {
    public let id: String
    public let preload: Bool?

    public init(id: String, preload: Bool? = nil) {
        self.id = id
        self.preload = preload
    }
}
