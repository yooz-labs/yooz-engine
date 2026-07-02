// TouchUpWireTypes.swift
// YoozEngineWire
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Processing intensity for the `/v1/touchup` pipeline. Single definition
/// shared by the server (`APIServer` / `LLMModule`'s `TouchUpEngine`), the
/// SDK, and the in-process transport — previously three independent enums
/// (`ServerTouchUpMode`, `LLMModule.TouchUpMode`, SDK `TouchUpMode`) kept in
/// sync by hand plus a translation layer at the wire boundary (#225).
public enum TouchUpMode: String, Codable, Sendable {
    /// Skip LLM entirely; regex voice-commands + whitespace only.
    case off
    /// Yooz-Light (0.5B): contractions, capitalization, simple grammar.
    case light
    /// Yooz-Quality when available, else Yooz-Light: standard proofreading.
    case standard
    /// Yooz-Quality comprehensive cleanup: self-corrections, fragments.
    case full
}

public struct TouchUpRequest: Codable, Sendable, Equatable {
    public let text: String
    public let mode: TouchUpMode
    public let language: String?
    /// Optional GPU-admission override (engine#228). Nil (the default)
    /// lets the engine classify this call as `.background` — see
    /// `GPUWorkloadClass`. Deliberately strict on decode: an unknown value
    /// rejects the request rather than silently downgrading (see the
    /// `GPUWorkloadClass` doc).
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

public struct TouchUpResponse: Codable, Sendable, Equatable {
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

// MARK: - Picker (canonical module-picker pattern; AGENTS.md "Module model
// picker pattern"). Single definition — previously separate copies in
// `YoozEngine/TouchUp/TouchUpPickerTypes.swift` (LLMModule) and
// `YoozEngineClient/Types/TouchUpTypes.swift` (SDK).

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

/// Response for `GET /v1/touchup/models`. `activeId` is the id of
/// the entry where `isActive == true` — surfaced separately so a
/// client that only cares about the current selection does not
/// have to scan the array.
public struct TouchUpModelsResponse: Codable, Sendable, Equatable {
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
public struct TouchUpSetModelRequest: Codable, Sendable, Equatable {
    public let id: String
    public let preload: Bool?

    public init(id: String, preload: Bool? = nil) {
        self.id = id
        self.preload = preload
    }
}
