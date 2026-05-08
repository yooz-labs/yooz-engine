// ModelPicker.swift
// EngineCore
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Coarse class for a model in any picker (TouchUp / STT / TTS /
/// future). Module-neutral name — earlier `TouchUpModelTier` was
/// renamed in #99 because the second adopter (STT) made the
/// TouchUp prefix nonsensical when STT backends carry a `tier`
/// field. The old name is retained as a typealias on the SDK for
/// one release to ease consumer migration.
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

/// Lifecycle state of a single picker row. Replaces three boolean
/// flags so the `loaded ⇒ cached ⇒ available` invariant is encoded
/// in the type system (illegal combinations like "loaded but not
/// cached" become unrepresentable).
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
