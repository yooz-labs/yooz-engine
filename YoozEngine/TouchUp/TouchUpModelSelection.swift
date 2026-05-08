// TouchUpModelSelection.swift
// YoozEngine
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Identifier for the active TouchUp model. Distinct from
/// `LLMModelType` because the picker exposes a third option —
/// Apple Intelligence (Foundation Models) — that is not an MLX
/// backend and therefore does not fit `LLMModelType`'s "MLX-only"
/// contract. Keeping the picker ids in their own enum prevents a
/// future MLX-side refactor from accidentally renaming a wire id
/// the consumer SDK depends on.
///
/// Stable wire ids (do not rename without bumping a major SDK):
/// - `yooz-light-v3` — `LLMModelType.yoozLight`
/// - `yooz-quality-v3` — `LLMModelType.yoozQuality`
/// - `foundation-models` — `FoundationModelsBackend` (macOS 26+ only)
///
/// All picker UX strings (display name, description, tier, size)
/// are owned by this enum, not delegated back to `LLMModelType`.
/// Picker presentation is a property of the *selection* — the
/// engine-side `LLMModelType` stays headless so a backend rename
/// cannot silently change picker text.
enum TouchUpModelSelection: String, Codable, Sendable, CaseIterable {
    case yoozLight = "yooz-light-v3"
    case yoozQuality = "yooz-quality-v3"
    case foundationModels = "foundation-models"

    /// Picker-visible name surfaced in consumer UIs.
    var displayName: String {
        switch self {
        case .yoozLight: return "Yooz-Light"
        case .yoozQuality: return "Yooz-Quality"
        case .foundationModels: return "Apple Intelligence"
        }
    }

    /// One-line description for picker subtitles.
    var description: String {
        switch self {
        case .yoozLight: return "Fast proofreading (~200ms)"
        case .yoozQuality: return "High quality proofreading (~310ms)"
        case .foundationModels: return "On-device 3B (macOS 26+, no download)"
        }
    }

    /// Coarse tier label for badge / sort UX. The wire side uses
    /// the typed `ModelTier` enum; this property is the
    /// engine-side mapping.
    var tier: ModelTier {
        switch self {
        case .yoozLight: return .light
        case .yoozQuality: return .quality
        case .foundationModels: return .premium
        }
    }

    /// Approximate on-disk size after first-run download. `nil` for
    /// FoundationModels because the OS owns the weights.
    var estimatedSize: Int64? {
        switch self {
        case .yoozLight: return 276 * 1024 * 1024  // ~276 MB
        case .yoozQuality: return 424 * 1024 * 1024  // ~424 MB
        case .foundationModels: return nil
        }
    }
}
