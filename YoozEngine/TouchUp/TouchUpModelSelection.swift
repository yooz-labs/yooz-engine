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
enum TouchUpModelSelection: String, Codable, Sendable, CaseIterable {
    case yoozLight = "yooz-light-v3"
    case yoozQuality = "yooz-quality-v3"
    case foundationModels = "foundation-models"

    /// Map MLX-tier selections back onto the underlying
    /// `LLMModelType`. Returns `nil` for `.foundationModels` because
    /// that tier is dispatched through `FoundationModelsBackend`,
    /// not the MLX path.
    var mlxModelType: LLMModelType? {
        switch self {
        case .yoozLight: return .yoozLight
        case .yoozQuality: return .yoozQuality
        case .foundationModels: return nil
        }
    }

    /// Human-readable name surfaced in picker UIs. Mirrors
    /// `LLMModelType.displayName` for the MLX tiers; "Apple
    /// Intelligence" for FoundationModels matches the macOS 26+ UX.
    var displayName: String {
        switch self {
        case .yoozLight: return LLMModelType.yoozLight.displayName
        case .yoozQuality: return LLMModelType.yoozQuality.displayName
        case .foundationModels: return "Apple Intelligence"
        }
    }

    /// One-line description for picker subtitles.
    var description: String {
        switch self {
        case .yoozLight: return LLMModelType.yoozLight.description
        case .yoozQuality: return LLMModelType.yoozQuality.description
        case .foundationModels: return "On-device 3B (macOS 26+, no download)"
        }
    }

    /// Coarse latency / size tier label for sorting and badge UX.
    /// `light` is the fast default; `quality` ships a Pro badge in
    /// whisper; `premium` is reserved for OS-provided backends.
    var tier: String {
        switch self {
        case .yoozLight: return "light"
        case .yoozQuality: return "quality"
        case .foundationModels: return "premium"
        }
    }

    /// Approximate on-disk size after first-run download. `nil` for
    /// FoundationModels because the OS owns the weights.
    var estimatedSize: Int64? {
        switch self {
        case .yoozLight: return LLMModelType.yoozLight.estimatedSize
        case .yoozQuality: return LLMModelType.yoozQuality.estimatedSize
        case .foundationModels: return nil
        }
    }
}
