// BuildVariant.swift
// EngineCore
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Which flavor of `YoozEngine.app` is currently running.
///
/// Read at runtime from `Bundle.main`'s `YoozBuildVariant` Info.plist key,
/// which is set per app target in `project.yml` via `INFOPLIST_KEY_*`.
///
/// **Why Info.plist instead of `SWIFT_ACTIVE_COMPILATION_CONDITIONS`:**
/// `BuildVariant` lives in the `EngineCore` framework, which is built once
/// and linked by all three app variants. Per-target compile conditions set
/// on `YoozEngine`, `YoozEngineWhisper`, or `YoozEngineLite` therefore never
/// reach `EngineCore.framework`'s compilation, and the static `#if` below
/// would always fall through to `.full`. Reading the host bundle's Info.plist
/// at runtime lets the same framework binary report the correct variant for
/// whichever app is linking it today.
///
/// The fallback is `.full` so unit-test bundles (which load EngineCore but
/// have no `YoozBuildVariant` key in `xctest`'s bundle) still behave as the
/// canonical "everything is present" variant.
public enum BuildVariant: String, Sendable, Codable {
    case full
    case whisper
    case lite

    /// Info.plist key written by `project.yml` per app target.
    public static let infoPlistKey = "YoozBuildVariant"

    /// Evaluated lazily (not at static init) so that tests injecting a custom
    /// `Bundle.main` override via swizzling see the new value. In production
    /// this is effectively a constant for the process lifetime.
    public static var current: BuildVariant {
        resolved(from: Bundle.main)
    }

    /// Seam for tests — resolves a variant from any bundle. Falls back to
    /// `.full` when the key is absent or unrecognised.
    public static func resolved(from bundle: Bundle) -> BuildVariant {
        guard
            let raw = bundle.object(forInfoDictionaryKey: infoPlistKey) as? String,
            let variant = BuildVariant(rawValue: raw)
        else {
            return .full
        }
        return variant
    }

    /// Whether MLX-based STT (Parakeet / FastConformer / Qwen3) is bundled
    /// into this variant. `.lite` drops it entirely and relies on Apple STT
    /// for transcription.
    public var includesMLXSTT: Bool {
        switch self {
        case .full, .whisper: return true
        case .lite: return false
        }
    }

    /// Whether the CoreML VAD model (`silero-vad-unified-v6.0.0`) is bundled
    /// into this variant. `.whisper` hosts its own embedded VAD because the
    /// ~64ms call rate makes an HTTP round-trip non-viable; `.lite` has no
    /// need for VAD on its hot path.
    public var includesVAD: Bool {
        switch self {
        case .full: return true
        case .whisper, .lite: return false
        }
    }

    /// Whether the LLM stack (MLX-Swift backends + Apple Intelligence when
    /// available) is bundled. All three variants ship LLM — it is the
    /// engine's primary value-add over native OS APIs.
    public var includesLLM: Bool { true }

    /// Grammar (`YoozTextCleanup` xcframework) is always linked. The Rust
    /// FFI loads on first reference; engine-side cost is ~0.
    public var includesGrammar: Bool { true }
}
