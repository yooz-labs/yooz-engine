// EngineConfig+STT.swift
// STTModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore
import Foundation

/// STT-specific `EngineConfig` extensions.
///
/// `EngineConfig` lives in `EngineCore`, which can't import `STTModule`
/// without creating a cycle. Properties that reference STT-only types
/// (`STTBackendID`, `STTLanguage`) live here so the modular boundary
/// stays one-way: `STTModule -> EngineCore`.
///
/// Both properties below are computed (not cached `static let`) so tests
/// can drive them via `setenv` / `unsetenv` without spawning a subprocess.
extension EngineConfig {
    /// Default STT backend resolved at startup. Driven by the
    /// `YOOZ_STT_BACKEND` env var so dev and tests can flip the flag
    /// without writing a config file. Unknown values fall back to
    /// `.parakeet` rather than crashing.
    public static var sttBackend: STTBackendID {
        guard
            let raw = ProcessInfo.processInfo.environment[
                "YOOZ_STT_BACKEND"
            ],
            let parsed = STTBackendID(rawValue: raw)
        else {
            return .parakeet
        }
        return parsed
    }

    /// Default STT language to eager-load on the full / whisper
    /// variants. Driven by `YOOZ_DEFAULT_STT_LANG` so a user with a
    /// different primary language pays the eager-load cost on the
    /// right model. Falls back to English on unknown values.
    public static var defaultSTTLanguage: STTLanguage {
        guard
            let raw = ProcessInfo.processInfo.environment[
                "YOOZ_DEFAULT_STT_LANG"
            ],
            let parsed = STTLanguage.fromCode(raw),
            parsed.isImplemented
        else {
            return .english
        }
        return parsed
    }
}
