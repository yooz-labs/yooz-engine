// BuildVariant.swift
// EngineCore
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Which flavor of `YoozEngine.app` is currently running.
///
/// Set at build time via `SWIFT_ACTIVE_COMPILATION_CONDITIONS`. The standalone
/// menu-bar app is `.full` (all modules); apps that embed a slim engine
/// helper use dedicated variants (`.whisper`, future `.notes`, `.voice`).
public enum BuildVariant: String, Sendable, Codable {
    case full
    case whisper
    case lite

    public static let current: BuildVariant = {
        #if VARIANT_WHISPER
        return .whisper
        #elseif VARIANT_LITE
        return .lite
        #else
        return .full
        #endif
    }()
}
