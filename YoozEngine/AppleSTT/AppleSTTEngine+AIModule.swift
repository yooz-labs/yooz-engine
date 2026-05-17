// AppleSTTEngine+AIModule.swift
// AppleSTTModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore
import Foundation

/// Conformance to `AIModule` so `ModuleRegistry` can discover this module.
///
/// The registered name is `apple_stt`, distinct from the MLX `stt` module so
/// both can coexist in full/whisper variants and be routed by engine type in
/// `/v1/stt/engine`. Lite variants register only this module.
extension AppleSTTEngine: AIModule, SessionResettable {
    public static var name: String { "apple_stt" }

    /// Detailed status for `/v1/modules`.
    ///
    /// Reports OS version (via `ProcessInfo.processInfo.operatingSystemVersionString`
    /// because the concrete `OperatingSystemVersion` struct isn't `Codable`),
    /// active backend kind, and current speech-recognition authorization.
    /// Authorization is surfaced here rather than as a hard load failure so
    /// consuming apps can discover the state and prompt the user via their
    /// own UI flow.
    public func healthCheck() async -> ModuleHealth {
        let osVersion = ProcessInfo.processInfo.operatingSystemVersionString
        let auth = AppleSTTEngine.authorizationStatus
        let kind = backendKind?.rawValue ?? "unresolved"
        let lang = currentLanguage.rawValue
        let bcp47 = currentLanguage.bcp47

        let loaded = isLoaded
        var error: String? = nil
        if !loaded {
            error = "AppleSTT engine not started; POST /v1/stt/engine or /v1/stt/load first"
        } else if auth != .authorized {
            error = "Speech recognition not authorized (status: \(auth.rawValue))"
        }

        return ModuleHealth(
            loaded: loaded && auth == .authorized,
            error: error,
            detail: [
                "os_version": osVersion,
                "backend_kind": kind,
                "authorization": auth.rawValue,
                "language": lang,
                "locale": bcp47,
                "has_built_in_vad": String(AppleSTTEngine.hasBuiltInVAD),
                "streaming": String(isStreaming)
            ]
        )
    }
}
