// ModuleHealth.swift
// EngineCore
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Structured health status for an AI module.
///
/// Returned by `AIModule.healthCheck()` and surfaced in the `/v1/modules`
/// API response. `detail` carries module-specific key/value pairs (e.g.
/// `stt` may set `{"language": "en", "model": "parakeet-tdt-0.6b"}`).
public struct ModuleHealth: Sendable, Codable, Equatable {
    public let loaded: Bool
    public let error: String?
    public let detail: [String: String]

    public init(loaded: Bool, error: String? = nil, detail: [String: String] = [:]) {
        self.loaded = loaded
        self.error = error
        self.detail = detail
    }
}
