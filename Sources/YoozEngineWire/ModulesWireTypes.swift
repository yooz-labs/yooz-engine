// ModulesWireTypes.swift
// YoozEngineWire
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Top-level response body for `GET /v1/modules`.
///
/// Describes every module compiled into the running build variant along with
/// its current health. Thin clients (yooz-whisper, yooz-notes, ...) read this
/// to decide which endpoints the engine can serve before issuing real
/// requests. Modules not bundled in the variant are simply absent; unbundled
/// endpoints return HTTP 501 (see A4 / #28).
public struct ModulesResponse: Sendable, Codable, Equatable {
    /// Engine semantic version (`EngineConfig.version`). Identical to each
    /// manifest's `version` under the unified-versioning scheme locked in A1.
    public let engineVersion: String

    /// Which build flavor this process is (`full`, `whisper`, ...). Matches
    /// `BuildVariant.current.rawValue` at the server.
    public let buildVariant: String

    /// Every registered module, sorted by `name` for deterministic output.
    public let modules: [ModuleManifest]

    public init(engineVersion: String, buildVariant: String, modules: [ModuleManifest]) {
        self.engineVersion = engineVersion
        self.buildVariant = buildVariant
        self.modules = modules
    }
}

/// Per-module entry inside `ModulesResponse.modules`.
///
/// `version` is the engine version (unified scheme). `loaded` and `error`
/// come from `AIModule.healthCheck()`. `detail` carries module-specific
/// key/value pairs (e.g. `{"language": "en"}` for STT); ordering is not
/// guaranteed by `[String: String]` at the Swift level, so the server
/// encodes with sorted keys for deterministic JSON output.
public struct ModuleManifest: Sendable, Codable, Equatable {
    public let name: String
    public let version: String
    public let loaded: Bool
    public let error: String?
    public let detail: [String: String]

    public init(name: String, version: String, loaded: Bool, error: String?, detail: [String: String]) {
        self.name = name
        self.version = version
        self.loaded = loaded
        self.error = error
        self.detail = detail
    }
}
