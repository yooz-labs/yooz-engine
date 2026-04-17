// ModuleNotBundledResponse.swift
// EngineCore
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Response body returned by the server when a client hits an endpoint whose
/// backing module is not linked into the current build variant.
///
/// Paired with HTTP 501 (Not Implemented). The shape is intentionally a
/// superset of `ErrorResponse` (`error` + `code`) with an extra `module` field
/// so thin clients can identify the missing capability without parsing
/// the human-readable message. See A4 / #28.
///
/// Example JSON:
/// ```json
/// {
///   "error": "Module 'vad' not bundled in this build variant (whisper)",
///   "code": "module_not_bundled",
///   "module": "vad"
/// }
/// ```
public struct ModuleNotBundledResponse: Sendable, Codable, Equatable {
    /// Human-readable message naming the module and variant.
    public let error: String
    /// Stable error code clients can switch on. Always `"module_not_bundled"`.
    public let code: String
    /// The unbundled module's name (`"stt"`, `"vad"`, `"llm"`, `"grammar"`, ...).
    public let module: String

    public init(module: String, buildVariant: String) {
        self.module = module
        self.code = Self.code
        self.error = "Module '\(module)' not bundled in this build variant (\(buildVariant))"
    }

    /// Explicit designated initializer used by `Codable` and tests that
    /// need to pin the exact `error` string.
    public init(error: String, code: String, module: String) {
        self.error = error
        self.code = code
        self.module = module
    }

    /// Stable error code emitted in `code` for every instance.
    public static let code = "module_not_bundled"
}
