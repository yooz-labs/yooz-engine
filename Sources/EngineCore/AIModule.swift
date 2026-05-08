// AIModule.swift
// EngineCore
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Protocol every yooz-engine AI module conforms to.
///
/// A module is a self-contained AI capability (STT, Grammar, LLM, VAD, TTS)
/// that ships as its own framework target. The `YoozEngine` app wires modules
/// into the HTTP/WS server via `ModuleRegistry`.
///
/// Build variants (see `BuildVariant`) determine which modules are linked.
/// Calls to unbundled modules return HTTP 501 from the server layer.
public protocol AIModule: Sendable {
    /// Stable identifier used in `/v1/modules` and internal routing.
    /// Must be lowercase, no spaces (e.g. "stt", "grammar", "llm", "vad", "tts").
    static var name: String { get }

    /// Whether this module is loaded and ready to accept requests.
    ///
    /// Implementations should be cheap (no network/FFI calls) so health checks
    /// stay fast. For modules that require explicit loading (STT model,
    /// LLM weights), this reflects whether the load completed successfully.
    var isReady: Bool { get async }

    /// Detailed health information for `/v1/modules` manifest entries.
    ///
    /// The returned `ModuleHealth` carries a loaded flag, optional error
    /// message, and module-specific detail keys (e.g. STT language,
    /// grammar rule count).
    func healthCheck() async -> ModuleHealth
}
