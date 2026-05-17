// YoozSTTEngine+AIModule.swift
// STTModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore

/// Conformance to `AIModule` so `ModuleRegistry` can discover this module.
///
/// STTModule exposes one logical capability (speech-to-text) even though it
/// bundles two model families (Parakeet TDT for English/European, FastConformer
/// for Arabic/Persian). The module name stays `stt`; `/v1/stt/*` endpoints
/// live here.
///
/// `YoozSTTEngine.isReady` already exists as a stored `@Published` flag, set
/// `true` on successful `start(language:)` and `false` on `stop()`. A sync
/// `Bool` satisfies the protocol's `get async` requirement with no override.
extension YoozSTTEngine: AIModule, SessionResettable {
    public static var name: String { "stt" }

    /// Per-recording-session reset (engine issue #114). Drops the active
    /// `StreamingTranscriber`'s accumulated audio buffer + finalized/draft
    /// token state so the next recording starts cold. Does NOT unload the
    /// Parakeet weights; `stop()` is the separate operation that does that.
    ///
    /// Idempotent: if no stream is active (the most common case at the very
    /// start of a recording, before the WS connect lands), `resetStream()`
    /// is a no-op.
    public func resetForNewSession() async {
        resetStream()
    }

    public func healthCheck() async -> ModuleHealth {
        let running = isRunning
        return ModuleHealth(
            loaded: running,
            error: running ? nil : "STT model not loaded; POST /v1/stt/load to load a language",
            detail: [
                "language": running ? currentLanguage.rawValue : "",
                "display_name": running ? currentLanguage.displayName : "",
                "model_identifier": running ? currentLanguage.modelIdentifier : "",
                "model_family": running ? currentLanguage.modelFamily.rawValue : "",
                "streaming": String(isStreaming)
            ]
        )
    }
}
