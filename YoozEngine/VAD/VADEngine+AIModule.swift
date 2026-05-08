// VADEngine+AIModule.swift
// VADModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore

/// Conformance to `AIModule` so `ModuleRegistry` can discover this module.
extension VADEngine: AIModule {
    public static var name: String { "vad" }

    /// Mirrors `isLoaded`. Async because `VADEngine` is an actor and the
    /// load flag is isolated state.
    public var isReady: Bool {
        get async { isLoaded }
    }

    public func healthCheck() async -> ModuleHealth {
        let loaded = isLoaded
        return ModuleHealth(
            loaded: loaded,
            error: loaded ? nil : "Silero VAD model not loaded; call load() first",
            detail: [
                "model": "silero-vad-unified-v6.0.0",
                "sample_rate": String(Self.sampleRate),
                "window_size": String(Self.windowSize)
            ]
        )
    }
}
