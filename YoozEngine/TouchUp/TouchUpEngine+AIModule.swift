// TouchUpEngine+AIModule.swift
// LLMModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore

/// Conformance to `AIModule` so `ModuleRegistry` can discover this module.
///
/// LLMModule exposes one logical capability (text touch-up via LLM) even
/// though it bundles up to three backends (Yooz-Light, Yooz-Quality,
/// Apple Intelligence). The module name stays `llm` since the public API
/// and `/v1/llm/generate` + `/v1/touchup` endpoints both live here.
extension TouchUpEngine: AIModule, SessionResettable {
    public static var name: String { "llm" }

    /// Mirrors `isPreloaded`. Async because `TouchUpEngine` is an actor and
    /// preload state is isolated.
    public var isReady: Bool {
        get async { isPreloaded }
    }

    public func healthCheck() async -> ModuleHealth {
        let preloaded = isPreloaded
        let lightLoaded = await isLightModelLoaded
        let qualityLoaded = await isQualityModelLoaded
        let foundationLoaded = await isFoundationModelsLoaded
        return ModuleHealth(
            loaded: preloaded,
            error: preloaded ? nil : "TouchUpEngine not preloaded; call preload() to load MLX models",
            detail: [
                "light_model": LLMModelType.yoozLight.rawValue,
                "light_loaded": String(lightLoaded),
                "quality_model": LLMModelType.yoozQuality.rawValue,
                "quality_loaded": String(qualityLoaded),
                "foundation_models_loaded": String(foundationLoaded)
            ]
        )
    }
}
