// ModulesResponse.swift
// EngineCore
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

// `ModulesResponse` / `ModuleManifest` moved to `YoozEngineWire` (#225) —
// visible here via `WireReexport.swift`. This file now only holds the
// `AIModule`-aware construction helper, which stays in `EngineCore` because
// it depends on `AIModule` / `ModuleHealth`, both `EngineCore`-only
// orchestration types `YoozEngineWire` deliberately does not know about.

public extension ModulesResponse {
    /// Build a response by running `healthCheck()` on each provided module.
    ///
    /// Callers pass an already-ordered list (the registry's `all()` returns
    /// modules sorted by name). The helper exists in `EngineCore` so tests
    /// and the HTTP route share one construction path; route code stays a
    /// thin wrapper.
    ///
    /// - Parameters:
    ///   - modules: modules to probe; typically `await ModuleRegistry.shared.all()`.
    ///   - engineVersion: engine semver; typically `EngineConfig.version`.
    ///   - buildVariant: variant id; typically `BuildVariant.current.rawValue`.
    static func build(
        from modules: [any AIModule],
        engineVersion: String,
        buildVariant: String
    ) async -> ModulesResponse {
        var manifests: [ModuleManifest] = []
        manifests.reserveCapacity(modules.count)
        for module in modules {
            let health = await module.healthCheck()
            manifests.append(ModuleManifest(
                name: type(of: module).name,
                version: engineVersion,
                loaded: health.loaded,
                error: health.error,
                detail: health.detail
            ))
        }
        return ModulesResponse(
            engineVersion: engineVersion,
            buildVariant: buildVariant,
            modules: manifests
        )
    }
}
