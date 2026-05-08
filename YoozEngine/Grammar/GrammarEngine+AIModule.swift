// GrammarEngine+AIModule.swift
// GrammarModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore

/// Conformance to `AIModule` so `ModuleRegistry` can discover this module.
extension GrammarEngine: AIModule {
    public static var name: String { "grammar" }

    public nonisolated var isReady: Bool { isAvailable }

    public nonisolated func healthCheck() -> ModuleHealth {
        ModuleHealth(
            loaded: isAvailable,
            error: isAvailable ? nil : "Rust FFI loaded no rules; grammar correction disabled",
            detail: [
                "rules_total": String(totalRuleCount),
                "rules_simple": String(simpleRuleCount),
                "rules_pos": String(posRuleCount),
                "rules_programmatic": String(programmaticRuleCount),
                "library_version": version
            ]
        )
    }
}
