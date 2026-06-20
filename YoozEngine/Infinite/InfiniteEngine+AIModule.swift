// InfiniteEngine+AIModule.swift
// InfiniteModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore

extension InfiniteEngine: AIModule, SessionResettable {
    public static var name: String { "infinite" }

    public var isReady: Bool {
        get async { isLoaded }
    }

    public func healthCheck() async -> ModuleHealth {
        let current = activeModel
        return ModuleHealth(
            loaded: isLoaded,
            error: isLoaded ? nil : "Infinite generation backend not loaded; sessions and adapter readiness are available",
            detail: [
                "active_model": current.rawValue,
                "backend_kind": current.backendKind,
                "adapter_kind": current.adapterKind,
                "hf_repo": current.huggingFaceID ?? "none",
                "active_sessions": String(status().activeSessions),
                "cleanup_policy": Self.cleanupPolicy,
                "max_context_tokens": String(current.maxContextTokens),
                "ram_tier": current.ramTier
            ]
        )
    }

    public func resetForNewSession() async {
        resetForRecordingBoundary()
    }
}
