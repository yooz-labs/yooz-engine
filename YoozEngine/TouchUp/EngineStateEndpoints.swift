// EngineStateEndpoints.swift
// LLMModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore
import Foundation

/// Table endpoint for `GET /v1/state` (engine#226): the cross-module
/// snapshot a picker UI fetches once, before opening `/v1/events`, so it
/// can render immediately instead of waiting on the first pushed event.
///
/// Lives in LLMModule — today's only contributor is the TouchUp picker
/// (`TouchUpEngine`). A future module adopting the engine-owned-selection
/// contract (STT engine, Infinite — see AGENTS.md "Module model picker
/// pattern") adds its own row-builder call here; `EngineStateSnapshot`'s
/// shape does not need to change, since `EngineModelSnapshotRow` only
/// carries the canonical picker fields every module already publishes.
public enum EngineStateEndpoints {
    public static func endpoints() -> [Endpoint] {
        [
            Endpoint(EndpointSpecs.engineState) { _ in
                let touchUp = await touchUpSnapshot()
                return try WireResponse.json(EngineStateSnapshot(modules: [touchUp]))
            },
        ]
    }

    static func touchUpSnapshot() async -> EngineModuleSnapshot {
        let models = await TouchUpEngine.shared.availableModels()
        let activeId = await TouchUpEngine.shared.activeModel.rawValue
        return EngineModuleSnapshot(
            module: TouchUpEngine.selectionStoreModule,
            models: models.map {
                EngineModelSnapshotRow(
                    id: $0.id, displayName: $0.displayName, description: $0.description,
                    tier: $0.tier, sizeBytes: $0.sizeBytes, loadState: $0.loadState,
                    isActive: $0.isActive
                )
            },
            activeId: activeId
        )
    }
}
