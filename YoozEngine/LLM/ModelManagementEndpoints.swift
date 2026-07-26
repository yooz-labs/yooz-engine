// ModelManagementEndpoints.swift
// LLMModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore
import Foundation
import os

private let logger = Logger(
    subsystem: "live.yooz.engine", category: "model-management-endpoints"
)

/// Table endpoints for the model-management (disk-hygiene) family
/// (engine#225 Phase B): `GET /v1/models`, `DELETE /v1/models/:id`,
/// `POST /v1/models/cleanup`.
///
/// Lives in LLMModule because the LLM side of the inventory
/// (`TouchUpEngine` picker state + `LLMModelCatalog` descriptors) is
/// module-owned; the one STT-owned input — the active STT model's hub repo
/// dir name, which the delete route refuses to remove — is injected as a
/// closure so this file does not depend on `STTModule` (whose linkage varies
/// by build variant; Lite has none and injects `{ nil }`). Before the table,
/// both transports carried a full copy of all three handlers plus the
/// `llmInventoryInputs` assembly; the only behavioral daylight between the
/// copies was error rendering (the in-process copy rethrew raw delete/cleanup
/// failures where the loopback wrapped them in `delete_failed` /
/// `cleanup_failed`) — both now speak the loopback's typed codes.
public enum ModelManagementEndpoints {
    public static func endpoints(
        activeSTTRepoDirName: @escaping @Sendable () -> String?
    ) -> [Endpoint] {
        [
            // `GET /v1/models` — the cross-module inventory with real
            // on-disk sizes: the friendly LLM catalog (with bundle
            // awareness) plus a disk-first sweep of every other hub repo
            // (Parakeet, legacy) so nothing consuming disk is hidden. See
            // `ModelStore.inventory`.
            Endpoint(EndpointSpecs.modelsInventory) { _ in
                let store = ModelStore()
                let rows = await store.inventory(
                    llm: llmInventoryInputs(),
                    activeSTTRepoDirName: activeSTTRepoDirName()
                )
                let models = rows.map {
                    ManagedModelInfo(
                        id: $0.id, module: $0.module, displayName: $0.displayName,
                        sizeBytes: $0.sizeBytes, cached: $0.cached, loaded: $0.loaded,
                        isActive: $0.isActive, deletable: $0.deletable
                    )
                }
                return try WireResponse.json(ManagedModelsResponse(models: models))
            },
            // `DELETE /v1/models/:id` — remove one model's reclaimable
            // on-disk copies. Unloads it from memory only after the disk
            // delete succeeds (a failed delete leaves the model fully
            // usable, not unloaded-but-present); refuses to delete the
            // active model (409).
            Endpoint(EndpointSpecs.modelsDelete) { request in
                // Defensive only: unreachable through either transport —
                // the table's matcher never dispatches this endpoint
                // without capturing `:id`, and the loopback adapter
                // extracts it via the router. Kept as a typed 400 (not a
                // force-unwrap) so a hypothetical future caller that
                // hand-builds a WireRequest fails loudly and safely.
                guard let id = request.pathParameters["id"] else {
                    throw WireError(
                        status: 400, code: "invalid_request",
                        message: "Missing 'id' path parameter"
                    )
                }
                let store = ModelStore()

                if let modelType = LLMModelType(rawValue: id) {
                    let active = await TouchUpEngine.shared.activeModel
                    if active.rawValue == id {
                        throw WireError(
                            status: 409, code: "model_active",
                            message: "Cannot delete the active model '\(id)'"
                        )
                    }
                    let descriptor = LLMModelCatalog.cacheDescriptors().first { $0.id == id }
                    do {
                        let reclaimed = try await store.deleteModel(
                            hfRepoDirName: descriptor?.hfRepoDirName,
                            modelsDirSubdir: descriptor?.modelsDirSubdir
                        )
                        await TouchUpEngine.shared.unload(modelType)
                        return try WireResponse.json(
                            DeleteModelResult(id: id, reclaimedBytes: reclaimed)
                        )
                    } catch {
                        logger.error(
                            "DELETE /v1/models/\(id) failed: \(error.localizedDescription)"
                        )
                        throw WireError(
                            status: 500, code: "delete_failed",
                            message: "Failed to delete '\(id)': \(error.localizedDescription)"
                        )
                    }
                }

                if id.hasPrefix("models--") {
                    if id == activeSTTRepoDirName() {
                        throw WireError(
                            status: 409, code: "model_active",
                            message: "Cannot delete the active model '\(id)'"
                        )
                    }
                    do {
                        let reclaimed = try await store.deleteModel(
                            hfRepoDirName: id, modelsDirSubdir: nil
                        )
                        return try WireResponse.json(
                            DeleteModelResult(id: id, reclaimedBytes: reclaimed)
                        )
                    } catch {
                        logger.error(
                            "DELETE /v1/models/\(id) failed: \(error.localizedDescription)"
                        )
                        throw WireError(
                            status: 500, code: "delete_failed",
                            message: "Failed to delete '\(id)': \(error.localizedDescription)"
                        )
                    }
                }

                throw WireError(
                    status: 404, code: "unknown_model",
                    message: "Unknown model '\(id)'"
                )
            },
            // `POST /v1/models/cleanup` — one-shot disk-hygiene migration:
            // collapse superseded snapshots + drop duplicates a
            // higher-priority copy supersedes. Idempotent.
            Endpoint(EndpointSpecs.modelsCleanup) { _ in
                let store = ModelStore()
                do {
                    let report = try await store.cleanupAll(
                        descriptors: LLMModelCatalog.cacheDescriptors()
                    )
                    return try WireResponse.json(ModelCleanupResult(
                        totalReclaimedBytes: report.totalReclaimedBytes,
                        perRepo: report.perRepo
                    ))
                } catch {
                    logger.error(
                        "POST /v1/models/cleanup failed: \(error.localizedDescription)"
                    )
                    throw WireError(
                        status: 500, code: "cleanup_failed",
                        message: "Cleanup failed: \(error.localizedDescription)"
                    )
                }
            },
        ]
    }

    /// LLM rows for the model-management inventory: the cache descriptors
    /// paired with live display/loaded/active state. Single copy — both
    /// transports previously duplicated this assembly.
    ///
    /// `loaded` comes from `getModelInfo()` (catalogue-wide, engine#303),
    /// not the TouchUp picker rows: the picker only ever names its three
    /// selectable entries (yooz-light-v3 / yooz-quality-v3 /
    /// foundation-models), so a generate-only catalogue model resident via
    /// `/v1/llm/generate` or `/v1/llm/preload` would otherwise report
    /// `loaded: false` here even while genuinely occupying memory.
    /// `displayName` / `isActive` stay picker-sourced — `isActive` is
    /// specifically "is this the TouchUp picker's active model", which is
    /// correctly always `false` for a model the picker cannot select.
    static func llmInventoryInputs() async -> [ModelStore.LLMInventoryInput] {
        let picker = await TouchUpEngine.shared.availableModels()
        let info = await TouchUpEngine.shared.getModelInfo()
        return LLMModelCatalog.cacheDescriptors().map { descriptor in
            let row = picker.first { $0.id == descriptor.id }
            let loaded = info.first { $0.type.rawValue == descriptor.id }?.isLoaded ?? false
            return ModelStore.LLMInventoryInput(
                descriptor: descriptor,
                displayName: row?.displayName ?? descriptor.id,
                loaded: loaded,
                isActive: row?.isActive ?? false
            )
        }
    }
}
