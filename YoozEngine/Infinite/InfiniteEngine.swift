// InfiniteEngine.swift
// InfiniteModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore
import Foundation

public enum InfiniteError: Error, LocalizedError, Sendable, Equatable {
    case invalidModel(String)
    case modelUnavailable(String)
    case modelSetFailed(String)

    public var errorDescription: String? {
        switch self {
        case .invalidModel(let id):
            return "Unknown Infinite model: \(id)"
        case .modelUnavailable(let id):
            return "Infinite model is unavailable on this system: \(id)"
        case .modelSetFailed(let reason):
            return "Failed to set Infinite model: \(reason)"
        }
    }
}

public actor InfiniteEngine {
    public static let shared = InfiniteEngine()

    public private(set) var activeModel: InfiniteModelSelection = .gemma4E4B1M
    private var loadedModel: InfiniteModelSelection?
    private var lastLoadError: String?
    private var activeSessionCount = 0

    private init() {}

    public var isLoaded: Bool {
        loadedModel == activeModel
    }

    public func availableModels() -> [InfiniteModelInfo] {
        let models = InfiniteModelSelection.allCases.map { info(for: $0) }
        precondition(
            models.filter(\.isActive).count == 1,
            "Infinite picker must expose exactly one active row"
        )
        return models
    }

    public func setActiveModel(
        _ selection: InfiniteModelSelection,
        preload: Bool
    ) async throws -> InfiniteModelInfo {
        guard isModelSelectable(selection) else {
            throw InfiniteError.modelUnavailable(selection.rawValue)
        }

        activeModel = selection
        lastLoadError = nil

        if preload {
            // Phase 1 is contract/scaffold only. Real backend loading lands
            // in Phase 2; preload=false is the route-test happy path.
            lastLoadError = "Infinite backend loading is not implemented in Phase 1"
            throw InfiniteError.modelSetFailed(lastLoadError!)
        }

        return info(for: selection)
    }

    public func status() -> InfiniteStatus {
        InfiniteStatus(
            loaded: isLoaded,
            modelId: activeModel.rawValue,
            progress: nil,
            state: isLoaded ? "ready" : "idle",
            activeSessions: activeSessionCount,
            maxContextTokens: activeModel.maxContextTokens,
            ramTier: activeModel.ramTier,
            backendKind: activeModel.backendKind,
            lastError: lastLoadError
        )
    }

    public func resetForRecordingBoundary() {
        // The engine-wide /v1/session/begin boundary is per recording.
        // It must not unload models or wipe future durable Infinite
        // contexts. Phase 3 will add explicit long-context session APIs.
        activeSessionCount = 0
    }

    private func info(for selection: InfiniteModelSelection) -> InfiniteModelInfo {
        InfiniteModelInfo(
            id: selection.rawValue,
            displayName: selection.displayName,
            description: selection.description,
            tier: selection.tier,
            sizeBytes: selection.sizeBytes,
            loadState: loadState(for: selection),
            isActive: selection == activeModel,
            maxContextTokens: selection.maxContextTokens,
            ramTier: selection.ramTier,
            backendKind: selection.backendKind,
            requiresAppleSilicon: true,
            evidenceRef: selection.evidenceRef
        )
    }

    private func loadState(for selection: InfiniteModelSelection) -> ModelLoadState {
        guard isModelSelectable(selection) else { return .unavailable }
        return loadedModel == selection ? .loaded : .available
    }

    private func isModelSelectable(_ selection: InfiniteModelSelection) -> Bool {
        #if arch(arm64)
        return InfiniteRAMTier.current.supports(required: selection.requiredRAMTier)
        #else
        return false
        #endif
    }
}
