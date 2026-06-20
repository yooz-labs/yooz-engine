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
    private var preparedBackend: InfiniteBackendHandle?
    private var lastLoadError: String?
    private var activeSessionCount = 0
    private let backendAdapter: any InfiniteBackendAdapter

    init(backendAdapter: any InfiniteBackendAdapter = CatalogInfiniteBackendAdapter()) {
        self.backendAdapter = backendAdapter
    }

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

        let nextPreparedBackend: InfiniteBackendHandle?
        if preload {
            do {
                nextPreparedBackend = try await backendAdapter.prepare(selection.descriptor)
            } catch {
                lastLoadError = error.localizedDescription
                throw InfiniteError.modelSetFailed(error.localizedDescription)
            }
        } else {
            nextPreparedBackend = preparedBackend?.selection == selection ? preparedBackend : nil
        }

        activeModel = selection
        preparedBackend = nextPreparedBackend
        lastLoadError = nil

        return info(for: selection)
    }

    public func status() -> InfiniteStatus {
        InfiniteStatus(
            loaded: isLoaded,
            modelId: activeModel.rawValue,
            progress: nil,
            state: state,
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
            nativeContextTokens: selection.nativeContextTokens,
            ramTier: selection.ramTier,
            backendKind: selection.backendKind,
            adapterKind: selection.adapterKind,
            huggingFaceID: selection.huggingFaceID,
            revision: selection.revision,
            requiresAppleSilicon: true,
            evidenceRef: selection.evidenceRef
        )
    }

    private func loadState(for selection: InfiniteModelSelection) -> ModelLoadState {
        guard isModelSelectable(selection) else { return .unavailable }
        if loadedModel == selection {
            return .loaded
        }
        if InfiniteCacheProbe.isCached(selection.descriptor) {
            return .cached
        }
        return .available
    }

    private func isModelSelectable(_ selection: InfiniteModelSelection) -> Bool {
        #if arch(arm64)
        return InfiniteRAMTier.current.supports(required: selection.requiredRAMTier)
        #else
        return false
        #endif
    }

    private var state: String {
        if isLoaded {
            return "ready"
        }
        if preparedBackend?.selection == activeModel {
            return "adapter_ready"
        }
        return "idle"
    }
}
