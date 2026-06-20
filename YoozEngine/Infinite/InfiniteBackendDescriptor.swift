// InfiniteBackendDescriptor.swift
// InfiniteModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

public struct InfiniteModelRepository: Codable, Sendable, Equatable {
    public let id: String
    public let revision: String

    public init(id: String, revision: String) {
        self.id = id
        self.revision = revision
    }
}

public enum InfiniteBackendKind: String, Codable, Sendable {
    case pagedKV = "paged-kv"
    case retrieval
}

public enum InfiniteAdapterKind: String, Codable, Sendable {
    case pagedKVMLX = "infinite-paged-kv-mlx-v1"
    case retrievalIndex = "infinite-retrieval-index-v1"
}

public struct InfiniteBackendDescriptor: Codable, Sendable, Equatable {
    public let selection: InfiniteModelSelection
    public let repository: InfiniteModelRepository?
    public let backendKind: InfiniteBackendKind
    public let adapterKind: InfiniteAdapterKind
    public let nativeContextTokens: Int
    public let targetContextTokens: Int
    public let requiredRAMTier: InfiniteRAMTier
    public let requiredCachedFiles: [String]

    public init(
        selection: InfiniteModelSelection,
        repository: InfiniteModelRepository?,
        backendKind: InfiniteBackendKind,
        adapterKind: InfiniteAdapterKind,
        nativeContextTokens: Int,
        targetContextTokens: Int,
        requiredRAMTier: InfiniteRAMTier,
        requiredCachedFiles: [String] = ["config.json", ".safetensors"]
    ) {
        self.selection = selection
        self.repository = repository
        self.backendKind = backendKind
        self.adapterKind = adapterKind
        self.nativeContextTokens = nativeContextTokens
        self.targetContextTokens = targetContextTokens
        self.requiredRAMTier = requiredRAMTier
        self.requiredCachedFiles = requiredCachedFiles
    }
}

public struct InfiniteBackendHandle: Sendable, Equatable {
    public let selection: InfiniteModelSelection
    public let adapterKind: InfiniteAdapterKind
    public let repository: InfiniteModelRepository?

    public init(
        selection: InfiniteModelSelection,
        adapterKind: InfiniteAdapterKind,
        repository: InfiniteModelRepository?
    ) {
        self.selection = selection
        self.adapterKind = adapterKind
        self.repository = repository
    }
}

public protocol InfiniteBackendAdapter: Sendable {
    func prepare(_ descriptor: InfiniteBackendDescriptor) async throws -> InfiniteBackendHandle
}

public struct CatalogInfiniteBackendAdapter: InfiniteBackendAdapter {
    public init() {}

    public func prepare(_ descriptor: InfiniteBackendDescriptor) async throws -> InfiniteBackendHandle {
        InfiniteBackendHandle(
            selection: descriptor.selection,
            adapterKind: descriptor.adapterKind,
            repository: descriptor.repository
        )
    }
}
