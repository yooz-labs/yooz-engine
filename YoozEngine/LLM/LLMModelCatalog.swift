// LLMModelCatalog.swift
// LLMModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore
import Foundation

/// Bridges the LLM model catalog into the EngineCore disk-hygiene layer.
///
/// `ModelStore` (EngineCore) is pure-Foundation and knows nothing about
/// `LLMModelType`; this maps each known LLM model to the on-disk locations its
/// weights can occupy (HF hub repo, models-directory copy, app bundle) so the
/// management routes and the cleanup migration can size, delete, and dedupe them.
public enum LLMModelCatalog {
    /// One `ModelCacheDescriptor` per known LLM model.
    public static func cacheDescriptors() -> [ModelCacheDescriptor] {
        LLMModelType.allCases.map { type in
            ModelCacheDescriptor(
                id: type.rawValue,
                module: "llm",
                hfRepoDirName: ModelCacheDescriptor.hubRepoDirName(
                    forHuggingFaceID: type.huggingFaceID
                ),
                // The LLM resolver probes `modelsDirectory/<rawValue>` (see
                // `MLXLLMBackend.bundledModelDirectory`).
                modelsDirSubdir: type.rawValue,
                isBundled: MLXLLMBackend.isBundled(type)
            )
        }
    }
}
