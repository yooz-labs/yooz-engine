// LLMModelCatalog.swift
// LLMModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore
import Foundation

/// One entry in the engine's curated LLM catalogue (engine#303): everything
/// `LLMModelType` needs, as plain stored data rather than an enum case.
struct LLMCatalogEntry: Sendable {
    let id: String
    let huggingFaceID: String
    let displayName: String
    let description: String
    let estimatedSize: Int64
    let latencyHintMs: Int
    let purpose: LLMModelPurpose
}

/// The engine's curated list of servable LLM models (engine#303), and the
/// bridge from that list into the EngineCore disk-hygiene layer.
///
/// Owner framing: "the engine determines the models to determine the
/// experience, similar to the ollama model" — each consumer app (whisper,
/// remi, ...) picks from this catalogue rather than the engine accepting
/// any identifier that merely round-trips a naming convention. Membership
/// here is what gates `POST /v1/llm/generate`'s `invalid_model` 400
/// (`LLMModelType(rawValue:)` resolves against `entries`) — adding a model
/// to `entries` is the ONLY step required to make it servable and visible
/// on `GET /v1/llm/models`.
///
/// `ModelStore` (EngineCore) is pure-Foundation and knows nothing about
/// `LLMModelType`; `cacheDescriptors()` maps each catalogued model to the
/// on-disk locations its weights can occupy (HF hub repo, models-directory
/// copy, app bundle) so the management routes and the cleanup migration can
/// size, delete, and dedupe them — this falls out of `LLMModelType.allCases`
/// for free, with no separate catalogue to keep in sync.
public enum LLMModelCatalog {
    /// Curated entries, in display order. `yooz-light-v3` / `yooz-quality-v3`
    /// are STABLE wire ids consumer SDKs depend on — never rename. Every
    /// `huggingFaceID` MUST start with `YoozLabs/`, asserted below: this is a
    /// curated catalogue, not a bare-prefix gate — not every `YoozLabs/` repo
    /// is an MLX causal LM this backend can load (e.g.
    /// `YoozLabs/Qwen3-ASR-1.7B-8bit` is ASR, `YoozLabs/YoozTextCleanup` is an
    /// xcframework), so a prefix-only rule would turn a wrong repo id into a
    /// multi-GB download that fails at load instead of a clean 400.
    static let entries: [LLMCatalogEntry] = {
        let curated: [LLMCatalogEntry] = [
            LLMCatalogEntry(
                id: "yooz-light-v3",
                huggingFaceID: "YoozLabs/Yooz-Light-v3-Qwen3.5-0.8B",
                displayName: "Yooz-Light",
                description: "Fast proofreading (~300ms)",
                estimatedSize: 605 * 1024 * 1024,   // ~605 MB (fused 6-bit)
                latencyHintMs: 300,
                purpose: .proofread
            ),
            LLMCatalogEntry(
                id: "yooz-quality-v3",
                huggingFaceID: "YoozLabs/Yooz-Quality-v3-Qwen3.5-4B",
                displayName: "Yooz-Quality",
                description: "High quality rewriting (~1s)",
                estimatedSize: 3277 * 1024 * 1024,  // ~3.2 GB (fused 6-bit)
                latencyHintMs: 1200,
                purpose: .proofread
            ),
            // First non-proofreading addition (engine#303): an untuned
            // QAT-lean KD base, same lineage as the tiers above but not a
            // TouchUp head. remi's auto-approve JSON-classify workload
            // measured 38/38 on the permission grid at 0 unparsable
            // responses, vs. yoozQuality's 6 silently-unparsable "passes"
            // (a proofreader echoing dangerous input back verbatim reads as
            // a safe verdict only by accident). yooz-benchmark research/issue-24.
            LLMCatalogEntry(
                id: "yooz-instruct-4b",
                huggingFaceID: "YoozLabs/Qwen3.5-4B-qat-lean-4bit-mlx",
                displayName: "Yooz-Instruct-4B",
                description: "General instruct / classify, untuned QAT-lean base (~1s)",
                estimatedSize: 2370 * 1024 * 1024,  // ~2.37 GB
                latencyHintMs: 1000,
                purpose: .general
            ),
        ]
        for entry in curated {
            precondition(
                entry.huggingFaceID.hasPrefix("YoozLabs/"),
                "LLMModelCatalog entry '\(entry.id)' huggingFaceID " +
                    "'\(entry.huggingFaceID)' must start with 'YoozLabs/'"
            )
        }
        return curated
    }()

    /// Resolve a catalogue entry by canonical wire id OR full Hugging Face
    /// repo id (alias resolution) — see `LLMModelType.init(rawValue:)`.
    static func entry(rawValue: String) -> LLMCatalogEntry? {
        entries.first { $0.id == rawValue || $0.huggingFaceID == rawValue }
    }

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
                isBundled: MLXLLMBackend.isBundled(type),
                // Carried so `GET /v1/models` can name the registered repo
                // alongside the canonical id (engine#308).
                huggingFaceID: type.huggingFaceID
            )
        }
    }
}
