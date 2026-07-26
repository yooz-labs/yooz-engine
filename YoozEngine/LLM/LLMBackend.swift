// LLMBackend.swift
// LLMModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore
import Foundation

// MARK: - Model Types

/// A servable Yooz LLM model, resolved against the engine's curated
/// catalogue (`LLMModelCatalog`, engine#303).
///
/// Catalogue-as-data, not identity-as-enum: the engine curates a list of
/// models it is willing to serve — same shape as ollama's library — and
/// this type is a typed handle onto one catalogue entry. `/v1/llm/generate`
/// accepts any catalogued model's `rawValue` as the `model` field, not just
/// a fixed pair of TouchUp proofreading tiers.
///
/// `init?(rawValue:)` resolves against `LLMModelCatalog.entries` by EITHER
/// the canonical wire id (e.g. `"yooz-light-v3"`) OR the model's full
/// Hugging Face repo id (alias resolution) — a caller that names the HF
/// repo directly keeps working without a repoint. Membership is curated,
/// not a bare `YoozLabs/` prefix rule: not every repo under the org is an
/// MLX causal LM this backend can serve (e.g. `YoozLabs/Qwen3-ASR-1.7B-8bit`
/// is ASR, `YoozLabs/YoozTextCleanup` is an xcframework) — see
/// `LLMModelCatalog` for the prefix invariant enforced on curated entries.
///
/// `yooz-light-v3` and `yooz-quality-v3` are STABLE wire ids; consumer SDKs
/// depend on them — never rename.
///
/// `Hashable`/`Equatable` are BY `rawValue` ALONE: `TouchUpEngine` uses this
/// type as a dictionary key (`loadStates`, `lastLoadErrors`,
/// `inFlightLoadTasks`, and its per-tier backend cache), and every non-nil
/// instance is catalogue-sourced (the memberwise fields cannot diverge from
/// `rawValue` in practice), so rawValue equality is both sufficient and the
/// deliberately narrow contract.
///
/// `public` because `APIServer` (a different target on the modular
/// build) consumes this type directly via the picker routes.
public struct LLMModelType: RawRepresentable, Hashable, Sendable, CaseIterable {
    public let rawValue: String
    /// Hugging Face model identifier. Pulled by
    /// `loadModelContainer(from: #hubDownloader(), …, configuration:)`
    /// on first load. `revision` defaults to `main`; pin a commit here
    /// only if a future upstream change breaks compatibility with our
    /// backend assumptions.
    public let huggingFaceID: String
    public let displayName: String
    public let description: String
    /// Approximate on-disk size after HF download (used for picker UX
    /// hints in consumer apps). Numbers are the published 4-bit MLX
    /// snapshot sizes, not raw weights.
    public let estimatedSize: Int64
    /// Best-effort per-model latency baseline in milliseconds for picker
    /// UX hints (`LLMModelInfo.latencyHintMs`); keep consistent with
    /// `description` above.
    public let latencyHintMs: Int
    /// Proofreading head vs. general/classify base (engine#303) — the
    /// whole point of catalogue-as-data: consumers can tell these apart
    /// instead of every catalogued model implicitly being a TouchUp tier.
    public let purpose: LLMModelPurpose

    private init(entry: LLMCatalogEntry) {
        self.rawValue = entry.id
        self.huggingFaceID = entry.huggingFaceID
        self.displayName = entry.displayName
        self.description = entry.description
        self.estimatedSize = entry.estimatedSize
        self.latencyHintMs = entry.latencyHintMs
        self.purpose = entry.purpose
    }

    public init?(rawValue: String) {
        guard let entry = LLMModelCatalog.entry(rawValue: rawValue) else { return nil }
        self.init(entry: entry)
    }

    public static var allCases: [LLMModelType] {
        LLMModelCatalog.entries.map(LLMModelType.init(entry:))
    }

    public static func == (lhs: LLMModelType, rhs: LLMModelType) -> Bool {
        lhs.rawValue == rhs.rawValue
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(rawValue)
    }

    // MARK: - Known catalogue entries
    //
    // Static members so every existing `.yoozLight` / `LLMModelType.yoozLight`
    // call site (dozens, across APIServer/TouchUpEngine/tests) keeps compiling
    // unchanged even though this is now a struct, not an enum with cases.
    // Force-unwrapped: each id is asserted present in `LLMModelCatalog.entries`
    // by `LLMModelCatalogTests`, so a typo here is a build-time-adjacent test
    // failure, not a runtime crash surface.

    /// Fast proofread tier. Yooz-Light v3, fused 6-bit on the KD
    /// Qwen3.5-0.8B QAT base (yooz-benchmark#29).
    public static let yoozLight = LLMModelType(rawValue: "yooz-light-v3")!
    /// High-quality rewrite tier. Yooz-Quality v3, fused 6-bit on the KD
    /// Qwen3.5-4B QAT base. Replaces the former Qwen3.5-9B fallback.
    public static let yoozQuality = LLMModelType(rawValue: "yooz-quality-v3")!
    /// General instruct / classify base — untuned QAT-lean KD, same lineage
    /// as the proofreading tiers but NOT a TouchUp head (`purpose == .general`).
    /// First non-proofreading catalogue addition (engine#303); measured
    /// 38/38 on remi's auto-approve permission grid with 0 unparsable
    /// responses, vs. `yoozQuality`'s 6 silently-unparsable "passes".
    public static let yoozInstruct4B = LLMModelType(rawValue: "yooz-instruct-4b")!
}

// MARK: - Errors

public enum LLMError: Error, LocalizedError, Sendable {
    case notLoaded
    case loadFailed(String)
    case generationFailed(String)
    case notAvailable(String)
    case downloadFailed(String)
    case parsingFailed(String)

    public var errorDescription: String? {
        switch self {
        case .notLoaded:
            return "Model not loaded"
        case .loadFailed(let reason):
            return "Failed to load model: \(reason)"
        case .generationFailed(let reason):
            return "Generation failed: \(reason)"
        case .notAvailable(let reason):
            return "Model not available: \(reason)"
        case .downloadFailed(let reason):
            return "Download failed: \(reason)"
        case .parsingFailed(let reason):
            return "Failed to parse response: \(reason)"
        }
    }
}

// MARK: - Protocol

/// Protocol for LLM backends used in touch-up processing.
///
/// Kept `internal` on purpose; `TouchUpEngine` is the only out-of-module
/// caller and it exposes its own domain API, not the backend abstraction.
protocol LLMBackend: Actor {
    var identifier: String { get }
    var modelType: LLMModelType { get }
    var isLoaded: Bool { get }

    func load() async throws
    func unload()
    /// `workloadClass` (engine#228) tells the backend whether this call is
    /// latency-sensitive (`.interactive`, admitted immediately) or
    /// throughput work that can queue/yield behind interactive activity
    /// (`.background`, the default every existing caller gets via
    /// `TouchUpProcessor.process`). No default here — the one existential
    /// call site (`TouchUpProcessor.process`) passes it explicitly so the
    /// classification is never accidentally implicit.
    /// `postProcess` selects the PROOFREADING salvage pass (engine#312).
    /// True for touch-up, where returning the original text when the model
    /// chatters is the desired "nothing to correct" behaviour. False for raw
    /// generation, where that same rule turns a refusal or an empty completion
    /// into a verbatim echo of the caller's prompt.
    func generate(
        prompt: String,
        systemPrompt: String,
        workloadClass: MLXWorkloadClass,
        postProcess: Bool
    ) async throws -> String
}
