// WireFixtureExportTests.swift
// YoozEngineClientTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import XCTest
@testable import YoozEngineClient

/// Captures v0.7.5-era wire JSON for the SDK-owned DTOs the #225 wire-type
/// consolidation moves into `YoozEngineWire`.
///
/// Not a normal assertion test: gated behind `EXPORT_WIRE_FIXTURES=1` and run
/// once, by hand, against the pre-refactor tree to freeze
/// `Tests/Fixtures/wire-v0.7.5/*.json`. Every other test target's
/// decode-compat suite reads the committed output; this generator does not
/// run in CI. Companion to `EngineCoreTests/WireFixtureExportTests.swift`
/// (which captures the `EngineCore`-owned DTOs). Kept in the repo so a future
/// DTO family migration can regenerate fixtures the same way.
final class WireFixtureExportTests: XCTestCase {
    func testExportSDKOwnedFixtures() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["EXPORT_WIRE_FIXTURES"] == "1",
            "set EXPORT_WIRE_FIXTURES=1 to (re)generate committed wire fixtures"
        )

        let dir = try fixturesDirectory()

        // MARK: Picker family

        try write(ModelTier.quality, "ModelTier", to: dir)
        try write(ModelLoadState.cached, "ModelLoadState", to: dir)

        let touchUpModelInfo = TouchUpModelInfo(
            id: "yooz-light-v2",
            displayName: "Yooz-Light",
            description: "Fast, on-device cleanup",
            tier: .light,
            sizeBytes: 550_000_000,
            loadState: .cached,
            isActive: true
        )
        try write(touchUpModelInfo, "TouchUpModelInfo", to: dir)
        try write(
            TouchUpModelsResponse(models: [touchUpModelInfo], activeId: "yooz-light-v2"),
            "TouchUpModelsResponse", to: dir
        )
        try write(
            TouchUpSetModelRequest(id: "yooz-light-v2", preload: true),
            "TouchUpSetModelRequest", to: dir
        )

        let sttBackendInfo = STTBackendInfo(
            id: "parakeet",
            displayName: "Parakeet TDT",
            description: "High-accuracy MLX STT",
            tier: .quality,
            sizeBytes: 600_000_000,
            loadState: .loaded,
            isActive: true,
            supportsBatch: true,
            supportsStreaming: true,
            supportedLanguages: ["en", "es"]
        )
        try write(sttBackendInfo, "STTBackendInfo", to: dir)
        try write(
            STTBackendsResponse(backends: [sttBackendInfo], activeId: "parakeet"),
            "STTBackendsResponse", to: dir
        )
        try write(
            STTSetBackendRequest(id: "parakeet", preload: true),
            "STTSetBackendRequest", to: dir
        )

        // MARK: Models/status/modules family

        let managedModelInfo = ManagedModelInfo(
            id: "yooz-light-v2",
            module: "llm",
            displayName: "Yooz-Light",
            sizeBytes: 550_000_000,
            cached: true,
            loaded: true,
            isActive: true,
            deletable: false
        )
        try write(managedModelInfo, "ManagedModelInfo", to: dir)
        try write(
            ManagedModelsResponse(models: [managedModelInfo]),
            "ManagedModelsResponse", to: dir
        )
        try write(
            DeleteModelResult(id: "models--foo--bar", reclaimedBytes: 123_456),
            "DeleteModelResult", to: dir
        )
        try write(
            ModelCleanupResult(totalReclaimedBytes: 999_999, perRepo: ["models--foo--bar": 999_999]),
            "ModelCleanupResult", to: dir
        )
        try write(LoadState.loading, "LoadState", to: dir)
        try write(
            LLMStatus(loaded: true, modelId: "yooz-light-v2", progress: nil, state: .ready, lastError: nil),
            "LLMStatus", to: dir
        )
        try write(
            STTStatus(loaded: true, language: "en", streaming: false, progress: nil, state: .ready, lastError: nil),
            "STTStatus", to: dir
        )
        try write(
            HealthStatus(
                status: "ok",
                version: "0.7.5",
                modules: ModuleStatus(
                    stt: true, llm: true, touchup: true, grammar: true,
                    vad: true, tts: false, infinite: nil
                )
            ),
            "HealthStatus", to: dir
        )

        // MARK: STT/LLM/TouchUp/Grammar/VAD bodies

        try write(
            LLMGenerateRequest(prompt: "hello world", model: "yooz-light-v2", systemPrompt: "be terse"),
            "LLMGenerateRequest", to: dir
        )
        try write(
            LLMGenerateResponse(text: "Hello, world.", model: "yooz-light-v2", tokensGenerated: 4, processingTimeMs: 120),
            "LLMGenerateResponse", to: dir
        )
        let llmModelInfo = LLMModelInfo(
            id: "yooz-light-v2", displayName: "Yooz-Light",
            sizeBytes: 550_000_000, loaded: true, latencyHintMs: 200
        )
        try write(llmModelInfo, "LLMModelInfo", to: dir)
        try write(
            LLMModelsResponse(current: "yooz-light-v2", available: [llmModelInfo]),
            "LLMModelsResponse", to: dir
        )
        try write(LLMModelSelection(model: "yooz-light-v2"), "LLMModelSelection", to: dir)

        try write(TouchUpMode.standard, "TouchUpMode", to: dir)
        try write(
            TouchUpRequest(text: "hello   world", mode: .standard, language: "en"),
            "TouchUpRequest", to: dir
        )
        try write(
            TouchUpResponse(
                result: "Hello, world.", mode: .standard, processingTimeMs: 80,
                modelUsed: "yooz-light-v2", warnings: nil
            ),
            "TouchUpResponse", to: dir
        )

        try write(
            GrammarCheckRequest(text: "he go home", categories: ["grammar"], usePOS: true),
            "GrammarCheckRequest", to: dir
        )
        try write(
            GrammarCheckResponse(result: "He goes home.", correctionsApplied: 1, ruleCount: 1560),
            "GrammarCheckResponse", to: dir
        )

        let speechSegment = SpeechSegment(startMs: 100, endMs: 900, probability: 0.97)
        try write(speechSegment, "SpeechSegment", to: dir)
        try write(VADResponse(segments: [speechSegment]), "VADResponse", to: dir)

        let alignedToken = AlignedToken(text: "hello", start: 0.0, end: 0.42)
        try write(alignedToken, "AlignedToken", to: dir)
        try write(
            TranscriptionResult(
                text: "hello world", finalized: "hello world", draft: "",
                language: "en", tokens: [alignedToken]
            ),
            "TranscriptionResult", to: dir
        )

        try write(
            STTLanguageInfo(code: "en", name: "English", implemented: true, family: "latin"),
            "STTLanguageInfo", to: dir
        )
        try write(
            STTLanguagesResponse(languages: [
                STTLanguageInfo(code: "en", name: "English", implemented: true, family: "latin"),
            ]),
            "STTLanguagesResponse", to: dir
        )
    }

    private func fixturesDirectory() throws -> URL {
        let thisFile = URL(fileURLWithPath: #filePath)
        let dir = thisFile
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/wire-v0.7.5")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func write<T: Encodable>(_ value: T, _ name: String, to dir: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        let data = try encoder.encode(value)
        try data.write(to: dir.appendingPathComponent("\(name).json"))
    }
}
