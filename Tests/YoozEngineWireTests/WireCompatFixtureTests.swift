// WireCompatFixtureTests.swift
// YoozEngineWireTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import XCTest
@testable import YoozEngineWire

/// Proves every DTO the #225 refactor moved into `YoozEngineWire` still
/// speaks the exact v0.7.5-era wire JSON its old, pre-refactor home (server
/// `APITypes.swift`, SDK `Types/*.swift`, or the in-process wire mirrors)
/// used — in BOTH directions:
///
/// - **decode**: the committed fixture decodes to the expected value, and
/// - **encode**: re-encoding that value (`.sortedKeys` + `.prettyPrinted`,
///   the generator's settings) reproduces the fixture byte-for-byte, so an
///   encoder-side regression (a dropped `encodeIfPresent`, a key rename in
///   a hand-rolled `encode(to:)`) fails here too, not just decode drift.
///
/// Response/entity fixtures were captured BEFORE the type move by
/// `EngineCoreTests/WireFixtureExportTests` and
/// `YoozEngineClientTests/WireFixtureExportTests` (`EXPORT_WIRE_FIXTURES=1`)
/// at pre-refactor commit `a6614bc` and are committed under
/// `Tests/Fixtures/wire-v0.7.5/`. The request-side fixtures the pre-refactor
/// tree had no public encoder for (`BatchSTTRequest`, `STTLoadRequest`,
/// `VADRequest`, `SessionBeginResponse`) plus the `TouchUpResponseWarnings`
/// variant are generator-written against the post-refactor encoders with
/// field layouts matching the pre-refactor wire (verified against the old
/// struct definitions at `a6614bc`).
///
/// A wire id, field name, or JSON shape change on a moved type breaks one of
/// these tests — that is the point: it is the compatibility contract engine
/// issue #225 requires ("no field renames").
final class WireCompatFixtureTests: XCTestCase {
    private func fixture(_ name: String) throws -> Data {
        let thisFile = URL(fileURLWithPath: #filePath)
        let url = thisFile
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/wire-v0.7.5/\(name).json")
        return try Data(contentsOf: url)
    }

    /// Decode the fixture, compare to `expected`, then re-encode and compare
    /// bytes back to the fixture (the round trip that pins the encoder).
    private func assertWireStable<T: Codable & Equatable>(
        _ type: T.Type,
        fixture name: String,
        expected: T,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let data = try fixture(name)
        let decoded = try JSONDecoder().decode(T.self, from: data)
        XCTAssertEqual(
            decoded, expected,
            "decoded \(name).json mismatched expected value", file: file, line: line
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        let reencoded = try encoder.encode(decoded)
        XCTAssertEqual(
            String(data: reencoded, encoding: .utf8),
            String(data: data, encoding: .utf8),
            "re-encoding \(name).json did not reproduce the fixture bytes",
            file: file, line: line
        )
    }

    // MARK: - Picker family

    func testModelTier() throws {
        try assertWireStable(ModelTier.self, fixture: "ModelTier", expected: .quality)
    }

    func testModelTierUnknownRawValueFallsBack() throws {
        // Forward compat: a newer engine shipping a fifth tier must decode
        // as `.unknown` on this SDK, never throw.
        let decoded = try JSONDecoder().decode(ModelTier.self, from: Data("\"turbo\"".utf8))
        XCTAssertEqual(decoded, .unknown)
    }

    func testModelLoadState() throws {
        try assertWireStable(ModelLoadState.self, fixture: "ModelLoadState", expected: .cached)
    }

    func testModelLoadStateUnknownRawValueFallsBack() throws {
        // Forward compat: unknown lifecycle states grey the row out
        // (`.unavailable`), never throw.
        let decoded = try JSONDecoder().decode(ModelLoadState.self, from: Data("\"warming\"".utf8))
        XCTAssertEqual(decoded, .unavailable)
    }

    func testTouchUpModelInfo() throws {
        try assertWireStable(
            TouchUpModelInfo.self, fixture: "TouchUpModelInfo",
            expected: TouchUpModelInfo(
                id: "yooz-light-v2", displayName: "Yooz-Light",
                description: "Fast, on-device cleanup", tier: .light,
                sizeBytes: 550_000_000, loadState: .cached, isActive: true
            )
        )
    }

    func testTouchUpModelsResponse() throws {
        try assertWireStable(
            TouchUpModelsResponse.self, fixture: "TouchUpModelsResponse",
            expected: TouchUpModelsResponse(
                models: [TouchUpModelInfo(
                    id: "yooz-light-v2", displayName: "Yooz-Light",
                    description: "Fast, on-device cleanup", tier: .light,
                    sizeBytes: 550_000_000, loadState: .cached, isActive: true
                )],
                activeId: "yooz-light-v2"
            )
        )
    }

    func testTouchUpSetModelRequest() throws {
        try assertWireStable(
            TouchUpSetModelRequest.self, fixture: "TouchUpSetModelRequest",
            expected: TouchUpSetModelRequest(id: "yooz-light-v2", preload: true)
        )
    }

    func testSTTBackendInfo() throws {
        try assertWireStable(
            STTBackendInfo.self, fixture: "STTBackendInfo",
            expected: STTBackendInfo(
                id: "parakeet", displayName: "Parakeet TDT",
                description: "High-accuracy MLX STT", tier: .quality,
                sizeBytes: 600_000_000, loadState: .loaded, isActive: true,
                supportsBatch: true, supportsStreaming: true,
                supportedLanguages: ["en", "es"]
            )
        )
    }

    func testSTTBackendsResponse() throws {
        try assertWireStable(
            STTBackendsResponse.self, fixture: "STTBackendsResponse",
            expected: STTBackendsResponse(
                backends: [STTBackendInfo(
                    id: "parakeet", displayName: "Parakeet TDT",
                    description: "High-accuracy MLX STT", tier: .quality,
                    sizeBytes: 600_000_000, loadState: .loaded, isActive: true,
                    supportsBatch: true, supportsStreaming: true,
                    supportedLanguages: ["en", "es"]
                )],
                activeId: "parakeet"
            )
        )
    }

    func testSTTSetBackendRequest() throws {
        try assertWireStable(
            STTSetBackendRequest.self, fixture: "STTSetBackendRequest",
            expected: STTSetBackendRequest(id: "parakeet", preload: true)
        )
    }

    // MARK: - Models/status/modules family

    func testModulesResponse() throws {
        try assertWireStable(
            ModulesResponse.self, fixture: "ModulesResponse",
            expected: ModulesResponse(
                engineVersion: "0.7.5",
                buildVariant: "full",
                modules: [
                    ModuleManifest(
                        name: "grammar", version: "0.7.5", loaded: true,
                        error: nil, detail: ["rules_total": "1560"]
                    ),
                    ModuleManifest(
                        name: "stt", version: "0.7.5", loaded: false,
                        error: "not loaded", detail: [:]
                    ),
                ]
            )
        )
    }

    func testManagedModelInfo() throws {
        try assertWireStable(
            ManagedModelInfo.self, fixture: "ManagedModelInfo",
            expected: ManagedModelInfo(
                id: "yooz-light-v2", module: "llm", displayName: "Yooz-Light",
                sizeBytes: 550_000_000, cached: true, loaded: true,
                isActive: true, deletable: false
            )
        )
    }

    func testManagedModelsResponse() throws {
        try assertWireStable(
            ManagedModelsResponse.self, fixture: "ManagedModelsResponse",
            expected: ManagedModelsResponse(models: [ManagedModelInfo(
                id: "yooz-light-v2", module: "llm", displayName: "Yooz-Light",
                sizeBytes: 550_000_000, cached: true, loaded: true,
                isActive: true, deletable: false
            )])
        )
    }

    func testDeleteModelResult() throws {
        try assertWireStable(
            DeleteModelResult.self, fixture: "DeleteModelResult",
            expected: DeleteModelResult(id: "models--foo--bar", reclaimedBytes: 123_456)
        )
    }

    func testModelCleanupResult() throws {
        try assertWireStable(
            ModelCleanupResult.self, fixture: "ModelCleanupResult",
            expected: ModelCleanupResult(
                totalReclaimedBytes: 999_999,
                perRepo: ["models--foo--bar": 999_999]
            )
        )
    }

    func testLoadState() throws {
        try assertWireStable(LoadState.self, fixture: "LoadState", expected: .loading)
    }

    func testLLMStatus() throws {
        try assertWireStable(
            LLMStatus.self, fixture: "LLMStatus",
            expected: LLMStatus(loaded: true, modelId: "yooz-light-v2", progress: nil, state: .ready, lastError: nil)
        )
    }

    func testSTTStatus() throws {
        try assertWireStable(
            STTStatus.self, fixture: "STTStatus",
            expected: STTStatus(loaded: true, language: "en", streaming: false, progress: nil, state: .ready, lastError: nil)
        )
    }

    // MARK: - STT/LLM/TouchUp/Grammar/VAD bodies

    func testLLMGenerateRequest() throws {
        try assertWireStable(
            LLMGenerateRequest.self, fixture: "LLMGenerateRequest",
            expected: LLMGenerateRequest(
                prompt: "hello world", model: "yooz-light-v2", systemPrompt: "be terse"
            )
        )
    }

    func testLLMGenerateResponse() throws {
        try assertWireStable(
            LLMGenerateResponse.self, fixture: "LLMGenerateResponse",
            expected: LLMGenerateResponse(
                text: "Hello, world.", model: "yooz-light-v2",
                tokensGenerated: 4, processingTimeMs: 120
            )
        )
    }

    func testGPUWorkloadClassIsStrict() throws {
        // Deliberately NOT tolerant-decode (unlike ModelTier): a declared
        // scheduling class is explicit caller intent; an unknown value must
        // reject the request, not silently downgrade. See GPUWorkloadClass.
        XCTAssertEqual(
            try JSONDecoder().decode(GPUWorkloadClass.self, from: Data("\"interactive\"".utf8)),
            .interactive
        )
        XCTAssertThrowsError(
            try JSONDecoder().decode(GPUWorkloadClass.self, from: Data("\"turbo\"".utf8))
        )
    }

    func testLLMModelInfo() throws {
        try assertWireStable(
            LLMModelInfo.self, fixture: "LLMModelInfo",
            expected: LLMModelInfo(
                id: "yooz-light-v2", displayName: "Yooz-Light",
                sizeBytes: 550_000_000, loaded: true, latencyHintMs: 200
            )
        )
    }

    func testLLMModelsResponse() throws {
        try assertWireStable(
            LLMModelsResponse.self, fixture: "LLMModelsResponse",
            expected: LLMModelsResponse(
                current: "yooz-light-v2",
                available: [LLMModelInfo(
                    id: "yooz-light-v2", displayName: "Yooz-Light",
                    sizeBytes: 550_000_000, loaded: true, latencyHintMs: 200
                )]
            )
        )
    }

    func testLLMModelSelection() throws {
        try assertWireStable(
            LLMModelSelection.self, fixture: "LLMModelSelection",
            expected: LLMModelSelection(model: "yooz-light-v2")
        )
    }

    func testTouchUpMode() throws {
        try assertWireStable(TouchUpMode.self, fixture: "TouchUpMode", expected: .standard)
    }

    func testTouchUpRequest() throws {
        try assertWireStable(
            TouchUpRequest.self, fixture: "TouchUpRequest",
            expected: TouchUpRequest(text: "hello   world", mode: .standard, language: "en")
        )
    }

    func testTouchUpResponse() throws {
        try assertWireStable(
            TouchUpResponse.self, fixture: "TouchUpResponse",
            expected: TouchUpResponse(
                result: "Hello, world.", mode: .standard, processingTimeMs: 80,
                modelUsed: "yooz-light-v2", warnings: nil
            )
        )
    }

    func testTouchUpResponseWithWarnings() throws {
        // The nil-warnings fixture above cannot catch a silent key rename of
        // an optional field the encoder omits; this populated variant can.
        try assertWireStable(
            TouchUpResponse.self, fixture: "TouchUpResponseWarnings",
            expected: TouchUpResponse(
                result: "Hello, world.", mode: .full, processingTimeMs: 310,
                modelUsed: "yooz-light-v2",
                warnings: ["yooz-quality-v2 not loaded; fell back to yooz-light-v2"]
            )
        )
    }

    func testGrammarCheckRequest() throws {
        try assertWireStable(
            GrammarCheckRequest.self, fixture: "GrammarCheckRequest",
            expected: GrammarCheckRequest(text: "he go home", categories: ["grammar"], usePOS: true)
        )
    }

    func testGrammarCheckResponse() throws {
        try assertWireStable(
            GrammarCheckResponse.self, fixture: "GrammarCheckResponse",
            expected: GrammarCheckResponse(result: "He goes home.", correctionsApplied: 1, ruleCount: 1560)
        )
    }

    func testVADRequest() throws {
        try assertWireStable(
            VADRequest.self, fixture: "VADRequest",
            expected: VADRequest(samples: [0.1, -0.2, 0.3], reset: true)
        )
    }

    func testSpeechSegment() throws {
        try assertWireStable(
            SpeechSegment.self, fixture: "SpeechSegment",
            expected: SpeechSegment(startMs: 100, endMs: 900, probability: 0.97)
        )
    }

    func testVADResponse() throws {
        try assertWireStable(
            VADResponse.self, fixture: "VADResponse",
            expected: VADResponse(segments: [SpeechSegment(startMs: 100, endMs: 900, probability: 0.97)])
        )
    }

    func testAlignedToken() throws {
        try assertWireStable(
            AlignedToken.self, fixture: "AlignedToken",
            expected: AlignedToken(text: "hello", start: 0.0, end: 0.42)
        )
    }

    func testTranscriptionResult() throws {
        try assertWireStable(
            TranscriptionResult.self, fixture: "TranscriptionResult",
            expected: TranscriptionResult(
                text: "hello world", finalized: "hello world", draft: "",
                language: "en", tokens: [AlignedToken(text: "hello", start: 0.0, end: 0.42)]
            )
        )
    }

    func testSTTLoadRequest() throws {
        try assertWireStable(
            STTLoadRequest.self, fixture: "STTLoadRequest",
            expected: STTLoadRequest(language: "en")
        )
    }

    func testSTTLoadRequestDefaultsMissingLanguage() throws {
        // Transport parity by construction (#225 review): the "missing
        // language means English" rule lives on the type, so the loopback
        // and in-process handlers cannot diverge on it. Decode-only: the
        // encoder always emits the resolved language.
        let decoded = try JSONDecoder().decode(
            STTLoadRequest.self, from: try fixture("STTLoadRequestDefaults")
        )
        XCTAssertEqual(decoded, STTLoadRequest(language: "en", allowFetch: false))
    }

    func testBatchSTTRequest() throws {
        try assertWireStable(
            BatchSTTRequest.self, fixture: "BatchSTTRequest",
            expected: BatchSTTRequest(samples: [0.1, -0.2], language: "en", mode: "normal")
        )
    }

    func testBatchSTTRequestDefaultsMissingLanguageAndMode() throws {
        // Same construction-level parity pin as STTLoadRequestDefaults, for
        // the batch body's `language` ("en") and `mode` ("normal") defaults.
        let decoded = try JSONDecoder().decode(
            BatchSTTRequest.self, from: try fixture("BatchSTTRequestDefaults")
        )
        XCTAssertEqual(decoded, BatchSTTRequest(samples: [], language: "en", mode: "normal"))
    }

    func testSTTLanguageInfo() throws {
        try assertWireStable(
            STTLanguageInfo.self, fixture: "STTLanguageInfo",
            expected: STTLanguageInfo(code: "en", name: "English", implemented: true, family: "latin")
        )
    }

    func testSTTLanguagesResponse() throws {
        try assertWireStable(
            STTLanguagesResponse.self, fixture: "STTLanguagesResponse",
            expected: STTLanguagesResponse(languages: [
                STTLanguageInfo(code: "en", name: "English", implemented: true, family: "latin"),
            ])
        )
    }

    func testSessionBeginResponse() throws {
        try assertWireStable(
            SessionBeginResponse.self, fixture: "SessionBeginResponse",
            expected: SessionBeginResponse(
                sessionId: "8f14e45f-ceea-467e-bd44-59c7f5c3c8f9",
                ts: "2026-07-02T09:00:00Z"
            )
        )
    }
}
