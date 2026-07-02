// WireCompatFixtureTests.swift
// YoozEngineWireTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import XCTest
@testable import YoozEngineWire

/// Proves every DTO the #225 refactor moved into `YoozEngineWire` still
/// decodes the exact v0.7.5-era JSON its old, pre-refactor home (server
/// `APITypes.swift`, SDK `Types/*.swift`, or the in-process wire mirrors)
/// used to emit. Fixtures were captured BEFORE the move by
/// `EngineCoreTests/WireFixtureExportTests` and
/// `YoozEngineClientTests/WireFixtureExportTests` (`EXPORT_WIRE_FIXTURES=1`)
/// and are committed under `Tests/Fixtures/wire-v0.7.5/`.
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

    private func assertDecodes<T: Decodable & Equatable>(
        _ type: T.Type,
        fixture name: String,
        expected: T,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let data = try fixture(name)
        let decoded = try JSONDecoder().decode(T.self, from: data)
        XCTAssertEqual(decoded, expected, "decoded \(name).json mismatched expected value", file: file, line: line)
    }

    // MARK: - Picker family

    func testModelTier() throws {
        let data = try fixture("ModelTier")
        XCTAssertEqual(try JSONDecoder().decode(ModelTier.self, from: data), .quality)
    }

    func testModelLoadState() throws {
        let data = try fixture("ModelLoadState")
        XCTAssertEqual(try JSONDecoder().decode(ModelLoadState.self, from: data), .cached)
    }

    func testTouchUpModelInfo() throws {
        try assertDecodes(
            TouchUpModelInfo.self, fixture: "TouchUpModelInfo",
            expected: TouchUpModelInfo(
                id: "yooz-light-v2", displayName: "Yooz-Light",
                description: "Fast, on-device cleanup", tier: .light,
                sizeBytes: 550_000_000, loadState: .cached, isActive: true
            )
        )
    }

    func testTouchUpModelsResponse() throws {
        try assertDecodes(
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
        try assertDecodes(
            TouchUpSetModelRequest.self, fixture: "TouchUpSetModelRequest",
            expected: TouchUpSetModelRequest(id: "yooz-light-v2", preload: true)
        )
    }

    func testSTTBackendInfo() throws {
        try assertDecodes(
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
        try assertDecodes(
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
        let data = try fixture("STTSetBackendRequest")
        let decoded = try JSONDecoder().decode(STTSetBackendRequest.self, from: data)
        XCTAssertEqual(decoded.id, "parakeet")
        XCTAssertEqual(decoded.preload, true)
    }

    // MARK: - Models/status/modules family

    func testModulesResponse() throws {
        let data = try fixture("ModulesResponse")
        let decoded = try JSONDecoder().decode(ModulesResponse.self, from: data)
        XCTAssertEqual(decoded.engineVersion, "0.7.5")
        XCTAssertEqual(decoded.buildVariant, "full")
        XCTAssertEqual(decoded.modules.count, 2)
        XCTAssertEqual(decoded.modules[0].name, "grammar")
        XCTAssertEqual(decoded.modules[0].detail["rules_total"], "1560")
        XCTAssertEqual(decoded.modules[1].name, "stt")
        XCTAssertEqual(decoded.modules[1].error, "not loaded")
    }

    func testManagedModelInfo() throws {
        try assertDecodes(
            ManagedModelInfo.self, fixture: "ManagedModelInfo",
            expected: ManagedModelInfo(
                id: "yooz-light-v2", module: "llm", displayName: "Yooz-Light",
                sizeBytes: 550_000_000, cached: true, loaded: true,
                isActive: true, deletable: false
            )
        )
    }

    func testManagedModelsResponse() throws {
        let data = try fixture("ManagedModelsResponse")
        let decoded = try JSONDecoder().decode(ManagedModelsResponse.self, from: data)
        XCTAssertEqual(decoded.models.count, 1)
        XCTAssertEqual(decoded.models[0].id, "yooz-light-v2")
    }

    func testDeleteModelResult() throws {
        try assertDecodes(
            DeleteModelResult.self, fixture: "DeleteModelResult",
            expected: DeleteModelResult(id: "models--foo--bar", reclaimedBytes: 123_456)
        )
    }

    func testModelCleanupResult() throws {
        try assertDecodes(
            ModelCleanupResult.self, fixture: "ModelCleanupResult",
            expected: ModelCleanupResult(
                totalReclaimedBytes: 999_999,
                perRepo: ["models--foo--bar": 999_999]
            )
        )
    }

    func testLoadState() throws {
        let data = try fixture("LoadState")
        XCTAssertEqual(try JSONDecoder().decode(LoadState.self, from: data), .loading)
    }

    func testLLMStatus() throws {
        try assertDecodes(
            LLMStatus.self, fixture: "LLMStatus",
            expected: LLMStatus(loaded: true, modelId: "yooz-light-v2", progress: nil, state: .ready, lastError: nil)
        )
    }

    func testSTTStatus() throws {
        try assertDecodes(
            STTStatus.self, fixture: "STTStatus",
            expected: STTStatus(loaded: true, language: "en", streaming: false, progress: nil, state: .ready, lastError: nil)
        )
    }

    // MARK: - STT/LLM/TouchUp/Grammar/VAD bodies

    func testLLMGenerateRequest() throws {
        let data = try fixture("LLMGenerateRequest")
        let decoded = try JSONDecoder().decode(LLMGenerateRequest.self, from: data)
        XCTAssertEqual(decoded.prompt, "hello world")
        XCTAssertEqual(decoded.model, "yooz-light-v2")
        XCTAssertEqual(decoded.systemPrompt, "be terse")
    }

    func testLLMGenerateRequestLegacySnakeCase() throws {
        // Not a captured fixture (the SDK never emits snake_case) — a
        // synthetic case pinning the legacy `system_prompt` decode leniency
        // this move folded into the canonical type from the old server-only
        // `LLMGenerateServerRequest`.
        let json = #"{"prompt":"hi","system_prompt":"be terse"}"#.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(LLMGenerateRequest.self, from: json)
        XCTAssertEqual(decoded.prompt, "hi")
        XCTAssertEqual(decoded.systemPrompt, "be terse")
    }

    func testLLMGenerateResponse() throws {
        let data = try fixture("LLMGenerateResponse")
        let decoded = try JSONDecoder().decode(LLMGenerateResponse.self, from: data)
        XCTAssertEqual(decoded.text, "Hello, world.")
        XCTAssertEqual(decoded.model, "yooz-light-v2")
        XCTAssertEqual(decoded.tokensGenerated, 4)
        XCTAssertEqual(decoded.processingTimeMs, 120)
    }

    func testLLMModelInfo() throws {
        try assertDecodes(
            LLMModelInfo.self, fixture: "LLMModelInfo",
            expected: LLMModelInfo(
                id: "yooz-light-v2", displayName: "Yooz-Light",
                sizeBytes: 550_000_000, loaded: true, latencyHintMs: 200
            )
        )
    }

    func testLLMModelsResponse() throws {
        let data = try fixture("LLMModelsResponse")
        let decoded = try JSONDecoder().decode(LLMModelsResponse.self, from: data)
        XCTAssertEqual(decoded.current, "yooz-light-v2")
        XCTAssertEqual(decoded.available.count, 1)
    }

    func testLLMModelSelection() throws {
        try assertDecodes(
            LLMModelSelection.self, fixture: "LLMModelSelection",
            expected: LLMModelSelection(model: "yooz-light-v2")
        )
    }

    func testTouchUpMode() throws {
        let data = try fixture("TouchUpMode")
        XCTAssertEqual(try JSONDecoder().decode(TouchUpMode.self, from: data), .standard)
    }

    func testTouchUpRequest() throws {
        let data = try fixture("TouchUpRequest")
        let decoded = try JSONDecoder().decode(TouchUpRequest.self, from: data)
        XCTAssertEqual(decoded.text, "hello   world")
        XCTAssertEqual(decoded.mode, .standard)
        XCTAssertEqual(decoded.language, "en")
    }

    func testTouchUpResponse() throws {
        let data = try fixture("TouchUpResponse")
        let decoded = try JSONDecoder().decode(TouchUpResponse.self, from: data)
        XCTAssertEqual(decoded.result, "Hello, world.")
        XCTAssertEqual(decoded.mode, .standard)
        XCTAssertEqual(decoded.processingTimeMs, 80)
        XCTAssertEqual(decoded.modelUsed, "yooz-light-v2")
        XCTAssertNil(decoded.warnings)
    }

    func testGrammarCheckRequest() throws {
        let data = try fixture("GrammarCheckRequest")
        let decoded = try JSONDecoder().decode(GrammarCheckRequest.self, from: data)
        XCTAssertEqual(decoded.text, "he go home")
        XCTAssertEqual(decoded.categories, ["grammar"])
        XCTAssertEqual(decoded.usePOS, true)
    }

    func testGrammarCheckResponse() throws {
        let data = try fixture("GrammarCheckResponse")
        let decoded = try JSONDecoder().decode(GrammarCheckResponse.self, from: data)
        XCTAssertEqual(decoded.result, "He goes home.")
        XCTAssertEqual(decoded.correctionsApplied, 1)
        XCTAssertEqual(decoded.ruleCount, 1560)
    }

    func testSpeechSegment() throws {
        try assertDecodes(
            SpeechSegment.self, fixture: "SpeechSegment",
            expected: SpeechSegment(startMs: 100, endMs: 900, probability: 0.97)
        )
    }

    func testVADResponse() throws {
        try assertDecodes(
            VADResponse.self, fixture: "VADResponse",
            expected: VADResponse(segments: [SpeechSegment(startMs: 100, endMs: 900, probability: 0.97)])
        )
    }

    func testAlignedToken() throws {
        try assertDecodes(
            AlignedToken.self, fixture: "AlignedToken",
            expected: AlignedToken(text: "hello", start: 0.0, end: 0.42)
        )
    }

    func testTranscriptionResult() throws {
        try assertDecodes(
            TranscriptionResult.self, fixture: "TranscriptionResult",
            expected: TranscriptionResult(
                text: "hello world", finalized: "hello world", draft: "",
                language: "en", tokens: [AlignedToken(text: "hello", start: 0.0, end: 0.42)]
            )
        )
    }

    func testSTTLanguageInfo() throws {
        try assertDecodes(
            STTLanguageInfo.self, fixture: "STTLanguageInfo",
            expected: STTLanguageInfo(code: "en", name: "English", implemented: true, family: "latin")
        )
    }

    func testSTTLanguagesResponse() throws {
        try assertDecodes(
            STTLanguagesResponse.self, fixture: "STTLanguagesResponse",
            expected: STTLanguagesResponse(languages: [
                STTLanguageInfo(code: "en", name: "English", implemented: true, family: "latin"),
            ])
        )
    }

    func testSessionBeginResponse() throws {
        try assertDecodes(
            SessionBeginResponse.self, fixture: "SessionBeginResponse",
            expected: SessionBeginResponse(
                sessionId: "8f14e45f-ceea-467e-bd44-59c7f5c3c8f9",
                ts: "2026-07-02T09:00:00Z"
            )
        )
    }
}
