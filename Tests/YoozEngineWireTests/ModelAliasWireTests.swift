// ModelAliasWireTests.swift
// YoozEngineWireTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Wire coverage for the `huggingFaceID` alias field (engine#308).
//
// `WireCompatFixtureTests` pins the v0.7.5 shape and would keep passing if the
// new field were dropped entirely — its fixtures predate it and the encoder
// omits a nil optional. So the field needs its own tests, in both directions:
// that it survives a round trip when SET, and that it stays absent when nil so
// an older decoder is unaffected.

import XCTest
@testable import YoozEngineWire

final class ModelAliasWireTests: XCTestCase {
    private let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.outputFormatting = [.sortedKeys]
        return e
    }()

    // MARK: - LLMModelInfo

    func testLLMModelInfoRoundTripsHuggingFaceID() throws {
        let model = LLMModelInfo(
            id: "yooz-instruct-4b",
            displayName: "Yooz-Instruct-4B",
            sizeBytes: 2_370 * 1_024 * 1_024,
            loaded: true,
            latencyHintMs: 1_000,
            purpose: .general,
            huggingFaceID: "YoozLabs/Qwen3.5-4B-qat-lean-4bit-mlx"
        )

        let json = try encoder.encode(model)
        let text = String(decoding: json, as: UTF8.self)
        XCTAssertTrue(
            text.contains("\"huggingFaceID\":\"YoozLabs\\/Qwen3.5-4B-qat-lean-4bit-mlx\""),
            "the alias must be on the wire, not just in the Swift value: \(text)"
        )

        let decoded = try JSONDecoder().decode(LLMModelInfo.self, from: json)
        XCTAssertEqual(decoded.huggingFaceID, "YoozLabs/Qwen3.5-4B-qat-lean-4bit-mlx")
        XCTAssertEqual(decoded, model)
    }

    /// A nil alias must not add a key. This is what keeps the v0.7.5 fixtures
    /// byte-identical, and what lets a backend with no HuggingFace repo behind
    /// it (Apple Intelligence, remote) stay silent rather than assert `null`.
    func testLLMModelInfoOmitsAbsentHuggingFaceID() throws {
        let model = LLMModelInfo(id: "some-model", displayName: "Some Model")
        let text = String(decoding: try encoder.encode(model), as: UTF8.self)
        XCTAssertFalse(text.contains("huggingFaceID"), text)
    }

    /// Forward compatibility in the other direction: a payload from an engine
    /// too old to know the field must still decode, with the alias absent.
    func testLLMModelInfoDecodesLegacyPayloadWithoutAlias() throws {
        let legacy = Data("""
        {"id":"yooz-light-v3","displayName":"Yooz-Light","loaded":false}
        """.utf8)
        let decoded = try JSONDecoder().decode(LLMModelInfo.self, from: legacy)
        XCTAssertNil(decoded.huggingFaceID)
        XCTAssertEqual(decoded.id, "yooz-light-v3")
    }

    // MARK: - ManagedModelInfo

    func testManagedModelInfoRoundTripsHuggingFaceID() throws {
        let row = ManagedModelInfo(
            id: "yooz-instruct-4b",
            module: "llm",
            displayName: "Yooz-Instruct-4B",
            sizeBytes: 2_370_000_000,
            cached: true,
            loaded: false,
            isActive: false,
            deletable: true,
            huggingFaceID: "YoozLabs/Qwen3.5-4B-qat-lean-4bit-mlx"
        )

        let json = try encoder.encode(row)
        let decoded = try JSONDecoder().decode(ManagedModelInfo.self, from: json)
        XCTAssertEqual(decoded.huggingFaceID, "YoozLabs/Qwen3.5-4B-qat-lean-4bit-mlx")
        XCTAssertEqual(decoded, row)
    }

    func testManagedModelInfoOmitsAbsentHuggingFaceID() throws {
        let row = ManagedModelInfo(
            id: "models--mlx-community--parakeet-tdt-0.6b-v3",
            module: "stt",
            displayName: "parakeet-tdt-0.6b-v3",
            sizeBytes: 600_000_000,
            cached: true,
            loaded: false,
            isActive: false,
            deletable: true
        )
        let text = String(decoding: try encoder.encode(row), as: UTF8.self)
        XCTAssertFalse(text.contains("huggingFaceID"), text)
    }
}
