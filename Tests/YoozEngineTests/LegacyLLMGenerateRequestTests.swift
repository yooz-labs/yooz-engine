// LegacyLLMGenerateRequestTests.swift
// YoozEngineTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import XCTest
@testable import YoozEngine

/// Pins the loopback-only legacy decode shim for `POST /v1/llm/generate`
/// (`LegacyLLMGenerateRequest`, `YoozEngine/Server/APITypes.swift`): the
/// snake_case `system_prompt` spelling some pre-SDK callers post must keep
/// decoding after #225 moved the canonical `LLMGenerateRequest` (camelCase
/// only) into `YoozEngineWire`. Mirrors `LegacySTTSetBackendRequest`'s
/// pattern of keeping transport-specific compat out of the shared DTO.
final class LegacyLLMGenerateRequestTests: XCTestCase {
    func testDecodesCanonicalCamelCase() throws {
        let json = #"{"prompt":"hi","model":"yooz-light-v2","systemPrompt":"be terse"}"#
        let decoded = try JSONDecoder().decode(
            LegacyLLMGenerateRequest.self, from: Data(json.utf8)
        )
        XCTAssertEqual(decoded.prompt, "hi")
        XCTAssertEqual(decoded.model, "yooz-light-v2")
        XCTAssertEqual(decoded.systemPrompt, "be terse")
        XCTAssertNil(decoded.workloadClass)
    }

    func testDecodesLegacySnakeCaseSystemPrompt() throws {
        let json = #"{"prompt":"hi","system_prompt":"be terse"}"#
        let decoded = try JSONDecoder().decode(
            LegacyLLMGenerateRequest.self, from: Data(json.utf8)
        )
        XCTAssertEqual(decoded.systemPrompt, "be terse")
    }

    func testCanonicalKeyWinsWhenBothSpellingsPresent() throws {
        let json = #"{"prompt":"hi","systemPrompt":"new","system_prompt":"old"}"#
        let decoded = try JSONDecoder().decode(
            LegacyLLMGenerateRequest.self, from: Data(json.utf8)
        )
        XCTAssertEqual(decoded.systemPrompt, "new")
    }

    func testWorkloadClassDecodesTyped() throws {
        let json = #"{"prompt":"hi","workloadClass":"interactive"}"#
        let decoded = try JSONDecoder().decode(
            LegacyLLMGenerateRequest.self, from: Data(json.utf8)
        )
        XCTAssertEqual(decoded.workloadClass, .interactive)
    }

    func testUnknownWorkloadClassFailsDecode() throws {
        // Route maps the decode failure to 400 `invalid_request`
        // (`GPUAdmissionRouteTests.testGenerateUnknownWorkloadClassIs400InvalidRequest`
        // covers the HTTP surface; this pins the decode layer).
        let json = #"{"prompt":"hi","workloadClass":"turbo"}"#
        XCTAssertThrowsError(
            try JSONDecoder().decode(LegacyLLMGenerateRequest.self, from: Data(json.utf8))
        )
    }
}
