// TouchUpContextWireTests.swift
// YoozEngineInProcessTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Two-struct-trap regression test (engine#280 Phase 4 / whisper#317): the
// in-process transport decodes `/v1/touchup` request bodies through its own
// local `TouchUpBody` shim, a SEPARATE type from the canonical
// `YoozEngineWire.TouchUpRequest` the loopback server decodes (#225 — see
// `InProcessTransport.swift`'s "TouchUp mode decode shim" doc). Adding a
// field to one does not add it to the other; without also adding
// `contextVocabulary`/`contextAppName` to `TouchUpBody` and forwarding them
// in `handleTouchUp`, whisper's actual shipping path (in-process, not
// loopback) would silently drop both fields even though the wire-compat
// fixture tests (`WireCompatFixtureTests`) prove the loopback-facing
// `TouchUpRequest` decodes them fine.
//
// Mirrors `GPUAdmissionInProcessTests`'s workloadClass wire-contract style:
// mode "off" is regex-only (no LLM load), so these stay fast and
// deterministic. This proves the JSON shape round-trips through the
// in-process decode path without erroring when the new keys — including an
// over-cap vocabulary list — are present. It does NOT by itself prove the
// fields reach the composed system prompt: that deeper claim needs a loaded
// model (mode "off" never reaches `selectPrompt`/`withContext`) and is
// covered by the gated `TouchUpFidelityEvalTests` context-block run plus the
// ungated pure-function tests on `TouchUpEngine.withContext` and
// `TouchUpEngine.cappedContextVocabulary`.
import XCTest

@testable import YoozEngineInProcess

final class TouchUpContextWireTests: XCTestCase {

    private func touchUpBody(vocabulary: [String]?, appName: String?) -> Data {
        var json = #"{"text":"hello world","mode":"off""#
        if let vocabulary {
            let escaped = vocabulary.map { "\"\($0)\"" }.joined(separator: ",")
            json += #","contextVocabulary":[\#(escaped)]"#
        }
        if let appName {
            json += #","contextAppName":"\#(appName)""#
        }
        json += "}"
        return Data(json.utf8)
    }

    func testTouchUpContextFieldsDecodeWithoutError() async throws {
        let transport = InProcessTransport()
        try await transport.connect()
        let data = try await transport.post(
            "/v1/touchup",
            body: touchUpBody(vocabulary: ["Robinhood", "Cloudflare"], appName: "Slack")
        )
        XCTAssertFalse(data.isEmpty)
    }

    func testTouchUpOverCapVocabularyDecodesWithoutError() async throws {
        let transport = InProcessTransport()
        try await transport.connect()
        let overCap = (0..<35).map { "term\($0)" }
        let data = try await transport.post(
            "/v1/touchup", body: touchUpBody(vocabulary: overCap, appName: nil)
        )
        XCTAssertFalse(data.isEmpty)
    }

    func testTouchUpOmittedContextFieldsDecodeWithoutError() async throws {
        let transport = InProcessTransport()
        try await transport.connect()
        let data = try await transport.post(
            "/v1/touchup", body: touchUpBody(vocabulary: nil, appName: nil)
        )
        XCTAssertFalse(data.isEmpty)
    }
}
