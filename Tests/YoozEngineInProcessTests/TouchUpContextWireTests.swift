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
// `resolvedTouchUpCallArguments(from:)` (review item 4) is the real
// regression seam: it decodes `TouchUpBody` and builds the exact
// `TouchUpCallArguments` `handleTouchUp` passes to
// `TouchUpEngine.processWithActiveModel`, WITHOUT calling that method — no
// model load, no transport round trip needed. Asserting equality on the
// built struct proves the fields are read and forwarded. An earlier version
// of this file only checked that a mode-"off" `/v1/touchup` call didn't
// throw, which would have passed even if forwarding silently regressed to
// `nil, nil`: mode "off" returns from `processWithActiveModel` before those
// fields are ever read, so a bare success/failure check could not tell the
// two apart.

import EngineCore
import XCTest
import YoozEngineWire

@testable import YoozEngineInProcess

final class TouchUpContextWireTests: XCTestCase {

    private func touchUpBody(mode: String = "off", vocabulary: [String]?, appName: String?) -> Data {
        var json = #"{"text":"hello world","mode":"\#(mode)""#
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

    // MARK: - Built-argument regression (the two-struct trap, proven directly)

    func testContextFieldsForwardIntoBuiltCallArguments() throws {
        let body = touchUpBody(
            mode: "standard", vocabulary: ["Robinhood", "Cloudflare"], appName: "Slack"
        )
        let arguments = try InProcessTransport.resolvedTouchUpCallArguments(from: body)
        XCTAssertEqual(
            arguments,
            TouchUpCallArguments(
                text: "hello world",
                mode: .standard,
                workloadClass: .background,
                contextVocabulary: ["Robinhood", "Cloudflare"],
                contextAppName: "Slack"
            )
        )
    }

    func testOmittedContextFieldsForwardAsNil() throws {
        let body = touchUpBody(mode: "standard", vocabulary: nil, appName: nil)
        let arguments = try InProcessTransport.resolvedTouchUpCallArguments(from: body)
        XCTAssertNil(arguments.contextVocabulary)
        XCTAssertNil(arguments.contextAppName)
    }

    func testOverCapVocabularyForwardsUncappedAtDecodeLayer() throws {
        // The 30-term cap is TouchUpEngine's job (applied inside
        // `process`/`processWithActiveModel`), not the transport's — this
        // decode/forward layer should pass every term through untouched.
        let overCap = (0..<35).map { "term\($0)" }
        let body = touchUpBody(mode: "standard", vocabulary: overCap, appName: nil)
        let arguments = try InProcessTransport.resolvedTouchUpCallArguments(from: body)
        XCTAssertEqual(arguments.contextVocabulary, overCap)
    }

    // MARK: - Full transport round trip (fast/deterministic, mode "off")
    //
    // These stay as a shallower, wire-shape-only check: mode "off" so the
    // call resolves without a model load, proving the JSON shape (including
    // an over-cap vocabulary list) round-trips through the real transport
    // without erroring end to end.

    func testTouchUpContextFieldsDecodeWithoutError() async throws {
        let transport = InProcessTransport()
        try await transport.connect()
        let data = try await transport.post(
            "/v1/touchup",
            body: touchUpBody(vocabulary: ["Robinhood", "Cloudflare"], appName: "Slack")
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
