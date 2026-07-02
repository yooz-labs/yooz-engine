// EndpointTableDispatchTests.swift
// YoozEngineInProcessTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore
import XCTest
import YoozEngineClient
@testable import LLMModule
@testable import YoozEngineInProcess

/// Behavioral pins for table-dispatched routes through `InProcessTransport`
/// (engine#225 Phase B) — the complement to `RouteParityTests`' pure
/// reachability sweep. These assert the CONTRACT the shared handlers carry:
/// the canonical picker wire codes (AGENTS.md "Wire codes") now surface
/// in-process as typed `YoozEngineError.serverError`s, where the pre-table
/// in-process handlers propagated raw errors (the drift the table closed).
/// Every request here fails validation before any disk/network/model work.
final class EndpointTableDispatchTests: XCTestCase {
    private func makeTransport() async throws -> InProcessTransport {
        let transport = InProcessTransport()
        try await transport.connect()
        return transport
    }

    private func assertServerError(
        _ body: @autoclosure () async throws -> Data,
        status: Int,
        code: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) async {
        do {
            _ = try await body()
            XCTFail("expected serverError(\(status), \(code)); call succeeded", file: file, line: line)
        } catch let error as YoozEngineError {
            guard case .serverError(let gotStatus, let gotCode, _) = error else {
                XCTFail("expected serverError, got \(error)", file: file, line: line)
                return
            }
            XCTAssertEqual(gotStatus, status, file: file, line: line)
            XCTAssertEqual(gotCode, code, file: file, line: line)
        } catch {
            XCTFail("expected YoozEngineError.serverError, got \(error)", file: file, line: line)
        }
    }

    func testPickerSetRejectsUndecodableBodyWithInvalidRequest() async throws {
        let transport = try await makeTransport()
        await assertServerError(
            try await transport.post("/v1/touchup/model", body: Data("{}".utf8)),
            status: 400, code: "invalid_request"
        )
    }

    func testPickerSetRejectsUnknownIdWithInvalidModel() async throws {
        let transport = try await makeTransport()
        await assertServerError(
            try await transport.post(
                "/v1/touchup/model", body: Data(#"{"id":"no-such-model"}"#.utf8)
            ),
            status: 400, code: "invalid_model"
        )
    }

    /// In-process witness for the 501 `model_unavailable` picker code —
    /// the loopback side is pinned by
    /// `TouchUpPickerRouteTests.testPostModelWithFoundationModelsOn26MinusReturns501`;
    /// one handler serves both transports, and this proves the second half
    /// of the code matrix. Same self-skip gate as the loopback test: the
    /// unavailable branch is the one under test.
    func testPickerSetFoundationModelsUnavailableReturns501() async throws {
        guard !FoundationModelsBackend().isAvailable() else {
            throw XCTSkip("FoundationModels available on this OS; unavailable branch not reachable")
        }
        let transport = try await makeTransport()
        await assertServerError(
            try await transport.post(
                "/v1/touchup/model",
                body: Data(#"{"id":"foundation-models","preload":true}"#.utf8)
            ),
            status: 501, code: "model_unavailable"
        )
    }

    func testModelsDeleteRejectsUnknownIdWith404() async throws {
        // A non-LLM id without the "models--" hub prefix 404s before any
        // disk work — cheap and deterministic.
        let transport = try await makeTransport()
        await assertServerError(
            try await transport.delete("/v1/models/endpoint-table-bogus"),
            status: 404, code: "unknown_model"
        )
    }

    func testSessionBeginReturnsWireShapeAndEndReturnsEmpty() async throws {
        let transport = try await makeTransport()
        let beginBody = try await transport.post("/v1/session/begin", body: Data())
        let begin = try JSONDecoder().decode(SessionBeginResponse.self, from: beginBody)
        XCTAssertFalse(begin.sessionId.isEmpty)
        XCTAssertFalse(begin.ts.isEmpty)

        let endBody = try await transport.post("/v1/session/end", body: Data())
        XCTAssertTrue(endBody.isEmpty, "204-equivalent must surface as empty Data")
    }
}
