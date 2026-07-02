// EndpointTableTests.swift
// EngineCoreTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import XCTest
@testable import EngineCore

/// Unit tests for the typed endpoint table machinery (engine#225 Phase B):
/// spec identity, path-parameter matching, and the invariant that
/// `RouteManifest` is a pure projection of the `EndpointSpecs` catalog.
final class EndpointTableTests: XCTestCase {
    private static let okHandler: WireHandler = { _ in WireResponse(status: 200) }

    // MARK: - EndpointSpec

    func testParameterNamesParsing() {
        XCTAssertEqual(EndpointSpec(.get, "/v1/models").parameterNames, [])
        XCTAssertEqual(EndpointSpec(.delete, "/v1/models/:id").parameterNames, ["id"])
        XCTAssertEqual(
            EndpointSpec(.post, "/v1/infinite/sessions/:sessionID/append").parameterNames,
            ["sessionID"]
        )
    }

    func testSpecKeyShapeMatchesManifestKeyShape() {
        // The two key derivations must stay interchangeable — the manifest
        // projection and the parity allowlist both join on this shape.
        let spec = EndpointSpec(.delete, "/v1/models/:id")
        XCTAssertEqual(spec.key, RouteManifestEntry(.delete, "/v1/models/:id").key)
    }

    // MARK: - Matching

    func testExactMatch() {
        let table = EndpointTable([Endpoint(EndpointSpec(.get, "/v1/models"), handler: Self.okHandler)])
        let match = table.match(method: .get, path: "/v1/models")
        XCTAssertNotNil(match)
        XCTAssertEqual(match?.pathParameters, [:])
    }

    func testMethodMismatchDoesNotMatch() {
        let table = EndpointTable([Endpoint(EndpointSpec(.get, "/v1/models"), handler: Self.okHandler)])
        XCTAssertNil(table.match(method: .post, path: "/v1/models"))
        XCTAssertNil(table.match(method: .delete, path: "/v1/models"))
    }

    func testUnknownPathDoesNotMatch() {
        let table = EndpointTable([Endpoint(EndpointSpec(.get, "/v1/models"), handler: Self.okHandler)])
        XCTAssertNil(table.match(method: .get, path: "/v1/models/extra"))
        XCTAssertNil(table.match(method: .get, path: "/v1"))
        XCTAssertNil(table.match(method: .get, path: "/v1/nope"))
    }

    func testParameterizedMatchCapturesAndDecodes() {
        let table = EndpointTable([
            Endpoint(EndpointSpec(.delete, "/v1/models/:id"), handler: Self.okHandler),
        ])
        let match = table.match(method: .delete, path: "/v1/models/models--YoozLabs--X%2DY")
        XCTAssertNotNil(match)
        XCTAssertEqual(match?.pathParameters["id"], "models--YoozLabs--X-Y")
    }

    func testParameterizedMatchRequiresSameSegmentCount() {
        let table = EndpointTable([
            Endpoint(EndpointSpec(.delete, "/v1/models/:id"), handler: Self.okHandler),
        ])
        XCTAssertNil(table.match(method: .delete, path: "/v1/models"))
        XCTAssertNil(table.match(method: .delete, path: "/v1/models/a/b"))
    }

    func testStaticSegmentsMustMatchAroundParameters() {
        let table = EndpointTable([
            Endpoint(
                EndpointSpec(.post, "/v1/infinite/sessions/:sessionID/append"),
                handler: Self.okHandler
            ),
        ])
        XCTAssertNotNil(table.match(method: .post, path: "/v1/infinite/sessions/abc/append"))
        XCTAssertNil(table.match(method: .post, path: "/v1/infinite/sessions/abc/generate"))
    }

    // MARK: - WireResponse / WireError

    func testWireResponseJSONEncodesWithDefaultEncoder() throws {
        let response = try WireResponse.json(SessionBeginResponse(sessionId: "s", ts: "t"))
        XCTAssertEqual(response.status, 200)
        let decoded = try JSONDecoder().decode(SessionBeginResponse.self, from: response.body)
        XCTAssertEqual(decoded, SessionBeginResponse(sessionId: "s", ts: "t"))
    }

    func testNoContentIs204WithEmptyBody() {
        XCTAssertEqual(WireResponse.noContent.status, 204)
        XCTAssertTrue(WireResponse.noContent.body.isEmpty)
    }

    func testInvalidRequestErrorShape() {
        struct Dummy: Error, LocalizedError {
            var errorDescription: String? { "boom" }
        }
        let error = WireError.invalidRequest(Dummy())
        XCTAssertEqual(error.status, 400)
        XCTAssertEqual(error.code, "invalid_request")
        XCTAssertEqual(error.message, "Invalid request body: boom")
    }

    // MARK: - Catalog / manifest projection

    func testSpecCatalogHasNoDuplicates() {
        let keys = EndpointSpecs.all.map(\.key)
        XCTAssertEqual(keys.count, Set(keys).count, "EndpointSpecs.all has duplicate entries")
    }

    func testConvertedAndLegacyAreDisjoint() {
        let converted = Set(EndpointSpecs.converted.map(\.key))
        let legacy = Set(EndpointSpecs.legacy.map(\.key))
        XCTAssertTrue(
            converted.isDisjoint(with: legacy),
            "a spec appears in both converted and legacy: \(converted.intersection(legacy))"
        )
    }

    /// `RouteManifest.all` must be exactly the spec catalog projected to
    /// manifest entries, plus the single WebSocket route — the #225
    /// acceptance criterion "the parity test's manifest is generated from
    /// the table".
    func testManifestIsAProjectionOfTheSpecCatalog() {
        let projected = Set(EndpointSpecs.all.map { "\($0.method.rawValue) \($0.path)" })
        let manifest = Set(RouteManifest.all.map(\.key))
        XCTAssertEqual(
            manifest.subtracting(projected),
            ["WS /v1/stt/stream"],
            "manifest carries entries beyond the catalog + the WS route"
        )
        XCTAssertTrue(
            projected.isSubset(of: manifest),
            "catalog specs missing from the manifest: \(projected.subtracting(manifest))"
        )
    }
}
