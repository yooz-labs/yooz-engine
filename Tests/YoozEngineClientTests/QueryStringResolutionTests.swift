// QueryStringResolutionTests.swift
// YoozEngineClientTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Pins the `?wait=true` URL construction contract: paths passed to
// `get` / `post` that carry a `?<query>` suffix must be split into a
// path component plus a real URL query string. The pre-fix behavior
// (`URL.appendingPathComponent("/v1/stt/load?wait=true")`) percent-
// encoded the `?` into the path itself, producing
// `/v1/stt/load%3Fwait%3Dtrue` which the Hummingbird router sees as
// a literal path with no registered handler — every consumer using
// `loadModel` / `preloadModel` was getting a 404 instead of the
// blocking-load behavior they expected.

import Foundation
import XCTest

@testable import YoozEngineClient

final class QueryStringResolutionTests: XCTestCase {

    /// `resolveURL` is the internal helper post / get both call. A
    /// path with a `?` suffix must produce a URL whose `.query`
    /// equals the suffix (no percent-encoding into the path).
    func testQuerySuffixSurfacesAsRealQuery() {
        let client = YoozEngineClient(host: "127.0.0.1", port: 19920)
        let url = client.resolveURLForTesting("/v1/stt/load?wait=true")
        XCTAssertEqual(url.path, "/v1/stt/load",
                       "Path must not include the query string")
        XCTAssertEqual(url.query, "wait=true",
                       "Query must surface as a real query string")
        XCTAssertEqual(url.absoluteString,
                       "http://127.0.0.1:19920/v1/stt/load?wait=true")
    }

    /// A path without a `?` must produce a clean URL — no spurious
    /// `?` or empty query.
    func testPathWithoutQuerySuffixIsUnchanged() {
        let client = YoozEngineClient(host: "127.0.0.1", port: 19920)
        let url = client.resolveURLForTesting("/v1/stt/status")
        XCTAssertEqual(url.path, "/v1/stt/status")
        XCTAssertNil(url.query)
        XCTAssertEqual(url.absoluteString,
                       "http://127.0.0.1:19920/v1/stt/status")
    }

    /// Multiple query params separated by `&` survive the round-trip.
    /// Not used by any current SDK call site but the contract is
    /// stable so future endpoints can rely on it.
    func testMultipleQueryParamsAreParsed() {
        let client = YoozEngineClient(host: "127.0.0.1", port: 19920)
        let url = client.resolveURLForTesting("/v1/stt/load?wait=true&lang=en")
        XCTAssertEqual(url.path, "/v1/stt/load")
        XCTAssertEqual(url.query, "wait=true&lang=en")
    }
}

extension YoozEngineClient {
    /// Test-only thin alias for `resolveURL` so the asserts read
    /// naturally without exposing a public surface change.
    func resolveURLForTesting(_ path: String) -> URL {
        return resolveURL(path)
    }
}
