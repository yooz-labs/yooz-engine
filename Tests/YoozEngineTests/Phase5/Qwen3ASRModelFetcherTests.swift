// Copyright 2026 Yooz Labs. All rights reserved.

import CryptoKit
import Foundation
import XCTest

@testable import YoozEngine

/// Phase 5 — `Qwen3ASRModelFetcher` against an injected
/// `HTTPDownloadClient` mock. CI must never hit the live HF Hub.
final class Qwen3ASRModelFetcherTests: XCTestCase {

    // MARK: - In-memory mock client

    /// Hand-rolled mock that serves a manifest + per-file payloads
    /// from in-memory dictionaries. Supports `Range` resume by
    /// honoring the `byteOffset` argument the fetcher passes.
    private final class MockClient: HTTPDownloadClient, @unchecked Sendable {
        let manifestData: Data
        let blobs: [String: Data]
        private(set) var rangeRequests: [(path: String, offset: Int64)] = []
        private(set) var fullRequests: [String] = []
        private let lock = NSLock()

        init(manifestData: Data, blobs: [String: Data]) {
            self.manifestData = manifestData
            self.blobs = blobs
        }

        func fetchData(
            url: URL, headers: [String: String]
        ) async throws -> Data {
            // Manifest endpoint: /api/models/<repo>/tree/<ref>
            if url.path.contains("/api/models/") {
                return manifestData
            }
            throw Qwen3ASRError.fetchFailed(
                "MockClient: unexpected fetchData URL \(url)"
            )
        }

        func downloadFile(
            url: URL,
            destination: URL,
            byteOffset: Int64,
            expectedTotalBytes: Int64?,
            progress: @Sendable (Int64) -> Void
        ) async throws {
            // Path like: /<repo>/resolve/<ref>/<file>
            let comps = url.pathComponents
            guard let resolveIdx = comps.firstIndex(of: "resolve"),
                  resolveIdx + 1 < comps.count
            else {
                throw Qwen3ASRError.fetchFailed(
                    "MockClient: malformed resolve URL \(url)"
                )
            }
            let path = comps[(resolveIdx + 2)...].joined(separator: "/")
            guard let blob = blobs[path] else {
                throw Qwen3ASRError.fetchFailed(
                    "MockClient: no blob for \(path)"
                )
            }

            lock.lock()
            if byteOffset > 0 {
                rangeRequests.append((path: path, offset: byteOffset))
            } else {
                fullRequests.append(path)
            }
            lock.unlock()

            // Slice off the resumed portion, append to disk.
            let payload = blob.subdata(in: Int(byteOffset)..<blob.count)
            if !FileManager.default.fileExists(atPath: destination.path) {
                FileManager.default.createFile(
                    atPath: destination.path, contents: nil
                )
            }
            let handle = try FileHandle(forWritingTo: destination)
            try handle.seekToEnd()
            try handle.write(contentsOf: payload)
            try handle.close()

            // Report progress in two chunks so the callback fires more
            // than once and we can assert monotonicity.
            let half = Int64(payload.count) / 2
            progress(byteOffset + half)
            progress(byteOffset + Int64(payload.count))
        }
    }

    /// Always-fails client used to exercise the error-mode matrix.
    /// Returns the supplied error from `fetchData(url:headers:)`
    /// (which the real fetcher calls for the manifest endpoint).
    private final class StaticErrorClient: HTTPDownloadClient,
        @unchecked Sendable
    {
        let manifestError: Error
        init(manifestError: Error) { self.manifestError = manifestError }

        func fetchData(
            url: URL, headers: [String: String]
        ) async throws -> Data {
            throw manifestError
        }

        func downloadFile(
            url: URL,
            destination: URL,
            byteOffset: Int64,
            expectedTotalBytes: Int64?,
            progress: @Sendable (Int64) -> Void
        ) async throws {
            throw manifestError
        }
    }

    // MARK: - Helpers

    private var tempDir: URL!

    override func setUp() {
        super.setUp()
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen3-fetch-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: tempDir, withIntermediateDirectories: true
        )
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: tempDir)
        super.tearDown()
    }

    /// Build a fake manifest + blob dictionary covering every required
    /// file plus an optional `tokenizer.json`. Each file gets a
    /// distinguishable byte payload so we can verify on-disk contents.
    /// Safetensors files carry an `lfs.oid` SHA-256 of their payload
    /// so the fetcher's integrity check passes; non-LFS artefacts
    /// (manifest JSON / configs / text) intentionally have no
    /// digest to exercise the size-only-validation warning path.
    private func fixtureClient() throws -> (MockClient, [String: Data]) {
        let allFiles =
            Qwen3ASRModelFetcher.requiredFiles
            + Qwen3ASRModelFetcher.optionalFiles
        var blobs: [String: Data] = [:]
        for path in allFiles {
            let payload = "fixture-\(path)\n".data(using: .utf8)!
            blobs[path] = payload + Data(repeating: 0x00, count: 1024)
        }

        let manifestEntries = blobs.map { path, data -> HFManifestEntry in
            let lfs: HFManifestLFS?
            if path.hasSuffix(".safetensors") {
                lfs = HFManifestLFS(oid: Self.sha256Hex(data))
            } else {
                lfs = nil
            }
            return HFManifestEntry(
                path: path,
                size: Int64(data.count),
                type: "file",
                lfs: lfs
            )
        }
        let manifestData = try JSONEncoder().encode(manifestEntries)
        return (MockClient(manifestData: manifestData, blobs: blobs), blobs)
    }

    /// SHA-256 hex digest helper for the test fixture. Mirrors the
    /// production fetcher's `validateSha256` (1 MB chunks) but
    /// computed eagerly because the test fixtures fit in memory.
    private static func sha256Hex(_ data: Data) -> String {
        var hasher = SHA256()
        hasher.update(data: data)
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    // MARK: - Tests

    /// End-to-end fetch into an empty directory: every required file
    /// lands on disk, sizes match, optional file is fetched too.
    func testDownloadsAllFiles() async throws {
        let (client, blobs) = try fixtureClient()
        let fetcher = Qwen3ASRModelFetcher(
            client: client,
            baseURL: URL(string: "https://mock.local")!
        )

        var progressCount = 0
        var sawDone = false
        for try await event in await fetcher.download(into: tempDir, runTokenizerPrep: false) {
            progressCount += 1
            if case .done = event { sawDone = true }
        }
        XCTAssertTrue(sawDone, "Expected a `.done` progress event")
        XCTAssertGreaterThan(progressCount, 0)

        for (path, blob) in blobs {
            let dest = tempDir.appendingPathComponent(path)
            XCTAssertTrue(
                FileManager.default.fileExists(atPath: dest.path),
                "Missing fetched file \(path)"
            )
            let onDisk = try Data(contentsOf: dest)
            XCTAssertEqual(onDisk.count, blob.count, "Size mismatch on \(path)")
        }

        // The fetcher does not re-pull files that already exist with
        // the right size; assert no resume requests happened on a
        // fresh dir.
        XCTAssertTrue(
            client.rangeRequests.isEmpty,
            "Fresh fetch must not issue Range requests"
        )
    }

    /// Pre-populate a partial `model.safetensors` on disk and confirm
    /// the fetcher resumes from `currentSize` rather than re-pulling.
    func testResumesPartialFile() async throws {
        let (client, blobs) = try fixtureClient()
        let fetcher = Qwen3ASRModelFetcher(
            client: client,
            baseURL: URL(string: "https://mock.local")!
        )

        // Drop the first 100 bytes of the model.safetensors blob into
        // the destination; the fetcher should issue a Range from 100.
        let dest = tempDir.appendingPathComponent("model.safetensors")
        let prefix = blobs["model.safetensors"]!.prefix(100)
        FileManager.default.createFile(
            atPath: dest.path, contents: Data(prefix)
        )

        for try await _ in await fetcher.download(into: tempDir, runTokenizerPrep: false) {}

        XCTAssertTrue(
            client.rangeRequests.contains { $0.path == "model.safetensors" && $0.offset == 100 },
            "Expected a Range request for model.safetensors at offset 100; got \(client.rangeRequests)"
        )
        // After resume the file must equal the full blob.
        let onDisk = try Data(contentsOf: dest)
        XCTAssertEqual(onDisk, blobs["model.safetensors"])
    }

    /// Progress callbacks must fire in monotonically non-decreasing
    /// order for each file. The Phase 5 fetcher feeds the
    /// per-handle byte counter through `DownloadProgress.fileBytes`.
    func testProgressIsMonotonicPerFile() async throws {
        let (client, _) = try fixtureClient()
        let fetcher = Qwen3ASRModelFetcher(
            client: client,
            baseURL: URL(string: "https://mock.local")!
        )

        var perFileMax: [String: Int64] = [:]
        for try await event in await fetcher.download(into: tempDir, runTokenizerPrep: false) {
            if case let .fileBytes(path, completed, _) = event {
                let prev = perFileMax[path] ?? 0
                XCTAssertGreaterThanOrEqual(
                    completed, prev,
                    "Progress went backwards for \(path)"
                )
                perFileMax[path] = max(prev, completed)
            }
        }

        XCTAssertFalse(perFileMax.isEmpty, "No fileBytes events observed")
    }

    /// `isModelDirReady` must report `false` until every required
    /// file is present, then flip to `true` after a successful fetch.
    func testIsModelDirReadyTransitions() async throws {
        let (client, _) = try fixtureClient()
        let fetcher = Qwen3ASRModelFetcher(
            client: client,
            baseURL: URL(string: "https://mock.local")!
        )

        let beforeReady = await fetcher.isModelDirReady(tempDir)
        XCTAssertFalse(beforeReady)

        for try await _ in await fetcher.download(into: tempDir, runTokenizerPrep: false) {}

        let afterReady = await fetcher.isModelDirReady(tempDir)
        XCTAssertTrue(afterReady)
    }

    // MARK: - HTTP error mode matrix

    /// Inject an HTTP 404 from the manifest endpoint. The fetcher
    /// must surface a typed `Qwen3ASRError.fetchFailed` with the
    /// `.httpStatus` payload — never a silent success.
    func testManifestHTTP404PropagatesAsTypedError() async throws {
        let mock = StaticErrorClient(
            manifestError: Qwen3ASRError.fetchFailed(
                .httpStatus(
                    code: 404,
                    url: URL(string: "https://mock.local")!
                )
            )
        )
        let fetcher = Qwen3ASRModelFetcher(
            client: mock,
            baseURL: URL(string: "https://mock.local")!
        )

        do {
            for try await _ in await fetcher.download(
                into: tempDir, runTokenizerPrep: false
            ) {}
            XCTFail("Expected fetchFailed for 404 manifest")
        } catch let asrError as Qwen3ASRError {
            guard case .fetchFailed(let failure) = asrError else {
                return XCTFail("Expected .fetchFailed, got \(asrError)")
            }
            guard case .httpStatus(let code, _) = failure else {
                return XCTFail(
                    "Expected .httpStatus payload, got \(failure)"
                )
            }
            XCTAssertEqual(code, 404)
        }

        let ready = await fetcher.isModelDirReady(tempDir)
        XCTAssertFalse(
            ready,
            "isModelDirReady must remain false after a fetch error."
        )
    }

    /// Inject an HTTP 401 (auth required / token expired). Verifies
    /// the same typed-error path as 404 — the fetcher does not
    /// branch on status code, the typed payload preserves it for
    /// the consumer.
    func testManifestHTTP401PropagatesAsTypedError() async throws {
        let mock = StaticErrorClient(
            manifestError: Qwen3ASRError.fetchFailed(
                .httpStatus(
                    code: 401,
                    url: URL(string: "https://mock.local")!
                )
            )
        )
        let fetcher = Qwen3ASRModelFetcher(
            client: mock,
            baseURL: URL(string: "https://mock.local")!
        )

        do {
            for try await _ in await fetcher.download(
                into: tempDir, runTokenizerPrep: false
            ) {}
            XCTFail("Expected fetchFailed for 401 manifest")
        } catch let asrError as Qwen3ASRError {
            guard case .fetchFailed(let failure) = asrError else {
                return XCTFail("Expected .fetchFailed, got \(asrError)")
            }
            guard case .httpStatus(let code, _) = failure else {
                return XCTFail(
                    "Expected .httpStatus payload, got \(failure)"
                )
            }
            XCTAssertEqual(code, 401)
        }
    }

    // MARK: - Restart-resume

    /// Real restart-resume: instantiate a fetcher, consume the
    /// stream up to the first `fileBytes` event, drop the stream
    /// (mid-download). Instantiate a NEW fetcher (singleton-
    /// discarded) and download again. The second pass must (a)
    /// issue a Range request, (b) end with the full blob on
    /// disk, (c) emit no double-prefix corruption.
    ///
    /// The byte-layer rangeIgnored guard is exercised separately
    /// in `testFetchValidationFailsWhenServerIgnoresRangeHeader`.
    /// This test asserts the higher-level invariant: resume after
    /// a partial download must issue a Range request, complete the
    /// file, and not double-prefix bytes.
    func testRestartResumePicksUpPartialFile() async throws {
        let (client, blobs) = try fixtureClient()
        let modelBlob = blobs["model.safetensors"]!
        // Pre-write a partial copy on disk to simulate "killed
        // mid-download a previous run". The first fetcher would
        // resume from that offset.
        let dest = tempDir.appendingPathComponent("model.safetensors")
        let partialBytes = 500
        let partial = modelBlob.prefix(partialBytes)
        try Data(partial).write(to: dest)

        // First fetcher: drop the stream after a few events, do
        // NOT consume it to completion. Cancellation propagates
        // via `continuation.onTermination`.
        do {
            let firstFetcher = Qwen3ASRModelFetcher(
                client: client,
                baseURL: URL(string: "https://mock.local")!
            )
            var consumed = 0
            for try await event in await firstFetcher.download(
                into: tempDir, runTokenizerPrep: false
            ) {
                consumed += 1
                if case .fileBytes = event {
                    if consumed > 1 { break }
                }
            }
        }

        // Second fetcher: identical args, should (a) see the
        // partial on disk, (b) issue a Range request from the
        // existing offset, (c) end with the full blob.
        let secondFetcher = Qwen3ASRModelFetcher(
            client: client,
            baseURL: URL(string: "https://mock.local")!
        )
        for try await _ in await secondFetcher.download(
            into: tempDir, runTokenizerPrep: false
        ) {}

        // model.safetensors must end up exactly as the canonical
        // blob — no duplication, no truncation.
        let final = try Data(contentsOf: dest)
        XCTAssertEqual(
            final, modelBlob,
            "Restart-resume produced corrupt model.safetensors."
        )

        // At least one Range request must have been issued for
        // model.safetensors during the run (the second fetcher
        // saw the partial on disk).
        let ranges = client.rangeRequests.filter {
            $0.path == "model.safetensors"
        }
        XCTAssertFalse(
            ranges.isEmpty,
            "Restart-resume must issue at least one Range request "
                + "for the partially-downloaded model file."
        )
    }

    /// Skip-when-already-complete: fetcher should NOT issue any
    /// downloads when every file is already on disk at the expected
    /// size. Tokenizer prep still runs (idempotent).
    func testSkipsCompleteFiles() async throws {
        let (client, blobs) = try fixtureClient()
        // Pre-write all files to disk.
        for (path, data) in blobs {
            let dest = tempDir.appendingPathComponent(path)
            try data.write(to: dest)
        }
        // Drop the tokenizer-prep sentinel so prep is a no-op too.
        try Data("ok\n".utf8).write(
            to: tempDir.appendingPathComponent(
                Qwen3ASRTokenizerPrep.sentinelFilename
            )
        )

        let fetcher = Qwen3ASRModelFetcher(
            client: client,
            baseURL: URL(string: "https://mock.local")!
        )
        for try await _ in await fetcher.download(into: tempDir, runTokenizerPrep: false) {}

        XCTAssertTrue(
            client.rangeRequests.isEmpty,
            "Already-complete dir must issue no Range requests"
        )
        XCTAssertTrue(
            client.fullRequests.isEmpty,
            "Already-complete dir must issue no full-file requests; got \(client.fullRequests)"
        )
    }

    /// SHA-256 mismatch on a safetensors file MUST surface as a typed
    /// `FetchFailure.checksumMismatch`, not be silently swallowed. A
    /// future regression that disabled SHA verification would let
    /// byte-flipped weights ship to disk; this test is the regression
    /// gate. (Round-2 silent-failure #B6 closure.)
    func testRejectsTamperedSafetensorsWithChecksumMismatch() async throws {
        let allFiles =
            Qwen3ASRModelFetcher.requiredFiles
            + Qwen3ASRModelFetcher.optionalFiles
        var blobs: [String: Data] = [:]
        for path in allFiles {
            let payload = "fixture-\(path)\n".data(using: .utf8)!
            blobs[path] = payload + Data(repeating: 0x00, count: 1024)
        }

        // Build the manifest with a *deliberately wrong* SHA for
        // every safetensors file — the fetcher should refuse to
        // accept the on-disk bytes after download because the
        // hash will not match.
        let bogusSha = String(repeating: "0", count: 64)
        let manifestEntries = blobs.map { path, data -> HFManifestEntry in
            let lfs: HFManifestLFS?
            if path.hasSuffix(".safetensors") {
                lfs = HFManifestLFS(oid: bogusSha)
            } else {
                lfs = nil
            }
            return HFManifestEntry(
                path: path,
                size: Int64(data.count),
                type: "file",
                lfs: lfs
            )
        }
        let manifestData = try JSONEncoder().encode(manifestEntries)
        let client = MockClient(manifestData: manifestData, blobs: blobs)

        let fetcher = Qwen3ASRModelFetcher(
            client: client,
            baseURL: URL(string: "https://mock.local")!
        )

        do {
            for try await _ in await fetcher.download(
                into: tempDir, runTokenizerPrep: false
            ) {
                // Drain. The throw happens during checksum verify
                // post-download; we expect to not reach a `.done`.
            }
            XCTFail(
                "Expected checksumMismatch to throw; the fetcher must "
                    + "not accept safetensors whose computed SHA-256 "
                    + "differs from the manifest's lfs.oid."
            )
        } catch let Qwen3ASRError.fetchFailed(payload) {
            switch payload {
            case .checksumMismatch(let path, let expected, _):
                XCTAssertTrue(
                    path.hasSuffix(".safetensors"),
                    "checksumMismatch must fire for a safetensors file; "
                        + "got \(path)"
                )
                XCTAssertEqual(
                    expected, bogusSha,
                    "Expected hash should equal the (bogus) lfs.oid we "
                        + "planted in the manifest"
                )
            default:
                XCTFail(
                    "Expected .checksumMismatch FetchFailure; got "
                        + "\(payload)"
                )
            }
        } catch {
            XCTFail(
                "Expected Qwen3ASRError.fetchFailed(.checksumMismatch); "
                    + "got \(error)"
            )
        }
    }
}
