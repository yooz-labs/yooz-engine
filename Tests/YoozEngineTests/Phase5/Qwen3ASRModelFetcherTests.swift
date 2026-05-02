// Copyright 2026 Yooz Labs. All rights reserved.

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
    private func fixtureClient() throws -> (MockClient, [String: Data]) {
        let allFiles =
            Qwen3ASRModelFetcher.requiredFiles
            + Qwen3ASRModelFetcher.optionalFiles
        var blobs: [String: Data] = [:]
        for path in allFiles {
            // `model.safetensors` mocked content is just a marker plus
            // some bytes; we never load it as a real safetensors here.
            // Other files get plausible placeholder content (the
            // tokenizer artifacts must still parse, so they need to
            // be valid JSON / plain-text where required by downstream
            // tests; this fixture is for fetcher mechanics only).
            let payload = "fixture-\(path)\n".data(using: .utf8)!
            blobs[path] = payload + Data(repeating: 0x00, count: 1024)
        }

        let manifestEntries = blobs.map { path, data in
            HFManifestEntry(
                path: path, size: Int64(data.count), type: "file"
            )
        }
        let manifestData = try JSONEncoder().encode(manifestEntries)
        return (MockClient(manifestData: manifestData, blobs: blobs), blobs)
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
}
