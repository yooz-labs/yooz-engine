// Copyright 2026 Yooz Labs. All rights reserved.

import CryptoKit
import Foundation
import os.log

// MARK: - Manifest types

/// LFS sub-object the Hub attaches to large blobs (model weights,
/// merges, etc.). Present for any file that shipped via Git LFS;
/// `nil` for inline blobs.
public struct HFManifestLFS: Codable, Sendable, Equatable {
    /// SHA-256 hex digest of the blob. Used to verify on-disk
    /// integrity after a download completes.
    public let oid: String?

    public init(oid: String?) {
        self.oid = oid
    }
}

/// Manifest entry returned by the Hugging Face Hub
/// `/api/models/<repo>/tree/<ref>` endpoint. The Hub returns one of
/// these per file; the fetcher picks the entries it needs to materialize
/// the full checkpoint on disk.
public struct HFManifestEntry: Codable, Sendable, Equatable {
    /// Path within the repo, e.g. `model.safetensors`.
    public let path: String
    /// File size in bytes when known. The Hub returns this for every
    /// blob in current API revisions; missing means "treat as
    /// non-resumable".
    public let size: Int64?
    /// `file` for normal blobs; `directory` for subdirs (we filter
    /// those out).
    public let type: String
    /// LFS payload when the blob shipped via Git LFS. Carries the
    /// SHA-256 the fetcher uses to verify the on-disk download.
    public let lfs: HFManifestLFS?

    public init(
        path: String,
        size: Int64?,
        type: String = "file",
        lfs: HFManifestLFS? = nil
    ) {
        self.path = path
        self.size = size
        self.type = type
        self.lfs = lfs
    }
}

// MARK: - Progress reporting

/// Progress event emitted by `Qwen3ASRModelFetcher.download`. The
/// fetcher publishes one event per fetched file (`fileStarted`,
/// `fileBytes`, `fileFinished`) plus a final `prep`/`done` pair.
public enum DownloadProgress: Sendable, Equatable {
    case manifestResolved(totalBytes: Int64, fileCount: Int)
    case fileStarted(path: String, bytes: Int64?)
    case fileBytes(path: String, completed: Int64, total: Int64?)
    case fileFinished(path: String, bytes: Int64)
    case tokenizerPrepStarted
    case tokenizerPrepFinished
    case done(modelDir: URL)
}

// MARK: - HTTP client abstraction

/// Minimal HTTP client surface the fetcher uses. Allows tests to
/// inject a `URLProtocol`-backed mock instead of hitting the live HF
/// Hub. The real implementation is `URLSessionHTTPDownloadClient`.
public protocol HTTPDownloadClient: Sendable {
    /// Fetch the JSON body at `url`. Used for manifest discovery.
    func fetchData(url: URL, headers: [String: String]) async throws -> Data

    /// Stream `url` into `destination`. Supports `Range` resume via
    /// `byteOffset`. The implementation appends bytes to the file
    /// (creating it if absent) and reports completed-byte counts via
    /// `progress`.
    func downloadFile(
        url: URL,
        destination: URL,
        byteOffset: Int64,
        expectedTotalBytes: Int64?,
        progress: @Sendable (Int64) -> Void
    ) async throws
}

/// Production HTTP client. Uses `URLSession` with delegate-driven
/// progress callbacks.
public struct URLSessionHTTPDownloadClient: HTTPDownloadClient {
    private let session: URLSession

    public init(session: URLSession = .shared) {
        self.session = session
    }

    public func fetchData(
        url: URL, headers: [String: String]
    ) async throws -> Data {
        var request = URLRequest(url: url)
        for (k, v) in headers { request.setValue(v, forHTTPHeaderField: k) }
        let (data, response): (Data, URLResponse)
        do {
            (data, response) = try await session.data(for: request)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw Qwen3ASRError.fetchFailed(
                .transport("\(url): \(error.localizedDescription)")
            )
        }
        guard let http = response as? HTTPURLResponse else {
            throw Qwen3ASRError.fetchFailed(
                .transport("Non-HTTP response from \(url)")
            )
        }
        guard (200..<300).contains(http.statusCode) else {
            throw Qwen3ASRError.fetchFailed(
                .httpStatus(code: http.statusCode, url: url)
            )
        }
        return data
    }

    public func downloadFile(
        url: URL,
        destination: URL,
        byteOffset: Int64,
        expectedTotalBytes: Int64?,
        progress: @Sendable (Int64) -> Void
    ) async throws {
        var request = URLRequest(url: url)
        if byteOffset > 0 {
            request.setValue("bytes=\(byteOffset)-", forHTTPHeaderField: "Range")
        }
        let bytes: URLSession.AsyncBytes
        let response: URLResponse
        do {
            (bytes, response) = try await session.bytes(for: request)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw Qwen3ASRError.fetchFailed(
                .transport("\(url): \(error.localizedDescription)")
            )
        }
        guard let http = response as? HTTPURLResponse else {
            throw Qwen3ASRError.fetchFailed(
                .transport("Non-HTTP response from \(url)")
            )
        }
        guard (200..<300).contains(http.statusCode) else {
            throw Qwen3ASRError.fetchFailed(
                .httpStatus(code: http.statusCode, url: url)
            )
        }
        // If we asked for a Range and the server returned 200 (full
        // body) instead of 206 (Partial Content), appending its
        // payload to our existing partial file would corrupt it.
        // Fail loud rather than silently produce a duplicate-prefix
        // file. Caller can retry with byteOffset=0 next time.
        if byteOffset > 0 && http.statusCode != 206 {
            throw Qwen3ASRError.fetchFailed(.rangeIgnored(url: url))
        }

        // When `byteOffset == 0` the caller wants a fresh download —
        // either the file didn't exist or it was corrupted (e.g. local
        // size > manifest size; see `fetchFile`). Truncate any existing
        // bytes before opening the handle so `seekToEnd` puts the
        // cursor at byte 0. Without this, an existing file's payload
        // gets prepended to the new download and the on-disk size
        // grows unbounded across retries — engine#144 reproduced this
        // as a `model.safetensors` blob 10x larger than its HF manifest
        // entry, which then failed the size-mismatch check and aborted
        // the run before the sentinel could be written, leaving the
        // dir not-ready for the next switch.
        //
        // Check `createFile`'s Bool: a `false` return means the
        // truncate/create failed (disk full, permission denied, or a
        // pre-existing file we can't replace). Without this guard the
        // subsequent `seekToEnd` would append on top of the
        // un-truncated bytes — reintroducing the very corruption this
        // path was added to fix.
        if byteOffset == 0 {
            guard FileManager.default.createFile(
                atPath: destination.path, contents: nil
            ) else {
                throw Qwen3ASRError.fetchFailed(
                    .other(
                        "Could not truncate destination file at "
                        + "\(destination.path) before fresh download; "
                        + "refusing to append on possibly-corrupted bytes."
                    )
                )
            }
        } else if !FileManager.default.fileExists(atPath: destination.path) {
            guard FileManager.default.createFile(
                atPath: destination.path, contents: nil
            ) else {
                throw Qwen3ASRError.fetchFailed(
                    .other(
                        "Could not create destination file at "
                        + "\(destination.path) for download."
                    )
                )
            }
        }
        let handle = try FileHandle(forWritingTo: destination)
        defer { try? handle.close() }
        try handle.seekToEnd()

        var written: Int64 = byteOffset
        var buffer = Data()
        buffer.reserveCapacity(64 * 1024)
        do {
            for try await byte in bytes {
                // Cooperative cancellation: an outer Task.cancel()
                // (e.g. from `AsyncThrowingStream.onTermination` in
                // `download(into:)`) propagates to this inner await
                // and lets us free partial-file state cleanly rather
                // than continuing to consume bytes after the caller
                // gave up.
                try Task.checkCancellation()
                buffer.append(byte)
                if buffer.count >= 64 * 1024 {
                    try handle.write(contentsOf: buffer)
                    written += Int64(buffer.count)
                    progress(written)
                    buffer.removeAll(keepingCapacity: true)
                }
            }
        } catch is CancellationError {
            if !buffer.isEmpty {
                try? handle.write(contentsOf: buffer)
            }
            throw CancellationError()
        } catch {
            throw Qwen3ASRError.fetchFailed(
                .transport("\(url): \(error.localizedDescription)")
            )
        }
        if !buffer.isEmpty {
            try handle.write(contentsOf: buffer)
            written += Int64(buffer.count)
            progress(written)
        }
    }
}

// MARK: - Fetcher actor

/// Fetches the canonical Qwen3-ASR checkpoint into the engine's
/// Application Support directory. Resumable per-file. Tokenizer prep
/// runs after the artifacts land. Idempotent: a fully-prepped model
/// directory short-circuits with `DownloadProgress.done`.
public actor Qwen3ASRModelFetcher {

    // MARK: - Singleton

    public static let shared = Qwen3ASRModelFetcher()

    // MARK: - Constants

    /// HuggingFace repo id served from the canonical mirror.
    public static let canonicalRepo = "mlx-community/Qwen3-ASR-1.7B-8bit"

    /// Files we materialize on disk. `tokenizer.json` is in
    /// `optionalFiles` because some checkpoint revisions ship it and
    /// some don't; the prep step at first-run is the single source of
    /// truth for tokenizer readiness — see `Qwen3ASRTokenizerPrep`.
    public static let requiredFiles: [String] = [
        "config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "preprocessor_config.json",
        "generation_config.json",
        "chat_template.json",
    ]

    /// Files that, when missing, do not fail the fetch.
    /// `tokenizer.json` is preferred when the upstream mirror ships
    /// it; the prep step validates it loads via
    /// `AutoTokenizer.from(modelFolder:)`. Today swift-transformers'
    /// loader REQUIRES `tokenizer.json`, so when the upstream
    /// revision omits it, prep surfaces a typed
    /// `tokenizerValidationFailed` error at first-run rather than
    /// silently producing wrong token IDs later. We deliberately do not
    /// synthesize the file ourselves; the canary encode in prep
    /// catches synthesis-vs-canonical drift if a future
    /// swift-transformers release adds the fallback path.
    public static let optionalFiles: [String] = [
        "tokenizer.json",
    ]

    // MARK: - State

    private let logger = Logger(
        subsystem: "live.yooz.engine",
        category: "Qwen3ASRModelFetcher"
    )

    private let client: HTTPDownloadClient
    private let baseURL: URL

    public init(
        client: HTTPDownloadClient = URLSessionHTTPDownloadClient(),
        baseURL: URL = URL(string: "https://huggingface.co")!
    ) {
        self.client = client
        self.baseURL = baseURL
    }

    // MARK: - Public surface

    /// Default on-disk location for the Qwen3-ASR checkpoint. The
    /// directory may not exist yet; `download(...)` creates it.
    ///
    /// Computed locally rather than via `EngineConfig.modelsDirectory`
    /// so this file compiles cleanly inside the Qwen3ASR SwiftPM
    /// target (which doesn't link the engine app). The resolved path
    /// is identical: `~/Library/Application Support/YoozEngine/Models/qwen3-asr-1.7b/`.
    public static var defaultModelDir: URL {
        if let override = ProcessInfo.processInfo.environment[
            "YOOZ_QWEN3_ASR_DIR"
        ] {
            return URL(fileURLWithPath: override)
        }
        guard let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first else {
            // Fall back to a sensible default rather than crashing.
            return URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("qwen3-asr-1.7b")
        }
        return appSupport
            .appendingPathComponent("YoozEngine/Models/qwen3-asr-1.7b")
    }

    /// True when every required file already exists on disk and the
    /// post-fetch sentinel has been dropped.
    ///
    /// Round-2 silent-failure #B9: a previous version did existence-
    /// only checks, so a partially-copied or truncated
    /// `model.safetensors` (e.g. operator pre-staging the dir to
    /// skip the 3.5 GB fetch) would short-circuit the fetcher and
    /// the loader would either crash with "malformed safetensors"
    /// or, worse, load garbage and return junk transcriptions.
    /// The sentinel is dropped by `runDownload` only after every
    /// file's size and SHA-256 (when available) have been
    /// validated, so its presence is the single proof point that
    /// the directory was filled by *us* with verified bytes.
    public func isModelDirReady(_ modelDir: URL) -> Bool {
        for file in Self.requiredFiles {
            let url = modelDir.appendingPathComponent(file)
            if !FileManager.default.fileExists(atPath: url.path) {
                return false
            }
        }
        let sentinel = modelDir.appendingPathComponent(
            Self.fetchCompleteSentinelName
        )
        return FileManager.default.fileExists(atPath: sentinel.path)
    }

    /// Sentinel file name dropped by the fetcher after every file's
    /// size + SHA-256 (when available) validates. Used by
    /// `isModelDirReady` to reject pre-staged dirs that haven't
    /// gone through the integrity check.
    fileprivate static let fetchCompleteSentinelName: String =
        ".yooz_qwen3_asr_fetch_complete"

    /// Fetch the canonical checkpoint into `modelDir`. Yields progress
    /// events on the returned `AsyncThrowingStream`. Caller awaits the
    /// stream to completion before treating the model as loadable.
    ///
    /// - Parameters:
    ///   - modelDir: target directory (created if missing).
    ///   - repo: HF repo id; defaults to `canonicalRepo`.
    ///   - revision: HF git revision; defaults to `main`.
    ///   - runTokenizerPrep: when `true` (production default), runs
    ///     `Qwen3ASRTokenizerPrep.prepare(modelDir:)` after the
    ///     downloads land. Tests with synthetic fixture content set
    ///     this to `false` because the prep step uses a real tokenizer
    ///     loader that would reject junk JSON.
    public func download(
        into modelDir: URL,
        repo: String = canonicalRepo,
        revision: String = "main",
        runTokenizerPrep: Bool = true
    ) -> AsyncThrowingStream<DownloadProgress, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.runDownload(
                        modelDir: modelDir,
                        repo: repo,
                        revision: revision,
                        runTokenizerPrep: runTokenizerPrep,
                        continuation: continuation
                    )
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: - Internals

    private func runDownload(
        modelDir: URL,
        repo: String,
        revision: String,
        runTokenizerPrep: Bool,
        continuation: AsyncThrowingStream<DownloadProgress, Error>.Continuation
    ) async throws {
        try FileManager.default.createDirectory(
            at: modelDir, withIntermediateDirectories: true
        )

        let manifest = try await fetchManifest(repo: repo, revision: revision)
        let needed = manifest.filter {
            Self.requiredFiles.contains($0.path)
                || Self.optionalFiles.contains($0.path)
        }

        let totalBytes = needed.compactMap(\.size).reduce(0, +)
        continuation.yield(
            .manifestResolved(totalBytes: totalBytes, fileCount: needed.count)
        )

        for entry in needed {
            try Task.checkCancellation()
            try await fetchFile(
                entry: entry,
                modelDir: modelDir,
                repo: repo,
                revision: revision,
                continuation: continuation
            )
        }

        // Tokenizer prep — idempotent. Skipped under unit tests that
        // feed the fetcher synthetic fixture data (the real prep step
        // uses a JSON-validating tokenizer loader that would reject
        // mocked content).
        if runTokenizerPrep {
            continuation.yield(.tokenizerPrepStarted)
            try await Qwen3ASRTokenizerPrep.prepare(modelDir: modelDir)
            continuation.yield(.tokenizerPrepFinished)
        }

        // Drop the integrity sentinel only after every per-file
        // validation passed. `isModelDirReady` consults this so a
        // pre-staged but unvalidated dir is treated as not-ready
        // and re-fetched (round-2 silent-failure #B9).
        let sentinel = modelDir.appendingPathComponent(
            Self.fetchCompleteSentinelName
        )
        try Data().write(to: sentinel, options: .atomic)

        continuation.yield(.done(modelDir: modelDir))
    }

    private func fetchManifest(
        repo: String, revision: String
    ) async throws -> [HFManifestEntry] {
        let path = "/api/models/\(repo)/tree/\(revision)?recursive=true"
        guard let url = URL(string: path, relativeTo: baseURL) else {
            throw Qwen3ASRError.fetchFailed(
                .other("Invalid manifest URL for repo \(repo)")
            )
        }
        let data = try await client.fetchData(
            url: url, headers: ["Accept": "application/json"]
        )
        do {
            let entries = try JSONDecoder().decode(
                [HFManifestEntry].self, from: data
            )
            return entries.filter { $0.type == "file" }
        } catch {
            throw Qwen3ASRError.fetchFailed(
                .manifestDecode("\(error)")
            )
        }
    }

    private func fetchFile(
        entry: HFManifestEntry,
        modelDir: URL,
        repo: String,
        revision: String,
        continuation: AsyncThrowingStream<DownloadProgress, Error>.Continuation
    ) async throws {
        let dest = modelDir.appendingPathComponent(entry.path)

        // Skip if already complete.
        if let expected = entry.size,
           let actual = try? FileManager.default.attributesOfItem(
               atPath: dest.path
           )[.size] as? Int64,
           actual == expected
        {
            continuation.yield(
                .fileStarted(path: entry.path, bytes: expected)
            )
            continuation.yield(
                .fileFinished(path: entry.path, bytes: expected)
            )
            return
        }

        // Determine resume offset.
        var offset: Int64 = 0
        if let expected = entry.size,
           let actual = try? FileManager.default.attributesOfItem(
               atPath: dest.path
           )[.size] as? Int64
        {
            if actual < expected {
                offset = actual
            } else if actual > expected {
                // Local file is larger than the manifest claims —
                // a prior aborted download left a corrupted blob
                // (engine#144: cumulative append-on-retry inflated
                // model.safetensors to 10x its expected size). Remove
                // it and re-fetch from scratch; `downloadFile` will
                // also truncate when `offset == 0` as belt-and-
                // suspenders for the same scenario.
                //
                // Throw on remove failure rather than swallowing —
                // a silent fall-through would let `downloadFile`
                // append on top of the oversized prefix again,
                // recreating the same bug a layer up. The truncate
                // guard in `downloadFile` catches most cases, but a
                // permission error or a busy file would slip through
                // both layers without surfacing a diagnostic.
                logger.warning("""
                    Removing oversized cached file \(entry.path, privacy: .public) \
                    (\(actual) bytes on disk vs \(expected) in manifest); \
                    re-fetching from scratch.
                    """)
                do {
                    try FileManager.default.removeItem(at: dest)
                } catch {
                    throw Qwen3ASRError.fetchFailed(
                        .other(
                            "Failed to remove oversized cached file "
                            + "\(entry.path) (\(actual) bytes vs "
                            + "\(expected) in manifest): "
                            + "\(error.localizedDescription)"
                        )
                    )
                }
            }
        } else if FileManager.default.fileExists(atPath: dest.path)
                  && entry.size == nil
        {
            // Unknown size and file exists — re-pull from scratch.
            try? FileManager.default.removeItem(at: dest)
        }

        let resolveURL = baseURL.appendingPathComponent(
            "/\(repo)/resolve/\(revision)/\(entry.path)"
        )

        continuation.yield(
            .fileStarted(path: entry.path, bytes: entry.size)
        )

        try await client.downloadFile(
            url: resolveURL,
            destination: dest,
            byteOffset: offset,
            expectedTotalBytes: entry.size
        ) { written in
            continuation.yield(
                .fileBytes(
                    path: entry.path,
                    completed: written,
                    total: entry.size
                )
            )
        }

        // Validate final size when known.
        if let expected = entry.size,
           let actual = try? FileManager.default.attributesOfItem(
               atPath: dest.path
           )[.size] as? Int64,
           actual != expected
        {
            throw Qwen3ASRError.fetchFailed(
                .sizeMismatch(
                    path: entry.path,
                    expected: expected,
                    actual: actual
                )
            )
        }

        // Validate streaming SHA-256 when the manifest carries an
        // LFS digest. Catches mid-stream corruption that the size
        // check cannot (e.g. TLS-MITM proxy that flips bytes but
        // returns the right total length).
        //
        // Round-2 silent-failure #B6: silently skipping SHA when the
        // manifest lacks an `lfs.oid` erodes the integrity guarantee
        // for the very files (large weights) that motivated adding
        // it. Require a digest for `*.safetensors` files; for the
        // smaller, non-LFS JSON / text artefacts (manifest, configs,
        // tokenizer files), accept absence with a loud warning so
        // operators see the gap in `os_log`.
        if let expectedSha = entry.lfs?.oid {
            try validateSha256(
                fileURL: dest,
                expected: expectedSha,
                path: entry.path
            )
        } else if entry.path.hasSuffix(".safetensors") {
            throw Qwen3ASRError.fetchFailed(
                .other(
                    "Manifest entry for required safetensors file "
                    + "\"\(entry.path)\" is missing the lfs.oid "
                    + "digest. Refusing to accept the download "
                    + "without an integrity check; the upstream "
                    + "manifest schema may have shifted."
                )
            )
        } else {
            let path = entry.path
            let warningMessage =
                "No LFS digest for size-only validation. Acceptable "
                + "for small JSON / text artefacts; surface this if "
                + "it appears for a weights file."
            logger.warning(
                "\(path, privacy: .public): \(warningMessage, privacy: .public)"
            )
        }

        continuation.yield(
            .fileFinished(
                path: entry.path,
                bytes: entry.size ?? 0
            )
        )
    }

    /// Streaming SHA-256 verification of `fileURL` against `expected`
    /// (hex digest). Reads the file in 1 MB chunks so a 3 GB blob
    /// doesn't pin RAM.
    private nonisolated func validateSha256(
        fileURL: URL,
        expected: String,
        path: String
    ) throws {
        let handle = try FileHandle(forReadingFrom: fileURL)
        defer { try? handle.close() }
        var hasher = SHA256()
        let chunk = 1 * 1024 * 1024
        while true {
            try Task.checkCancellation()
            let data = try handle.read(upToCount: chunk) ?? Data()
            if data.isEmpty { break }
            hasher.update(data: data)
        }
        let actual = hasher.finalize().map {
            String(format: "%02x", $0)
        }.joined()
        let normalizedExpected = expected.lowercased()
        if actual != normalizedExpected {
            throw Qwen3ASRError.fetchFailed(
                .checksumMismatch(
                    path: path,
                    expected: normalizedExpected,
                    actual: actual
                )
            )
        }
    }
}
