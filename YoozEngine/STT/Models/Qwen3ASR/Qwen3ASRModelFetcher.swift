// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import os.log

// MARK: - Manifest types

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

    public init(path: String, size: Int64?, type: String = "file") {
        self.path = path
        self.size = size
        self.type = type
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
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw Qwen3ASRError.fetchFailed("Non-HTTP response from \(url)")
        }
        guard (200..<300).contains(http.statusCode) else {
            throw Qwen3ASRError.fetchFailed(
                "HTTP \(http.statusCode) from \(url)"
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
        let (bytes, response) = try await session.bytes(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw Qwen3ASRError.fetchFailed("Non-HTTP response from \(url)")
        }
        guard (200..<300).contains(http.statusCode) else {
            throw Qwen3ASRError.fetchFailed(
                "HTTP \(http.statusCode) from \(url)"
            )
        }

        // Append-mode write. Create the file if it doesn't exist.
        if !FileManager.default.fileExists(atPath: destination.path) {
            FileManager.default.createFile(
                atPath: destination.path, contents: nil
            )
        }
        let handle = try FileHandle(forWritingTo: destination)
        defer { try? handle.close() }
        try handle.seekToEnd()

        var written: Int64 = byteOffset
        var buffer = Data()
        buffer.reserveCapacity(64 * 1024)
        for try await byte in bytes {
            buffer.append(byte)
            if buffer.count >= 64 * 1024 {
                try handle.write(contentsOf: buffer)
                written += Int64(buffer.count)
                progress(written)
                buffer.removeAll(keepingCapacity: true)
            }
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

    /// Files we materialize on disk. `tokenizer.json` is optional
    /// because some checkpoint revisions ship it and others rely on
    /// the prep step — see the Phase 4 carryover note.
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

    /// Files that, when missing, do not fail the fetch. The prep
    /// step generates `tokenizer.json` if absent.
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
    public static var defaultModelDir: URL {
        if let override = ProcessInfo.processInfo.environment[
            "YOOZ_QWEN3_ASR_DIR"
        ] {
            return URL(fileURLWithPath: override)
        }
        return EngineConfig.modelsDirectory
            .appendingPathComponent("qwen3-asr-1.7b")
    }

    /// True when every required file already exists on disk.
    public func isModelDirReady(_ modelDir: URL) -> Bool {
        for file in Self.requiredFiles {
            let url = modelDir.appendingPathComponent(file)
            if !FileManager.default.fileExists(atPath: url.path) {
                return false
            }
        }
        return true
    }

    /// Fetch the canonical checkpoint into `modelDir`. Yields progress
    /// events on the returned `AsyncThrowingStream`. Caller awaits the
    /// stream to completion before treating the model as loadable.
    ///
    /// - Parameters:
    ///   - modelDir: target directory (created if missing).
    ///   - repo: HF repo id; defaults to `canonicalRepo`.
    ///   - revision: HF git revision; defaults to `main`.
    public func download(
        into modelDir: URL,
        repo: String = canonicalRepo,
        revision: String = "main"
    ) -> AsyncThrowingStream<DownloadProgress, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.runDownload(
                        modelDir: modelDir,
                        repo: repo,
                        revision: revision,
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

        // Tokenizer prep — idempotent.
        continuation.yield(.tokenizerPrepStarted)
        try await Qwen3ASRTokenizerPrep.prepare(modelDir: modelDir)
        continuation.yield(.tokenizerPrepFinished)

        continuation.yield(.done(modelDir: modelDir))
    }

    private func fetchManifest(
        repo: String, revision: String
    ) async throws -> [HFManifestEntry] {
        let path = "/api/models/\(repo)/tree/\(revision)?recursive=true"
        guard let url = URL(string: path, relativeTo: baseURL) else {
            throw Qwen3ASRError.fetchFailed(
                "Invalid manifest URL for repo \(repo)"
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
                "Manifest decode failed: \(error)"
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
           )[.size] as? Int64,
           actual < expected
        {
            offset = actual
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
            throw Qwen3ASRError.fetchValidationFailed(
                "\(entry.path): expected \(expected) bytes, got \(actual)"
            )
        }

        continuation.yield(
            .fileFinished(
                path: entry.path,
                bytes: entry.size ?? 0
            )
        )
    }
}
