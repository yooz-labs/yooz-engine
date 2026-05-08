// STTModelDownloader.swift
// STTModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore
import Foundation
import os.log

private let logger = Logger(subsystem: "live.yooz.engine", category: "STTModelDownloader")

// MARK: - STT Model Descriptor

/// Identifies an STT weights bundle that the engine can obtain on demand.
///
/// The identifier is the on-disk directory name under
/// `EngineConfig.modelsDirectory` (which is what `YoozSTTEngine.getModelDirectory`
/// already searches). `ghcrPackage` is the GHCR OCI package name; `ghcrArtifact`
/// is the tag/artifact within that package.
///
/// `legacyWhisperSlug` is the per-model subpath (e.g.
/// `Models/parakeet-tdt-0.6b-en`) that stable/dev/beta whisper variants
/// wrote under their Application Support directories. It's paired with a
/// per-descriptor resolver that scans every `live.yooz.*` app support dir
/// for that slug, so a dev whisper install (`live.yooz.whisper.dev`) gets
/// picked up automatically. See `STTModelDescriptor.resolveLegacyPaths`.
public struct STTModelDescriptor: Sendable, Equatable {
    public let identifier: String
    public let ghcrPackage: String
    public let ghcrArtifact: String
    public let estimatedSize: Int64
    public let legacyWhisperSlug: String?

    public init(
        identifier: String,
        ghcrPackage: String,
        ghcrArtifact: String,
        estimatedSize: Int64,
        legacyWhisperSlug: String? = nil
    ) {
        self.identifier = identifier
        self.ghcrPackage = ghcrPackage
        self.ghcrArtifact = ghcrArtifact
        self.estimatedSize = estimatedSize
        self.legacyWhisperSlug = legacyWhisperSlug
    }
}

extension STTModelDescriptor {
    /// The canonical Parakeet TDT 0.6B weights used for English + European
    /// languages. Other model families (FastConformer, CJK) will get their
    /// own descriptors when they wire in; keeping this table small until the
    /// GHCR publish story is confirmed avoids faking URLs that don't exist.
    ///
    /// NOTE: GHCR and GitHub Releases artifacts for this package are not yet
    /// published (tracked as engine#41). Until that ships, resolution on a
    /// fresh machine relies on the legacy whisper path resolver; the
    /// GHCR/Releases paths exist so the code is ready the moment the publish
    /// pipeline lands.
    public static let parakeetTDT06B: STTModelDescriptor = {
        STTModelDescriptor(
            identifier: "parakeet-tdt",
            ghcrPackage: "yooz-models",
            ghcrArtifact: "stt-parakeet-tdt-0.6b",
            estimatedSize: 700 * 1024 * 1024, // ~700 MB packed
            legacyWhisperSlug: "Models/parakeet-tdt-0.6b-en"
        )
    }()

    /// All candidate legacy whisper directories for this descriptor, in
    /// priority order. Pure; takes the Application Support root as a
    /// parameter so it can be unit-tested against a tmp fixture without
    /// touching the user's real `~/Library/Application Support`.
    ///
    /// Search scope (broadened in the v0.6.0 epic after a dev whisper
    /// build came up on a fresh machine and the engine couldn't find the
    /// `.dev` Application Support copy):
    ///   1. `live.yooz.whisper/<slug>`           — stable whisper
    ///   2. `live.yooz.whisper.dev/<slug>`       — dev whisper
    ///   3. `live.yooz.whisper.beta/<slug>`      — beta whisper
    ///   4. Any `live.yooz.*` directory with the slug under it — catches
    ///      future variants (e.g. `live.yooz.whisper.internal`) without
    ///      needing a code change.
    ///
    /// Entries are deduplicated while preserving first-seen order so an
    /// explicit `.dev` entry from the fixed list isn't shadowed by the
    /// wildcard scan.
    ///
    /// Returns `[]` when `legacyWhisperSlug` is nil, meaning this model
    /// has no legacy fallback.
    public func resolveLegacyPaths(appSupportRoot: URL) -> [URL] {
        guard let slug = legacyWhisperSlug else { return [] }

        var seen: Set<String> = []
        var out: [URL] = []

        func addIfNew(_ url: URL) {
            let key = url.standardizedFileURL.path
            if seen.insert(key).inserted {
                out.append(url)
            }
        }

        // Fixed, high-priority candidates.
        for suffix in ["live.yooz.whisper",
                       "live.yooz.whisper.dev",
                       "live.yooz.whisper.beta"] {
            addIfNew(
                appSupportRoot
                    .appendingPathComponent(suffix, isDirectory: true)
                    .appendingPathComponent(slug)
            )
        }

        // Wildcard: enumerate every `live.yooz.*` app support dir that
        // holds this slug. Non-throwing; we just skip if the root dir is
        // unreadable (fresh account, enumerator failure, sandbox).
        if let entries = try? FileManager.default.contentsOfDirectory(
            at: appSupportRoot,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) {
            for entry in entries {
                let name = entry.lastPathComponent
                guard name.hasPrefix("live.yooz.") else { continue }
                // Defensive: skip the engine's own dir so we never try to
                // "import" models from ourselves.
                if name == "YoozEngine" || name.hasPrefix("live.yooz.engine") {
                    continue
                }
                let candidate = entry.appendingPathComponent(slug)
                addIfNew(candidate)
            }
        }

        return out
    }
}

// MARK: - STT Model Downloader

/// Resolves STT model weights to an on-disk location inside
/// `EngineConfig.modelsDirectory`.
///
/// Resolution order on a cache miss:
///   1. Copy from the legacy yooz-whisper `Application Support` path if it
///      exists. Keeps the first live integration run working before GHCR
///      artifacts are published.
///   2. Download the OCI manifest from GHCR and stream the blob to a tarball,
///      then extract. Mirrors the LLM `ModelDownloader` flow.
///   3. Fall through to GitHub Releases `tar.gz` if the OCI manifest 404s.
///
/// Errors propagate as `STTDownloadError` so `YoozSTTEngine.start` can
/// surface a precise code to the API server instead of a generic
/// `load_failed`.
public actor STTModelDownloader {

    public static let shared = STTModelDownloader()

    private static let ghcrAPI = "https://ghcr.io/v2"
    private static let owner = "yooz-labs"

    private let session: URLSession
    private let fileManager = FileManager.default

    public init() {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 3600
        self.session = URLSession(configuration: config)
    }

    // MARK: - Public API

    /// Directory where the model ends up after successful resolution. The
    /// STT engine expects this directory to contain `config.json` at its
    /// root.
    public nonisolated func modelDirectory(for descriptor: STTModelDescriptor) -> URL {
        EngineConfig.modelsDirectory
            .appendingPathComponent(descriptor.identifier, isDirectory: true)
    }

    /// A model is considered cached only when its directory contains both
    /// `config.json` **and** at least one weights file (`.safetensors` or
    /// `.npz`) at depth 1. Requiring a weights file prevents a torn
    /// download (where the config lands but the network drops before the
    /// blob finishes) from being treated as a hit on the next launch —
    /// Parakeet would otherwise raise a cryptic runtime error deep inside
    /// MLX.
    ///
    /// This check is on the hot path of every model load, so it does a
    /// single `fileExists` for `config.json` + a single non-recursive
    /// `contentsOfDirectory` scan for weights. No full tree walk.
    public nonisolated func isCached(_ descriptor: STTModelDescriptor) -> Bool {
        let dir = modelDirectory(for: descriptor)
        guard fileManager.fileExists(atPath: dir.appendingPathComponent("config.json").path) else {
            return false
        }
        guard let entries = try? fileManager.contentsOfDirectory(atPath: dir.path) else {
            return false
        }
        for entry in entries {
            let ext = (entry as NSString).pathExtension.lowercased()
            if ext == "safetensors" || ext == "npz" {
                return true
            }
        }
        return false
    }

    /// Ensure the model is on disk and return its directory. No-op when
    /// cached. Tries the legacy whisper path first, then GHCR OCI, then
    /// GitHub Releases.
    public func ensureAvailable(
        _ descriptor: STTModelDescriptor,
        progressHandler: @escaping @Sendable (Double) -> Void = { _ in }
    ) async throws -> URL {
        let modelDir = modelDirectory(for: descriptor)

        if isCached(descriptor) {
            logger.info("STT model \(descriptor.identifier, privacy: .public) already cached")
            progressHandler(1.0)
            return modelDir
        }

        try fileManager.createDirectory(
            at: EngineConfig.modelsDirectory,
            withIntermediateDirectories: true
        )

        // 1. Legacy yooz-whisper location(s) — copy (not symlink) without
        //    network. v0.6.0 broadens this from a single
        //    `live.yooz.whisper` path to also include `.dev`, `.beta`,
        //    and any `live.yooz.*` app-support dir that holds the slug.
        //    See `STTModelDescriptor.resolveLegacyPaths`.
        let appSupport = fileManager
            .urls(for: .applicationSupportDirectory, in: .userDomainMask)
            .first
        if let appSupport = appSupport {
            let candidates = descriptor.resolveLegacyPaths(appSupportRoot: appSupport)
            for legacy in candidates {
                guard Self.legacyCandidateIsComplete(legacy) else { continue }
                logger.info(
                    "Importing STT model from legacy whisper path: \(legacy.path, privacy: .public)"
                )
                try adoptLegacyModel(from: legacy, to: modelDir)
                progressHandler(1.0)
                return modelDir
            }
        }

        // 2/3. Network path — OCI with GitHub Releases fallback.
        logger.info("Downloading STT model \(descriptor.identifier, privacy: .public) from GHCR")
        let archive = try await downloadArchive(
            for: descriptor,
            progressHandler: progressHandler
        )
        defer { try? fileManager.removeItem(at: archive) }
        try extractArchive(archive, to: modelDir)

        guard isCached(descriptor) else {
            // Leaving a half-extracted modelDir on disk would make the next
            // launch look like "cached" to anything less strict than
            // isCached, and would waste the next download attempt's
            // chance to succeed. Log what we found before removing so ops
            // can correlate with the failing archive.
            let contents = (try? fileManager.contentsOfDirectory(atPath: modelDir.path)) ?? []
            let contentsSummary = contents.joined(separator: ",")
            logger.error("STT archive extracted but verification failed for \(descriptor.identifier, privacy: .public); contents=[\(contentsSummary, privacy: .public)]; cleaning up \(modelDir.path, privacy: .public)")
            try? fileManager.removeItem(at: modelDir)
            throw STTDownloadError.archiveMissingConfig
        }

        logger.info("STT model ready at \(modelDir.path, privacy: .public)")
        return modelDir
    }

    // MARK: - Legacy adoption

    /// Verify a legacy candidate directory holds a usable model: a
    /// `config.json` at the root plus at least one weights file
    /// (`.safetensors` or `.npz`) anywhere in the tree. Mirrors the
    /// `isCached` contract so a torn legacy download (config landed,
    /// weights didn't) isn't adopted and then misdiagnosed as a cache
    /// hit on the next launch.
    static func legacyCandidateIsComplete(_ dir: URL) -> Bool {
        let fm = FileManager.default
        guard fm.fileExists(atPath: dir.appendingPathComponent("config.json").path) else {
            return false
        }
        guard let enumerator = fm.enumerator(
            at: dir,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) else {
            return false
        }
        for case let url as URL in enumerator {
            let ext = url.pathExtension.lowercased()
            if ext == "safetensors" || ext == "npz" {
                return true
            }
        }
        return false
    }

    /// Copy the weights from the old yooz-whisper location into the engine's
    /// models directory. We copy rather than symlink so that if the host
    /// user uninstalls yooz-whisper the engine keeps working.
    private func adoptLegacyModel(from source: URL, to destination: URL) throws {
        if fileManager.fileExists(atPath: destination.path) {
            try fileManager.removeItem(at: destination)
        }
        try fileManager.copyItem(at: source, to: destination)
    }

    // MARK: - GHCR / GitHub Releases

    private func getAnonymousToken(for descriptor: STTModelDescriptor) async throws -> String {
        let scope = "repository:\(Self.owner)/\(descriptor.ghcrPackage)/\(descriptor.ghcrArtifact):pull"
        guard let tokenURL = URL(string: "https://ghcr.io/token?service=ghcr.io&scope=\(scope)") else {
            throw STTDownloadError.invalidURL
        }
        let (data, response) = try await session.data(from: tokenURL)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw STTDownloadError.authFailed
        }
        struct TokenResponse: Decodable { let token: String }
        return try JSONDecoder().decode(TokenResponse.self, from: data).token
    }

    private func downloadArchive(
        for descriptor: STTModelDescriptor,
        progressHandler: @escaping @Sendable (Double) -> Void
    ) async throws -> URL {
        // Any failure whose root cause is "GHCR manifest unreachable"
        // (DNS, TLS, timeout, 404, 5xx, transient token-endpoint failure)
        // should fall through to the GitHub Releases tarball. Only
        // 401/403 short-circuits — those mean package visibility or
        // scope is wrong, which is an auth bug the Releases path won't
        // fix.
        do {
            let token = try await getAnonymousToken(for: descriptor)

            guard let manifestURL = URL(
                string: "\(Self.ghcrAPI)/\(Self.owner)/\(descriptor.ghcrPackage)/\(descriptor.ghcrArtifact)/manifests/latest"
            ) else {
                throw STTDownloadError.invalidURL
            }

            var manifestRequest = URLRequest(url: manifestURL)
            manifestRequest.setValue(
                "application/vnd.oci.image.manifest.v1+json",
                forHTTPHeaderField: "Accept"
            )
            manifestRequest.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")

            let (manifestData, manifestResponse) = try await session.data(for: manifestRequest)
            guard let http = manifestResponse as? HTTPURLResponse else {
                throw STTDownloadError.invalidResponse
            }
            if http.statusCode == 401 || http.statusCode == 403 {
                throw STTDownloadError.httpError(http.statusCode)
            }
            if !(200...299).contains(http.statusCode) {
                logger.warning(
                    "GHCR manifest HTTP \(http.statusCode, privacy: .public) for \(descriptor.identifier, privacy: .public); falling back to Releases"
                )
                return try await downloadFromGitHubReleases(
                    descriptor: descriptor,
                    progressHandler: progressHandler
                )
            }

            let manifest = try JSONDecoder().decode(OCIManifest.self, from: manifestData)
            guard let layer = manifest.layers.first else {
                throw STTDownloadError.noLayersInManifest
            }
            guard let blobURL = URL(
                string: "\(Self.ghcrAPI)/\(Self.owner)/\(descriptor.ghcrPackage)/\(descriptor.ghcrArtifact)/blobs/\(layer.digest)"
            ) else {
                throw STTDownloadError.invalidURL
            }
            return try await downloadFile(
                from: blobURL,
                expectedSize: layer.size,
                token: token,
                progressHandler: progressHandler
            )
        } catch STTDownloadError.httpError(let code) where code == 401 || code == 403 {
            // Auth bug — let it surface.
            throw STTDownloadError.httpError(code)
        } catch {
            logger.warning(
                "GHCR manifest resolution failed for \(descriptor.identifier, privacy: .public) (\(error.localizedDescription, privacy: .public)); falling back to Releases"
            )
            return try await downloadFromGitHubReleases(
                descriptor: descriptor,
                progressHandler: progressHandler
            )
        }
    }

    private func downloadFromGitHubReleases(
        descriptor: STTModelDescriptor,
        progressHandler: @escaping @Sendable (Double) -> Void
    ) async throws -> URL {
        guard let releaseURL = URL(
            string: "https://github.com/\(Self.owner)/\(descriptor.ghcrPackage)/releases/latest/download/\(descriptor.ghcrArtifact).tar.gz"
        ) else {
            throw STTDownloadError.invalidURL
        }
        logger.info("Falling back to GitHub Releases for STT model \(descriptor.identifier, privacy: .public)")
        return try await downloadFile(
            from: releaseURL,
            expectedSize: descriptor.estimatedSize,
            progressHandler: progressHandler
        )
    }

    /// Downloads a blob to a temp file using
    /// `URLSession.download(for:delegate:)`. Progress is streamed via a
    /// per-task `URLSessionDownloadDelegate` which the system fills from
    /// a dedicated I/O thread — no per-byte Swift loop, no in-memory
    /// accumulation, no O(n) `Data.append` for a 700 MB blob.
    ///
    /// The 5%-threshold progress contract is preserved for callers.
    private func downloadFile(
        from url: URL,
        expectedSize: Int64,
        token: String? = nil,
        progressHandler: @escaping @Sendable (Double) -> Void
    ) async throws -> URL {
        let tempFile = EngineConfig.cacheDirectory
            .appendingPathComponent("stt-\(UUID().uuidString).tar.gz")
        try fileManager.createDirectory(
            at: EngineConfig.cacheDirectory,
            withIntermediateDirectories: true
        )

        var request = URLRequest(url: url)
        if let token = token {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        let progressDelegate = DownloadProgressDelegate(
            expectedSize: expectedSize,
            progressHandler: progressHandler
        )

        progressHandler(0.0)
        // `URLSession.download(for:delegate:)` (macOS 12+) accepts a
        // per-task delegate for progress without needing a separate
        // session instance. The returned URL is a system temp file we
        // must move before the call site returns.
        let (tempDownloadURL, response) = try await session.download(
            for: request,
            delegate: progressDelegate
        )

        guard let http = response as? HTTPURLResponse,
              (200...299).contains(http.statusCode) else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            try? fileManager.removeItem(at: tempDownloadURL)
            throw STTDownloadError.httpError(code)
        }

        try? fileManager.removeItem(at: tempFile)
        try fileManager.moveItem(at: tempDownloadURL, to: tempFile)
        progressHandler(1.0)
        return tempFile
    }

    /// Extract a tarball into `destination`. The engine's packaging
    /// contract is that the archive contains exactly one top-level
    /// directory whose contents are the model tree; we verify this
    /// explicitly rather than trusting a blind `--strip-components=1`.
    ///
    /// On a contract violation, the staging directory is preserved so
    /// ops can inspect what the publisher actually produced; on
    /// success, the staging directory is cleaned up.
    private func extractArchive(_ archive: URL, to destination: URL) throws {
        if fileManager.fileExists(atPath: destination.path) {
            try fileManager.removeItem(at: destination)
        }

        let stagingDir = EngineConfig.cacheDirectory
            .appendingPathComponent("stt-stage-\(UUID().uuidString)", isDirectory: true)
        try fileManager.createDirectory(at: stagingDir, withIntermediateDirectories: true)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
        process.arguments = ["-xzf", archive.path, "-C", stagingDir.path]
        let stderr = Pipe()
        process.standardError = stderr
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            let data = stderr.fileHandleForReading.readDataToEndOfFile()
            let msg = String(data: data, encoding: .utf8) ?? "unknown error"
            // tar failed outright — staging is empty / garbage; safe to clean.
            try? fileManager.removeItem(at: stagingDir)
            throw STTDownloadError.extractionFailed("tar exit \(process.terminationStatus): \(msg)")
        }

        // Enforce the single-top-level-directory contract.
        let entries = try fileManager.contentsOfDirectory(atPath: stagingDir.path)
            .filter { !$0.hasPrefix(".") }
        guard entries.count == 1 else {
            // Preserve staging dir for debugging.
            throw STTDownloadError.extractionFailed(
                "archive layout violation: expected 1 top-level entry, found \(entries.count): \(entries); preserved at \(stagingDir.path)"
            )
        }
        let topLevel = stagingDir.appendingPathComponent(entries[0])
        var isDir: ObjCBool = false
        guard fileManager.fileExists(atPath: topLevel.path, isDirectory: &isDir), isDir.boolValue else {
            throw STTDownloadError.extractionFailed(
                "archive layout violation: top-level entry \(entries[0]) is not a directory; preserved at \(stagingDir.path)"
            )
        }

        // Move the single top-level directory into place as `destination`.
        try fileManager.moveItem(at: topLevel, to: destination)
        try? fileManager.removeItem(at: stagingDir)
    }
}

// MARK: - URLSession download progress delegate

/// Streams progress from `URLSession.download(for:)` to a Sendable
/// handler. Kept non-actor and thread-safe via a serial lock so
/// `URLSession`'s I/O-thread callbacks can drive the 5%-threshold
/// contract without an actor hop per chunk.
private final class DownloadProgressDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let expectedSize: Int64
    private let progressHandler: @Sendable (Double) -> Void
    private let lock = NSLock()
    private var lastReportedPercent: Int = 0

    init(expectedSize: Int64, progressHandler: @escaping @Sendable (Double) -> Void) {
        self.expectedSize = expectedSize
        self.progressHandler = progressHandler
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        let total = totalBytesExpectedToWrite > 0 ? totalBytesExpectedToWrite : expectedSize
        guard total > 0 else { return }
        let ratio = Double(totalBytesWritten) / Double(total)
        let pct = Int(ratio * 100)
        lock.lock()
        let shouldReport = pct >= lastReportedPercent + 5
        if shouldReport { lastReportedPercent = pct }
        lock.unlock()
        if shouldReport {
            progressHandler(min(max(ratio, 0.0), 1.0))
        }
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        // The caller receives the temp URL via `download(for:)`. No work here.
    }
}

// MARK: - Errors

public enum STTDownloadError: Error, LocalizedError, Sendable {
    case invalidURL
    case invalidResponse
    case httpError(Int)
    case noLayersInManifest
    case authFailed
    case extractionFailed(String)
    case archiveMissingConfig

    public var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid download URL"
        case .invalidResponse:
            return "Invalid HTTP response from model registry"
        case .httpError(let code):
            return "Model registry returned HTTP \(code)"
        case .noLayersInManifest:
            return "OCI manifest contained no layers"
        case .authFailed:
            return "Failed to acquire anonymous GHCR token"
        case .extractionFailed(let msg):
            return "Failed to extract STT model archive: \(msg)"
        case .archiveMissingConfig:
            return "Downloaded STT archive did not contain config.json at its root"
        }
    }
}

// MARK: - OCI Manifest (private mirror of LLM module's type)

private struct OCIManifest: Decodable {
    let schemaVersion: Int
    let mediaType: String?
    let layers: [OCILayer]
}

private struct OCILayer: Decodable {
    let mediaType: String
    let digest: String
    let size: Int64
}
