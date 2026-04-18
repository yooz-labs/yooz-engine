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
/// is the tag/artifact within that package. `legacyWhisperPath` is a concrete
/// absolute path on disk that the old yooz-whisper app wrote to, used as a
/// transient fallback while GHCR artifacts are being published.
public struct STTModelDescriptor: Sendable, Equatable {
    public let identifier: String
    public let ghcrPackage: String
    public let ghcrArtifact: String
    public let estimatedSize: Int64
    public let legacyWhisperPath: URL?

    public init(
        identifier: String,
        ghcrPackage: String,
        ghcrArtifact: String,
        estimatedSize: Int64,
        legacyWhisperPath: URL? = nil
    ) {
        self.identifier = identifier
        self.ghcrPackage = ghcrPackage
        self.ghcrArtifact = ghcrArtifact
        self.estimatedSize = estimatedSize
        self.legacyWhisperPath = legacyWhisperPath
    }
}

extension STTModelDescriptor {
    /// The canonical Parakeet TDT 0.6B weights used for English + European
    /// languages. Other model families (FastConformer, CJK) will get their
    /// own descriptors when they wire in; keeping this table small until the
    /// GHCR publish story is confirmed avoids faking URLs that don't exist.
    public static let parakeetTDT06B: STTModelDescriptor = {
        let legacy = FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask)
            .first?
            .appendingPathComponent("live.yooz.whisper/Models/parakeet-tdt-0.6b-en")
        return STTModelDescriptor(
            identifier: "parakeet-tdt",
            ghcrPackage: "yooz-models",
            ghcrArtifact: "stt-parakeet-tdt-0.6b",
            estimatedSize: 700 * 1024 * 1024, // ~700 MB packed
            legacyWhisperPath: legacy
        )
    }()
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
    /// `.npz`). Requiring a weights file prevents a torn download (where
    /// the config lands but the network drops before the blob finishes)
    /// from being treated as a hit on the next launch — Parakeet would
    /// otherwise raise a cryptic runtime error deep inside MLX.
    public nonisolated func isCached(_ descriptor: STTModelDescriptor) -> Bool {
        let dir = modelDirectory(for: descriptor)
        guard fileManager.fileExists(atPath: dir.appendingPathComponent("config.json").path) else {
            return false
        }
        guard let enumerator = fileManager.enumerator(
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

        // 1. Legacy yooz-whisper location — copy/symlink without network.
        if let legacy = descriptor.legacyWhisperPath,
           fileManager.fileExists(atPath: legacy.appendingPathComponent("config.json").path) {
            logger.info("Importing STT model from legacy whisper path: \(legacy.path, privacy: .public)")
            try adoptLegacyModel(from: legacy, to: modelDir)
            progressHandler(1.0)
            return modelDir
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
            throw STTDownloadError.archiveMissingConfig
        }

        logger.info("STT model ready at \(modelDir.path, privacy: .public)")
        return modelDir
    }

    // MARK: - Legacy adoption

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
        if http.statusCode == 404 {
            return try await downloadFromGitHubReleases(
                descriptor: descriptor,
                progressHandler: progressHandler
            )
        }
        guard http.statusCode == 200 else {
            throw STTDownloadError.httpError(http.statusCode)
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

        let (asyncBytes, response) = try await session.bytes(for: request)
        guard let http = response as? HTTPURLResponse,
              (200...299).contains(http.statusCode) else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            throw STTDownloadError.httpError(code)
        }

        let totalSize = http.expectedContentLength > 0 ? http.expectedContentLength : expectedSize

        try? fileManager.removeItem(at: tempFile)
        fileManager.createFile(atPath: tempFile.path, contents: nil)
        let outputHandle = try FileHandle(forWritingTo: tempFile)

        var success = false
        defer {
            if !success {
                try? outputHandle.close()
                try? fileManager.removeItem(at: tempFile)
            }
        }

        var bytesReceived: Int64 = 0
        var lastReported = 0
        var buffer = Data()
        let bufferSize = 65536

        progressHandler(0.0)
        for try await byte in asyncBytes {
            buffer.append(byte)
            bytesReceived += 1
            if buffer.count >= bufferSize {
                try outputHandle.write(contentsOf: buffer)
                buffer.removeAll(keepingCapacity: true)
                let pct = Int((Double(bytesReceived) / Double(max(totalSize, 1))) * 100)
                if pct >= lastReported + 5 {
                    lastReported = pct
                    progressHandler(Double(bytesReceived) / Double(max(totalSize, 1)))
                }
            }
        }
        if !buffer.isEmpty {
            try outputHandle.write(contentsOf: buffer)
        }
        try outputHandle.close()
        success = true
        progressHandler(1.0)
        return tempFile
    }

    private func extractArchive(_ archive: URL, to destination: URL) throws {
        if fileManager.fileExists(atPath: destination.path) {
            try fileManager.removeItem(at: destination)
        }
        try fileManager.createDirectory(at: destination, withIntermediateDirectories: true)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
        process.arguments = ["-xzf", archive.path, "-C", destination.path, "--strip-components=1"]
        let stderr = Pipe()
        process.standardError = stderr
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            let data = stderr.fileHandleForReading.readDataToEndOfFile()
            let msg = String(data: data, encoding: .utf8) ?? "unknown error"
            throw STTDownloadError.extractionFailed("tar exit \(process.terminationStatus): \(msg)")
        }
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
