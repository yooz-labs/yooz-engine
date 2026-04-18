// ModelDownloader.swift
// LLMModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore
import Foundation
import os.log

private let logger = Logger(subsystem: "live.yooz.engine", category: "ModelDownloader")

/// Downloads LLM models from GitHub Container Registry (GHCR).
/// Models are stored as OCI artifacts at ghcr.io/yooz-labs/yooz-models/.
/// Falls back to GitHub Releases (tar.gz) if the OCI download fails.
actor ModelDownloader {

    // MARK: - Configuration

    private static let ghcrAPI = "https://ghcr.io/v2"
    private static let owner = "yooz-labs"

    // MARK: - Properties

    private let session: URLSession
    private let fileManager = FileManager.default

    let bundleIdentifier: String

    var cacheDirectory: URL {
        EngineConfig.cacheDirectory.appendingPathComponent("models", isDirectory: true)
    }

    // MARK: - Initialization

    init(bundleIdentifier: String = "live.yooz.engine") {
        self.bundleIdentifier = bundleIdentifier
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 3600
        self.session = URLSession(configuration: config)
    }

    // MARK: - Public API

    func isModelCached(_ modelType: LLMModelType) -> Bool {
        let modelDir = modelDirectory(for: modelType)
        let configFile = modelDir.appendingPathComponent("config.json")
        let weightsFile = modelDir.appendingPathComponent("model.safetensors")
        return fileManager.fileExists(atPath: configFile.path) &&
               fileManager.fileExists(atPath: weightsFile.path)
    }

    func modelDirectory(for modelType: LLMModelType) -> URL {
        return cacheDirectory.appendingPathComponent(modelType.rawValue, isDirectory: true)
    }

    func downloadModel(
        _ modelType: LLMModelType,
        progressHandler: @escaping @Sendable (Double) -> Void
    ) async throws -> URL {
        let modelDir = modelDirectory(for: modelType)

        if isModelCached(modelType) {
            logger.info("Model \(modelType.rawValue) already cached")
            progressHandler(1.0)
            return modelDir
        }

        logger.info("Downloading \(modelType.rawValue)...")

        try fileManager.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)

        let archiveURL = try await downloadArchive(for: modelType, progressHandler: progressHandler)

        try extractArchive(archiveURL, to: modelDir)

        do {
            try fileManager.removeItem(at: archiveURL)
        } catch {
            logger.warning("Failed to remove archive \(archiveURL.lastPathComponent): \(error.localizedDescription)")
        }

        logger.info("Model \(modelType.rawValue) ready at \(modelDir.path)")
        return modelDir
    }

    func deleteModel(_ modelType: LLMModelType) throws {
        let modelDir = modelDirectory(for: modelType)
        if fileManager.fileExists(atPath: modelDir.path) {
            try fileManager.removeItem(at: modelDir)
            logger.info("Deleted \(modelType.rawValue)")
        }
    }

    func cachedModelsSize() -> Int64 {
        var totalSize: Int64 = 0
        for modelType in LLMModelType.allCases {
            if isModelCached(modelType) {
                totalSize += directorySize(modelDirectory(for: modelType))
            }
        }
        return totalSize
    }

    // MARK: - Private Methods

    private func getAnonymousToken(for modelType: LLMModelType) async throws -> String {
        let scope = "repository:\(Self.owner)/\(modelType.packageName)/\(modelType.rawValue):pull"
        guard let tokenURL = URL(string: "https://ghcr.io/token?service=ghcr.io&scope=\(scope)") else {
            throw DownloadError.authFailed
        }

        let (data, response) = try await session.data(from: tokenURL)

        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw DownloadError.authFailed
        }

        struct TokenResponse: Decodable {
            let token: String
        }

        let tokenResponse = try JSONDecoder().decode(TokenResponse.self, from: data)
        return tokenResponse.token
    }

    private func downloadArchive(
        for modelType: LLMModelType,
        progressHandler: @escaping @Sendable (Double) -> Void
    ) async throws -> URL {
        let token = try await getAnonymousToken(for: modelType)

        guard let manifestURL = URL(string: "\(Self.ghcrAPI)/\(Self.owner)/\(modelType.packageName)/\(modelType.rawValue)/manifests/latest") else {
            throw DownloadError.invalidResponse
        }

        var manifestRequest = URLRequest(url: manifestURL)
        manifestRequest.setValue("application/vnd.oci.image.manifest.v1+json", forHTTPHeaderField: "Accept")
        manifestRequest.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")

        let (manifestData, manifestResponse) = try await session.data(for: manifestRequest)

        guard let httpResponse = manifestResponse as? HTTPURLResponse else {
            throw DownloadError.invalidResponse
        }

        if httpResponse.statusCode == 404 {
            return try await downloadFromGitHubReleases(modelType: modelType, progressHandler: progressHandler)
        }

        guard httpResponse.statusCode == 200 else {
            logger.error("Manifest request failed with status \(httpResponse.statusCode)")
            throw DownloadError.httpError(httpResponse.statusCode)
        }

        let manifest = try JSONDecoder().decode(OCIManifest.self, from: manifestData)

        guard let layer = manifest.layers.first else {
            throw DownloadError.noLayersInManifest
        }

        guard let blobURL = URL(string: "\(Self.ghcrAPI)/\(Self.owner)/\(modelType.packageName)/\(modelType.rawValue)/blobs/\(layer.digest)") else {
            throw DownloadError.invalidResponse
        }

        return try await downloadFile(from: blobURL, expectedSize: layer.size, token: token, progressHandler: progressHandler)
    }

    private func downloadFromGitHubReleases(
        modelType: LLMModelType,
        progressHandler: @escaping @Sendable (Double) -> Void
    ) async throws -> URL {
        guard let releaseURL = URL(string: "https://github.com/\(Self.owner)/\(modelType.packageName)/releases/latest/download/\(modelType.rawValue).tar.gz") else {
            throw DownloadError.invalidResponse
        }

        logger.info("Falling back to GitHub Releases...")
        return try await downloadFile(from: releaseURL, expectedSize: modelType.estimatedSize, progressHandler: progressHandler)
    }

    /// Downloads a blob to a temp file using
    /// `URLSession.download(for:delegate:)`. Progress is streamed via a
    /// per-task `URLSessionDownloadDelegate`; the system writes directly
    /// to a temp file — no per-byte Swift loop and no in-memory
    /// accumulation. See `STTModelDownloader.downloadFile` for the same
    /// pattern in the STT module.
    private func downloadFile(
        from url: URL,
        expectedSize: Int64,
        token: String? = nil,
        progressHandler: @escaping @Sendable (Double) -> Void
    ) async throws -> URL {
        let tempFile = cacheDirectory.appendingPathComponent(UUID().uuidString + ".tar.gz")

        try fileManager.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)

        var request = URLRequest(url: url)
        if let token = token {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        let progressDelegate = LLMDownloadProgressDelegate(
            expectedSize: expectedSize,
            progressHandler: progressHandler
        )

        progressHandler(0.0)
        let (tempDownloadURL, response) = try await session.download(
            for: request,
            delegate: progressDelegate
        )

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            let statusCode = (response as? HTTPURLResponse)?.statusCode ?? 0
            logger.error("Download failed with status \(statusCode)")
            try? fileManager.removeItem(at: tempDownloadURL)
            throw DownloadError.httpError(statusCode)
        }

        try? fileManager.removeItem(at: tempFile)
        try fileManager.moveItem(at: tempDownloadURL, to: tempFile)
        progressHandler(1.0)
        return tempFile
    }

    private func extractArchive(_ archiveURL: URL, to destination: URL) throws {
        if fileManager.fileExists(atPath: destination.path) {
            try fileManager.removeItem(at: destination)
        }

        try fileManager.createDirectory(at: destination, withIntermediateDirectories: true)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
        process.arguments = ["-xzf", archiveURL.path, "-C", destination.path, "--strip-components=1"]

        let stderrPipe = Pipe()
        process.standardError = stderrPipe

        try process.run()
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
            let stderr = String(data: stderrData, encoding: .utf8) ?? "unknown error"
            throw DownloadError.extractionFailed("tar exit code \(process.terminationStatus): \(stderr)")
        }
    }

    private func directorySize(_ url: URL) -> Int64 {
        var size: Int64 = 0
        let enumerator = fileManager.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey])
        while let fileURL = enumerator?.nextObject() as? URL {
            if let fileSize = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                size += Int64(fileSize)
            }
        }
        return size
    }
}

// MARK: - URLSession download progress delegate

/// Streams progress from `URLSession.download(for:delegate:)` to a
/// Sendable handler. Mirrors `DownloadProgressDelegate` in the STT
/// module — kept separate so each module owns its own progress
/// translation without cross-module imports.
private final class LLMDownloadProgressDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
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
        // `download(for:delegate:)` returns the temp URL to the caller; no work needed here.
    }
}

// MARK: - OCI Manifest

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

// MARK: - Errors

enum DownloadError: Error, LocalizedError, Sendable {
    case invalidResponse
    case httpError(Int)
    case noLayersInManifest
    case extractionFailed(String)
    case authFailed

    var errorDescription: String? {
        switch self {
        case .invalidResponse:
            return "Invalid response from server"
        case .httpError(let code):
            return "HTTP error: \(code)"
        case .noLayersInManifest:
            return "No layers found in manifest"
        case .extractionFailed(let reason):
            return "Failed to extract model archive: \(reason)"
        case .authFailed:
            return "Failed to authenticate with GHCR"
        }
    }
}
