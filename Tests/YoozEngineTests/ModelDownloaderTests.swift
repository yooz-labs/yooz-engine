// ModelDownloaderTests.swift
// YoozEngineTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import XCTest
import CryptoKit
import Network
@testable import YoozEngine

/// Exercises `ModelDownloader.downloadFile` end-to-end against a localhost
/// fixture server, covering issue #22 (chunked reads replacing the prior
/// byte-by-byte loop). Asserts:
/// - Final on-disk size matches the served content length.
/// - SHA-256 of the downloaded file matches the original.
/// - Progress callbacks fire with monotonically non-decreasing fractions
///   in the [0, 1] range and the final 1.0 sentinel is delivered.
final class ModelDownloaderTests: XCTestCase {

    private var server: LocalHTTPFileServer?

    override func tearDown() {
        server?.stop()
        server = nil
        super.tearDown()
    }

    func testChunkedDownloadProducesByteIdenticalFileWithProgress() async throws {
        // 4 MiB fixture: large enough to cross multiple 1 MiB chunk boundaries
        // and trigger several progress callbacks, but small enough to stay
        // bounded for CI.
        let fixtureSize = 4 * 1024 * 1024
        let fixture = Self.makeFixture(byteCount: fixtureSize)
        let expectedDigest = SHA256.hash(data: fixture)

        let server = try LocalHTTPFileServer(payload: fixture)
        self.server = server
        try server.start()

        let url = server.url(path: "/fixture.bin")

        // Capture progress values in a Sendable, thread-safe collector.
        let collector = ProgressCollector()
        let downloader = ModelDownloader(bundleIdentifier: "live.yooz.engine.tests")

        let resultURL = try await downloader.downloadFile(
            from: url,
            expectedSize: Int64(fixtureSize),
            token: nil,
            progressHandler: { fraction in
                collector.append(fraction)
            }
        )

        defer { try? FileManager.default.removeItem(at: resultURL) }

        // 1. Byte-exact size match.
        let downloadedAttributes = try FileManager.default.attributesOfItem(atPath: resultURL.path)
        let downloadedSize = downloadedAttributes[.size] as? Int ?? -1
        XCTAssertEqual(downloadedSize, fixtureSize, "Downloaded size must equal served content length")

        // 2. SHA-256 integrity.
        let downloaded = try Data(contentsOf: resultURL)
        let downloadedDigest = SHA256.hash(data: downloaded)
        XCTAssertEqual(
            Data(downloadedDigest),
            Data(expectedDigest),
            "SHA-256 of downloaded bytes must match the served fixture"
        )

        // 3. Progress callback semantics:
        //    - At least one callback fired.
        //    - Final value is 1.0.
        //    - Values are in [0, 1] and non-decreasing.
        let progressValues = collector.snapshot()
        XCTAssertFalse(progressValues.isEmpty, "Progress handler must be invoked at least once")
        XCTAssertEqual(progressValues.last, 1.0, "Final progress callback must be 1.0")
        for value in progressValues {
            XCTAssertGreaterThanOrEqual(value, 0.0)
            XCTAssertLessThanOrEqual(value, 1.0)
        }
        for (a, b) in zip(progressValues, progressValues.dropFirst()) {
            XCTAssertLessThanOrEqual(a, b, "Progress must be non-decreasing")
        }
    }

    func testHTTPErrorStatusIsSurfaced() async throws {
        // Server that always responds 404 — confirms HTTP status handling
        // survived the rewrite.
        let server = try LocalHTTPFileServer(payload: Data(), forceStatus: 404)
        self.server = server
        try server.start()

        let url = server.url(path: "/missing.bin")
        let downloader = ModelDownloader(bundleIdentifier: "live.yooz.engine.tests")

        do {
            _ = try await downloader.downloadFile(
                from: url,
                expectedSize: 100,
                token: nil,
                progressHandler: { _ in }
            )
            XCTFail("Expected DownloadError.httpError to be thrown for 404 response")
        } catch DownloadError.httpError(let code) {
            XCTAssertEqual(code, 404)
        }
    }

    // MARK: - Helpers

    private static func makeFixture(byteCount: Int) -> Data {
        // Deterministic, non-trivial payload — repeating LFSR-like sequence
        // so chunk boundaries don't mask off-by-one errors with all-zero data.
        var bytes = [UInt8]()
        bytes.reserveCapacity(byteCount)
        var state: UInt32 = 0xDEAD_BEEF
        for _ in 0..<byteCount {
            state = state &* 1_103_515_245 &+ 12_345
            bytes.append(UInt8(truncatingIfNeeded: state >> 16))
        }
        return Data(bytes)
    }
}

// MARK: - Progress Collector

/// Thread-safe append-only buffer for progress callback values.
private final class ProgressCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [Double] = []

    func append(_ value: Double) {
        lock.lock()
        values.append(value)
        lock.unlock()
    }

    func snapshot() -> [Double] {
        lock.lock()
        defer { lock.unlock() }
        return values
    }
}

// MARK: - Local HTTP fixture server

/// Minimal HTTP/1.1 server used only by tests. Serves a fixed payload (or
/// a fixed status code) for any request. Binds to 127.0.0.1 on an
/// OS-assigned port. Intentionally tiny — no real routing, no keep-alive.
private final class LocalHTTPFileServer: @unchecked Sendable {
    private let listener: NWListener
    private let payload: Data
    private let forceStatus: Int?
    private let queue = DispatchQueue(label: "live.yooz.engine.tests.LocalHTTPFileServer")
    private var connections: [NWConnection] = []
    private let connectionsLock = NSLock()

    init(payload: Data, forceStatus: Int? = nil) throws {
        self.payload = payload
        self.forceStatus = forceStatus
        let parameters = NWParameters.tcp
        parameters.allowLocalEndpointReuse = true
        // Port 0 = OS-assigned ephemeral port.
        self.listener = try NWListener(using: parameters, on: .any)
    }

    func start() throws {
        let started = DispatchSemaphore(value: 0)
        listener.stateUpdateHandler = { [weak started] state in
            switch state {
            case .ready:
                started?.signal()
            case .failed, .cancelled:
                started?.signal()
            default:
                break
            }
        }
        listener.newConnectionHandler = { [weak self] conn in
            self?.handle(connection: conn)
        }
        listener.start(queue: queue)

        if started.wait(timeout: .now() + 5) == .timedOut {
            throw NSError(
                domain: "LocalHTTPFileServer",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Listener failed to reach .ready within 5s"]
            )
        }
        guard listener.state == .ready else {
            throw NSError(
                domain: "LocalHTTPFileServer",
                code: -2,
                userInfo: [NSLocalizedDescriptionKey: "Listener state is \(listener.state)"]
            )
        }
    }

    func stop() {
        listener.cancel()
        connectionsLock.lock()
        let conns = connections
        connections.removeAll()
        connectionsLock.unlock()
        for conn in conns {
            conn.cancel()
        }
    }

    func url(path: String) -> URL {
        guard let port = listener.port?.rawValue else {
            fatalError("Listener has no port; call start() first")
        }
        return URL(string: "http://127.0.0.1:\(port)\(path)")!
    }

    // MARK: - Connection handling

    private func handle(connection: NWConnection) {
        connectionsLock.lock()
        connections.append(connection)
        connectionsLock.unlock()

        connection.start(queue: queue)
        receive(on: connection, accumulated: Data())
    }

    private func receive(on connection: NWConnection, accumulated: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65_536) { [weak self] data, _, isComplete, error in
            guard let self else { return }
            var buffer = accumulated
            if let data = data, !data.isEmpty {
                buffer.append(data)
            }

            // Headers end at \r\n\r\n. Bodies are not used here.
            if let range = buffer.range(of: Data("\r\n\r\n".utf8)) {
                _ = buffer[..<range.lowerBound]
                self.respond(on: connection)
                return
            }

            if isComplete || error != nil {
                connection.cancel()
                return
            }
            self.receive(on: connection, accumulated: buffer)
        }
    }

    private func respond(on connection: NWConnection) {
        let status = forceStatus ?? 200
        let reason: String
        switch status {
        case 200: reason = "OK"
        case 404: reason = "Not Found"
        default:  reason = "Unknown"
        }

        let body = (status == 200) ? payload : Data()
        var head = "HTTP/1.1 \(status) \(reason)\r\n"
        head += "Content-Type: application/octet-stream\r\n"
        head += "Content-Length: \(body.count)\r\n"
        head += "Connection: close\r\n"
        head += "\r\n"

        var responseData = Data(head.utf8)
        responseData.append(body)

        connection.send(content: responseData, completion: .contentProcessed { _ in
            connection.send(content: nil, contentContext: .finalMessage, isComplete: true, completion: .contentProcessed { _ in
                connection.cancel()
            })
        })
    }
}
