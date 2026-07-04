// InfiniteSessionStoreTests.swift
// InfiniteModuleTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import XCTest
@testable import InfiniteModule

final class InfiniteSessionStoreTests: XCTestCase {

    private var tempRoot: URL!
    private var store: InfiniteSessionStore!

    override func setUpWithError() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("InfiniteSessionStoreTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        tempRoot = dir
        setenv("YOOZ_INFINITE_SESSIONS_DIR", dir.path, 1)
        store = InfiniteSessionStore()
    }

    override func tearDownWithError() throws {
        unsetenv("YOOZ_INFINITE_SESSIONS_DIR")
        if let tempRoot, FileManager.default.fileExists(atPath: tempRoot.path) {
            try FileManager.default.removeItem(at: tempRoot)
        }
        tempRoot = nil
        store = nil
    }

    // MARK: - Env override

    func testDefaultRootHonorsEnvOverride() {
        XCTAssertEqual(InfiniteSessionStore.defaultRoot.path, tempRoot.path)
        XCTAssertEqual(store.root.path, tempRoot.path)
    }

    // MARK: - Manifest round trip

    func testManifestRoundTrip() throws {
        let manifest = makeManifest(tokenIdsSHA256: "abc123")
        try store.writeManifest(manifest, session: "s1", checkpoint: "c1")
        let read = try store.readManifest(session: "s1", checkpoint: "c1")
        XCTAssertEqual(read, manifest)
    }

    // MARK: - tokens.bin round trip

    func testTokensRoundTripEmpty() throws {
        try store.writeTokens([], session: "s1", checkpoint: "c1")
        let read = try store.readTokens(session: "s1", checkpoint: "c1")
        XCTAssertEqual(read, [])
    }

    func testTokensRoundTripAboveOneHundredThousandIds() throws {
        let tokenIds = (0..<150_000).map { $0 % 32_000 }
        try store.writeTokens(tokenIds, session: "s1", checkpoint: "c1")
        let read = try store.readTokens(session: "s1", checkpoint: "c1")
        XCTAssertEqual(read, tokenIds)
    }

    func testEncodeTokensRejectsNegativeId() {
        XCTAssertThrowsError(try InfiniteSessionStore.encodeTokens([-1])) { error in
            XCTAssertEqual(error as? TokensCodecError, .tokenIdOutOfRange(-1))
        }
    }

    func testDecodeTokensRejectsTruncatedByteCount() {
        XCTAssertThrowsError(try InfiniteSessionStore.decodeTokens(Data([0x01, 0x02, 0x03]))) { error in
            XCTAssertEqual(error as? TokensCodecError, .truncatedTokensFile(byteCount: 3))
        }
    }

    // MARK: - Integrity verification

    func testVerifySucceedsOnMatchingFacts() throws {
        try writeCheckpoint(session: "s1", checkpoint: "c1", tokenIds: [1, 2, 3, 4])
        let manifest = try store.readManifest(session: "s1", checkpoint: "c1")
        try store.verify(
            manifest: manifest,
            against: liveFacts(matching: manifest),
            session: "s1",
            checkpoint: "c1"
        )
    }

    func testVerifyThrowsOnEachManifestFieldMismatch() throws {
        try writeCheckpoint(session: "s1", checkpoint: "c1", tokenIds: [1, 2, 3])
        let manifest = try store.readManifest(session: "s1", checkpoint: "c1")
        let matching = liveFacts(matching: manifest)

        XCTAssertThrowsError(try store.verify(
            manifest: manifest,
            against: LiveSessionFacts(
                schemaVersion: matching.schemaVersion + 1,
                repoId: matching.repoId,
                revision: matching.revision,
                tokenizerHash: matching.tokenizerHash
            ),
            session: "s1",
            checkpoint: "c1"
        )) { error in
            XCTAssertEqual(
                error as? SessionIntegrityError,
                .schemaVersionMismatch(expected: manifest.schemaVersion, actual: manifest.schemaVersion + 1)
            )
        }

        XCTAssertThrowsError(try store.verify(
            manifest: manifest,
            against: LiveSessionFacts(
                schemaVersion: matching.schemaVersion,
                repoId: "other/repo",
                revision: matching.revision,
                tokenizerHash: matching.tokenizerHash
            ),
            session: "s1",
            checkpoint: "c1"
        )) { error in
            XCTAssertEqual(
                error as? SessionIntegrityError,
                .repoIdMismatch(expected: manifest.model.repoId, actual: "other/repo")
            )
        }

        XCTAssertThrowsError(try store.verify(
            manifest: manifest,
            against: LiveSessionFacts(
                schemaVersion: matching.schemaVersion,
                repoId: matching.repoId,
                revision: "other-revision",
                tokenizerHash: matching.tokenizerHash
            ),
            session: "s1",
            checkpoint: "c1"
        )) { error in
            XCTAssertEqual(
                error as? SessionIntegrityError,
                .revisionMismatch(expected: manifest.model.revision, actual: "other-revision")
            )
        }

        XCTAssertThrowsError(try store.verify(
            manifest: manifest,
            against: LiveSessionFacts(
                schemaVersion: matching.schemaVersion,
                repoId: matching.repoId,
                revision: matching.revision,
                tokenizerHash: "other-hash"
            ),
            session: "s1",
            checkpoint: "c1"
        )) { error in
            XCTAssertEqual(
                error as? SessionIntegrityError,
                .tokenizerHashMismatch(expected: manifest.tokenizerHash, actual: "other-hash")
            )
        }
    }

    func testVerifyThrowsTokenIdsSHA256MismatchOnCorruptedTokensFile() throws {
        try writeCheckpoint(session: "s1", checkpoint: "c1", tokenIds: [1, 2, 3])
        let manifest = try store.readManifest(session: "s1", checkpoint: "c1")

        // Corrupt tokens.bin directly on disk, bypassing the store API, to
        // simulate a checkpoint whose token file no longer matches its
        // manifest-recorded digest.
        let tamperedTokens = try InfiniteSessionStore.encodeTokens([9, 9, 9])
        let tokensPath = store.sessionDirectory("s1")
            .appendingPathComponent("c1", isDirectory: true)
            .appendingPathComponent("tokens.bin", isDirectory: false)
        try tamperedTokens.write(to: tokensPath)
        let actualSHA = InfiniteSessionStore.sha256Hex(tamperedTokens)

        XCTAssertThrowsError(try store.verify(
            manifest: manifest,
            against: liveFacts(matching: manifest),
            session: "s1",
            checkpoint: "c1"
        )) { error in
            XCTAssertEqual(
                error as? SessionIntegrityError,
                .tokenIdsSHA256Mismatch(expected: manifest.tokenIdsSHA256, actual: actualSHA)
            )
        }
    }

    // MARK: - Fork

    func testForkProducesIndependentFilesWithParentCheckpointId() throws {
        let originalTokenIds = [10, 20, 30]
        try writeCheckpoint(session: "s1", checkpoint: "root", tokenIds: originalTokenIds)

        try store.fork(
            session: "s1",
            checkpoint: "root",
            intoSession: "s1",
            newCheckpointId: "fork-a",
            label: "branch a"
        )

        let forkedManifest = try store.readManifest(session: "s1", checkpoint: "fork-a")
        XCTAssertEqual(forkedManifest.parentCheckpointId, "root")
        XCTAssertEqual(forkedManifest.label, "branch a")
        XCTAssertEqual(try store.readTokens(session: "s1", checkpoint: "fork-a"), originalTokenIds)

        // Mutating the fork must not perturb the original checkpoint's bytes.
        try store.writeTokens([99, 98, 97], session: "s1", checkpoint: "fork-a")
        XCTAssertEqual(try store.readTokens(session: "s1", checkpoint: "root"), originalTokenIds)
    }

    func testForkIntoADifferentSession() throws {
        let originalTokenIds = [1, 2, 3]
        try writeCheckpoint(session: "s1", checkpoint: "root", tokenIds: originalTokenIds)

        try store.fork(session: "s1", checkpoint: "root", intoSession: "s2", newCheckpointId: "c1", label: nil)

        XCTAssertEqual(try store.readTokens(session: "s2", checkpoint: "c1"), originalTokenIds)
        // The source session is untouched.
        XCTAssertEqual(try store.readTokens(session: "s1", checkpoint: "root"), originalTokenIds)
    }

    func testForkThrowsWhenSourceCheckpointMissing() {
        XCTAssertThrowsError(try store.fork(
            session: "missing",
            checkpoint: "missing",
            intoSession: "s1",
            newCheckpointId: "c1",
            label: nil
        )) { error in
            XCTAssertEqual(
                error as? InfiniteSessionStoreError,
                .checkpointNotFound(session: "missing", checkpoint: "missing")
            )
        }
    }

    func testForkThrowsWhenDestinationCheckpointAlreadyExists() throws {
        try writeCheckpoint(session: "s1", checkpoint: "root", tokenIds: [1])
        try writeCheckpoint(session: "s1", checkpoint: "taken", tokenIds: [2])

        XCTAssertThrowsError(try store.fork(
            session: "s1",
            checkpoint: "root",
            intoSession: "s1",
            newCheckpointId: "taken",
            label: nil
        )) { error in
            XCTAssertEqual(
                error as? InfiniteSessionStoreError,
                .checkpointAlreadyExists(session: "s1", checkpoint: "taken")
            )
        }
    }

    // MARK: - Listing

    func testListCheckpointsOrderingAndLatest() throws {
        let base = Date()
        try writeCheckpoint(
            session: "s1", checkpoint: "c-newest", tokenIds: [1], createdAt: base.addingTimeInterval(20)
        )
        try writeCheckpoint(
            session: "s1", checkpoint: "c-oldest", tokenIds: [2], createdAt: base
        )
        try writeCheckpoint(
            session: "s1", checkpoint: "c-middle", tokenIds: [3], createdAt: base.addingTimeInterval(10)
        )

        let checkpoints = try store.listCheckpoints(session: "s1")
        XCTAssertEqual(checkpoints.map { $0.id }, ["c-oldest", "c-middle", "c-newest"])

        let latest = try store.latestCheckpoint(session: "s1")
        XCTAssertEqual(latest?.id, "c-newest")
    }

    func testListCheckpointsEmptyForUnknownSession() throws {
        XCTAssertTrue(try store.listCheckpoints(session: "does-not-exist").isEmpty)
        XCTAssertNil(try store.latestCheckpoint(session: "does-not-exist"))
    }

    // MARK: - Delete

    func testDeleteSessionCleans() throws {
        try writeCheckpoint(session: "s1", checkpoint: "c1", tokenIds: [1])
        XCTAssertTrue(FileManager.default.fileExists(atPath: store.sessionDirectory("s1").path))

        try store.deleteSession("s1")
        XCTAssertFalse(FileManager.default.fileExists(atPath: store.sessionDirectory("s1").path))

        // Idempotent: deleting an already-deleted session is a no-op, not an error.
        XCTAssertNoThrow(try store.deleteSession("s1"))
    }

    // MARK: - Helpers

    private func makeManifest(
        tokenCount: Int = 3,
        tokenIdsSHA256: String,
        createdAt: Date = Date(),
        parentCheckpointId: String? = nil,
        label: String? = nil
    ) -> InfiniteSessionManifest {
        InfiniteSessionManifest(
            model: ModelIdentity(
                selectionId: "gemma4-e4b-1m",
                repoId: "mlx-community/gemma-4-e4b-it-1m-4bit",
                revision: "b4966f32e71f9f4976a78f74bc8944b1d064bcbf"
            ),
            tokenizerHash: "deadbeefcafe",
            tokenCount: tokenCount,
            tokenIdsSHA256: tokenIdsSHA256,
            cacheConfig: SessionKnobs(kvBits: 4, kvGroupSize: 64, kvScheme: "affine"),
            createdAt: createdAt,
            parentCheckpointId: parentCheckpointId,
            label: label
        )
    }

    private func writeCheckpoint(
        session: String,
        checkpoint: String,
        tokenIds: [Int],
        createdAt: Date = Date(),
        parentCheckpointId: String? = nil,
        label: String? = nil
    ) throws {
        let tokensData = try InfiniteSessionStore.encodeTokens(tokenIds)
        let manifest = makeManifest(
            tokenCount: tokenIds.count,
            tokenIdsSHA256: InfiniteSessionStore.sha256Hex(tokensData),
            createdAt: createdAt,
            parentCheckpointId: parentCheckpointId,
            label: label
        )
        try store.writeTokens(tokenIds, session: session, checkpoint: checkpoint)
        try store.writeManifest(manifest, session: session, checkpoint: checkpoint)
    }

    private func liveFacts(matching manifest: InfiniteSessionManifest) -> LiveSessionFacts {
        LiveSessionFacts(
            schemaVersion: manifest.schemaVersion,
            repoId: manifest.model.repoId,
            revision: manifest.model.revision,
            tokenizerHash: manifest.tokenizerHash
        )
    }
}
