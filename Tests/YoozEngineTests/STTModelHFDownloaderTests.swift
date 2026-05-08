// STTModelHFDownloaderTests.swift
// YoozEngineTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Unit coverage for the HF auto-download surface. The real
// `HubClient.downloadSnapshot` path is exercised by integration smoke
// tests; these tests pin the offline branches and the API shape that
// the route handler relies on for wire-code mapping.

import XCTest
@testable import YoozEngine
import HuggingFace
@testable import STTModule

final class STTModelHFDownloaderTests: XCTestCase {

    /// Languages with no HF mirror (FastConformer Arabic/Persian/
    /// Hebrew, every CJK member) must throw the typed
    /// `unsupportedLanguage` error so `APIServer.mapSTTLoadError` can
    /// produce a `language_unmirrored` 501 instead of letting the
    /// downloader reach for `HubClient` with no repo id.
    func testSnapshotThrowsUnsupportedLanguageWhenNoMirrorWired() async {
        let unsupported: [STTLanguage] = [.persian, .arabic, .hebrew, .chinese]
        for language in unsupported {
            do {
                _ = try await STTModelHFDownloader.snapshot(for: language)
                XCTFail("\(language.rawValue) should have thrown unsupportedLanguage")
            } catch let error as STTHFDownloadError {
                guard case .unsupportedLanguage(let lang) = error else {
                    XCTFail("Expected unsupportedLanguage, got \(error)")
                    continue
                }
                XCTAssertEqual(lang, language)
            } catch {
                XCTFail("Unexpected error type for \(language.rawValue): \(error)")
            }
        }
    }

    /// `isCached` is the cheap predicate the picker UX in yooz-whisper
    /// reads to decide whether to display a "this will download X MB"
    /// hint. For families without a mirror, it must always say `false`
    /// — there is no way to cache a repo whose id does not exist.
    func testIsCachedFalseForUnsupportedLanguages() {
        let unsupported: [STTLanguage] = [.persian, .arabic, .hebrew, .chinese, .japanese]
        for language in unsupported {
            XCTAssertFalse(
                STTModelHFDownloader.isCached(for: language),
                "\(language.rawValue) cannot be cached because it has no HF mirror"
            )
        }
    }

    /// The error description forms part of the wire body returned by
    /// `/v1/stt/load`. Pinning a sentinel substring catches an
    /// accidental rewrite that would lose the language code or family
    /// name — both are useful for debugging from the consumer side.
    func testUnsupportedLanguageErrorMessageContainsContext() {
        let error = STTHFDownloadError.unsupportedLanguage(.persian)
        let message = error.errorDescription ?? ""
        XCTAssertTrue(
            message.contains("fa"),
            "Expected error to mention language code 'fa'; got: \(message)"
        )
        XCTAssertTrue(
            message.contains("fast-conformer"),
            "Expected error to mention model family 'fast-conformer'; got: \(message)"
        )
    }

    /// `repoID(for:)` is the single source of truth for namespace/name
    /// splitting; both `snapshot(for:)` and `isCached(for:)` go through
    /// it. A drift between them (the original code split inline in two
    /// places) silently broke the cache lookup. This test pins the
    /// happy path so a future inlining doesn't regress.
    func testRepoIDForParakeetParsesNamespaceAndName() throws {
        let repo = try XCTUnwrap(STTModelHFDownloader.repoID(for: .english))
        XCTAssertEqual(repo.namespace, "mlx-community")
        XCTAssertEqual(repo.name, "parakeet-tdt-0.6b-v3")
    }

    /// Symmetric: families without a mirror return nil so callers can
    /// short-circuit before any FileManager / HubClient work.
    func testRepoIDForUnmirroredFamilyIsNil() {
        XCTAssertNil(STTModelHFDownloader.repoID(for: .persian))
        XCTAssertNil(STTModelHFDownloader.repoID(for: .chinese))
    }
}
