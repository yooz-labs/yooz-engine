// STTModelHFDownloaderTests.swift
// YoozEngineTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Unit coverage for the issue #41 HF auto-download surface. The actual
// HubApi.snapshot path is exercised by integration smoke tests (gated on
// network availability + the user's HF cache), not here — these tests
// pin the shape of the API and the offline branches:
//
//   - `unsupportedLanguage` is thrown for families with no mirror.
//   - `isCached(for:)` returns `false` for the unsupported families.
//   - The error description carries enough context to debug a wire
//     failure surfaced as `model_not_found` from `/v1/stt/load`.
//
// We deliberately do not test the cached-true path against a real HF
// cache directory — doing so would either require a network fetch in CI
// or a mock that drifts from `swift-huggingface`'s actual cache layout.

import XCTest
@testable import YoozEngine

final class STTModelHFDownloaderTests: XCTestCase {

    /// Persian, Arabic, Hebrew, and the CJK group all return `nil`
    /// from `huggingFaceID`. The downloader must turn that into the
    /// typed `unsupportedLanguage` error rather than reaching for
    /// `HubApi` with an empty repo id (which would surface as a
    /// confusing 404 rather than a structured error).
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
    /// `/v1/stt/load` (see `APIServer.swift`'s YoozSTTError → 404
    /// mapping). Pinning a sentinel substring catches an accidental
    /// rewrite that would lose the language code or family name —
    /// both are useful for debugging from the consumer side.
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
}
