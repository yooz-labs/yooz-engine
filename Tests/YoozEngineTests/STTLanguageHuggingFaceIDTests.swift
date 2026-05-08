// STTLanguageHuggingFaceIDTests.swift
// YoozEngineTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Pins the `STTLanguage.huggingFaceID` contract for issue #41.
// Renaming the Parakeet repo or accidentally enabling FastConformer
// before the YoozLabs MLX mirror is published would silently break
// `/v1/stt/load`; these tests turn that into a build-time failure.

import XCTest
@testable import YoozEngine

final class STTLanguageHuggingFaceIDTests: XCTestCase {

    /// All Parakeet TDT languages share the multilingual 0.6B v3 mirror.
    /// Pinning the exact repo id catches a typo / fork rename that would
    /// otherwise surface only as a runtime 404 from `/v1/stt/load`.
    func testParakeetLanguagesMapToParakeetTDTV3Mirror() {
        let parakeetLanguages: [STTLanguage] = [
            .english, .spanish, .french, .german, .italian,
            .portuguese, .dutch, .polish, .russian, .ukrainian
        ]
        for language in parakeetLanguages {
            XCTAssertEqual(
                language.huggingFaceID,
                "mlx-community/parakeet-tdt-0.6b-v3",
                "\(language.rawValue) should resolve to the multilingual Parakeet v3 mirror"
            )
        }
    }

    /// FastConformer (Persian / Arabic / Hebrew) and CJK have no MLX
    /// mirror published yet (tracked separately — see #41 body).
    /// Returning `nil` is the explicit signal that
    /// `STTModelHFDownloader.snapshot(for:)` should throw
    /// `unsupportedLanguage` rather than 404 deep inside HubApi.
    func testFastConformerAndCJKReturnNil() {
        let unsupportedLanguages: [STTLanguage] = [
            .arabic, .persian, .hebrew,
            .chinese, .japanese, .korean, .cantonese
        ]
        for language in unsupportedLanguages {
            XCTAssertNil(
                language.huggingFaceID,
                "\(language.rawValue) should not have a HF mirror wired yet"
            )
        }
    }

    /// Total coverage of `STTLanguage.allCases` lives in this single
    /// switch so adding a new case forces the developer to consciously
    /// pick a mirror (or `nil`). If a future case slips through with a
    /// default `nil`, this test still passes — but `huggingFaceID`'s
    /// own `switch modelFamily` is exhaustive on `ModelFamily`, so the
    /// compiler catches new families. This test additionally guards
    /// against silently re-enabling a previously-disabled family by
    /// accident.
    func testEveryLanguageHasDeterministicMapping() {
        for language in STTLanguage.allCases {
            switch language.modelFamily {
            case .parakeetTDT:
                XCTAssertNotNil(language.huggingFaceID, "\(language.rawValue) parakeet mapping missing")
            case .fastConformer, .cjk, .apple:
                XCTAssertNil(language.huggingFaceID, "\(language.rawValue) should not yet have a mirror")
            }
        }
    }
}
