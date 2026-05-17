// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import XCTest

@testable import STTModule
@testable import YoozEngine

/// Phase 6 — display helpers on `STTBackendID`. Engine-side source of
/// truth for downstream UIs (Whisper, Notes).
final class STTBackendIDDisplayTests: XCTestCase {

    // MARK: - displayName

    func testDisplayNameForEachBackend() {
        XCTAssertEqual(
            STTBackendID.parakeet.displayName, "Parakeet (Recommended)"
        )
        XCTAssertEqual(
            STTBackendID.fastConformer.displayName,
            "FastConformer (Arabic / Persian / Hebrew)"
        )
        XCTAssertEqual(
            STTBackendID.appleSTT.displayName, "Apple Speech (On-device)"
        )
        XCTAssertEqual(
            STTBackendID.qwen3ASRPreview.displayName,
            "Multilingual (Preview)"
        )
    }

    func testDisplayNameIsExhaustive() {
        // Adding a new STTBackendID without a displayName mapping
        // crashes any switch that doesn't cover the new case;
        // iterate to exercise every case.
        for backend in STTBackendID.allCases {
            let name = backend.displayName
            XCTAssertFalse(
                name.isEmpty,
                "displayName empty for \(backend.rawValue)"
            )
        }
    }

    // MARK: - isPreview

    func testOnlyQwen3IsPreview() {
        for backend in STTBackendID.allCases {
            switch backend {
            case .qwen3ASRPreview:
                XCTAssertTrue(backend.isPreview)
            case .parakeet, .fastConformer, .appleSTT:
                XCTAssertFalse(backend.isPreview)
            }
        }
    }

    // MARK: - estimatedDownloadMB

    func testEstimatedDownloadMBOnlyForPreview() {
        XCTAssertNil(STTBackendID.parakeet.estimatedDownloadMB)
        XCTAssertNil(STTBackendID.fastConformer.estimatedDownloadMB)
        XCTAssertNil(STTBackendID.appleSTT.estimatedDownloadMB)
        XCTAssertEqual(
            STTBackendID.qwen3ASRPreview.estimatedDownloadMB, 3_500
        )
    }

    func testEstimatedDownloadMBIsPreviewIff() {
        for backend in STTBackendID.allCases {
            XCTAssertEqual(
                backend.estimatedDownloadMB != nil, backend.isPreview,
                "Mismatch between estimatedDownloadMB and isPreview "
                    + "for \(backend.rawValue)"
            )
        }
    }

    // MARK: - defaultModelVariant tag is engine-controlled

    func testDefaultModelVariantTagsAreNonEmpty() {
        for backend in STTBackendID.allCases {
            let tag = STTBackendMetrics.defaultModelVariant(for: backend)
            XCTAssertFalse(
                tag.isEmpty,
                "Empty modelVariant tag for \(backend.rawValue)"
            )
            // Tags must be ASCII identifiers — no user content.
            XCTAssertNil(
                tag.rangeOfCharacter(from: .whitespacesAndNewlines),
                "modelVariant tag '\(tag)' contains whitespace; tags "
                    + "must be ASCII identifiers."
            )
        }
    }
}
