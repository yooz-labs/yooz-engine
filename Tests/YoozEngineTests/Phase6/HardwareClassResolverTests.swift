// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import XCTest

@testable import STTModule
@testable import YoozEngine

/// Phase 6 — exhaustive classification tests for `HardwareClass`.
///
/// The classifier is pure (`HardwareClass.classify(brandString:)`)
/// so we drive it directly with a stub resolver rather than relying
/// on `sysctl`. This keeps the suite host-independent.
final class HardwareClassResolverTests: XCTestCase {

    // MARK: - Apple Silicon (the four supported generations)

    func testAppleM1ProClassifiesAsM1() {
        let resolver = StaticHardwareClassResolver(
            brandString: "Apple M1 Pro"
        )
        XCTAssertEqual(resolver.resolve(), .appleSiliconM1)
    }

    func testAppleM1ClassifiesAsM1() {
        XCTAssertEqual(
            StaticHardwareClassResolver(brandString: "Apple M1").resolve(),
            .appleSiliconM1
        )
    }

    func testAppleM2MaxClassifiesAsM2() {
        XCTAssertEqual(
            StaticHardwareClassResolver(brandString: "Apple M2 Max").resolve(),
            .appleSiliconM2
        )
    }

    func testAppleM2UltraClassifiesAsM2() {
        XCTAssertEqual(
            StaticHardwareClassResolver(brandString: "Apple M2 Ultra").resolve(),
            .appleSiliconM2
        )
    }

    func testAppleM3ClassifiesAsM3() {
        XCTAssertEqual(
            StaticHardwareClassResolver(brandString: "Apple M3").resolve(),
            .appleSiliconM3
        )
    }

    func testAppleM3MaxClassifiesAsM3() {
        XCTAssertEqual(
            StaticHardwareClassResolver(brandString: "Apple M3 Max").resolve(),
            .appleSiliconM3
        )
    }

    func testAppleM4ClassifiesAsM4() {
        XCTAssertEqual(
            StaticHardwareClassResolver(brandString: "Apple M4").resolve(),
            .appleSiliconM4
        )
    }

    func testAppleM4ProClassifiesAsM4() {
        XCTAssertEqual(
            StaticHardwareClassResolver(brandString: "Apple M4 Pro").resolve(),
            .appleSiliconM4
        )
    }

    // MARK: - Future Apple Silicon (M5 onward)

    func testFutureAppleSiliconClassifiesAsAppleSiliconUnknown() {
        // M5 doesn't exist yet; the classifier should still bucket
        // it into Apple-Silicon-unknown rather than M4 or unknown.
        XCTAssertEqual(
            StaticHardwareClassResolver(brandString: "Apple M5 Ultra").resolve(),
            .appleSiliconUnknown
        )
    }

    func testGenericAppleBrandClassifiesAsAppleSiliconUnknown() {
        XCTAssertEqual(
            StaticHardwareClassResolver(brandString: "Apple silicon").resolve(),
            .appleSiliconUnknown
        )
    }

    // MARK: - Intel

    func testIntelBrandClassifiesAsIntel() {
        XCTAssertEqual(
            StaticHardwareClassResolver(
                brandString: "Genuine Intel(R) CPU @ 2.30GHz"
            ).resolve(),
            .intel
        )
    }

    func testIntelCoreI7ClassifiesAsIntel() {
        XCTAssertEqual(
            StaticHardwareClassResolver(
                brandString: "Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz"
            ).resolve(),
            .intel
        )
    }

    // MARK: - Unknown / empty

    func testEmptyStringClassifiesAsUnknown() {
        XCTAssertEqual(
            StaticHardwareClassResolver(brandString: "").resolve(),
            .unknown
        )
    }

    func testWhitespaceOnlyClassifiesAsUnknown() {
        XCTAssertEqual(
            StaticHardwareClassResolver(brandString: "   \n\t").resolve(),
            .unknown
        )
    }

    func testGarbageStringClassifiesAsUnknown() {
        XCTAssertEqual(
            StaticHardwareClassResolver(brandString: "RISC-V Holographic 9000").resolve(),
            .unknown
        )
    }

    // MARK: - System resolver (host machine)

    func testSystemResolverReturnsKnownClassOnThisMachine() {
        // The CI host is an Apple Silicon Mac. The resolver should
        // return one of the Apple Silicon variants, not `.unknown`.
        // We do NOT pin to a specific generation — that would couple
        // the test to whichever Mac is running it.
        let cls = SystemHardwareClassResolver().resolve()
        let appleSilicon: Set<HardwareClass> = [
            .appleSiliconM1, .appleSiliconM2,
            .appleSiliconM3, .appleSiliconM4,
            .appleSiliconUnknown,
        ]
        XCTAssertTrue(
            appleSilicon.contains(cls) || cls == .intel,
            "System resolver returned \(cls) on a Mac. Expected an "
                + "Apple Silicon class or .intel."
        )
    }
}
