// STTModuleTests.swift
// STTModuleTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// These tests exercise real STTModule code: the domain enums, public API
// surface, AIModule conformance, and invariants that survive without a
// loaded model. No mocks, per yooz project policy. Tests that would
// require loading a 600MB+ Parakeet or FastConformer checkpoint are
// skipped with XCTSkipUnless and can be un-skipped by setting
// YOOZ_STT_LOAD_MODELS=1 on a machine that has the weights available;
// mirrors the VAD + LLM gating patterns.

import XCTest
import EngineCore
@testable import STTModule

final class STTModuleTests: XCTestCase {

    /// Gate model-dependent tests behind an env var so CI never tries to
    /// load 600MB+ of STT weights. Mirrors `YOOZ_LLM_LOAD_MODELS` and
    /// the Silero mlpackage gate in VADModuleTests.
    private var shouldLoadRealModels: Bool {
        ProcessInfo.processInfo.environment["YOOZ_STT_LOAD_MODELS"] == "1"
    }

    // MARK: - AIModule conformance (always runs)

    func testAIModuleName() {
        XCTAssertEqual(YoozSTTEngine.name, "stt")
    }

    func testAIModuleIsReadyMirrorsIsRunning() async {
        // YoozSTTEngine.isReady is a stored @Published flag set true only
        // after a successful start(). Before any load, both must report
        // the default-false state; after load, both flip together.
        let engine = YoozSTTEngine.shared
        let ready = await engine.isReady
        XCTAssertEqual(ready, engine.isRunning,
                       "isReady must reflect the model-loaded state exactly")
    }

    func testHealthCheckReportsLoadedFlag() async {
        let engine = YoozSTTEngine.shared
        let health = await engine.healthCheck()
        XCTAssertEqual(health.loaded, engine.isRunning,
                       "healthCheck().loaded must match engine.isRunning")
    }

    func testHealthCheckDetailKeysPresent() async {
        let health = await YoozSTTEngine.shared.healthCheck()
        let expected: Set<String> = [
            "language",
            "display_name",
            "model_identifier",
            "model_family",
            "streaming"
        ]
        XCTAssertEqual(Set(health.detail.keys), expected,
                       "healthCheck detail must report current-language descriptors")
    }

    func testHealthCheckWhenNotLoaded() async throws {
        let engine = YoozSTTEngine.shared
        try XCTSkipIf(engine.isRunning,
                      "shared engine already loaded; cannot assert not-loaded detail")

        let health = await engine.healthCheck()
        XCTAssertFalse(health.loaded)
        XCTAssertNotNil(health.error, "unloaded engine should surface an error message")
        XCTAssertEqual(health.detail["streaming"], "false")
    }

    // MARK: - STTLanguage (always runs)

    func testAllLanguagesHaveDisplayNames() {
        // APIServer's /v1/stt/languages response reads displayName for every
        // case. An empty string would render a broken UI.
        for lang in STTLanguage.allCases {
            XCTAssertFalse(lang.displayName.isEmpty,
                           "\(lang.rawValue) has empty displayName")
        }
    }

    func testAllLanguagesHaveModelIdentifiers() {
        for lang in STTLanguage.allCases {
            XCTAssertFalse(lang.modelIdentifier.isEmpty,
                           "\(lang.rawValue) has empty modelIdentifier")
        }
    }

    func testFromCodeRoundTrip() {
        // APIServer.parseLanguage uses fromCode to turn wire strings into
        // the domain enum. The case-insensitive upper-case path must also
        // resolve; clients occasionally send "EN" or "FA".
        XCTAssertEqual(STTLanguage.fromCode("en"), .english)
        XCTAssertEqual(STTLanguage.fromCode("EN"), .english,
                       "fromCode should be case-insensitive")
        XCTAssertEqual(STTLanguage.fromCode("fa"), .persian)
        XCTAssertEqual(STTLanguage.fromCode("ar"), .arabic)
        XCTAssertNil(STTLanguage.fromCode("zz-unknown"),
                     "unknown codes must return nil, not a default")
    }

    func testRawValueStabilityForKnownCodes() {
        // Wire protocol: raw values are part of the public /v1/stt/* API.
        // Any rename breaks clients.
        XCTAssertEqual(STTLanguage.english.rawValue, "en")
        XCTAssertEqual(STTLanguage.persian.rawValue, "fa")
        XCTAssertEqual(STTLanguage.arabic.rawValue, "ar")
    }

    func testImplementedFlagsMatchA1Contract() {
        // Per the module design doc + STTLanguage.isImplemented, today the
        // implemented set is: every Parakeet TDT language (English/European)
        // plus Arabic + Persian via FastConformer. Hebrew + CJK are not
        // implemented yet.
        XCTAssertTrue(STTLanguage.english.isImplemented)
        XCTAssertTrue(STTLanguage.arabic.isImplemented)
        XCTAssertTrue(STTLanguage.persian.isImplemented)
        XCTAssertFalse(STTLanguage.hebrew.isImplemented,
                       "Hebrew currently unimplemented; flipping this flag requires model weights")
        XCTAssertFalse(STTLanguage.chinese.isImplemented)
        XCTAssertFalse(STTLanguage.japanese.isImplemented)
        XCTAssertFalse(STTLanguage.korean.isImplemented)
    }

    func testImplementedLanguagesSetMatchesAllCasesFilter() {
        // STTLanguage.implemented is surface-level UI convenience; ensure
        // it actually equals allCases.filter(\.isImplemented).
        let computed = STTLanguage.allCases.filter(\.isImplemented)
        XCTAssertEqual(Set(STTLanguage.implemented), Set(computed))
    }

    func testModelFamilyRouting() {
        XCTAssertEqual(STTLanguage.english.modelFamily, .parakeetTDT)
        XCTAssertEqual(STTLanguage.persian.modelFamily, .fastConformer)
        XCTAssertEqual(STTLanguage.arabic.modelFamily, .fastConformer)
        XCTAssertEqual(STTLanguage.chinese.modelFamily, .cjk)
    }

    // MARK: - ParakeetResult (always runs)

    func testParakeetResultEmptyDefault() {
        XCTAssertEqual(ParakeetResult.empty.text, "")
        XCTAssertEqual(ParakeetResult.empty.finalized, "")
        XCTAssertEqual(ParakeetResult.empty.draft, "")
        XCTAssertTrue(ParakeetResult.empty.isEmpty)
    }

    func testParakeetResultPublicInit() {
        // APIServer's WebSocket handler constructs WSSTTResult from a
        // ParakeetResult's public fields; the type must stay publicly
        // constructible so tests and future consumers can shape one.
        let result = ParakeetResult(text: "hello world", finalized: "hello", draft: "world")
        XCTAssertEqual(result.text, "hello world")
        XCTAssertEqual(result.finalized, "hello")
        XCTAssertEqual(result.draft, "world")
        XCTAssertFalse(result.isEmpty)
    }

    func testParakeetResultIsEmptyIgnoresWhitespace() {
        // The streaming + batch paths both trim whitespace before publishing
        // `text`; make sure isEmpty treats whitespace-only as empty.
        let result = ParakeetResult(text: "   ", finalized: "", draft: "")
        XCTAssertTrue(result.isEmpty)
    }

    // MARK: - YoozSTTError (always runs)

    func testYoozSTTErrorDescriptions() {
        let errors: [YoozSTTError] = [
            .modelNotFound("/tmp/missing"),
            .modelLoadFailed("bad weights"),
            .notReady,
            .streamError("underrun"),
            .languageNotSupported(.hebrew)
        ]
        for err in errors {
            XCTAssertNotNil(err.errorDescription, "missing description for \(err)")
            XCTAssertFalse(err.errorDescription!.isEmpty, "empty description for \(err)")
        }
    }

    // MARK: - YoozSTTEngine public API shape (always runs)

    func testSharedSingletonExists() {
        // The .shared accessor must be public; if access regresses to
        // internal this test won't compile.
        let engine = YoozSTTEngine.shared
        _ = engine.isRunning
    }

    func testDefaultStateBeforeLoad() throws {
        let engine = YoozSTTEngine.shared
        try XCTSkipIf(engine.isRunning,
                      "shared engine already loaded; cannot assert default state")

        XCTAssertFalse(engine.isRunning)
        XCTAssertFalse(engine.isReady)
        XCTAssertFalse(engine.isStreaming)
        XCTAssertEqual(engine.currentLanguage, .english,
                       "default currentLanguage should be .english")
    }

    func testCreateBatchTranscriberReturnsNilBeforeLoad() throws {
        // APIServer's WebSocket config branch calls createBatchTranscriber
        // after a successful start(); before load it must return nil, not
        // trap or throw. This locks that graceful-degrade contract.
        let engine = YoozSTTEngine.shared
        try XCTSkipIf(engine.isRunning,
                      "shared engine already loaded; cannot test nil-before-load path")

        XCTAssertNil(engine.createBatchTranscriber(mode: .normal))
    }

    func testBatchTranscribeAlignedThrowsWhenModelNotLoaded() async throws {
        // Regression guard for I2 in .context/pr_review_engine_block.md:
        // the aligned entry point used to return an empty
        // TranscriptionResult(tokens: []) when no model was loaded, which
        // APIServer turned into a 200 OK with empty text/tokens —
        // indistinguishable from silent input. It must now throw
        // YoozSTTError.notReady so the server can return 503 stt_not_loaded.
        let engine = YoozSTTEngine.shared
        try XCTSkipIf(engine.isRunning,
                      "shared engine already loaded; cannot assert not-ready throw path")

        let samples = [Float](repeating: 0, count: 1600)
        do {
            _ = try await engine.batchTranscribeAligned(samples: samples, mode: .normal)
            XCTFail("batchTranscribeAligned must throw when no model is loaded")
        } catch YoozSTTError.notReady {
            // expected
        } catch {
            XCTFail("expected YoozSTTError.notReady; got \(error)")
        }
    }

    func testAvailableLanguagesMatchesImplemented() {
        XCTAssertEqual(
            Set(YoozSTTEngine.shared.availableLanguages),
            Set(STTLanguage.implemented),
            "availableLanguages should mirror STTLanguage.implemented"
        )
    }

    // MARK: - AudioMode (always runs)

    func testAudioModeRawValues() {
        // Raw values are on the wire via /v1/stt/batch `mode` and the
        // WebSocket config message; changing them is a breaking change.
        XCTAssertEqual(AudioMode.normal.rawValue, "normal")
        XCTAssertNotNil(AudioMode(rawValue: "normal"))
    }

    // MARK: - Model-dependent tests (gated by YOOZ_STT_LOAD_MODELS)

    func testStartLoadsEnglishModel() async throws {
        try XCTSkipUnless(shouldLoadRealModels,
                          "Set YOOZ_STT_LOAD_MODELS=1 to exercise the 600MB+ Parakeet load path")

        let engine = YoozSTTEngine.shared
        try await engine.start(language: .english)
        XCTAssertTrue(engine.isRunning)
        XCTAssertEqual(engine.currentLanguage, .english)

        let health = await engine.healthCheck()
        XCTAssertTrue(health.loaded)
        XCTAssertEqual(health.detail["language"], "en")
    }

    // MARK: - Aligned transcription (engine#34, gated by YOOZ_STT_LOAD_MODELS)

    func testBatchTranscribeAlignedReturnsMonotonicTokens() async throws {
        try XCTSkipUnless(shouldLoadRealModels,
                          "Set YOOZ_STT_LOAD_MODELS=1 to exercise the aligned transcription path")

        // Two seconds of silence isn't a useful transcription input, so we
        // build a deterministic sine sweep that the Parakeet preprocessor
        // will at least chunk through the encoder. We don't assert on the
        // text (which would be flaky); we assert the alignment invariants
        // that whisper's chunk-boundary dedup relies on in #154.
        let engine = YoozSTTEngine.shared
        try await engine.start(language: .english)

        let sampleRate = 16_000
        let duration: Float = 2.0
        var samples = [Float](repeating: 0, count: Int(Float(sampleRate) * duration))
        for i in 0..<samples.count {
            let t = Float(i) / Float(sampleRate)
            samples[i] = 0.1 * sinf(2 * .pi * 440 * t)
        }

        let aligned = try await engine.batchTranscribeAligned(samples: samples, mode: .normal)

        // All token timestamps must be within the audio window.
        for token in aligned.tokens {
            XCTAssertGreaterThanOrEqual(token.start, 0)
            XCTAssertLessThanOrEqual(token.end, duration + 1.0,
                                     "token \(token.text) end \(token.end) exceeds audio window")
            XCTAssertGreaterThanOrEqual(token.duration, 0)
        }

        // Monotonicity: whisper's ChunkProcessor relies on token starts being
        // non-decreasing so `.filter { $0.end > ctx + eps }` preserves order.
        let starts = aligned.tokens.map(\.start)
        XCTAssertEqual(starts, starts.sorted(),
                       "AlignedToken.start must be monotonically non-decreasing")
    }

    // MARK: - Legacy whisper path resolution (always runs, tmp fixture)

    /// Build a tmp Application Support fixture with a whisper variant that
    /// holds a minimal "model" (config.json + empty .safetensors).
    /// Returns the root so tests can point `resolveLegacyPaths` at it
    /// without touching the real user ~/Library.
    private func makeLegacyFixture(
        variants: [String],
        slug: String
    ) throws -> URL {
        let root = FileManager.default
            .temporaryDirectory
            .appendingPathComponent("yooz-stt-legacy-\(UUID().uuidString)", isDirectory: true)
        for variant in variants {
            let dir = root
                .appendingPathComponent(variant, isDirectory: true)
                .appendingPathComponent(slug, isDirectory: true)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            try Data("{}".utf8).write(to: dir.appendingPathComponent("config.json"))
            try Data().write(to: dir.appendingPathComponent("model.safetensors"))
        }
        return root
    }

    func testResolveLegacyPathsFindsDevVariant() throws {
        // Regression for the dev whisper build: fresh machine, only the
        // `.dev` variant is installed, the engine must still adopt the
        // model from Application Support rather than attempting a
        // 700MB GHCR download.
        let descriptor = STTModelDescriptor.parakeetTDT06B
        let slug = try XCTUnwrap(descriptor.legacyWhisperSlug)
        let root = try makeLegacyFixture(
            variants: ["live.yooz.whisper.dev"],
            slug: slug
        )
        defer { try? FileManager.default.removeItem(at: root) }

        let found = descriptor.resolveLegacyPaths(appSupportRoot: root)
        let expected = root
            .appendingPathComponent("live.yooz.whisper.dev")
            .appendingPathComponent(slug)
        XCTAssertTrue(
            found.contains { $0.standardizedFileURL.path == expected.standardizedFileURL.path },
            "dev variant path must be in the resolved candidate list; got \(found.map(\.path))"
        )
    }

    func testResolveLegacyPathsOrdersStableDevBeta() throws {
        // All three fixed variants present. Priority order is
        // stable → dev → beta because if a stable build is installed
        // we assume it's the user's "real" model; the dev/beta entries
        // are fallbacks for machines that never had the stable build.
        let descriptor = STTModelDescriptor.parakeetTDT06B
        let slug = try XCTUnwrap(descriptor.legacyWhisperSlug)
        let root = try makeLegacyFixture(
            variants: [
                "live.yooz.whisper",
                "live.yooz.whisper.dev",
                "live.yooz.whisper.beta"
            ],
            slug: slug
        )
        defer { try? FileManager.default.removeItem(at: root) }

        let found = descriptor.resolveLegacyPaths(appSupportRoot: root)
        // Substring match is slug-independent: any change to the slug
        // (single- vs multi-component) keeps the test meaningful.
        let topThree = found.prefix(3).map(\.path)
        XCTAssertEqual(topThree.count, 3)
        XCTAssertTrue(topThree[0].contains("/live.yooz.whisper/"),
                      "first fixed candidate must be stable whisper; got \(topThree[0])")
        XCTAssertTrue(topThree[1].contains("/live.yooz.whisper.dev/"),
                      "second fixed candidate must be dev whisper; got \(topThree[1])")
        XCTAssertTrue(topThree[2].contains("/live.yooz.whisper.beta/"),
                      "third fixed candidate must be beta whisper; got \(topThree[2])")
    }

    func testResolveLegacyPathsPicksUpUnknownLiveYoozVariant() throws {
        // Future variant discovery: a `live.yooz.whisper.internal` build
        // should be found via the wildcard scan without a code change.
        let descriptor = STTModelDescriptor.parakeetTDT06B
        let slug = try XCTUnwrap(descriptor.legacyWhisperSlug)
        let root = try makeLegacyFixture(
            variants: ["live.yooz.whisper.internal"],
            slug: slug
        )
        defer { try? FileManager.default.removeItem(at: root) }

        let found = descriptor.resolveLegacyPaths(appSupportRoot: root)
        let expected = root
            .appendingPathComponent("live.yooz.whisper.internal")
            .appendingPathComponent(slug)
        XCTAssertTrue(
            found.contains { $0.standardizedFileURL.path == expected.standardizedFileURL.path },
            "wildcard scan must surface future live.yooz.* variants; got \(found.map(\.path))"
        )
    }

    func testResolveLegacyPathsSkipsEngineOwnDir() throws {
        // Defensive: the engine's own app-support dir must never be
        // treated as a legacy whisper candidate, even if someone
        // accidentally creates a `Models/parakeet-tdt-0.6b-en` tree
        // under `live.yooz.engine` (which would otherwise be picked up
        // by the `live.yooz.*` wildcard scan).
        let descriptor = STTModelDescriptor.parakeetTDT06B
        let slug = try XCTUnwrap(descriptor.legacyWhisperSlug)
        let root = try makeLegacyFixture(
            variants: ["live.yooz.engine"],
            slug: slug
        )
        defer { try? FileManager.default.removeItem(at: root) }

        let found = descriptor.resolveLegacyPaths(appSupportRoot: root)
        for url in found {
            XCTAssertFalse(
                url.path.contains("live.yooz.engine"),
                "resolver must never return the engine's own app-support dir; got \(url.path)"
            )
        }
    }

    func testResolveLegacyPathsEmptyWhenSlugNil() {
        // A descriptor without a legacy slug must return an empty list,
        // not the three fixed URLs with a trailing nil path component.
        let descriptor = STTModelDescriptor(
            identifier: "no-legacy",
            ghcrPackage: "yooz-models",
            ghcrArtifact: "no-legacy",
            estimatedSize: 0,
            legacyWhisperSlug: nil
        )
        let root = FileManager.default.temporaryDirectory
        XCTAssertTrue(descriptor.resolveLegacyPaths(appSupportRoot: root).isEmpty)
    }

    func testLegacyCandidateIsCompleteRequiresConfigAndWeights() throws {
        // Config without weights should not be adopted (torn download).
        let root = FileManager.default
            .temporaryDirectory
            .appendingPathComponent("yooz-stt-torn-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data("{}".utf8).write(to: root.appendingPathComponent("config.json"))
        XCTAssertFalse(
            STTModelDownloader.legacyCandidateIsComplete(root),
            "config-only directory must not be treated as a valid legacy candidate"
        )

        try Data().write(to: root.appendingPathComponent("model.safetensors"))
        XCTAssertTrue(
            STTModelDownloader.legacyCandidateIsComplete(root),
            "config + .safetensors must count as a valid legacy candidate"
        )
    }
}
