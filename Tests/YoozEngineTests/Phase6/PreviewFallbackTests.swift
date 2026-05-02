// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import XCTest

@testable import YoozEngine

/// Phase 6 — `PreviewFallbackHook` cold-start auto-fallback contract.
///
/// Cold-start failure modes that MUST trigger a Parakeet fallback:
///
/// 1. fetcher.ensureModelOnDisk() throws
/// 2. preview.ensureLoaded() throws
/// 3. preview.transcribe() throws on the very first call
///
/// Once the preview backend has succeeded once (`hasSucceededOnce`
/// flips to `true`), a subsequent transcribe-throw must propagate as
/// an empty result — NOT trigger another fallback.
///
/// Each fallback path records exactly one metric with
/// `fellBackFromPreview = true`.
final class PreviewFallbackTests: XCTestCase {

    // MARK: - Test doubles

    /// Spy fetcher: optionally throws on `ensureModelOnDisk()`.
    actor SpyFetcher: PreviewModelFetcherAdapter {
        let shouldThrow: Bool
        var calls: Int = 0

        init(shouldThrow: Bool) { self.shouldThrow = shouldThrow }

        func ensureModelOnDisk() async throws {
            calls += 1
            if shouldThrow {
                throw NSError(
                    domain: "test", code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "fetch boom"]
                )
            }
        }

        func callCount() -> Int { calls }
    }

    /// Spy preview backend with three knobs:
    ///   - throwOnLoad: ensureLoaded() throws
    ///   - throwOnFirstTranscribe: first transcribe call throws
    ///   - throwOnSubsequentTranscribe: second+ transcribe call throws
    ///
    /// Returns a fixed `ParakeetResult` on success.
    actor SpyPreviewBackend: PreviewBackendAdapter {
        var loadCalls: Int = 0
        var transcribeCalls: Int = 0
        let throwOnLoad: Bool
        let throwOnFirstTranscribe: Bool
        let throwOnSubsequentTranscribe: Bool

        init(
            throwOnLoad: Bool = false,
            throwOnFirstTranscribe: Bool = false,
            throwOnSubsequentTranscribe: Bool = false
        ) {
            self.throwOnLoad = throwOnLoad
            self.throwOnFirstTranscribe = throwOnFirstTranscribe
            self.throwOnSubsequentTranscribe = throwOnSubsequentTranscribe
        }

        func ensureLoaded() async throws {
            loadCalls += 1
            if throwOnLoad {
                throw NSError(
                    domain: "test", code: 2,
                    userInfo: [NSLocalizedDescriptionKey: "load boom"]
                )
            }
        }

        func transcribe(
            samples: [Float], language: STTLanguage
        ) async throws -> ParakeetResult {
            transcribeCalls += 1
            if transcribeCalls == 1 && throwOnFirstTranscribe {
                throw NSError(
                    domain: "test", code: 3,
                    userInfo: [NSLocalizedDescriptionKey: "transcribe boom"]
                )
            }
            if transcribeCalls > 1 && throwOnSubsequentTranscribe {
                throw NSError(
                    domain: "test", code: 4,
                    userInfo: [
                        NSLocalizedDescriptionKey:
                            "post-warmup transcribe boom"
                    ]
                )
            }
            return ParakeetResult(
                text: "preview hello",
                finalized: "preview hello",
                draft: ""
            )
        }

        func snapshot() -> (load: Int, transcribe: Int) {
            (loadCalls, transcribeCalls)
        }
    }

    /// Stub fallback backend always returns a fixed Parakeet result.
    struct StubFallback: FallbackBackendAdapter {
        let text: String
        func transcribe(
            samples: [Float], language: STTLanguage
        ) async -> ParakeetResult {
            ParakeetResult(text: text, finalized: text, draft: "")
        }
    }

    /// In-memory metric collector used in place of the JSONL sink.
    actor InMemoryMetricsSink: STTMetricsSink {
        private(set) var records: [STTBackendMetrics] = []
        func record(_ metric: STTBackendMetrics) async {
            records.append(metric)
        }
        func snapshot() -> [STTBackendMetrics] { records }
    }

    /// Deterministic clock that increments by `step` ms each call.
    final class FakeClock: PreviewFallbackClock, @unchecked Sendable {
        private var t: UInt64 = 0
        private let step: UInt64
        init(step: UInt64 = 10) { self.step = step }
        func nowMs() -> UInt64 { defer { t += step }; return t }
    }

    // MARK: - Helpers

    private func makeHook(
        fetcher: SpyFetcher,
        preview: SpyPreviewBackend,
        fallback: FallbackBackendAdapter,
        sink: any STTMetricsSink,
        resolver: HardwareClassResolver = StaticHardwareClassResolver(
            brandString: "Apple M3"
        )
    ) -> PreviewFallbackHook {
        PreviewFallbackHook(
            fetcher: fetcher,
            preview: preview,
            fallback: fallback,
            sink: sink,
            resolver: resolver,
            clock: FakeClock()
        )
    }

    // MARK: - Cold-start failure: fetch fails

    func testFallbackFiresWhenFetchThrows() async {
        let fetcher = SpyFetcher(shouldThrow: true)
        let preview = SpyPreviewBackend()
        let fallback = StubFallback(text: "parakeet hello")
        let sink = InMemoryMetricsSink()
        let hook = makeHook(
            fetcher: fetcher,
            preview: preview,
            fallback: fallback,
            sink: sink
        )

        let outcome = await hook.attemptPreviewWithFallback(
            samples: [Float](repeating: 0.0, count: 16_000),
            language: .english
        )

        XCTAssertTrue(outcome.fellBack)
        XCTAssertEqual(outcome.backendUsed, .parakeet)
        XCTAssertEqual(outcome.result.text, "parakeet hello")

        let preview_calls = await preview.snapshot()
        XCTAssertEqual(preview_calls.load, 0)
        XCTAssertEqual(preview_calls.transcribe, 0)

        let metrics = await sink.snapshot()
        XCTAssertEqual(metrics.count, 1)
        XCTAssertEqual(metrics[0].backend, .parakeet)
        XCTAssertTrue(metrics[0].fellBackFromPreview)
    }

    // MARK: - Cold-start failure: load fails

    func testFallbackFiresWhenLoadThrows() async {
        let fetcher = SpyFetcher(shouldThrow: false)
        let preview = SpyPreviewBackend(throwOnLoad: true)
        let fallback = StubFallback(text: "parakeet load fallback")
        let sink = InMemoryMetricsSink()
        let hook = makeHook(
            fetcher: fetcher,
            preview: preview,
            fallback: fallback,
            sink: sink
        )

        let outcome = await hook.attemptPreviewWithFallback(
            samples: [Float](repeating: 0.0, count: 16_000),
            language: .english
        )

        XCTAssertTrue(outcome.fellBack)
        XCTAssertEqual(outcome.backendUsed, .parakeet)
        XCTAssertEqual(outcome.result.text, "parakeet load fallback")

        let metrics = await sink.snapshot()
        XCTAssertEqual(metrics.count, 1)
        XCTAssertTrue(metrics[0].fellBackFromPreview)
    }

    // MARK: - Cold-start failure: first transcribe throws

    func testFallbackFiresWhenFirstTranscribeThrows() async {
        let fetcher = SpyFetcher(shouldThrow: false)
        let preview = SpyPreviewBackend(
            throwOnLoad: false, throwOnFirstTranscribe: true
        )
        let fallback = StubFallback(text: "parakeet first-transcribe fallback")
        let sink = InMemoryMetricsSink()
        let hook = makeHook(
            fetcher: fetcher,
            preview: preview,
            fallback: fallback,
            sink: sink
        )

        let outcome = await hook.attemptPreviewWithFallback(
            samples: [Float](repeating: 0.0, count: 16_000),
            language: .english
        )

        XCTAssertTrue(outcome.fellBack)
        XCTAssertEqual(outcome.backendUsed, .parakeet)
        XCTAssertEqual(
            outcome.result.text, "parakeet first-transcribe fallback"
        )
    }

    // MARK: - Happy path: preview succeeds

    func testPreviewSucceedsWithoutFallback() async {
        let fetcher = SpyFetcher(shouldThrow: false)
        let preview = SpyPreviewBackend()
        let fallback = StubFallback(text: "should not be used")
        let sink = InMemoryMetricsSink()
        let hook = makeHook(
            fetcher: fetcher,
            preview: preview,
            fallback: fallback,
            sink: sink
        )

        let outcome = await hook.attemptPreviewWithFallback(
            samples: [Float](repeating: 0.0, count: 16_000),
            language: .english
        )

        XCTAssertFalse(outcome.fellBack)
        XCTAssertEqual(outcome.backendUsed, .qwen3ASRPreview)
        XCTAssertEqual(outcome.result.text, "preview hello")

        let metrics = await sink.snapshot()
        XCTAssertEqual(metrics.count, 1)
        XCTAssertEqual(metrics[0].backend, .qwen3ASRPreview)
        XCTAssertFalse(metrics[0].fellBackFromPreview)
    }

    // MARK: - Post-warmup: failure does NOT trigger fallback

    func testPostWarmupTranscribeFailureDoesNotFallback() async {
        let fetcher = SpyFetcher(shouldThrow: false)
        let preview = SpyPreviewBackend(
            throwOnSubsequentTranscribe: true
        )
        let fallback = StubFallback(text: "should not be used")
        let sink = InMemoryMetricsSink()
        let hook = makeHook(
            fetcher: fetcher,
            preview: preview,
            fallback: fallback,
            sink: sink
        )

        // Call 1: succeeds, flips hasSucceededOnce.
        let first = await hook.attemptPreviewWithFallback(
            samples: [Float](repeating: 0.0, count: 16_000),
            language: .english
        )
        XCTAssertFalse(first.fellBack)
        XCTAssertEqual(first.backendUsed, .qwen3ASRPreview)

        // Call 2: throws. MUST propagate as empty result, not as
        // a Parakeet fallback. The fallback adapter must not be
        // touched.
        let second = await hook.attemptPreviewWithFallback(
            samples: [Float](repeating: 0.0, count: 16_000),
            language: .english
        )
        XCTAssertFalse(
            second.fellBack,
            "Post-warmup failure must NOT trigger fallback."
        )
        XCTAssertEqual(second.backendUsed, .qwen3ASRPreview)
        XCTAssertTrue(second.result.isEmpty)

        let metrics = await sink.snapshot()
        XCTAssertEqual(metrics.count, 2)
        XCTAssertFalse(metrics[0].fellBackFromPreview)
        XCTAssertFalse(metrics[1].fellBackFromPreview)
    }

    // MARK: - Cold-start state machine

    func testColdStartCompletesAfterFallback() async {
        let fetcher = SpyFetcher(shouldThrow: true)
        let preview = SpyPreviewBackend()
        let fallback = StubFallback(text: "fallback")
        let sink = InMemoryMetricsSink()
        let hook = makeHook(
            fetcher: fetcher,
            preview: preview,
            fallback: fallback,
            sink: sink
        )

        _ = await hook.attemptPreviewWithFallback(
            samples: [Float](repeating: 0.0, count: 16_000),
            language: .english
        )
        let completed = await hook.coldStartCompletedForTesting()
        XCTAssertTrue(
            completed,
            "Cold-start state must flip to completed after a "
                + "fallback so the hook stops thrashing the cold-start "
                + "path on every subsequent request when the "
                + "underlying issue is sticky. The user's preview "
                + "selection is honored at next process start."
        )
    }

    func testNoFurtherFallbackAfterColdStartCompleted() async {
        // First request: fetcher throws, hook falls back, flips
        // coldStartCompleted = true. Second request goes to the
        // post-warmup branch — preview transcribe failure must NOT
        // re-trigger fallback.
        let fetcher = SpyFetcher(shouldThrow: true)
        let preview = SpyPreviewBackend(throwOnSubsequentTranscribe: true)
        let fallback = StubFallback(text: "fallback")
        let sink = InMemoryMetricsSink()
        let hook = makeHook(
            fetcher: fetcher,
            preview: preview,
            fallback: fallback,
            sink: sink
        )

        // First request: fetcher throws, falls back.
        _ = await hook.attemptPreviewWithFallback(
            samples: [Float](repeating: 0.0, count: 16_000),
            language: .english
        )

        // Second request: post-warmup branch. Preview's transcribe
        // throws; hook propagates with empty result, no fallback.
        let outcome = await hook.attemptPreviewWithFallback(
            samples: [Float](repeating: 0.0, count: 16_000),
            language: .english
        )
        XCTAssertFalse(
            outcome.fellBack,
            "Post-warmup transcribe failure must NOT trigger fallback."
        )
        XCTAssertEqual(outcome.backendUsed, .qwen3ASRPreview)
    }

    // MARK: - Audio duration calculation

    func testAudioDurationMsZeroSampleRateReturnsZero() async {
        let fetcher = SpyFetcher(shouldThrow: false)
        let preview = SpyPreviewBackend()
        let fallback = StubFallback(text: "n/a")
        let sink = InMemoryMetricsSink()
        let hook = makeHook(
            fetcher: fetcher,
            preview: preview,
            fallback: fallback,
            sink: sink
        )

        // Defensive guard: zero sample rate would otherwise divide
        // by zero. The hook returns 0 ms instead of trapping.
        _ = await hook.attemptPreviewWithFallback(
            samples: [Float](repeating: 0.0, count: 16_000),
            language: .english,
            sampleRate: 0
        )

        let metrics = await sink.snapshot()
        XCTAssertEqual(metrics.count, 1)
        XCTAssertEqual(metrics[0].audioDurationMs, 0)
    }

    func testAudioDurationMsReflectsSampleCount() async {
        let fetcher = SpyFetcher(shouldThrow: false)
        let preview = SpyPreviewBackend()
        let fallback = StubFallback(text: "n/a")
        let sink = InMemoryMetricsSink()
        let hook = makeHook(
            fetcher: fetcher,
            preview: preview,
            fallback: fallback,
            sink: sink
        )

        // 32_000 samples at 16 kHz = 2 000 ms.
        _ = await hook.attemptPreviewWithFallback(
            samples: [Float](repeating: 0.0, count: 32_000),
            language: .english,
            sampleRate: 16_000
        )

        let metrics = await sink.snapshot()
        XCTAssertEqual(metrics.count, 1)
        XCTAssertEqual(metrics[0].audioDurationMs, 2_000)
    }
}
