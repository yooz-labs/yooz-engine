// GPUAdmissionContentionBenchTests.swift
// LLMModuleTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Live bench for engine#228, reproducing the whisper#263 contention shape:
// an active "interactive" workload (a live streaming STT session) running
// concurrently with a "background" MLX TouchUp/LLM generation. Unlike
// `MLXAdmissionGateTests` (EngineCoreTests, fake workloads, no weights),
// this suite drives the REAL `MLXLLMBackend.generate()` path through the
// REAL `MLXAdmissionGate.shared` singleton — the same gate instance
// `TouchUpEngine`, the loopback WS handler, and `InProcessSTTStreamSession`
// all use in production — so the bench numbers reflect the actual wiring,
// not a simulation of it.
//
// Gated behind `GPU_CONTENTION_LIVE=1` (mirrors `KVCOMPRESSION_LIVE`,
// `YOOZ_LLM_LOAD_MODELS`, `YOOZ_INFINITE_LIVE`): CI never runs this. Locally,
// the test additionally checks the Yooz-Light snapshot is already present in
// the HF cache before running — it never triggers a download, and skips
// cleanly (not silently) when the weights are absent.
//
// Run locally (weights already cached under ~/.cache/huggingface/hub):
//
//     GPU_CONTENTION_LIVE=1 swift test --filter LLMModuleTests.GPUAdmissionContentionBenchTests

import EngineCore
import XCTest

@testable import LLMModule

final class GPUAdmissionContentionBenchTests: XCTestCase {
    private var shouldRun: Bool {
        ProcessInfo.processInfo.environment["GPU_CONTENTION_LIVE"] == "1"
    }

    /// Wall-clock hold for the "interactive session is briefly active"
    /// scenario. Well under the gate's production aging ceiling
    /// (`EngineConfig.gpuAdmissionAgingSeconds`, default 2s) so this
    /// scenario exercises the "queues, then clears" path, not the
    /// starvation guard.
    private let interactiveHoldSeconds = 0.5

    private func elapsedSeconds(since start: CFAbsoluteTime) -> Double {
        CFAbsoluteTimeGetCurrent() - start
    }

    // MARK: - Scenario A: bounded queuing under a brief interactive session

    /// Reproduces the #263 shape at bench scale: a streaming session is
    /// "active" (a bare `beginInteractive()` — the same primitive the real
    /// STT streaming paths call) while a background TouchUp-style
    /// generation runs concurrently. Before engine#228, both would race for
    /// the GPU (the #263 evidence: 6-33s touch-up vs <2s idle, a 3-15x
    /// regression). With the gate wired in, the background call queues for
    /// (at most, roughly) the interactive hold duration and then completes
    /// — bounded, not runaway.
    func testBackgroundGenerationQueuesBehindBriefInteractiveSessionThenCompletes() async throws {
        try XCTSkipUnless(shouldRun, "Set GPU_CONTENTION_LIVE=1 to run the #263 contention bench")

        let backend = MLXLLMBackend.create(for: .yoozLight)
        let cached = await backend.isModelCached
        try XCTSkipUnless(
            cached,
            "Yooz-Light-v3 weights not found in ~/.cache/huggingface; skipping rather than downloading"
        )
        try await backend.load()

        let systemPrompt = """
        You are a transcription cleanup assistant. Fix punctuation and \
        capitalization in the user's text. Return only the corrected text, \
        with no commentary.
        """

        // Warm-up call: pays first-call overhead (tokenizer / KV-cache
        // priming) outside both measured windows.
        _ = try await backend.generate(
            prompt: "warm up the model before timing",
            systemPrompt: systemPrompt,
            workloadClass: .background
        )

        // Idle baseline: no interactive workload active.
        let idleStart = CFAbsoluteTimeGetCurrent()
        let idleText = try await backend.generate(
            prompt: "schedule a meeting with the team on friday",
            systemPrompt: systemPrompt,
            workloadClass: .background
        )
        let idleSeconds = elapsedSeconds(since: idleStart)
        XCTAssertFalse(idleText.isEmpty, "idle baseline generation produced no text")

        // Contended: an interactive session is active for `interactiveHoldSeconds`
        // while the background generation is submitted concurrently.
        await MLXAdmissionGate.shared.beginInteractive()
        var interactiveEnded = false
        defer {
            if !interactiveEnded {
                Task { await MLXAdmissionGate.shared.endInteractive() }
            }
        }

        let contendedStart = CFAbsoluteTimeGetCurrent()
        async let contendedResult: String = backend.generate(
            prompt: "the weather in tokyo is rainy today so bring an umbrella",
            systemPrompt: systemPrompt,
            workloadClass: .background
        )

        try await Task.sleep(for: .seconds(interactiveHoldSeconds))
        await MLXAdmissionGate.shared.endInteractive()
        interactiveEnded = true

        let contendedText = try await contendedResult
        let contendedSeconds = elapsedSeconds(since: contendedStart)
        XCTAssertFalse(contendedText.isEmpty, "contended generation produced no text")

        print(
            """
            [GPUAdmissionContentionBench] idle=\(String(format: "%.3f", idleSeconds))s \
            contended=\(String(format: "%.3f", contendedSeconds))s \
            hold=\(String(format: "%.3f", interactiveHoldSeconds))s
            """
        )

        // The queued call must reflect the hold — proves it actually queued
        // rather than racing ahead of the interactive marker. Small
        // tolerance for scheduling jitter.
        XCTAssertGreaterThanOrEqual(
            contendedSeconds, interactiveHoldSeconds - 0.05,
            "contended generation completed before the interactive hold cleared; the gate did not queue it"
        )

        // Bounded: contended latency should track idle + hold, with slack
        // for real generation variance — NOT the 3-15x-style blowup #263
        // measured pre-gate. Generous multiplier (4x + hold) keeps this
        // robust to a slow CI/dev machine while still catching a runaway
        // regression.
        let bound = (idleSeconds * 4) + interactiveHoldSeconds + 2.0
        XCTAssertLessThanOrEqual(
            contendedSeconds, bound,
            "contended generation (\(contendedSeconds)s) exceeded the bounded-queuing ceiling "
                + "(\(bound)s derived from idle=\(idleSeconds)s + hold=\(interactiveHoldSeconds)s); "
                + "looks like unbounded GPU contention, not gated queuing"
        )
    }

    // MARK: - Scenario B: starvation guard (no deadlock)

    /// Acceptance criterion: "interactive work cannot starve background
    /// work forever." Holds an interactive session active past the gate's
    /// aging ceiling and asserts the background generation still completes
    /// (force-admitted) rather than hanging — proven with a real MLX call,
    /// not just the fake-workload unit tests in `MLXAdmissionGateTests`.
    func testBackgroundGenerationIsForceAdmittedAfterAgingCeiling() async throws {
        try XCTSkipUnless(shouldRun, "Set GPU_CONTENTION_LIVE=1 to run the #263 contention bench")

        let backend = MLXLLMBackend.create(for: .yoozLight)
        let cached = await backend.isModelCached
        try XCTSkipUnless(
            cached,
            "Yooz-Light-v3 weights not found in ~/.cache/huggingface; skipping rather than downloading"
        )
        try await backend.load()

        let systemPrompt = """
        You are a transcription cleanup assistant. Fix punctuation and \
        capitalization in the user's text. Return only the corrected text, \
        with no commentary.
        """
        _ = try await backend.generate(
            prompt: "warm up the model before timing",
            systemPrompt: systemPrompt,
            workloadClass: .background
        )

        let agingSeconds = EngineConfig.gpuAdmissionAgingSeconds
        let stateBefore = await MLXAdmissionGate.shared.queueState

        await MLXAdmissionGate.shared.beginInteractive()
        var interactiveEnded = false
        // Guaranteed cleanup: the interactive marker must not leak into
        // later tests sharing the process-wide `.shared` gate, even if an
        // assertion below fails.
        defer {
            if !interactiveEnded {
                Task { await MLXAdmissionGate.shared.endInteractive() }
            }
        }

        let start = CFAbsoluteTimeGetCurrent()
        // Deliberately never clear the interactive marker before awaiting
        // the generation — the starvation guard, not a cleared signal, must
        // be what unblocks this call. Bounded ceiling so a regression that
        // removes the guard fails this test instead of hanging the suite.
        let ceilingSeconds = agingSeconds + 30
        let text = try await withDeadline(seconds: ceilingSeconds) {
            try await backend.generate(
                prompt: "remember to buy almonds walnuts and pistachios",
                systemPrompt: systemPrompt,
                workloadClass: .background
            )
        }
        let elapsed = elapsedSeconds(since: start)

        await MLXAdmissionGate.shared.endInteractive()
        interactiveEnded = true

        XCTAssertFalse(text.isEmpty, "force-admitted generation produced no text")

        let stateAfter = await MLXAdmissionGate.shared.queueState
        print(
            """
            [GPUAdmissionContentionBench] starvation-guard elapsed=\(String(format: "%.3f", elapsed))s \
            agingSeconds=\(agingSeconds)s \
            forcedByAging(before->after)=\(stateBefore.backgroundForcedByAging)->\(stateAfter.backgroundForcedByAging)
            """
        )

        XCTAssertGreaterThan(
            stateAfter.backgroundForcedByAging, stateBefore.backgroundForcedByAging,
            "expected the starvation guard to force-admit at least once while interactive stayed active"
        )
        // No deadlock: completed at roughly the aging ceiling, not
        // instantly (proves it actually queued) and not never (proves the
        // guard fired).
        XCTAssertGreaterThanOrEqual(
            elapsed, agingSeconds - 0.1,
            "force-admitted generation completed before the aging ceiling; the guard fired too early"
        )
    }
}

/// Races `operation` against a timeout, throwing if `operation` has not
/// completed within `seconds`. Used so a regression that removes the
/// starvation guard fails this test with a clear timeout error instead of
/// hanging the suite indefinitely.
private func withDeadline<T: Sendable>(
    seconds: Double,
    operation: @escaping () async throws -> T
) async throws -> T {
    try await withThrowingTaskGroup(of: T.self) { group in
        group.addTask { try await operation() }
        group.addTask {
            try await Task.sleep(for: .seconds(seconds))
            throw BenchTimeoutError()
        }
        let result = try await group.next()!
        group.cancelAll()
        return result
    }
}

private struct BenchTimeoutError: Error, CustomStringConvertible {
    var description: String { "operation exceeded the bench deadline" }
}
