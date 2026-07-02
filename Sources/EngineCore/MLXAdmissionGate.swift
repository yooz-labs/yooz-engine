// MLXAdmissionGate.swift
// EngineCore
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import os

/// A workload's latency sensitivity for MLX/GPU submission. Declared
/// alongside `MLXResidency` (engine#216/#217, memory-space arbitration) as
/// its compute-time complement: `MLXResidency` decides which model weights
/// are resident at once, `MLXAdmissionGate` decides which resident model's
/// compute gets the GPU right now (engine#228).
///
/// - `interactive`: latency-sensitive, small — a live streaming STT session,
///   a short grammar call. Never queues at the gate; always admitted
///   immediately. While active, it signals `background` submissions to
///   queue or yield.
/// - `background`: throughput work that can tolerate queuing — batch
///   transcription, TouchUp/LLM generation, Infinite append/generate.
public enum MLXWorkloadClass: String, Codable, Sendable {
    case interactive
    case background
}

/// Process-wide admission gate arbitrating MLX/GPU compute between
/// concurrently-running workloads.
///
/// Why this exists: `MLXResidency` solved co-residency thrash — two MLX
/// model categories can be resident at once without evicting each other's
/// buffers. But residency says nothing about which resident model's
/// *compute* runs right now. whisper#263 proved a latency-sensitive live
/// streaming session (Apple Speech `.fastResults`, itself GPU/Metal work
/// though not an MLX submitter) can starve a concurrently-running MLX
/// TouchUp generation from <2s to 6-33s. Per-app workarounds (making the
/// overlay slower) stop being viable once N independent modules share one
/// engine process and no module can see another's GPU load — this gate is
/// the engine-level scheduling primitive that replaces them.
///
/// Policy v1 (engine#228): while at least one `interactive` workload is
/// active, a `background` submission calling `checkpoint()` queues (polls)
/// until either the interactive load clears, or the aging deadline passes —
/// at which point it is force-admitted regardless of interactive load. That
/// is the starvation guard: interactive activity can delay background work
/// but can never block it forever. The deadline is anchored to the
/// *enclosing unit of work* (see `checkpoint(workStartedAt:)`), so a
/// multi-chunk generation shares one aging budget across all its
/// checkpoints and defers at most `agingInterval` total under sustained
/// interactive load.
///
/// `interactive` submissions never call `checkpoint()`. They signal via
/// `beginInteractive()` / `endInteractive()` and always run immediately —
/// the gate never queues latency-sensitive work, only background work that
/// might contend with it.
///
/// Granularity: callers are expected to call `checkpoint()` once before
/// starting a unit of GPU work, and again between chunks of a
/// longer-running submission (e.g. each token batch of an LLM generation —
/// see `MLXLLMBackend.generate`). That is what makes an in-flight background
/// generation pause between chunks rather than only queuing at submission
/// time.
///
/// Thread-safety: actor-isolated. `checkpoint()` is safe to call from any
/// Task, including repeatedly within a single long-running generation, and
/// from multiple concurrent background generations at once — each
/// independently polls and ages.
public actor MLXAdmissionGate {
    /// Shared process-wide instance used by the real STT streaming and
    /// LLM/TouchUp/Infinite generation paths.
    public static let shared = MLXAdmissionGate()

    /// Snapshot of gate activity. Exposed so route handlers / callers can
    /// render "waiting for GPU" and so tests can assert admission behavior
    /// without depending on signpost output.
    public struct QueueState: Equatable, Sendable {
        /// Number of currently-active interactive workloads. Not clamped to
        /// 1 — overlapping interactive sessions (e.g. two streaming STT
        /// connections) coexist without underflow on teardown.
        public let interactiveActive: Int
        /// Number of background `checkpoint()` calls currently queued —
        /// polling, waiting for interactive load to clear or to age out.
        public let backgroundWaiting: Int
        /// Cumulative count of `checkpoint()` calls admitted immediately
        /// (no interactive workload was active at call time).
        public let backgroundAdmittedImmediately: Int
        /// Cumulative count of `checkpoint()` calls that queued and were
        /// then admitted because interactive load cleared before aging out.
        public let backgroundAdmittedAfterWait: Int
        /// Cumulative count of `checkpoint()` calls force-admitted by the
        /// starvation guard (aged out while interactive load was still
        /// active). Non-zero here is the observable signal that background
        /// work is being bounded by the aging ceiling rather than by
        /// interactive load actually clearing — see `agingInterval`.
        public let backgroundForcedByAging: Int

        /// Internal on purpose: a `QueueState` is a read-only observability
        /// snapshot that should only originate from
        /// `MLXAdmissionGate.queueState` — external callers cannot fabricate
        /// one (e.g. with negative counters) that never came from a gate.
        init(
            interactiveActive: Int,
            backgroundWaiting: Int,
            backgroundAdmittedImmediately: Int,
            backgroundAdmittedAfterWait: Int,
            backgroundForcedByAging: Int
        ) {
            self.interactiveActive = interactiveActive
            self.backgroundWaiting = backgroundWaiting
            self.backgroundAdmittedImmediately = backgroundAdmittedImmediately
            self.backgroundAdmittedAfterWait = backgroundAdmittedAfterWait
            self.backgroundForcedByAging = backgroundForcedByAging
        }
    }

    /// Wall-clock ceiling a `checkpoint()` call queues before being
    /// force-admitted regardless of interactive load — the starvation
    /// guard backing the "no deadlock" acceptance criterion. Production
    /// default is `EngineConfig.gpuAdmissionAgingSeconds`; tests inject a
    /// small value so the aging path exercises in milliseconds rather than
    /// a real multi-second wait.
    private let agingInterval: Duration
    /// Poll granularity while queued. Small relative to `agingInterval` so
    /// the aging deadline is honored closely without busy-spinning the
    /// actor.
    private let pollInterval: Duration

    private var interactiveActive = 0
    private var backgroundWaiting = 0
    private var backgroundAdmittedImmediately = 0
    private var backgroundAdmittedAfterWait = 0
    private var backgroundForcedByAging = 0

    private let signposter = OSSignposter(
        logger: Logger(subsystem: "live.yooz.engine", category: "GPUAdmission")
    )

    /// - Parameters:
    ///   - agingInterval: starvation-guard ceiling. Defaults to
    ///     `EngineConfig.gpuAdmissionAgingSeconds` (production: 2s, matching
    ///     the #263 target of bounding background latency to within 2x of
    ///     idle under an active interactive session). Note: `.shared` reads
    ///     the env-var-backed default once, at its lazy first access —
    ///     mutating `YOOZ_GPU_ADMISSION_AGING_SEC` after that is ignored for
    ///     the process lifetime. Bench harnesses that need a different value
    ///     must set the env var before anything touches `.shared`.
    ///   - pollInterval: queued-wait poll granularity. Defaults to 20ms.
    public init(
        agingInterval: Duration = .seconds(EngineConfig.gpuAdmissionAgingSeconds),
        pollInterval: Duration = .milliseconds(20)
    ) {
        precondition(
            agingInterval > .zero && pollInterval > .zero,
            "MLXAdmissionGate intervals must be positive: a non-positive agingInterval silently defeats the starvation guard; a non-positive pollInterval busy-spins the actor"
        )
        self.agingInterval = agingInterval
        self.pollInterval = pollInterval
    }

    public var queueState: QueueState {
        QueueState(
            interactiveActive: interactiveActive,
            backgroundWaiting: backgroundWaiting,
            backgroundAdmittedImmediately: backgroundAdmittedImmediately,
            backgroundAdmittedAfterWait: backgroundAdmittedAfterWait,
            backgroundForcedByAging: backgroundForcedByAging
        )
    }

    /// Signal an interactive workload (live streaming STT session, short
    /// grammar call) has started. Never queues — the caller proceeds
    /// immediately; this only raises the signal `checkpoint()` callers
    /// observe. Callers must pair with a later `endInteractive()` (a
    /// `defer` at the call site is the recommended pattern so every exit
    /// path — clean close, abort, error — clears the signal exactly once).
    public func beginInteractive() {
        interactiveActive += 1
        signposter.emitEvent("interactive_begin")
    }

    /// Clear one interactive signal. Ignored (does not underflow) if called
    /// with no matching `beginInteractive()` — an over-balanced call is a
    /// caller bug, not a reason to corrupt the counter for every other
    /// concurrent interactive session.
    public func endInteractive() {
        guard interactiveActive > 0 else { return }
        interactiveActive -= 1
        signposter.emitEvent("interactive_end")
    }

    /// Admission checkpoint for background MLX work. Call once before
    /// starting a unit of GPU work, and again between chunks of a
    /// longer-running submission so an in-flight background submission
    /// yields to a newly-active interactive workload rather than only
    /// queuing at start.
    ///
    /// Returns immediately when no interactive workload is active.
    /// Otherwise queues (polling `pollInterval`) until interactive load
    /// clears or the aging deadline passes, whichever comes first — the
    /// starvation guard. Cooperatively cancellable: propagates
    /// `CancellationError` if the calling Task is cancelled while queued.
    ///
    /// - Parameter workStartedAt: when the enclosing unit of background work
    ///   (e.g. one whole LLM generation) began. The aging deadline is
    ///   `workStartedAt + agingInterval`, so all the checkpoints of one
    ///   generation share a single aging budget: under *sustained*
    ///   interactive load the generation defers at most `agingInterval`
    ///   total, not `agingInterval` per chunk (which for a multi-chunk
    ///   generation would multiply into the unbounded-latency regression
    ///   the #263 acceptance criterion forbids). Once the shared budget is
    ///   spent, subsequent checkpoints of that generation force-admit
    ///   immediately. Pass `nil` (the default) for a standalone checkpoint
    ///   whose budget starts now.
    public func checkpoint(
        workStartedAt: ContinuousClock.Instant? = nil
    ) async throws {
        guard interactiveActive > 0 else {
            backgroundAdmittedImmediately += 1
            return
        }

        backgroundWaiting += 1
        let waitID = signposter.makeSignpostID()
        let waitState = signposter.beginInterval("background_wait", id: waitID)
        defer {
            backgroundWaiting -= 1
            signposter.endInterval("background_wait", waitState)
        }

        let deadline = (workStartedAt ?? ContinuousClock.now)
            .advanced(by: agingInterval)
        var agedOut = false
        while interactiveActive > 0 {
            if ContinuousClock.now >= deadline {
                agedOut = true
                break
            }
            try Task.checkCancellation()
            try await Task.sleep(for: pollInterval)
        }

        if agedOut {
            backgroundForcedByAging += 1
            signposter.emitEvent("background_forced_by_aging")
        } else {
            backgroundAdmittedAfterWait += 1
        }
    }
}
