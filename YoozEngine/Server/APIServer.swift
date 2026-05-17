#if canImport(AppleSTTModule)
import AppleSTTModule
#endif
import EngineCore
import Foundation
import GrammarModule
import HuggingFace
import Hummingbird
import HummingbirdWebSocket
#if canImport(LLMModule)
import LLMModule
#endif
import Logging
import NIOCore
#if canImport(STTModule)
import STTModule
#endif
#if canImport(VADModule)
import VADModule
#endif

/// Request context used by the HTTP router. The only material difference
/// from `BasicRequestContext` is the body-size limit: Hummingbird 2's
/// default is 2 MiB, which is below the JSON-encoded size of medium-long
/// audio chunks coming through `/v1/stt/batch`. Whisper sends Float
/// sample arrays serialized to JSON; ~17 s of 16 kHz audio is already
/// over 2 MiB, so the default rejected long recordings with 400
/// `invalid_request` before the handler ever ran (engine #111,
/// yooz-whisper #176). 64 MiB covers roughly 5 minutes of audio with
/// headroom for the alignment-token response payload.
///
/// A binary wire format would shrink the payload ~5× and remove the
/// JSON cost, but that's a coordinated protocol change deferred to a
/// later epic. For now we raise the ceiling.
struct YoozEngineRequestContext: RequestContext {
    /// Per-request upload ceiling, in bytes. Exposed as a constant so
    /// tests and clients can reason about the limit without hard-coding
    /// the number. See the type doc for the audio-duration rationale.
    static let maxUploadBytes: Int = 64 * 1024 * 1024

    var coreContext: CoreRequestContextStorage

    init(source: ApplicationRequestContextSource) {
        self.coreContext = .init(source: source)
    }

    var maxUploadSize: Int { Self.maxUploadBytes }
}

@MainActor
final class APIServer: ObservableObject {
    enum State: Equatable {
        case stopped
        case starting
        case running
        case stopping
        /// The server task terminated unexpectedly (non-cancellation error).
        /// UI may offer a "Restart Engine" affordance when in this state.
        case crashed
    }

    @Published var state: State = .stopped
    @Published var lastError: String?

    var isRunning: Bool { state == .running }

    private var app: (any ApplicationProtocol)?
    private var serverTask: Task<Void, any Error>?
    let logger = Logger(label: "live.yooz.engine.server")

    /// Active STT backend selection. Defaulted lazily from the bundled modules
    /// when the server starts: full/whisper → `.parakeet`, lite → `.appleSTT`.
    /// `POST /v1/stt/engine` updates this; `/v1/stt/batch` and the WebSocket
    /// router consult it to route requests. See `AppleSTT + Lite Variant`
    /// section of `.context/phase5_epic.md`.
    private var currentSTTEngine: STTEngineCode = APIServer.defaultSTTEngine()

    /// Cancels the currently-running WebSocket STT session, if any. The WS
    /// handler stores a cancel closure here on config; a `POST /v1/stt/engine`
    /// switch invokes it to close the socket before the new engine takes over.
    private var sttStreamCancel: (@Sendable () -> Void)?

    /// Posted when the running server task terminates with a non-cancellation
    /// error. The `userInfo` carries the localized error string under
    /// `APIServer.crashErrorKey`.
    static let crashedNotification = Notification.Name("live.yooz.engine.server.crashed")
    static let crashErrorKey = "error"

    /// Module teardown watchdog: if module unloads don't complete in this
    /// interval we log, cancel the task, and force the state to `.stopped`
    /// so the user isn't stuck in `.stopping` forever.
    private let stopWatchdogSeconds: UInt64 = 5

    // MARK: - Lifecycle

    func start() async throws {
        // Allow starting from `.stopped` OR `.crashed` (recovery path).
        guard state == .stopped || state == .crashed else { return }
        state = .starting
        lastError = nil

        let port = EngineConfig.port
        let host = EngineConfig.host

        // Pre-check the port. A `bind()` + catch-EADDRINUSE probe is
        // cheaper, has no race with a task-group (which previously
        // dead-locked on `task.value` since Hummingbird's `run()` never
        // returns on the happy path), and gives us a distinct error
        // before we spend the cost of spinning up the router.
        //
        // There's a TOCTOU window between this check and Hummingbird
        // binding the port — but the failure mode we care about (a
        // stale engine from a previous run still holding the port) is
        // persistent, not transient, so the race in practice is a
        // non-issue.
        if PortDiagnostics.isPortInUse(port) {
            let pid = PortDiagnostics.pidHoldingPort(port)
            state = .stopped
            let startError = ServerStartError.portInUse(port: port, pid: pid)
            lastError = startError.errorDescription
            logger.error("Cannot start: port \(port) already in use (pid=\(pid.map(String.init) ?? "unknown"))")
            throw startError
        }

        let httpRouter = buildRouter()
        let wsRouter = buildWebSocketRouter()

        let app = Application(
            router: httpRouter,
            server: .http1WebSocketUpgrade(webSocketRouter: wsRouter),
            configuration: .init(
                address: .hostname(host, port: port),
                serverName: "YoozEngine/\(EngineConfig.version)"
            ),
            logger: logger
        )

        self.app = app

        let task: Task<Void, any Error> = Task { [weak self] in
            do {
                try await app.run()
            } catch is CancellationError {
                // Expected during stop(). Swallow + return normally.
                return
            } catch {
                // Propagate everything else; the crash watcher reads
                // this via `await task.value`.
                throw error
            }
        }
        self.serverTask = task

        // Verify Hummingbird actually came up. A short probe loop of
        // `/v1/health`; per-attempt timeout of 0.5s, up to ~2s total.
        do {
            try await APIServer.waitForHealthy(host: host, port: port)
        } catch {
            // Health probe failed. Cancel the task (it'll exit on the
            // next NIO event) and surface the most informative error.
            task.cancel()
            serverTask = nil
            self.app = nil
            state = .stopped

            // If the task died with a bind-error during our wait we
            // can still surface EADDRINUSE here. `Task.result` is
            // awaited with a brief deadline to avoid hanging if the
            // server is still coming up — we've already cancelled it.
            let observedError = await Self.observeTaskExit(task, deadlineMs: 500)
            if let observed = observedError, PortDiagnostics.isAddressInUse(observed) {
                let pid = PortDiagnostics.pidHoldingPort(port)
                let startError = ServerStartError.portInUse(port: port, pid: pid)
                lastError = startError.errorDescription
                throw startError
            }

            if let startError = error as? ServerStartError {
                lastError = startError.errorDescription
                throw startError
            }
            let wrapped = ServerStartError.failedToBind(error.localizedDescription)
            lastError = wrapped.errorDescription
            throw wrapped
        }

        // Watch the running task for unexpected termination.
        installCrashWatcher(for: task)

        state = .running
        logger.info("Yooz Engine started on \(EngineConfig.host):\(EngineConfig.port)")

        // Always apply variant gating so the snapshot reflects the
        // active build variant (e.g. VAD `unavailable` on whisper)
        // even when eager-load is disabled in tests. This is cheap
        // (touches the readiness map only); no module loads run.
        await ModuleEagerLoader.shared.markVariantUnavailableModules(
            variant: EngineConfig.variant
        )

        // Kick off the variant-aware module eager-load so modules
        // are primed by the time thin clients hit them. The kickoff
        // itself is non-blocking — it spawns a background TaskGroup
        // and returns immediately, so the server is fully responsive
        // while loads run.
        //
        // Tying this to `start()` (rather than only to the app's
        // `applicationDidFinishLaunching` hook) makes the menu Stop
        // -> Start path symmetric: `stop()` resets the loader,
        // `start()` re-kicks. Without this, a second `start()`
        // would leave every module in the previous run's terminal
        // state while the underlying engines are unloaded.
        //
        // Disabled in test runs (`EngineConfig.eagerLoadOnLaunch`
        // checks for XCTest env vars) so route tests boot fast and
        // don't pay the model-fetch cost.
        if EngineConfig.eagerLoadOnLaunch {
            await ModuleEagerLoader.shared.kickoff(
                variant: EngineConfig.variant
            )
        }
    }

    func stop() async {
        guard state == .running || state == .crashed else { return }
        state = .stopping

        // Run module teardown as a detached task. Structured-concurrency
        // task groups don't help here because the module unloads
        // (`YoozSTTEngine.stop`, `TouchUpEngine.unload`, etc.) don't
        // check for Task cancellation — calling `group.cancelAll()`
        // would not unblock them, so `stop()` would still be pinned
        // waiting for the teardown child to return. Instead we signal
        // completion through an actor flag and poll with a bounded
        // deadline. If the deadline passes first we log and move on;
        // the detached task keeps running until it completes or the
        // process exits.
        let completionFlag = TeardownCompletionFlag()
        let logger = self.logger
        Task.detached {
            await Self.performTeardown(logger: logger)
            await completionFlag.markDone()
        }

        let deadlineNs = stopWatchdogSeconds * 1_000_000_000
        let pollNs: UInt64 = 100_000_000  // 100 ms
        var elapsedNs: UInt64 = 0
        while elapsedNs < deadlineNs {
            if await completionFlag.isDone { break }
            try? await Task.sleep(nanoseconds: pollNs)
            elapsedNs += pollNs
        }
        if await !completionFlag.isDone {
            logger.error(
                "Module teardown exceeded \(stopWatchdogSeconds)s watchdog; forcing server shutdown"
            )
        }

        // Reset the eager loader so a subsequent `start()` (e.g. the
        // user toggled Stop -> Start from the menu bar) re-runs the
        // load policy against the now-unloaded modules. Without this
        // reset, the second start would leave the loader thinking
        // every module is `ready` (its last terminal state) while
        // the underlying modules are actually unloaded — health
        // would lie until the user hit each route.
        await ModuleEagerLoader.shared.reset()

        serverTask?.cancel()
        _ = await serverTask?.result
        serverTask = nil
        app = nil

        state = .stopped
        logger.info("Yooz Engine stopped")
    }

    /// One-shot "done" flag consumed by the stop() watchdog. The
    /// detached teardown task sets it; `stop()` polls it.
    private actor TeardownCompletionFlag {
        private(set) var isDone: Bool = false
        func markDone() { isDone = true }
    }

    /// Tear down the engine and start it again. Used by the "Restart
    /// Engine" menu item after a crash, or by operators in response to
    /// a stuck server. Idempotent: if already running it stops first.
    func restart() async throws {
        if state == .running || state == .crashed {
            await stop()
        }
        try await start()
    }

    // MARK: - Lifecycle helpers

    /// Module teardown in a well-defined order: STT → AppleSTT → LLM → VAD.
    /// Each step is independently isolated so one module failing cannot
    /// starve the others. The `canImport` guards let slim variants (e.g.
    /// Lite, which drops MLX STT) compile without pulling the absent module.
    ///
    /// `static` + `nonisolated` so `Task.detached` in `stop()` can call
    /// it without capturing the MainActor-isolated instance. Logger is
    /// `Sendable` (swift-log) and is the only state required.
    nonisolated private static func performTeardown(logger: Logger) async {
        // Stop the MLX STT engine to free GPU memory
        #if canImport(STTModule)
        YoozSTTEngine.shared.stop()
        #endif

        // Release the Apple STT backend reference (no model weights to free)
        #if canImport(AppleSTTModule)
        await AppleSTTEngine.shared.stop()
        #endif

        // Unload LLM models to free GPU memory
        await TouchUpEngine.shared.unload()

        // Reset VAD hidden/cell state to free memory
        #if canImport(VADModule)
        do {
            try await VADEngine.shared.reset()
        } catch {
            logger.error("Failed to reset VAD state: \(error)")
        }
        #endif
    }

    /// Spawn a background watcher that waits for `task` to terminate.
    /// If termination is due to a non-cancellation error and the server
    /// was in `.running`, flip to `.crashed` and post the notification.
    private func installCrashWatcher(for task: Task<Void, any Error>) {
        Task { [weak self] in
            do {
                _ = try await task.value
                // Task returned cleanly (cancelled via stop() or similar).
                // No crash bookkeeping required.
                return
            } catch is CancellationError {
                return
            } catch {
                await MainActor.run {
                    guard let self = self else { return }
                    // Only escalate to .crashed if we were actually
                    // running. A start() failure path takes care of its
                    // own state transitions.
                    guard self.state == .running else { return }
                    self.logger.error("Server terminated unexpectedly: \(error)")
                    self.state = .crashed
                    self.lastError = error.localizedDescription
                    NotificationCenter.default.post(
                        name: APIServer.crashedNotification,
                        object: self,
                        userInfo: [APIServer.crashErrorKey: error.localizedDescription]
                    )
                }
            }
        }
    }

    /// Poll `/v1/health` until a 200 OK arrives, or ~2s has elapsed.
    /// Uses a short per-attempt timeout so a single failed probe cannot
    /// swallow the whole budget. Throws `ServerStartError.healthCheckFailed`
    /// when the budget is exhausted.
    nonisolated private static func waitForHealthy(host: String, port: Int) async throws {
        let url = URL(string: "http://\(host):\(port)/v1/health")!
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 0.5
        config.timeoutIntervalForResource = 2.0
        let session = URLSession(configuration: config)
        defer { session.invalidateAndCancel() }

        // Up to 10 attempts × 200ms ≈ 2s budget. Larger than the old
        // 200ms sleep so a slow cold-start bind doesn't false-negative.
        for _ in 0..<10 {
            try Task.checkCancellation()
            do {
                let (_, response) = try await session.data(from: url)
                if let http = response as? HTTPURLResponse, http.statusCode == 200 {
                    return
                }
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                // Connection refused / timeout — loop and retry.
            }
            try? await Task.sleep(nanoseconds: 200_000_000)
        }
        throw ServerStartError.healthCheckFailed
    }

    /// Briefly wait for the server task to surface its exit error (if
    /// any). Returns nil when the task has not finished within
    /// `deadlineMs` or exited cleanly. Used by `start()` to upgrade a
    /// generic health-check timeout into the more informative
    /// `portInUse` branch when Hummingbird already failed to bind.
    ///
    /// Callers must have `task.cancel()`ed before invoking this so
    /// Hummingbird is guaranteed to unwind within the deadline —
    /// without the cancel, the task's `app.run()` never returns and
    /// `task.value` would block indefinitely.
    ///
    /// Implementation uses an actor-backed one-shot slot polled from
    /// a single await loop. Avoids structured-concurrency group
    /// semantics that would otherwise wait for BOTH children before
    /// the group body returns.
    nonisolated private static func observeTaskExit(
        _ task: Task<Void, any Error>,
        deadlineMs: UInt64
    ) async -> Error? {
        let slot = ExitSlot()
        Task.detached {
            do {
                _ = try await task.value
                await slot.set(nil)
            } catch {
                await slot.set(error)
            }
        }

        let pollNs: UInt64 = 25_000_000  // 25 ms
        let deadlineNs = deadlineMs * 1_000_000
        var elapsed: UInt64 = 0
        while elapsed < deadlineNs {
            if await slot.isSet { return await slot.value }
            try? await Task.sleep(nanoseconds: pollNs)
            elapsed += pollNs
        }
        return nil
    }

    /// Single-assignment storage for `observeTaskExit`.
    private actor ExitSlot {
        private(set) var isSet: Bool = false
        private(set) var value: Error?
        func set(_ error: Error?) {
            guard !isSet else { return }
            value = error
            isSet = true
        }
    }

    // MARK: - Helpers

    private nonisolated func errorResponse(
        status: HTTPResponse.Status,
        message: String,
        code: String
    ) -> Response {
        let body = (try? JSONEncoder().encode(ErrorResponse(error: message, code: code))) ?? Data()
        return Response(
            status: status,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(data: body))
        )
    }

    /// HTTP 501 response for routes whose backing module isn't linked into
    /// this build variant. See A4 / #28.
    ///
    /// The body follows `ModuleNotBundledResponse`: `error`, `code`, `module`.
    /// Clients switch on `code == "module_not_bundled"` and use `module` to
    /// identify which capability is missing; the `error` string carries the
    /// build-variant name for humans.
    nonisolated func moduleNotBundled(_ module: String) -> Response {
        let payload = ModuleNotBundledResponse(
            module: module,
            buildVariant: BuildVariant.current.rawValue
        )
        let body = (try? JSONEncoder().encode(payload)) ?? Data()
        return Response(
            status: .notImplemented,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(data: body))
        )
    }

    private nonisolated func jsonResponse<T: Encodable>(
        _ value: T,
        status: HTTPResponse.Status = .ok
    ) throws -> Response {
        let data = try JSONEncoder().encode(value)
        return Response(
            status: status,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(data: data))
        )
    }

    #if canImport(STTModule)
    /// Map a Qwen3-ASR backend error to the right HTTP error
    /// response. Mirrors the parakeet/fast_conformer error mapping
    /// so clients get a consistent shape across backends.
    ///
    /// `FetchFailure` payloads are demuxed into distinct HTTP codes
    /// so clients can branch on root cause: 401 for auth, 404 for
    /// repo missing, 502 for transient transport, 422 for local
    /// integrity violations (size/checksum mismatch), 500 for
    /// programmer error (`.other`, manifest decode bugs).
    nonisolated func mapQwen3BatchError(
        _ error: Qwen3ASRError
    ) -> Response {
        switch error {
        case .invalidInput(let detail):
            return errorResponse(
                status: .badRequest,
                message: detail,
                code: "invalid_input"
            )
        case .invalidConfig(let detail):
            return errorResponse(
                status: .internalServerError,
                message: "Qwen3 config invalid: \(detail)",
                code: "model_config_invalid"
            )
        case .pipelineNotLoaded:
            return errorResponse(
                status: .serviceUnavailable,
                message:
                    "Qwen3 pipeline is not loaded; call /v1/stt/load "
                    + "before /v1/stt/batch.",
                code: "pipeline_not_loaded"
            )
        case .fileNotFound, .noAudioTowerWeights:
            return errorResponse(
                status: .serviceUnavailable,
                message: error.description,
                code: "model_not_loaded"
            )
        case .malformedSafetensors,
             .missingTensor,
             .shapeMismatch,
             .dtypeMismatch,
             .unexpectedTensor:
            return errorResponse(
                status: .internalServerError,
                message: error.description,
                code: "model_load_failed"
            )
        case .fetchFailed(let failure):
            return mapFetchFailure(failure, message: error.description)
        case .tokenizerValidationFailed(let detail):
            return errorResponse(
                status: .internalServerError,
                message: detail,
                code: "tokenizer_validation_failed"
            )
        }
    }

    /// Demux a structured `FetchFailure` payload into its HTTP code.
    /// Shared between `/v1/stt/batch` and `/v1/stt/load` so clients
    /// see a single mapping regardless of which endpoint surfaced
    /// the failure.
    nonisolated func mapFetchFailure(
        _ failure: FetchFailure,
        message: String? = nil
    ) -> Response {
        let body = message ?? failure.description
        switch failure {
        case .httpStatus(let code, _):
            switch code {
            case 401:
                return errorResponse(
                    status: .unauthorized,
                    message: body,
                    code: "model_fetch_unauthorized"
                )
            case 404:
                return errorResponse(
                    status: .notFound,
                    message: body,
                    code: "model_fetch_not_found"
                )
            case 500..<600:
                return errorResponse(
                    status: .badGateway,
                    message: body,
                    code: "model_fetch_upstream_5xx"
                )
            default:
                return errorResponse(
                    status: .badGateway,
                    message: body,
                    code: "model_fetch_failed"
                )
            }
        case .checksumMismatch, .sizeMismatch:
            // Local integrity violation. Not a transient gateway
            // problem; retrying the same gateway will likely
            // surface the same bytes. Surface as 422 with a
            // dedicated code so retry policy doesn't spin.
            return errorResponse(
                status: .unprocessableContent,
                message: body,
                code: "model_corruption_detected"
            )
        case .rangeIgnored:
            return errorResponse(
                status: .conflict,
                message: body,
                code: "model_fetch_resume_failed"
            )
        case .manifestDecode:
            return errorResponse(
                status: .badGateway,
                message: body,
                code: "model_manifest_invalid"
            )
        case .other:
            // Internal contract violation (typo'd repo URL, etc.) —
            // not a gateway problem. 500 so it surfaces as a server
            // bug rather than an upstream blame.
            return errorResponse(
                status: .internalServerError,
                message: body,
                code: "model_fetch_internal_error"
            )
        case .transport:
            return errorResponse(
                status: .badGateway,
                message: body,
                code: "model_fetch_failed"
            )
        }
    }

    /// Demux a `/v1/stt/load` failure into a structured wire response.
    /// Mirrors `mapFetchFailure`'s code surface so a Parakeet/HF-cache
    /// failure looks identical on the wire to a Qwen3 fetch failure
    /// of the same root cause. Codes are stable contract — adding a
    /// new one is fine, renaming an existing one is a breaking change.
    nonisolated func mapSTTLoadError(_ error: any Error) -> Response {
        // Cooperative cancellation. 499 (client-closed-request) is a
        // de-facto convention for "the awaiter went away"; falling
        // back to 503 would conflate it with server unavailability.
        if error is CancellationError {
            return errorResponse(
                status: .init(code: 499, reasonPhrase: "Client Closed Request"),
                message: "Load cancelled",
                code: "cancelled"
            )
        }

        // No HF mirror wired for this language family. 501 is the
        // right code: the request was well-formed and the language
        // is recognised, but the server has no implementation path
        // for it on this build.
        if let download = error as? STTHFDownloadError {
            switch download {
            case .unsupportedLanguage(let language):
                return errorResponse(
                    status: .notImplemented,
                    message: download.errorDescription ?? "Language unmirrored: \(language.rawValue)",
                    code: "language_unmirrored"
                )
            }
        }

        // Offline-mode cache miss (`HubClient.downloadSnapshot` with
        // `localFilesOnly: true` and no cached snapshot). Distinct
        // from a generic "not found" — the client retries with
        // `allow_fetch=true` to recover.
        if let cacheError = error as? HubCacheError {
            switch cacheError {
            case .cachedPathResolutionFailed:
                return errorResponse(
                    status: .notFound,
                    message: cacheError.errorDescription ?? "STT model not in HF cache; retry with allow_fetch=true.",
                    code: "model_not_cached"
                )
            default:
                return errorResponse(
                    status: .internalServerError,
                    message: cacheError.errorDescription ?? "HF cache error",
                    code: "model_cache_error"
                )
            }
        }

        // HTTP-layer failures from the HF download. Demux on status
        // code so callers can distinguish auth-required from gone
        // from upstream-5xx; the codes match `mapFetchFailure` so
        // the Qwen3 and Parakeet paths look the same on the wire.
        if let httpError = error as? HTTPClientError {
            switch httpError {
            case .responseError(let response, let detail):
                let body = "\(detail) (HTTP \(response.statusCode) for \(response.url?.absoluteString ?? "unknown"))"
                switch response.statusCode {
                case 401:
                    return errorResponse(
                        status: .unauthorized,
                        message: body,
                        code: "model_fetch_unauthorized"
                    )
                case 404:
                    return errorResponse(
                        status: .notFound,
                        message: body,
                        code: "model_fetch_not_found"
                    )
                case 500..<600:
                    return errorResponse(
                        status: .badGateway,
                        message: body,
                        code: "model_fetch_upstream_5xx"
                    )
                default:
                    return errorResponse(
                        status: .badGateway,
                        message: body,
                        code: "model_fetch_failed"
                    )
                }
            case .requestError(let detail), .unexpectedError(let detail):
                return errorResponse(
                    status: .badGateway,
                    message: detail,
                    code: "model_fetch_failed"
                )
            case .decodingError(let response, let detail):
                return errorResponse(
                    status: .badGateway,
                    message: "\(detail) (HTTP \(response.statusCode))",
                    code: "model_fetch_failed"
                )
            }
        }

        // Network-layer failures (offline, DNS, timeout) — caller
        // should retry; not a server bug.
        if let urlError = error as? URLError {
            return errorResponse(
                status: .badGateway,
                message: urlError.localizedDescription,
                code: "model_fetch_network_error"
            )
        }

        // Engine-side validation. `languageNotSupported` is a 400
        // (the client sent a code we don't implement); any other
        // YoozSTTError surfaces as a generic load failure.
        if let sttError = error as? YoozSTTError {
            switch sttError {
            case .languageNotSupported:
                return errorResponse(
                    status: .badRequest,
                    message: sttError.errorDescription ?? "Language not supported",
                    code: "language_not_supported"
                )
            case .modelNotFound(let message):
                return errorResponse(
                    status: .notFound,
                    message: message,
                    code: "model_not_found"
                )
            default:
                return errorResponse(
                    status: .internalServerError,
                    message: sttError.errorDescription ?? "Load failed",
                    code: "load_failed"
                )
            }
        }

        return errorResponse(
            status: .internalServerError,
            message: error.localizedDescription,
            code: "load_failed"
        )
    }

    /// Build the canonical picker row for an STT backend. Mirrors
    /// `TouchUpEngine.row(for:loadState:)` so the two pickers
    /// produce identical wire shapes (modulo the STT-specific
    /// capability extensions).
    ///
    /// `activeLoaded` must be the result of
    /// `await sttEngine.isCurrentBackendLoaded()` resolved on the
    /// caller's side — `nonisolated` helpers can't reach the
    /// MainActor-bound `sttEngine` property. The active row reports
    /// `.loaded` when that flag is true, `.available` otherwise;
    /// non-active rows always report `.available` (lazy loading).
    nonisolated func sttBackendInfo(
        for backend: STTBackendID,
        active: STTBackendID,
        activeLoaded: Bool
    ) -> STTBackendInfo {
        let isActive = backend == active
        let loadState: ModelLoadState =
            (isActive && activeLoaded) ? .loaded : .available
        return STTBackendInfo(
            id: backend.rawValue,
            displayName: backend.displayName,
            description: backend.pickerDescription,
            tier: backend.pickerTier,
            sizeBytes: backend.estimatedDownloadMB
                .map { Int64($0) * 1_048_576 },
            loadState: loadState,
            isActive: isActive,
            supportsBatch: backend.supportsBatch,
            supportsStreaming: backend.supportsStreaming,
            supportedLanguages: backend.supportedLanguages.map(\.rawValue)
        )
    }

    #endif

    /// Build the canonical picker row for a server-level STT engine selector.
    /// Used by Lite, where `STTModule` is intentionally absent but Apple STT
    /// still needs the same `/v1/stt/engine` response shape.
    nonisolated func sttEngineInfo(
        for backend: STTEngineCode,
        active: STTEngineCode,
        activeLoaded: Bool
    ) -> STTBackendInfo {
        let isActive = backend == active
        let loadState: ModelLoadState =
            (isActive && activeLoaded) ? .loaded : .available

        let displayName: String
        let description: String
        let tier: ModelTier
        let sizeBytes: Int64?
        let supportsBatch: Bool
        let supportsStreaming: Bool
        let supportedLanguages: [String]

        switch backend {
        case .parakeet:
            displayName = "Parakeet (Recommended)"
            description = "Multilingual Latin / European"
            tier = .quality
            sizeBytes = nil
            supportsBatch = true
            supportsStreaming = true
            supportedLanguages = ["en"]
        case .fastConformer:
            displayName = "FastConformer (Arabic / Persian / Hebrew)"
            description = "Optimised for Arabic / Persian / Hebrew"
            tier = .quality
            sizeBytes = nil
            supportsBatch = true
            supportsStreaming = true
            supportedLanguages = ["ar", "fa", "he"]
        case .appleSTT:
            displayName = "Apple Speech (On-device)"
            description = "On-device, no download"
            tier = .premium
            sizeBytes = nil
            supportsBatch = true
            supportsStreaming = false
            #if canImport(AppleSTTModule)
            supportedLanguages = AppleSTTLanguage.allCases.map(\.rawValue)
            #else
            supportedLanguages = ["en"]
            #endif
        }

        return STTBackendInfo(
            id: backend.rawValue,
            displayName: displayName,
            description: description,
            tier: tier,
            sizeBytes: sizeBytes,
            loadState: loadState,
            isActive: isActive,
            supportsBatch: supportsBatch,
            supportsStreaming: supportsStreaming,
            supportedLanguages: supportedLanguages
        )
    }

    #if canImport(STTModule)
    private nonisolated func parseLanguage(_ code: String?) throws -> STTLanguage {
        let languageCode = code ?? "en"
        guard let language = STTLanguage.fromCode(languageCode) else {
            throw STTRequestError.invalidLanguage(languageCode)
        }
        guard language.isImplemented else {
            throw STTRequestError.notImplemented(language.displayName)
        }
        return language
    }
    #endif

    // MARK: - HTTP Router

    /// Build a router for testing without spinning up the full server
    /// task. Used by HTTP integration tests that exercise the route
    /// handlers via `Application.test(.router) { ... }`.
    func makeTestRouter() -> Router<YoozEngineRequestContext> {
        buildRouter()
    }

    /// Build a WebSocket router for testing.
    func makeTestWebSocketRouter() -> Router<BasicWebSocketRequestContext> {
        buildWebSocketRouter()
    }

    private func buildRouter() -> Router<YoozEngineRequestContext> {
        let router = Router(context: YoozEngineRequestContext.self)
        #if canImport(STTModule)
        let sttEngine = YoozSTTEngine.shared
        // Hoist the metrics sink + preview fallback hook so they
        // share lifecycle with the router (one engine `start()` call
        // = one sink + one hook, surviving across requests). The
        // hook's cold-start state machine relies on this — a
        // per-request hook would treat every request as a new
        // cold-start.
        let metricsSink: any STTMetricsSink = makeSTTMetricsSink(
            optedIn: EngineConfig.telemetryOptedIn,
            fileURL: EngineConfig.sttMetricsFileURL
        )
        let previewFallbackHook = PreviewFallbackHook(
            fetcher: Qwen3ASRPreviewFetcherAdapter(),
            preview: Qwen3ASRPreviewBackendAdapter(),
            fallback: ParakeetFallbackAdapter(),
            sink: metricsSink
        )
        #endif

        // Health
        router.get("/v1/health") { _, _ in
            // Pull the variant-aware readiness map from the eager
            // loader. The map is the source of truth for the new
            // `loading` / `error` / `unavailable` states; the legacy
            // bool fields stay for SDK back-compat (true iff the
            // module reports `.ready`).
            //
            // We also OR in the engines' current "is loaded" flags
            // so a thin client that called `/v1/stt/load` or
            // `/v1/llm/generate` and bypassed the eager loader still
            // sees `true` here. The eager loader only writes its
            // own state; it does not poll the engines after kickoff.
            let detail = await ModuleEagerLoader.shared.snapshot()
            let llmLoaded = await TouchUpEngine.shared.isLightModelLoaded
            let touchupReady = await TouchUpEngine.shared.isPreloaded
            #if canImport(VADModule)
            let vadLoaded = await VADEngine.shared.isLoaded
            #else
            let vadLoaded = false
            #endif
            #if canImport(STTModule)
            let sttLoaded = YoozSTTEngine.shared.isRunning
            #elseif canImport(AppleSTTModule)
            let sttLoaded = await AppleSTTEngine.shared.isLoaded
            #else
            let sttLoaded = false
            #endif

            func isReady(_ id: ModuleID, fallback: Bool) -> Bool {
                if detail[id.rawValue]?.state == .ready { return true }
                return fallback
            }

            return HealthResponse(
                status: "ok",
                version: EngineConfig.version,
                modules: EngineModules(
                    stt: isReady(.stt, fallback: sttLoaded),
                    llm: isReady(.llm, fallback: llmLoaded),
                    touchup: isReady(.touchup, fallback: touchupReady),
                    grammar: isReady(
                        .grammar,
                        fallback: GrammarEngine.shared.isAvailable
                    ),
                    vad: isReady(.vad, fallback: vadLoaded),
                    tts: isReady(.tts, fallback: false),
                    detail: detail
                )
            )
        }

        // Modules manifest: full per-module health + build variant.
        // Uses a local sorted-keys encoder so `detail` dictionaries (and all
        // JSON field names) come out in stable order for deterministic client
        // consumption. `/v1/health` stays unchanged.
        router.get("/v1/modules") { _, _ in
            let modules = await ModuleRegistry.shared.all()
            let response = await ModulesResponse.build(
                from: modules,
                engineVersion: EngineConfig.version,
                buildVariant: BuildVariant.current.rawValue
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = .sortedKeys
            let data = try encoder.encode(response)
            return Response(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: ByteBuffer(data: data))
            )
        }

        // Models
        router.get("/v1/models") { _, _ in
            var models: [ModelInfo] = []
            #if canImport(STTModule)
            let sttEngine = YoozSTTEngine.shared
            if sttEngine.isRunning {
                models.append(ModelInfo(
                    name: sttEngine.currentLanguage.modelIdentifier,
                    module: "stt",
                    loaded: true,
                    sizeBytes: nil
                ))
            }
            #endif
            #if canImport(AppleSTTModule)
            let appleEngine = AppleSTTEngine.shared
            if await appleEngine.isLoaded {
                let lang = await appleEngine.currentLanguage
                models.append(ModelInfo(
                    name: "apple-stt-\(lang.bcp47)",
                    module: "apple_stt",
                    loaded: true,
                    sizeBytes: nil
                ))
            }
            #endif
            let llmInfo = await TouchUpEngine.shared.getModelInfo()
            models.append(ModelInfo(
                name: llmInfo.light.type.rawValue,
                module: "llm",
                loaded: llmInfo.light.isLoaded,
                sizeBytes: llmInfo.light.type.estimatedSize
            ))
            models.append(ModelInfo(
                name: llmInfo.quality.type.rawValue,
                module: "llm",
                loaded: llmInfo.quality.isLoaded,
                sizeBytes: llmInfo.quality.type.estimatedSize
            ))
            let fmLoaded = await TouchUpEngine.shared.isFoundationModelsLoaded
            if fmLoaded {
                models.append(ModelInfo(
                    name: "foundation-models",
                    module: "llm",
                    loaded: true,
                    sizeBytes: nil
                ))
            }
            return ModelsResponse(models: models)
        }

        // LLM: Generate
        router.post("/v1/llm/generate") { [self] request, context in
            guard await ModuleRegistry.shared.isBundled("llm") else {
                return moduleNotBundled("llm")
            }
            let body: LLMGenerateServerRequest
            do {
                body = try await request.decode(as: LLMGenerateServerRequest.self, context: context)
            } catch {
                return errorResponse(
                    status: .badRequest,
                    message: "Invalid request body: \(error.localizedDescription)",
                    code: "invalid_request"
                )
            }

            // Resolve model type from name (default to light)
            let modelType: LLMModelType
            if let modelName = body.model {
                guard let resolved = LLMModelType(rawValue: modelName) else {
                    return errorResponse(
                        status: .badRequest,
                        message: "Unknown model: \(modelName). Available: \(LLMModelType.allCases.map(\.rawValue).joined(separator: ", "))",
                        code: "invalid_model"
                    )
                }
                modelType = resolved
            } else {
                modelType = .yoozLight
            }

            let startTime = CFAbsoluteTimeGetCurrent()
            do {
                let result = try await TouchUpEngine.shared.generate(
                    prompt: body.prompt,
                    systemPrompt: body.systemPrompt ?? "",
                    modelType: modelType,
                    kvCompression: body.kvCompression
                )
                let timeMs = Int((CFAbsoluteTimeGetCurrent() - startTime) * 1000)
                return try jsonResponse(LLMGenerateServerResponse(
                    text: result,
                    model: modelType.rawValue,
                    tokensGenerated: nil,
                    processingTimeMs: timeMs
                ))
            } catch {
                return errorResponse(
                    status: .internalServerError,
                    message: error.localizedDescription,
                    code: "generation_failed"
                )
            }
        }

        // LLM: List + manage models (catalogue + preferred model +
        // per-model preload / unload). Added for whisper's AI tab
        // dropdown; see `.context/llm_sdk_handoff.md`. All routes guard
        // on the `llm` module registry entry so slim variants return a
        // uniform 501 instead of 404.

        router.get("/v1/llm/models") { [self] _, _ -> Response in
            guard await ModuleRegistry.shared.isBundled("llm") else {
                return moduleNotBundled("llm")
            }
            let response = await Self.buildLLMModelsResponse()
            return try jsonResponse(response)
        }

        router.post("/v1/llm/model") { [self] request, context in
            guard await ModuleRegistry.shared.isBundled("llm") else {
                return moduleNotBundled("llm")
            }
            let body: LLMModelSelectionRequest
            do {
                body = try await request.decode(
                    as: LLMModelSelectionRequest.self,
                    context: context
                )
            } catch {
                return errorResponse(
                    status: .badRequest,
                    message: "Invalid request body: \(error.localizedDescription)",
                    code: "invalid_request"
                )
            }
            guard let modelType = LLMModelType(rawValue: body.model) else {
                return errorResponse(
                    status: .badRequest,
                    message: "Unknown model: \(body.model). Available: \(LLMModelType.allCases.map(\.rawValue).joined(separator: ", "))",
                    code: "invalid_model"
                )
            }
            await TouchUpEngine.shared.setPreferredModel(modelType)
            // Mirror GET's shape so whisper can reuse the decoder; the
            // `current` field reflects the new selection.
            let response = await Self.buildLLMModelsResponse()
            return try jsonResponse(response)
        }

        router.post("/v1/llm/preload") { [self] request, context in
            guard await ModuleRegistry.shared.isBundled("llm") else {
                return moduleNotBundled("llm")
            }
            let body: LLMModelSelectionRequest
            do {
                body = try await request.decode(
                    as: LLMModelSelectionRequest.self,
                    context: context
                )
            } catch {
                return errorResponse(
                    status: .badRequest,
                    message: "Invalid request body: \(error.localizedDescription)",
                    code: "invalid_request"
                )
            }
            guard let modelType = LLMModelType(rawValue: body.model) else {
                return errorResponse(
                    status: .badRequest,
                    message: "Unknown model: \(body.model)",
                    code: "invalid_model"
                )
            }
            do {
                try await TouchUpEngine.shared.preloadModel(modelType)
            } catch {
                return errorResponse(
                    status: .internalServerError,
                    message: error.localizedDescription,
                    code: "preload_failed"
                )
            }
            return try jsonResponse(Self.infoEntry(for: modelType, loaded: true))
        }

        router.post("/v1/llm/unload") { [self] request, context in
            guard await ModuleRegistry.shared.isBundled("llm") else {
                return moduleNotBundled("llm")
            }
            let body: LLMModelSelectionRequest
            do {
                body = try await request.decode(
                    as: LLMModelSelectionRequest.self,
                    context: context
                )
            } catch {
                return errorResponse(
                    status: .badRequest,
                    message: "Invalid request body: \(error.localizedDescription)",
                    code: "invalid_request"
                )
            }
            guard let modelType = LLMModelType(rawValue: body.model) else {
                return errorResponse(
                    status: .badRequest,
                    message: "Unknown model: \(body.model)",
                    code: "invalid_model"
                )
            }
            await TouchUpEngine.shared.unload(modelType)
            return try jsonResponse(Self.infoEntry(for: modelType, loaded: false))
        }

        // TouchUp: Process text
        router.post("/v1/touchup") { [self] request, context in
            // TouchUp lives in LLMModule; gate by the LLM registry entry.
            guard await ModuleRegistry.shared.isBundled("llm") else {
                return moduleNotBundled("llm")
            }
            let body: TouchUpServerRequest
            do {
                body = try await request.decode(as: TouchUpServerRequest.self, context: context)
            } catch {
                return errorResponse(
                    status: .badRequest,
                    message: "Invalid request body: \(error.localizedDescription)",
                    code: "invalid_request"
                )
            }

            if body.mode == .off {
                // Off mode: regex-only processing (no LLM)
                let result = await TouchUpEngine.shared.processRegexOnly(text: body.text)
                return try jsonResponse(TouchUpServerResponse(
                    result: result.text,
                    mode: .off,
                    processingTimeMs: Int(result.latencyMs),
                    modelUsed: result.modelUsed.rawValue,
                    warnings: nil
                ))
            }

            // Route through the picker-aware path so the model the
            // user picked via `POST /v1/touchup/model` is honored.
            // For callers that never set a model, this is identical
            // to the legacy `process(...)` path (default
            // `activeModel == .yoozLight`).
            let result = await TouchUpEngine.shared.processWithActiveModel(
                text: body.text,
                mode: body.mode.asDomain
            )

            var warnings: [String]? = nil
            if let reason = result.fallbackReason {
                warnings = [reason]
            }

            return try jsonResponse(TouchUpServerResponse(
                result: result.text,
                mode: body.mode,
                processingTimeMs: Int(result.latencyMs),
                modelUsed: result.modelUsed.rawValue,
                warnings: warnings
            ))
        }

        // TouchUp: List available models for the picker UI.
        // See AGENTS.md "Module model picker pattern" — this is the
        // canonical shape every module's picker endpoint should
        // follow (id / displayName / description / tier / sizeBytes
        // / isAvailable / isCached / isLoaded / isActive).
        router.get("/v1/touchup/models") { _, _ in
            let models = await TouchUpEngine.shared.availableModels()
            let activeId = await TouchUpEngine.shared.activeModel.rawValue
            return TouchUpModelsResponse(models: models, activeId: activeId)
        }

        // TouchUp: Set active model. `preload` defaults to true so a
        // one-shot picker change is enough to warm the model before
        // the next `/v1/touchup` call. Returns the new active row so
        // clients don't need a follow-up GET.
        router.post("/v1/touchup/model") { [self] request, context in
            let body: TouchUpSetModelRequest
            do {
                body = try await request.decode(
                    as: TouchUpSetModelRequest.self, context: context
                )
            } catch {
                return errorResponse(
                    status: .badRequest,
                    message: "Invalid request body: \(error.localizedDescription)",
                    code: "invalid_request"
                )
            }

            guard let selection = TouchUpModelSelection(rawValue: body.id) else {
                let known = TouchUpModelSelection.allCases.map(\.rawValue).joined(separator: ", ")
                return errorResponse(
                    status: .badRequest,
                    message: "Unknown TouchUp model id '\(body.id)'. Known: \(known).",
                    code: "invalid_model"
                )
            }

            do {
                let active = try await TouchUpEngine.shared.setActiveModel(
                    selection,
                    preload: body.preload ?? true
                )
                return try jsonResponse(active)
            } catch let error as LLMError {
                // `notAvailable` is the FoundationModels-on-pre-26
                // case — surface as 501 so the picker UI can render
                // "not supported on this Mac" cleanly. Any other
                // load failure (network, OOM) is 500.
                switch error {
                case .notAvailable(let detail):
                    return errorResponse(
                        status: .notImplemented,
                        message: detail,
                        code: "model_unavailable"
                    )
                default:
                    return errorResponse(
                        status: .internalServerError,
                        message: error.localizedDescription,
                        code: "model_set_failed"
                    )
                }
            } catch {
                return errorResponse(
                    status: .internalServerError,
                    message: error.localizedDescription,
                    code: "model_set_failed"
                )
            }
        }

        // Grammar: Check text
        router.post("/v1/grammar/check") { [self] request, context in
            guard await ModuleRegistry.shared.isBundled("grammar") else {
                return moduleNotBundled("grammar")
            }
            let body: GrammarCheckServerRequest
            do {
                body = try await request.decode(as: GrammarCheckServerRequest.self, context: context)
            } catch {
                return errorResponse(
                    status: .badRequest,
                    message: "Invalid request body: \(error.localizedDescription)",
                    code: "invalid_request"
                )
            }

            guard GrammarEngine.shared.isAvailable else {
                return errorResponse(
                    status: .serviceUnavailable,
                    message: "Grammar rules not loaded",
                    code: "grammar_not_available"
                )
            }

            let usePOS = body.usePOS ?? true
            let result = await GrammarEngine.shared.check(
                text: body.text,
                categories: body.categories,
                usePOS: usePOS
            )
            return try jsonResponse(GrammarCheckServerResponse(
                result: result.result,
                correctionsApplied: result.correctionsApplied,
                ruleCount: GrammarEngine.shared.ruleCount
            ))
        }

        // VAD: Detect speech segments
        //
        // The route itself is always registered so that in slim variants
        // (e.g. whisper, which embeds its own local VAD) clients get a
        // uniform HTTP 501 `module_not_bundled` response from A4 / #28
        // rather than Hummingbird's default 404. The type-level references
        // to `VADEngine` / `VADDetectServerRequest` are still compile-time
        // gated — the latter lives in the app target, but `VADEngine` only
        // exists when `VADModule` is linked, so the handler body needs the
        // `#if canImport(VADModule)` gate around the VAD-type call sites.
        router.post("/v1/vad/detect") { [self] request, context in
            guard await ModuleRegistry.shared.isBundled("vad") else {
                return moduleNotBundled("vad")
            }
            #if canImport(VADModule)
            let body: VADDetectServerRequest
            do {
                body = try await request.decode(as: VADDetectServerRequest.self, context: context)
            } catch {
                return errorResponse(
                    status: .badRequest,
                    message: "Invalid request body: \(error.localizedDescription)",
                    code: "invalid_request"
                )
            }

            guard await VADEngine.shared.isLoaded else {
                return errorResponse(
                    status: .serviceUnavailable,
                    message: "VAD model not loaded",
                    code: "vad_not_loaded"
                )
            }

            guard body.samples.count >= VADEngine.windowSize else {
                return errorResponse(
                    status: .badRequest,
                    message: "Samples too short; need at least \(VADEngine.windowSize) (32ms at 16kHz)",
                    code: "samples_too_short"
                )
            }

            do {
                let shouldReset = body.reset ?? true
                let segments = try await VADEngine.shared.detect(
                    samples: body.samples,
                    resetState: shouldReset
                )
                let responseSegments = segments.map { seg in
                    VADSegment(startMs: seg.startMs, endMs: seg.endMs, probability: seg.probability)
                }
                return try jsonResponse(VADDetectServerResponse(segments: responseSegments))
            } catch {
                return errorResponse(
                    status: .internalServerError,
                    message: error.localizedDescription,
                    code: "vad_detection_failed"
                )
            }
            #else
            // Unreachable: the registry guard above returns before we get
            // here in a variant that omits VADModule. Kept for exhaustive
            // compilation in the slim variant.
            return moduleNotBundled("vad")
            #endif
        }

        // STT: Backend picker (canonical pattern, second adopter
        // of #97). Same `models / activeId` shape as
        // `/v1/touchup/models` so consumer apps can template a
        // single ModelPickerStore<T> across pickers. STT-specific
        // capability flags ride along as optional extensions. The
        // legacy `STTEngineServerResponse` (current/available/
        // hasBuiltInVAD) shape is intentionally dropped here — the
        // capability bits live on `STTBackendInfo` so the SDK does
        // not need a parallel codepath per shape.
        router.get("/v1/stt/engine") { [self] _, _ in
            #if canImport(STTModule)
            let active = sttEngine.currentBackend
            let activeLoaded = await sttEngine.isCurrentBackendLoaded()
            let backends = STTBackendID.allCases.map { backend in
                self.sttBackendInfo(
                    for: backend,
                    active: active,
                    activeLoaded: activeLoaded
                )
            }
            return STTBackendsResponse(
                backends: backends,
                activeId: active.rawValue
            )
            #elseif canImport(AppleSTTModule)
            let active = await MainActor.run { currentSTTEngine }
            let activeLoaded = await AppleSTTEngine.shared.isLoaded
            let backends = APIServer.availableSTTEngines().map { backend in
                self.sttEngineInfo(
                    for: backend,
                    active: active,
                    activeLoaded: activeLoaded
                )
            }
            return STTBackendsResponse(
                backends: backends,
                activeId: active.rawValue
            )
            #else
            return moduleNotBundled("stt")
            #endif
        }

        router.post("/v1/stt/engine") { [self] request, context in
            #if canImport(STTModule)
            let body: STTSetBackendRequest
            do {
                body = try await request.decode(
                    as: STTSetBackendRequest.self, context: context
                )
            } catch {
                return errorResponse(
                    status: .badRequest,
                    message: "Invalid request body: \(error.localizedDescription)",
                    code: "invalid_request"
                )
            }

            // Accept canonical `id` first; fall back to legacy
            // `engine` so a one-week-old SDK that posts
            // `{ "engine": "parakeet" }` still works through one
            // release. New clients post
            // `{ "id": "parakeet", "preload": true }`.
            //
            // Reject conflicting values up-front — silently
            // dropping `engine` when the client posts both
            // `{"id":"parakeet","engine":"apple_stt"}` would
            // mid-migration eat half a build script's intent
            // with no diagnostic. 400 makes the bug observable.
            if let id = body.id, let legacy = body.engine, id != legacy {
                return errorResponse(
                    status: .badRequest,
                    message:
                        "Both 'id' and legacy 'engine' set with conflicting values "
                        + "('\(id)' vs '\(legacy)'); use 'id' only.",
                    code: "invalid_request"
                )
            }
            let rawId = body.id ?? body.engine
            guard let resolvedId = rawId else {
                return errorResponse(
                    status: .badRequest,
                    message: "Missing 'id' (or legacy 'engine') field",
                    code: "invalid_request"
                )
            }
            guard let backend = STTBackendID(rawValue: resolvedId) else {
                return errorResponse(
                    status: .badRequest,
                    message: "Unknown engine '\(resolvedId)'. Available: "
                        + STTBackendID.allCases.map(\.rawValue).joined(separator: ", "),
                    code: "invalid_model"
                )
            }

            await sttEngine.setBackend(backend)
            await MainActor.run {
                switch backend {
                case .parakeet, .qwen3ASRPreview:
                    currentSTTEngine = .parakeet
                case .fastConformer:
                    currentSTTEngine = .fastConformer
                case .appleSTT:
                    currentSTTEngine = .appleSTT
                }
                sttStreamCancel?()
            }
            // Note: `preload` is accepted on the wire for shape
            // parity with `/v1/touchup/model`, but actual STT
            // model loading happens lazily on the first
            // `/v1/stt/load` or `/v1/stt/batch` call. Eager
            // preload on backend switch is a future enhancement
            // that requires routing the call through the
            // current language — punt for now, document via the
            // `preload` field's no-op behavior.
            let activeLoaded = await sttEngine.isCurrentBackendLoaded()
            let info = sttBackendInfo(
                for: sttEngine.currentBackend,
                active: sttEngine.currentBackend,
                activeLoaded: activeLoaded
            )
            return try jsonResponse(info)
            #elseif canImport(AppleSTTModule)
            let body: STTSetBackendRequest
            do {
                body = try await request.decode(
                    as: STTSetBackendRequest.self, context: context
                )
            } catch {
                return errorResponse(
                    status: .badRequest,
                    message: "Invalid request body: \(error.localizedDescription)",
                    code: "invalid_request"
                )
            }
            if let id = body.id, let legacy = body.engine, id != legacy {
                return errorResponse(
                    status: .badRequest,
                    message:
                        "Both 'id' and legacy 'engine' set with conflicting values "
                        + "('\(id)' vs '\(legacy)'); use 'id' only.",
                    code: "invalid_request"
                )
            }
            let rawId = body.id ?? body.engine
            guard let resolvedId = rawId else {
                return errorResponse(
                    status: .badRequest,
                    message: "Missing 'id' (or legacy 'engine') field",
                    code: "invalid_request"
                )
            }
            guard resolvedId == STTEngineCode.appleSTT.rawValue else {
                return errorResponse(
                    status: .badRequest,
                    message: "Unknown engine '\(resolvedId)'. Available: apple_stt",
                    code: "invalid_model"
                )
            }
            await MainActor.run {
                currentSTTEngine = .appleSTT
                sttStreamCancel?()
            }
            let activeLoaded = await AppleSTTEngine.shared.isLoaded
            let info = sttEngineInfo(
                for: .appleSTT,
                active: .appleSTT,
                activeLoaded: activeLoaded
            )
            return try jsonResponse(info)
            #else
            return moduleNotBundled("stt")
            #endif
        }

        // STT: Available languages
        router.get("/v1/stt/languages") { [self] _, _ -> Response in
            #if canImport(STTModule)
            let sttBundled = await ModuleRegistry.shared.isBundled("stt")
            let appleBundled = await ModuleRegistry.shared.isBundled("apple_stt")
            guard sttBundled || appleBundled else {
                return moduleNotBundled("stt")
            }
            let payload = STTLanguagesResponse(languages: STTLanguage.allCases.map { lang in
                STTLanguageInfo(
                    code: lang.rawValue,
                    name: lang.displayName,
                    implemented: lang.isImplemented,
                    family: lang.modelFamily.rawValue
                )
            })
            return try jsonResponse(payload)
            #elseif canImport(AppleSTTModule)
            // Lite variant: surface the Apple STT language list without
            // pulling STTModule's STTLanguage type.
            guard await ModuleRegistry.shared.isBundled("apple_stt") else {
                return moduleNotBundled("apple_stt")
            }
            let payload = STTLanguagesResponse(languages: AppleSTTLanguage.allCases.map { lang in
                STTLanguageInfo(
                    code: lang.rawValue,
                    name: lang.rawValue.uppercased(),
                    implemented: AppleSTTBackend.isAvailable(localeIdentifier: lang.bcp47),
                    family: "apple"
                )
            })
            return try jsonResponse(payload)
            #else
            return moduleNotBundled("stt")
            #endif
        }

        // STT: Status — variant-aware dispatch between MLX and
        // Apple STT engines. Main's unified `isCurrentBackendLoaded()`
        // doesn't apply on the modular tree because AppleSTTEngine is
        // a separate actor from YoozSTTEngine.
        router.get("/v1/stt/status") { [self] _, _ -> Response in
            let active = await MainActor.run { currentSTTEngine }
            switch active {
            case .parakeet, .fastConformer:
                #if canImport(STTModule)
                guard await ModuleRegistry.shared.isBundled("stt") else {
                    return moduleNotBundled("stt")
                }
                let engine = YoozSTTEngine.shared
                let running = engine.isRunning
                let payload = STTStatusResponse(
                    loaded: running,
                    language: running ? engine.currentLanguage.rawValue : nil,
                    streaming: engine.isStreaming,
                    progress: engine.downloadProgress
                )
                return try jsonResponse(payload)
                #else
                return moduleNotBundled("stt")
                #endif
            case .appleSTT:
                #if canImport(AppleSTTModule)
                guard await ModuleRegistry.shared.isBundled("apple_stt") else {
                    return moduleNotBundled("apple_stt")
                }
                let engine = AppleSTTEngine.shared
                let loaded = await engine.isLoaded
                let lang = await engine.currentLanguage
                let streaming = await engine.isStreaming
                let payload = STTStatusResponse(
                    loaded: loaded,
                    language: loaded ? lang.rawValue : nil,
                    streaming: streaming,
                    progress: nil
                )
                return try jsonResponse(payload)
                #else
                return moduleNotBundled("apple_stt")
                #endif
            }
        }

        // STT: Load model (MLX path) — kept for back-compat with whisper
        // clients that call `STTClient.loadModel` ahead of the first batch.
        // Lite variants (no MLX) return 501 for this route; use
        // `POST /v1/stt/engine` to select Apple STT and the Apple backend
        // will load on the first batch request.
        router.post("/v1/stt/load") { [self] request, context in
            #if canImport(STTModule)
            guard await ModuleRegistry.shared.isBundled("stt") else {
                return moduleNotBundled("stt")
            }
            let body = try await request.decode(as: STTLoadRequest.self, context: context)
            let language: STTLanguage
            do {
                language = try parseLanguage(body.language)
            } catch let error as STTRequestError {
                return errorResponse(
                    status: .badRequest,
                    message: error.message,
                    code: error.code
                )
            }

            // When the active backend is qwen3, ensure the model
            // directory is materialized on disk before calling
            // start(). The fetch is opt-in via `allowFetch` (default
            // true for ergonomics; CI/tests can disable it).
            if sttEngine.currentBackend == .qwen3ASRPreview {
                let modelDir = Qwen3ASRModelFetcher.defaultModelDir
                let fetcher = Qwen3ASRModelFetcher.shared
                let ready = await fetcher.isModelDirReady(modelDir)
                if !ready {
                    let allow = body.allowFetch ?? true
                    if !allow {
                        return errorResponse(
                            status: .notFound,
                            message: "Qwen3-ASR model not on disk and "
                                + "allow_fetch=false; bring the model "
                                + "directory into place first.",
                            code: "model_not_found"
                        )
                    }
                    do {
                        let stream = await fetcher.download(into: modelDir)
                        for try await _ in stream {
                            // Drain the progress stream; HTTP clients
                            // get a single response after the fetch
                            // completes. Streaming progress to clients
                            // is a future enhancement (issue #59).
                        }
                    } catch let asrError as Qwen3ASRError {
                        // Route through the same demux as
                        // /v1/stt/batch so a fetch failure surfaces
                        // the same HTTP code regardless of which
                        // endpoint the client hit.
                        return mapQwen3BatchError(asrError)
                    } catch let fetchFailure as FetchFailure {
                        return mapFetchFailure(fetchFailure)
                    } catch {
                        return errorResponse(
                            status: .internalServerError,
                            message: "Model fetch failed: \(error.localizedDescription)",
                            code: "model_fetch_failed"
                        )
                    }
                } else {
                    // Even when the directory is ready, run the
                    // tokenizer prep step in case a previous run
                    // missed it (e.g. user pre-staged the files
                    // manually). Idempotent.
                    do {
                        try await Qwen3ASRTokenizerPrep.prepare(
                            modelDir: modelDir
                        )
                    } catch let asrError as Qwen3ASRError {
                        // Same mapping as the batch path so the wire
                        // shape is consistent
                        // (`tokenizer_validation_failed` /
                        // `model_fetch_failed` / etc).
                        return mapQwen3BatchError(asrError)
                    } catch {
                        return errorResponse(
                            status: .internalServerError,
                            message: "Tokenizer prep failed: \(error.localizedDescription)",
                            code: "tokenizer_prep_failed"
                        )
                    }
                }
            }

            // See `YoozSTTEngine.getModelDirectory` for resolution
            // order. `allowFetch` defaults to true; clients that want
            // to fail fast on a missing model pass `allow_fetch=false`.
            let allowFetch = body.allowFetch ?? true
            do {
                try await sttEngine.start(
                    language: language,
                    allowFetch: allowFetch
                )
                return try jsonResponse(STTStatusResponse(
                    loaded: true,
                    language: language.rawValue,
                    streaming: false,
                    progress: sttEngine.downloadProgress
                ))
            } catch {
                return mapSTTLoadError(error)
            }
            #else
            return moduleNotBundled("stt")
            #endif
        }

        // STT: Batch transcribe — routes to the currently selected engine.
        router.post("/v1/stt/batch") { [self] request, context in
            let active = await MainActor.run { currentSTTEngine }
            let body: BatchSTTRequest
            do {
                body = try await request.decode(as: BatchSTTRequest.self, context: context)
            } catch {
                return errorResponse(
                    status: .badRequest,
                    message: "Invalid request body: \(error.localizedDescription)",
                    code: "invalid_request"
                )
            }

            switch active {
            case .parakeet, .fastConformer:
                #if canImport(STTModule)
                guard await ModuleRegistry.shared.isBundled("stt") else {
                    return moduleNotBundled("stt")
                }
                let language: STTLanguage
                do {
                    language = try parseLanguage(body.language)
                } catch let error as STTRequestError {
                    return errorResponse(
                        status: .badRequest,
                        message: error.message,
                        code: error.code
                    )
                }
                let engine = YoozSTTEngine.shared
                do {
                    try await engine.start(language: language)
                } catch {
                    return errorResponse(
                        status: .internalServerError,
                        message: error.localizedDescription,
                        code: "model_load_failed"
                    )
                }
                let mode = AudioMode(rawValue: body.mode ?? "normal") ?? .normal
                let wantsAligned = body.aligned ?? false
                if wantsAligned {
                    // Parakeet already computes aligned tokens internally for
                    // the finalized/draft split; `batchTranscribeAligned`
                    // surfaces them. We still derive the full text from the
                    // token stream so the top-level `text` field stays
                    // consistent with the alignment.
                    let aligned: TranscriptionResult
                    do {
                        aligned = try await engine.batchTranscribeAligned(
                            samples: body.samples,
                            mode: mode
                        )
                    } catch YoozSTTError.notReady {
                        // Distinguishable from silent-input 200s: callers
                        // get an explicit 503 when the model never came up.
                        return errorResponse(
                            status: .serviceUnavailable,
                            message: "STT model is not loaded; call /v1/stt/load first",
                            code: "stt_not_loaded"
                        )
                    } catch {
                        return errorResponse(
                            status: .internalServerError,
                            message: error.localizedDescription,
                            code: "stt_aligned_failed"
                        )
                    }
                    let wireTokens = aligned.tokens.map { tok in
                        AlignedTokenWire(
                            text: tok.text,
                            start: tok.start,
                            end: tok.end
                        )
                    }
                    let fullText = aligned.text
                    return try jsonResponse(BatchSTTResponse(
                        text: fullText,
                        finalized: fullText,
                        draft: "",
                        language: language.rawValue,
                        tokens: wireTokens
                    ))
                }
                let result = await engine.batchTranscribe(samples: body.samples, mode: mode)
                return try jsonResponse(BatchSTTResponse(
                    text: result.text,
                    finalized: result.finalized,
                    draft: result.draft,
                    language: language.rawValue
                ))
                #else
                return moduleNotBundled("stt")
                #endif
            case .appleSTT:
                #if canImport(AppleSTTModule)
                guard await ModuleRegistry.shared.isBundled("apple_stt") else {
                    return moduleNotBundled("apple_stt")
                }
                let languageCode = body.language ?? "en"
                guard let language = AppleSTTLanguage.from(rawCode: languageCode) else {
                    return errorResponse(
                        status: .badRequest,
                        message: "Unknown language for Apple STT: \(languageCode)",
                        code: "invalid_language"
                    )
                }
                let engine = AppleSTTEngine.shared
                do {
                    try await engine.start(language: language)
                } catch {
                    return errorResponse(
                        status: .internalServerError,
                        message: error.localizedDescription,
                        code: "apple_stt_start_failed"
                    )
                }
                let wantsAligned = body.aligned ?? false
                do {
                    if wantsAligned {
                        let aligned = try await engine.batchTranscribeAligned(samples: body.samples)
                        let wireTokens = aligned.tokens.map { tok in
                            AlignedTokenWire(
                                text: tok.text,
                                start: tok.start,
                                end: tok.end
                            )
                        }
                        return try jsonResponse(BatchSTTResponse(
                            text: aligned.transcription,
                            finalized: aligned.transcription,
                            draft: "",
                            language: language.rawValue,
                            tokens: wireTokens
                        ))
                    }
                    let text = try await engine.batchTranscribe(samples: body.samples)
                    return try jsonResponse(BatchSTTResponse(
                        text: text,
                        finalized: text,
                        draft: "",
                        language: language.rawValue
                    ))
                } catch {
                    return errorResponse(
                        status: .internalServerError,
                        message: error.localizedDescription,
                        code: "apple_stt_failed"
                    )
                }
                #else
                return moduleNotBundled("apple_stt")
                #endif
            }
        }

        return router
    }

    // MARK: - WebSocket Router

    private func buildWebSocketRouter() -> Router<BasicWebSocketRequestContext> {
        let wsRouter = Router(context: BasicWebSocketRequestContext.self)
        let sttLogger = Logger(label: "live.yooz.engine.stt.stream")
        #if canImport(STTModule)
        // One sink instance per server lifetime — file handle stays
        // owned by the actor. Per-request sinks would race on the
        // file URL and confuse lifecycle ("is this sink alive?").
        let metricsSink: any STTMetricsSink = makeSTTMetricsSink(
            optedIn: EngineConfig.telemetryOptedIn,
            fileURL: EngineConfig.sttMetricsFileURL
        )
        #endif

        wsRouter.ws("/v1/stt/stream") { [weak self] inbound, outbound, context in
            let encoder = JSONEncoder()

            // Encode a WSSTTError and send it to the client. Used both
            // before the per-backend session setup (for early-fail
            // signalling) and inside the message loop. The `code` param
            // is optional so this matches the legacy single-arg call
            // sites; new code can pass typed `WSSTTErrorCode` values.
            func sendError(
                _ message: String, code: WSSTTErrorCode? = nil
            ) async {
                do {
                    let json = try encoder.encode(
                        WSSTTError(
                            type: "error", message: message, code: code
                        )
                    )
                    guard let jsonStr = String(data: json, encoding: .utf8) else { return }
                    try await outbound.write(.text(jsonStr))
                } catch {
                    sttLogger.error("Failed to send error message: \(error)")
                }
            }

            // Resolve the active engine on connect; `POST /v1/stt/engine`
            // during the session cancels the task via `sttStreamCancel`
            // (installed below), which flips `shouldAbort` and the loop
            // exits at the next iteration so we cleanly close the socket.
            let active: STTEngineCode = await MainActor.run { [weak self] in
                self?.currentSTTEngine ?? .parakeet
            }

            // Wire cooperative cancellation. The closure we register in
            // `sttStreamCancel` flips an `AbortBox`; the handler checks it on
            // each message turn. We keep the flag in a reference type so the
            // closure and the loop share mutable state without awaits.
            let abort = AbortBox()
            await MainActor.run { [weak self] in
                self?.sttStreamCancel = { abort.flag = true }
            }

            // Apple STT streaming is not implemented yet; return an explicit
            // error so clients fail fast instead of waiting on dropped audio.
            if active == .appleSTT {
                await sendError("Streaming STT is not implemented for Apple STT; use /v1/stt/batch")
                await MainActor.run { [weak self] in self?.sttStreamCancel = nil }
                return
            }

            #if canImport(STTModule)
            let sttEngine = YoozSTTEngine.shared
            var transcriber: StreamingTranscriber?
            // Per-connection streaming session for the qwen3
            // backend. Owns its own audio buffer; the underlying
            // backend actor serializes concurrent transcribe calls.
            var qwen3Session: Qwen3ASRStreamingSession?
            // Wall clock at session start for the telemetry record
            // emitted on `final`. Set when the qwen3 session is
            // configured (matches "stream start" semantics).
            var qwen3StreamStartedMs: UInt64?
            // True once we've sent the buffer-cap warning frame.
            // The frame is one-shot per connection so a long
            // dictation past the cap doesn't flood the client with
            // identical warnings.
            var qwen3BufferCapWarned = false

            // Encode a WSSTTResult to JSON and send via WebSocket
            func sendResult(_ result: WSSTTResult) async {
                do {
                    let json = try encoder.encode(result)
                    guard let jsonStr = String(data: json, encoding: .utf8) else {
                        sttLogger.error("Failed to convert encoded result to UTF-8 string")
                        return
                    }
                    try await outbound.write(.text(jsonStr))
                } catch {
                    sttLogger.error("Failed to send STT result: \(error)")
                }
            }

            // Send a one-shot non-fatal warning frame. `code` is a
            // typed enum; the wire still carries the snake_case
            // rawValue ("buffer_cap_reached", etc.) so clients
            // don't see a wire-shape change.
            func sendWarning(
                code: WSSTTWarningCode, message: String
            ) async {
                do {
                    let json = try encoder.encode(
                        WSSTTWarning(
                            type: "warning",
                            code: code,
                            message: message
                        )
                    )
                    guard let jsonStr = String(data: json, encoding: .utf8) else { return }
                    try await outbound.write(.text(jsonStr))
                } catch {
                    sttLogger.error("Failed to send warning: \(error)")
                }
            }

            // Build a Qwen3 telemetry record. `audioMs` is the
            // amount of audio actually consumed by the session
            // (post-cap if truncation happened). `errored` flips
            // when the stream tore down via an exception path so
            // dashboards can distinguish clean closes from drops.
            func buildQwen3Metric(
                audioMs: UInt32, fellBack: Bool, errored: Bool
            ) -> STTBackendMetrics {
                let endMs = DispatchTime.now().uptimeNanoseconds / 1_000_000
                let elapsed: UInt32
                if let started = qwen3StreamStartedMs {
                    elapsed = UInt32(min(endMs &- started, UInt64(UInt32.max)))
                } else {
                    elapsed = 0
                }
                let variant = STTBackendMetrics.defaultModelVariant(
                    for: .qwen3ASRPreview
                )
                return STTBackendMetrics(
                    backend: .qwen3ASRPreview,
                    modelVariant: variant,
                    audioDurationMs: audioMs,
                    timeToFirstTokenMs: nil,
                    endToEndLatencyMs: elapsed,
                    hardwareClass: SystemHardwareClassResolver().resolve(),
                    fellBackFromPreview: fellBack,
                    streamAborted: errored,
                    timestampUTC: Date()
                )
            }

            sttLogger.info("STT WebSocket client connected")

            // Use message-level API to handle fragmented frames automatically.
            // Max message size: 10MB (enough for ~2.7 min of 16kHz Float32 audio).
            //
            // We wrap the loop in do/catch so a thrown error inside
            // `inbound.messages(...)` (oversized frame, abrupt
            // disconnect mid-frame, framer decode error) does NOT
            // bypass the cleanup block below — without this, the
            // qwen3 session never gets `discard()`-ed, no telemetry
            // record fires, and the audio buffer (up to ~19 MB at
            // the 5-min cap) lingers on the actor's executor.
            do {
            for try await message in inbound.messages(maxSize: 10 * 1024 * 1024) {
                if abort.flag {
                    await sendError("Stream cancelled: STT engine was switched")
                    break
                }
                switch message {
                case .text(let text):
                    // Config message: {"type":"config","language":"en","mode":"normal"}
                    let config: WSSTTConfig
                    do {
                        guard let data = text.data(using: .utf8) else {
                            sttLogger.warning("Received non-UTF8 text message")
                            continue
                        }
                        config = try JSONDecoder().decode(WSSTTConfig.self, from: data)
                    } catch {
                        sttLogger.warning("Invalid text message: \(error)")
                        await sendError(
                            "Invalid message format",
                            code: .invalidMessageFormat
                        )
                        continue
                    }

                    if config.type == "config" {
                        let languageCode = config.language ?? "en"
                        guard let language = STTLanguage.fromCode(languageCode) else {
                            await sendError(
                                "Unknown language: \(languageCode)",
                                code: .unknownLanguage
                            )
                            continue
                        }

                        guard language.isImplemented else {
                            await sendError(
                                "Language not implemented: \(language.displayName)",
                                code: .languageNotImplemented
                            )
                            continue
                        }

                        // Branch on the active backend at config
                        // time. The qwen3 path opens a streaming
                        // session against the (already-loaded)
                        // backend actor; the Parakeet/FastConformer
                        // path creates a per-connection
                        // `StreamingTranscriber` exactly as before.
                        if sttEngine.currentBackend == .qwen3ASRPreview {
                            // Allow only languages the backend
                            // declares; the engine's `start` would
                            // otherwise refuse later anyway, but
                            // surfacing the error here keeps the
                            // protocol response immediate.
                            guard
                                STTBackendID.qwen3ASRPreview.supportedLanguages
                                    .contains(language)
                            else {
                                await sendError(
                                    "Language not supported by qwen3_asr_preview: "
                                        + language.displayName,
                                    code: .languageNotSupportedByBackend
                                )
                                continue
                            }

                            do {
                                try await sttEngine.start(language: language)
                            } catch {
                                await sendError(
                                    "Failed to load model: \(error.localizedDescription)",
                                    code: .modelLoadFailed
                                )
                                continue
                            }

                            // Discard any prior session — clients
                            // sending a second `config` reset the
                            // transcript. Mirrors how the legacy
                            // path overwrites `transcriber`.
                            if let existing = qwen3Session {
                                await existing.discard()
                            }
                            qwen3Session = Qwen3ASRStreamingSession(
                                languageHint: language.qwen3LanguageHint
                            )
                            qwen3StreamStartedMs =
                                DispatchTime.now().uptimeNanoseconds / 1_000_000
                            // Each `config` message starts a brand-new
                            // session with an empty buffer; the cap
                            // warning is one-shot per session, not per
                            // WS connection, so the prior session's
                            // flag must not silence the next session's
                            // truncation.
                            qwen3BufferCapWarned = false

                            do {
                                let ready = try encoder.encode(
                                    WSSTTReady(type: "ready", language: languageCode)
                                )
                                if let readyStr = String(data: ready, encoding: .utf8) {
                                    try await outbound.write(.text(readyStr))
                                }
                            } catch {
                                sttLogger.error("Failed to send ready message: \(error)")
                            }
                            sttLogger.info(
                                "STT stream configured: language=\(languageCode), backend=qwen3_asr_preview"
                            )
                            continue
                        }

                        do {
                            try await sttEngine.start(language: language)
                        } catch {
                            await sendError(
                                "Failed to load model: \(error.localizedDescription)",
                                code: .modelLoadFailed
                            )
                            continue
                        }

                        let mode = AudioMode(rawValue: config.mode ?? "normal") ?? .normal
                        transcriber = sttEngine.createBatchTranscriber(mode: mode)

                        do {
                            let ready = try encoder.encode(WSSTTReady(type: "ready", language: languageCode))
                            if let readyStr = String(data: ready, encoding: .utf8) {
                                try await outbound.write(.text(readyStr))
                            }
                        } catch {
                            sttLogger.error("Failed to send ready message: \(error)")
                        }
                        sttLogger.info("STT stream configured: language=\(languageCode), mode=\(mode.rawValue)")
                    }

                case .binary(var buffer):
                    let byteCount = buffer.readableBytes
                    // Reject obviously-malformed frames early. The
                    // protocol expects a multiple of `Float32`-sized
                    // bytes; partial-sample frames are dropped with a
                    // typed error so the client can correct rather
                    // than silently corrupting the buffer.
                    if byteCount % MemoryLayout<Float>.size != 0 {
                        await sendError(
                            "Audio frame byte count (\(byteCount)) is not a "
                                + "multiple of Float32 size; expected raw "
                                + "Float32 PCM at 16 kHz mono.",
                            code: .invalidAudioFrame
                        )
                        continue
                    }
                    let sampleCount = byteCount / MemoryLayout<Float>.size
                    guard sampleCount > 0 else { continue }

                    // Safe copy into aligned Float array via raw byte copy
                    let samples: [Float] = buffer.withUnsafeReadableBytes { ptr in
                        [Float](unsafeUninitializedCapacity: sampleCount) { dest, initializedCount in
                            _ = UnsafeMutableRawBufferPointer(dest).copyBytes(
                                from: UnsafeRawBufferPointer(ptr).prefix(sampleCount * MemoryLayout<Float>.size)
                            )
                            initializedCount = sampleCount
                        }
                    }

                    if let session = qwen3Session {
                        let outcome: Qwen3ASRStreamingSession.PushOutcome
                        do {
                            outcome = try await session.push(samples: samples)
                        } catch let sessionError as Qwen3ASRStreamingSession.SessionError {
                            await sendError(
                                "Streaming session error: \(sessionError.description)",
                                code: .sessionError
                            )
                            continue
                        } catch {
                            await sendError(
                                "Streaming session error: \(error.localizedDescription)",
                                code: .sessionError
                            )
                            continue
                        }
                        // Detect a buffer-cap hit: push() silently
                        // truncates past the soft cap. Fire a
                        // one-shot warning frame the first time
                        // truncation happens — including the very
                        // first chunk that partially crosses the cap,
                        // not just subsequent fully-rejected chunks.
                        // The flag is reset when the client sends a
                        // second `config` (transcript reset), so a
                        // long dictation that re-configs mid-stream
                        // still gets a warning per session.
                        if outcome.truncated && !qwen3BufferCapWarned {
                            qwen3BufferCapWarned = true
                            await sendWarning(
                                code: .bufferCapReached,
                                message:
                                    "Audio buffer reached the streaming "
                                    + "cap. Additional audio is being "
                                    + "discarded. The transcript on "
                                    + "`final` reflects the buffered "
                                    + "audio only."
                            )
                        }
                        // Heartbeat partial — qwen3's streaming model
                        // doesn't expose intermediate text without a
                        // re-prefill on every chunk, so the partial
                        // text fields stay empty until `final`.
                        await sendResult(WSSTTResult(
                            type: "partial",
                            text: "",
                            finalized: "",
                            draft: ""
                        ))
                        continue
                    }

                    guard let transcriber = transcriber else {
                        sttLogger.warning("Audio received before config message")
                        continue
                    }

                    let result = transcriber.addAudio(samples: samples)

                    await sendResult(WSSTTResult(
                        type: "partial",
                        text: result.text,
                        finalized: result.finalized,
                        draft: result.draft
                    ))
                }
            }
            } catch is CancellationError {
                // Outer Task was cancelled (server stop, peer
                // dropped, etc.). Run the same cleanup as a clean
                // disconnect so the qwen3 session is `discard()`-ed
                // and a metric record fires with the audio actually
                // consumed before the cancel.
                if let session = qwen3Session {
                    let audioMs = await session.audioDurationMs
                    await session.discard()
                    qwen3Session = nil
                    let metric = buildQwen3Metric(
                        audioMs: audioMs,
                        fellBack: false,
                        errored: true
                    )
                    await metricsSink.record(metric)
                }
                sttLogger.info("STT WebSocket cancelled")
                throw CancellationError()
            } catch {
                // The message loop threw — oversized frame past the
                // 10 MB cap, abrupt connection drop with a partial
                // frame in flight, framer decode error. Run the
                // qwen3 session cleanup we would otherwise miss,
                // emit one error frame to the client (best-effort),
                // and emit a metric record so dashboards see the
                // drop rate.
                sttLogger.error(
                    "STT WebSocket loop threw: \(String(describing: error))"
                )
                await sendError(
                    "Stream aborted: \(error.localizedDescription)",
                    code: .streamAborted
                )
                if let session = qwen3Session {
                    let audioMs = await session.audioDurationMs
                    await session.discard()
                    qwen3Session = nil
                    let metric = buildQwen3Metric(
                        audioMs: audioMs,
                        fellBack: false,
                        errored: true
                    )
                    await metricsSink.record(metric)
                }
                sttLogger.info("STT WebSocket client disconnected (error)")
                return
            }

            // Client disconnected cleanly; finalize whichever
            // session is active.
            if let session = qwen3Session {
                do {
                    let final = try await session.finalize()
                    await sendResult(WSSTTResult(
                        type: "final",
                        text: final.text,
                        finalized: final.text,
                        draft: ""
                    ))
                    // Telemetry — opt-in via EngineConfig. The
                    // hoisted sink lives for the server's lifetime;
                    // a `DiscardingMetricsSink` swallows the record
                    // when the user hasn't enabled local telemetry.
                    let metric = buildQwen3Metric(
                        audioMs: final.audioDurationMs,
                        fellBack: false,
                        errored: false
                    )
                    await metricsSink.record(metric)
                } catch Qwen3ASRStreamingSession.SessionError.empty {
                    // Client disconnected without sending audio. Emit
                    // an empty `final` for protocol symmetry.
                    await sendResult(WSSTTResult(
                        type: "final",
                        text: "",
                        finalized: "",
                        draft: ""
                    ))
                } catch let sessionError as Qwen3ASRStreamingSession.SessionError {
                    sttLogger.error(
                        "Qwen3 streaming finalize failed: \(sessionError.description)"
                    )
                    await sendError(
                        "Streaming finalize failed: \(sessionError.description)",
                        code: .finalizeFailed
                    )
                    let metric = buildQwen3Metric(
                        audioMs: await session.audioDurationMs,
                        fellBack: false,
                        errored: true
                    )
                    await metricsSink.record(metric)
                } catch {
                    sttLogger.error(
                        "Qwen3 streaming finalize failed: \(error.localizedDescription)"
                    )
                    await sendError(
                        "Streaming finalize failed: \(error.localizedDescription)",
                        code: .finalizeFailed
                    )
                    let metric = buildQwen3Metric(
                        audioMs: await session.audioDurationMs,
                        fellBack: false,
                        errored: true
                    )
                    await metricsSink.record(metric)
                }
            } else if let transcriber = transcriber {
                let finalResult = transcriber.finalize()
                await sendResult(WSSTTResult(
                    type: "final",
                    text: finalResult.text,
                    finalized: finalResult.finalized,
                    draft: finalResult.draft
                ))
            }

            sttLogger.info("STT WebSocket client disconnected")
            await MainActor.run { [weak self] in self?.sttStreamCancel = nil }
            #else
            // Lite variant: no MLX STT linked, so streaming isn't available.
            await sendError("Streaming STT is not bundled in this build variant")
            await MainActor.run { [weak self] in self?.sttStreamCancel = nil }
            #endif
        }

        return wsRouter
    }
}

enum STTRequestError: Error {
    case invalidLanguage(String)
    case notImplemented(String)

    var message: String {
        switch self {
        case .invalidLanguage(let code): return "Unknown language: \(code)"
        case .notImplemented(let name): return "Language not implemented: \(name)"
        }
    }

    var code: String {
        switch self {
        case .invalidLanguage: return "invalid_language"
        case .notImplemented: return "not_implemented"
        }
    }
}

// MARK: - WebSocket cancel box

/// Shared single-flag abort signal for WebSocket STT sessions. Writes come
/// from the `sttStreamCancel` closure set by `POST /v1/stt/engine`; reads
/// happen at each loop turn in the WebSocket handler. The class is
/// intentionally unchecked-Sendable — it holds a single `Bool` and the
/// WebSocket handler never mutates it, so the "only flips" contract is
/// safe without a lock.
final class AbortBox: @unchecked Sendable {
    var flag: Bool = false
}

// MARK: - STT Engine selector

/// Server-side mirror of `STTEngineType` in the client SDK. Kept separate so
/// the server target doesn't import `YoozEngineClient` (client depends on
/// server, not the other way around). Raw values match the SDK so JSON
/// encode/decode is round-trip compatible.
enum STTEngineCode: String, Codable, Sendable {
    case parakeet
    case fastConformer = "fast_conformer"
    case appleSTT = "apple_stt"

    var isMLXBacked: Bool {
        switch self {
        case .parakeet, .fastConformer: return true
        case .appleSTT: return false
        }
    }

    /// `true` when the backend performs its own endpointing. Pure — kept off
    /// `APIServer` so the `@MainActor`-isolated router helpers don't need to
    /// `await` just to read a capability flag.
    var hasBuiltInVAD: Bool {
        switch self {
        case .parakeet, .fastConformer: return false
        case .appleSTT: return true
        }
    }
}

/// Wire body for `POST /v1/stt/engine`.
struct STTEngineSwitchServerRequest: Decodable, Sendable {
    let engine: STTEngineCode
}

/// Wire body for `GET /v1/stt/engine`.
struct STTEngineServerResponse: Encodable, Sendable {
    let current: STTEngineCode
    let available: [STTEngineCode]
    let hasBuiltInVAD: Bool
}

extension APIServer {
    /// Default backend for a fresh server. Preference order:
    /// 1. `.parakeet` when MLX STT is linked (full/whisper),
    /// 2. `.appleSTT` when the Apple module is linked but MLX isn't (lite).
    /// 3. `.parakeet` as a last resort (will 501 at call time — caller will
    ///    see the module_not_bundled response from the route handler).
    nonisolated static func defaultSTTEngine() -> STTEngineCode {
        #if canImport(STTModule)
        return .parakeet
        #elseif canImport(AppleSTTModule)
        return .appleSTT
        #else
        return .parakeet
        #endif
    }

    /// STT backends linked into this build variant, in a stable order.
    nonisolated static func availableSTTEngines() -> [STTEngineCode] {
        var out: [STTEngineCode] = []
        #if canImport(STTModule)
        out.append(.parakeet)
        out.append(.fastConformer)
        #endif
        #if canImport(AppleSTTModule)
        out.append(.appleSTT)
        #endif
        return out
    }

    #if canImport(LLMModule)
    /// Build a single-model entry for the `/v1/llm/*` responses.
    /// `latencyHintMs` is a best-effort per-model baseline drawn from
    /// LLMModelType.description (e.g. "~200ms"). Keeps the JSON wire
    /// shape homogeneous across GET + preload + unload responses.
    nonisolated static func infoEntry(
        for modelType: LLMModelType,
        loaded: Bool
    ) -> LLMModelInfoServer {
        let hint: Int
        switch modelType {
        case .yoozLight: hint = 200
        case .yoozQuality: hint = 490
        }
        return LLMModelInfoServer(
            id: modelType.rawValue,
            displayName: modelType.displayName,
            sizeBytes: modelType.estimatedSize,
            loaded: loaded,
            latencyHintMs: hint
        )
    }

    /// Build the full `GET /v1/llm/models` body. Async because the load
    /// state + preferred-model flag live inside the TouchUpEngine actor.
    nonisolated static func buildLLMModelsResponse() async -> LLMModelsServerResponse {
        let info = await TouchUpEngine.shared.getModelInfo()
        let current = await TouchUpEngine.shared.preferredModel.rawValue
        return LLMModelsServerResponse(
            current: current,
            available: [
                infoEntry(for: .yoozLight, loaded: info.light.isLoaded),
                infoEntry(for: .yoozQuality, loaded: info.quality.isLoaded)
            ]
        )
    }
    #endif
}
