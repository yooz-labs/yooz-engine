#if canImport(AppleSTTModule)
import AppleSTTModule
#endif
import EngineCore
import Foundation
import GrammarModule
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
        logger.info("Yooz Engine started on \(host):\(port)")
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

    // MARK: - HTTP Router

    private func buildRouter() -> Router<BasicRequestContext> {
        let router = Router()

        // Health
        router.get("/v1/health") { _, _ in
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
            return HealthResponse(
                status: "ok",
                version: EngineConfig.version,
                modules: EngineModules(
                    stt: sttLoaded,
                    llm: llmLoaded,
                    touchup: touchupReady,
                    grammar: GrammarEngine.shared.isAvailable,
                    vad: vadLoaded,
                    tts: false
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
                    modelType: modelType
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

            let result = await TouchUpEngine.shared.process(
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

        // STT: Engine picker — current + available + capability bits.
        // Unlike the other `/v1/stt/*` routes we don't gate on the `stt`
        // module registry entry: a Lite build serves this endpoint with the
        // `apple_stt` module, returning `available: ["apple_stt"]`.
        router.get("/v1/stt/engine") { [self] _, _ -> Response in
            let current = await currentSTTEngine
            let available = APIServer.availableSTTEngines()
            let payload = STTEngineServerResponse(
                current: current,
                available: available,
                hasBuiltInVAD: current.hasBuiltInVAD
            )
            return try jsonResponse(payload)
        }

        // STT: Engine switch. Cancels any in-flight WebSocket stream, then
        // updates `currentSTTEngine`. Returns 400 on an unknown engine, 501
        // `module_not_bundled` when the target backend isn't linked in this
        // variant.
        router.post("/v1/stt/engine") { [self] request, context in
            let body: STTEngineSwitchServerRequest
            do {
                body = try await request.decode(as: STTEngineSwitchServerRequest.self, context: context)
            } catch {
                return errorResponse(
                    status: .badRequest,
                    message: "Invalid request body: \(error.localizedDescription)",
                    code: "invalid_request"
                )
            }
            guard APIServer.availableSTTEngines().contains(body.engine) else {
                let moduleName = (body.engine == .appleSTT) ? "apple_stt" : "stt"
                return moduleNotBundled(moduleName)
            }
            // Mutations to `sttStreamCancel` and `currentSTTEngine` touch
            // `@MainActor`-isolated state from this `@Sendable` handler,
            // so the updates run inside `MainActor.run`.
            await MainActor.run {
                self.sttStreamCancel?()
                self.sttStreamCancel = nil
                self.currentSTTEngine = body.engine
            }
            let payload = STTEngineServerResponse(
                current: body.engine,
                available: APIServer.availableSTTEngines(),
                hasBuiltInVAD: body.engine.hasBuiltInVAD
            )
            return try jsonResponse(payload)
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

        // STT: Status
        router.get("/v1/stt/status") { [self] _, _ -> Response in
            let active = await currentSTTEngine
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
                    streaming: engine.isStreaming
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
                    streaming: streaming
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
            do {
                try await YoozSTTEngine.shared.start(language: language)
                return try jsonResponse(STTStatusResponse(
                    loaded: true,
                    language: language.rawValue,
                    streaming: false
                ))
            } catch {
                return errorResponse(
                    status: .internalServerError,
                    message: error.localizedDescription,
                    code: "load_failed"
                )
            }
            #else
            return moduleNotBundled("stt")
            #endif
        }

        // STT: Batch transcribe — routes to the currently selected engine.
        router.post("/v1/stt/batch") { [self] request, context in
            let active = await currentSTTEngine
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

        wsRouter.ws("/v1/stt/stream") { [weak self] inbound, outbound, context in
            let encoder = JSONEncoder()

            // Encode a WSSTTError and send it to the client.
            func sendError(_ message: String) async {
                do {
                    let json = try encoder.encode(WSSTTError(type: "error", message: message))
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

            sttLogger.info("STT WebSocket client connected")

            // Use message-level API to handle fragmented frames automatically
            // Max message size: 10MB (enough for ~2.7 min of 16kHz Float32 audio)
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
                        await sendError("Invalid message format")
                        continue
                    }

                    if config.type == "config" {
                        let languageCode = config.language ?? "en"
                        guard let language = STTLanguage.fromCode(languageCode) else {
                            await sendError("Unknown language: \(languageCode)")
                            continue
                        }

                        guard language.isImplemented else {
                            await sendError("Language not implemented: \(language.displayName)")
                            continue
                        }

                        do {
                            try await sttEngine.start(language: language)
                        } catch {
                            await sendError("Failed to load model: \(error.localizedDescription)")
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
                    // Audio data: raw Float32 samples at 16kHz
                    guard let transcriber = transcriber else {
                        sttLogger.warning("Audio received before config message")
                        continue
                    }

                    let byteCount = buffer.readableBytes
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

                    let result = transcriber.addAudio(samples: samples)

                    await sendResult(WSSTTResult(
                        type: "partial",
                        text: result.text,
                        finalized: result.finalized,
                        draft: result.draft
                    ))
                }
            }

            // Client disconnected; finalize if a transcriber is active
            if let transcriber = transcriber {
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
