import AppleSTTModule
import EngineCore
import Foundation
import GrammarModule
import LLMModule
import OSLog
import STTModule
import VADModule
import YoozEngineClient

/// `EngineTransport` that serves the `YoozEngineClient` SDK surface by calling
/// the engine module actors directly — no loopback socket (epic #192 Phase 2).
///
/// A standalone App Store app links `YoozEngineInProcess`, builds its client as
/// `YoozEngineClient(transport: InProcessTransport())`, and gets the identical
/// SDK API in-sandbox.
///
/// ## Scope
///
/// Implemented (the full standalone-app surface): `GET /v1/health`,
/// `GET /v1/modules`; STT `GET /v1/stt/{status,languages,engine}`,
/// `POST /v1/stt/{batch,engine,load}`, and streaming via `openSTTStream`;
/// `POST /v1/grammar/check`; `POST /v1/vad/detect`; LLM
/// `GET /v1/llm/{status,models}`,
/// `POST /v1/llm/{generate,model,preload,unload,clear-cache}`;
/// TouchUp `GET /v1/touchup/models`, `POST /v1/touchup{,/model}`;
/// `POST /v1/session/{begin,end}` — the per-recording session-reset boundary
/// (engine issue #114 / #222), fanned out via the shared
/// `EngineCore.SessionCoordinator` so this transport and the loopback server
/// behave identically; `GET /v1/state` and streaming via `openEvents`
/// (engine#226) — the cross-module snapshot + live event feed backing
/// `EngineStateStore`, sourced from the same `EngineEventBus` the loopback
/// WS route reads from.
///
/// Reported as `unsupportedOperation`:
///   - **Streaming qwen3 preview** — loopback/dev only (unstable; engine#154).
///   - **Infinite** (`/v1/infinite/*`) — its consumer is the loopback host.
///
/// `RouteParityAllowlist.loopbackOnly` (EngineCore, `RouteManifest.swift`) is
/// the reviewable, tested source of truth for this list — this comment is a
/// human-readable summary of it, not a second authority; update the allowlist
/// first and let this comment follow (#223).
///
/// Each handler decodes the same wire body the SDK sub-client encoded and emits
/// the same wire shape the sub-client decodes, so the SDK is byte-for-byte
/// agnostic to which transport served it.
public final class InProcessTransport: EngineTransport {
    /// Non-routable placeholder — there is no socket. Present only because
    /// `baseURL` is part of the long-standing public SDK surface.
    public let baseURL = URL(string: "inprocess://engine")!
    public let port = 0

    private let host: EngineInProcessHost

    /// STT batch forensics (yooz-labs/yooz-whisper#280): same
    /// subsystem/category as `XPCServiceHandler`'s request-forensics logger
    /// so `log stream --predicate 'subsystem == "live.yooz.engine"'` shows
    /// the full XPC request -> batch handler timeline in one stream. This
    /// path is reached both in-process and (via `XPCServiceHandler` ->
    /// `InProcessTransport`) over XPC, so the log lines carry no
    /// transport tag of their own — cross-reference against the XPC
    /// request logger's timestamps to attribute elapsed time to
    /// marshaling vs. transcription.
    private static let sttLogger = Logger(subsystem: "live.yooz.engine", category: "xpc")

    /// The typed endpoint table (engine#225 Phase B): converted route
    /// families dispatch through this before the legacy switches below. The
    /// handler bodies are the same closures `APIServer` registers on the
    /// loopback router — declared once, in the module that owns each family
    /// (`SessionEndpoints` in EngineCore; `TouchUpEndpoints` /
    /// `ModelManagementEndpoints` in LLMModule).
    static let endpointTable = EndpointTable.trusted(
        SessionEndpoints.endpoints()
            + TouchUpEndpoints.pickerEndpoints()
            + ModelManagementEndpoints.endpoints(
                activeSTTRepoDirName: { InProcessTransport.activeSTTRepoDirName() }
            )
            + EngineStateEndpoints.endpoints()
            + sttDownloadEndpoints()
    )


    /// STT download/cancel table entries (engine#291), gated on the STT
    /// module being linked (the Lite variant ships without it).
    private static func sttDownloadEndpoints() -> [Endpoint] {
        #if canImport(STTModule)
        return STTDownloadEndpoints.endpoints()
        #else
        return []
        #endif
    }

    public init(host: EngineInProcessHost = .shared) {
        self.host = host
    }

    /// Dispatch a request through the endpoint table. Returns nil when the
    /// route is not (yet) table-converted, so the caller falls through to
    /// its legacy switch. `WireError`s are rethrown as
    /// `YoozEngineError.serverError` — the same typed error surface this
    /// transport has always exposed to the SDK.
    private func dispatchViaTable(
        _ method: RouteMethod, _ path: String, body: Data = Data()
    ) async throws -> Data? {
        guard let (endpoint, params) = Self.endpointTable.match(
            method: method, path: route(path)
        ) else { return nil }
        do {
            let response = try await endpoint.handler(
                WireRequest(body: body, pathParameters: params, query: Self.query(of: path))
            )
            return response.body
        } catch let error as WireError {
            throw YoozEngineError.serverError(
                statusCode: error.status, code: error.code, message: error.message
            )
        }
    }

    /// The raw query substring of `path` (no leading `?`), or nil.
    private static func query(of path: String) -> String? {
        guard let q = path.firstIndex(of: "?") else { return nil }
        return String(path[path.index(after: q)...])
    }

    /// Await `task`, converting a `LoadDeadlineExceeded` timeout into a
    /// typed `YoozEngineError` (PR #255 review, finding I4). The raw
    /// `LoadDeadlineExceeded` struct is not a `YoozEngineError`, so
    /// `XPCErrorBridge.toNSError` falls through to its non-typed-error
    /// branch and the client-side `toYoozEngineError` collapses it to
    /// `.engineNotReachable` — a `modelLoadDeadlineSeconds` (600s) load
    /// timeout would then look exactly like a dead service to the caller,
    /// which can trigger a reconnect storm instead of a "still loading"
    /// message. Shared by every call site that bounds an `enqueueLoad`
    /// task with `awaitLoadTask` (`handleBatch`, `openSTTStream`,
    /// `handleSTTLoad`) so the mapping can't drift between them.
    ///
    /// Internal (not `private`), not for any external caller — only so
    /// `Tests/YoozEngineInProcessTests` can exercise the conversion
    /// directly via `@testable import` (PR #255 review, finding I5).
    func awaitLoadOrTypedDeadline(
        _ task: Task<Void, Error>, deadlineSeconds: Double
    ) async throws {
        do {
            try await awaitLoadTask(task, deadlineSeconds: deadlineSeconds)
        } catch is LoadDeadlineExceeded {
            throw YoozEngineError.serverError(
                statusCode: 504, code: "load_deadline_exceeded",
                message: "STT model load did not complete within \(Int(deadlineSeconds))s"
            )
        }
    }

    public func connect() async throws {
        await host.bootstrap()
    }

    public func isReachable() async throws -> Bool {
        // Modules are linked in-process; once bootstrapped the engine is always
        // reachable. Bootstrap is idempotent.
        await host.bootstrap()
        return true
    }

    public func get(_ path: String) async throws -> Data {
        try await connect()
        if let tableResponse = try await dispatchViaTable(.get, path) {
            return tableResponse
        }
        switch route(path) {
        case "/v1/health":
            return try await handleHealth()
        case "/v1/modules":
            return try await handleModules()
        case "/v1/stt/status":
            return try await handleSTTStatus()
        case "/v1/stt/languages":
            return try await handleSTTLanguages()
        case "/v1/stt/engine":
            return try await handleSTTEngine()
        case "/v1/llm/status":
            return try await handleLLMStatus()
        case "/v1/llm/models":
            return try await handleLLMModels()
        default:
            throw YoozEngineError.unsupportedOperation(operation: "GET \(route(path))")
        }
    }

    // Handler contract (#223): every POST handler dispatched below MUST
    // validate its request (JSON-decode the body, or an equivalent cheap
    // guard) BEFORE doing any disk/network/model work. RouteParityTests
    // drives each route with a minimal invalid request and relies on that
    // fail-fast gate to prove dispatch reachability without side effects.
    // Handlers with no request body to gate on (`/v1/models/cleanup`,
    // `/v1/session/*`) DO run their real work on every test sweep, so they
    // must stay cheap and side-effect-safe in a bare test process — the test
    // additionally redirects the HF cache env vars so cleanup can never
    // touch the machine's real model cache. `/v1/llm/clear-cache`
    // (engine#299) is the same shape for an EMPTY body specifically: an
    // absent `model` is valid input (clear every loaded tier), not a
    // validation failure, so RouteParityTests's empty-body probe also runs
    // the real (but in-memory, side-effect-safe) clear path rather than
    // failing fast on a decode error.
    public func post(_ path: String, body: Data) async throws -> Data {
        try await connect()
        if let tableResponse = try await dispatchViaTable(.post, path, body: body) {
            return tableResponse
        }
        switch route(path) {
        case "/v1/grammar/check":
            return try await handleGrammar(body)
        case "/v1/vad/detect":
            return try await handleVAD(body)
        case "/v1/stt/batch":
            return try await handleBatch(body)
        case "/v1/stt/load":
            return try await handleSTTLoad(body, wait: Self.parseWaitQuery(path))
        case "/v1/stt/engine":
            return try await handleSetSTTEngine(body)
        case "/v1/llm/generate":
            return try await handleLLM(body)
        case "/v1/llm/model":
            return try await handleSetLLMModel(body)
        case "/v1/llm/preload":
            return try await handleLLMPreload(body)
        case "/v1/llm/unload":
            return try await handleLLMUnload(body)
        case "/v1/llm/clear-cache":
            return try await handleLLMClearCache(body)
        case "/v1/touchup":
            return try await handleTouchUp(body)
        default:
            throw YoozEngineError.unsupportedOperation(operation: "POST \(route(path))")
        }
    }

    public func delete(_ path: String) async throws -> Data {
        try await connect()
        if let tableResponse = try await dispatchViaTable(.delete, path) {
            return tableResponse
        }
        throw YoozEngineError.unsupportedOperation(operation: "DELETE \(route(path))")
    }

    /// `/v1/events` (engine#226): subscribe directly to the shared
    /// `EngineEventBus` — no socket, no translation layer. The same
    /// `EngineEvent`s the loopback WS route serializes are handed to the
    /// caller as-is.
    @available(macOS 14.0, iOS 17.0, *)
    public func openEvents() async throws -> AsyncStream<EngineEvent> {
        try await connect()
        return await EngineEventBus.shared.subscribe()
    }

    @available(macOS 14.0, iOS 17.0, *)
    public func openSTTStream(language: String, mode: String) async throws -> any STTStreamSession {
        try await connect()
        guard let lang = STTModule.STTLanguage.fromCode(language) else {
            throw YoozEngineError.serverError(
                statusCode: 400,
                code: "invalid_language",
                message: "Unknown STT language '\(language)'"
            )
        }
        // Unknown mode falls back to .normal for parity with the loopback WS
        // handler (which also coerces). normal/whispered are the only cases and
        // share rawValues across the SDK and engine enums.
        let audioMode = STTModule.AudioMode(rawValue: mode) ?? .normal

        switch YoozSTTEngine.shared.currentBackend {
        case .appleSTT:
            guard let appleLang = AppleSTTLanguage.from(rawCode: language) else {
                throw YoozEngineError.serverError(
                    statusCode: 400, code: "invalid_language",
                    message: "Language '\(language)' is not supported by Apple STT"
                )
            }
            try await AppleSTTEngine.shared.start(language: appleLang)
            return InProcessSTTStreamSession(backend: .apple(AppleSTTEngine.shared))

        case .qwen3ASRPreview:
            // The preview backend is loopback/dev only (unstable; engine#154).
            throw YoozEngineError.unsupportedOperation(operation: "streaming qwen3 preview")

        case .parakeet, .fastConformer:
            // Await any in-flight warmup first (engine#252, PR #255 review
            // finding C2) so a stream open racing it runs on already-
            // compiled kernels instead of contending with the warmup's own
            // JIT-compiling dummy transcription. No-op if none is in
            // flight.
            await YoozSTTEngine.shared.awaitWarmupIfNeeded(
                deadlineSeconds: EngineConfig.modelLoadDeadlineSeconds
            )
            // Bound the (possibly first-run) load so a stream open can't hang
            // indefinitely; routes through the same cancellable `enqueueLoad`
            // primitive the load endpoint uses.
            let task = await YoozSTTEngine.shared.enqueueLoad(language: lang) {
                try await YoozSTTEngine.shared.start(language: lang)
            }
            try await awaitLoadOrTypedDeadline(
                task, deadlineSeconds: EngineConfig.modelLoadDeadlineSeconds
            )
            guard let transcriber = YoozSTTEngine.shared.createBatchTranscriber(mode: audioMode) else {
                throw YoozEngineError.serverError(
                    statusCode: 503,
                    code: "stt_not_loaded",
                    message: "STT model failed to load for language '\(language)'"
                )
            }
            return InProcessSTTStreamSession(backend: .parakeet(transcriber))
        }
    }

    // MARK: - Routing

    /// Strip a `?query` suffix so routing matches on the path only.
    private func route(_ path: String) -> String {
        if let q = path.firstIndex(of: "?") {
            return String(path[..<q])
        }
        return path
    }

    /// Parse the `?wait=...` flag a load caller may append. Mirrors the loopback
    /// `APIServer.parseWaitQuery`: `loadModel` posts `/v1/stt/load?wait=true`
    /// (blocking), `loadModelAsync` posts `/v1/stt/load` (fire-and-forget). The
    /// in-process `route()` strips the query for matching, so the original path
    /// is parsed here to recover the flag.
    static func parseWaitQuery(_ path: String) -> Bool {
        guard let q = path.firstIndex(of: "?") else { return false }
        let query = path[path.index(after: q)...]
        for pair in query.split(separator: "&") {
            let kv = pair.split(separator: "=", maxSplits: 1)
            guard kv.first == "wait" else { continue }
            if kv.count == 1 { return true }
            return kv[1] == "true" || kv[1] == "1"
        }
        return false
    }

    // MARK: - Handlers

    private func handleHealth() async throws -> Data {
        let grammarReady = GrammarEngine.shared.isAvailable
        let llmReady = await TouchUpEngine.shared.isPreloaded
        let vadReady = await VADEngine.shared.isLoaded
        let sttReady = YoozSTTEngine.shared.isRunning

        let status = HealthStatus(
            status: "ok",
            version: EngineConfig.version,
            modules: ModuleStatus(
                stt: sttReady,
                llm: llmReady,
                touchup: llmReady,
                grammar: grammarReady,
                vad: vadReady,
                tts: false,
                infinite: nil
            )
        )
        return try JSONEncoder().encode(status)
    }

    private func handleModules() async throws -> Data {
        let modules = await ModuleRegistry.shared.all()
        var manifests: [ModuleManifest] = []
        for module in modules {
            let health = await module.healthCheck()
            manifests.append(
                ModuleManifest(
                    name: type(of: module).name,
                    version: EngineConfig.version,
                    loaded: health.loaded,
                    error: health.error,
                    detail: health.detail
                )
            )
        }
        manifests.sort { $0.name < $1.name }

        let response = ModulesResponse(
            engineVersion: EngineConfig.version,
            buildVariant: BuildVariant.current.rawValue,
            modules: manifests
        )
        return try JSONEncoder().encode(response)
    }

    // MARK: - Session boundary
    //
    // `/v1/session/{begin,end}` dispatch through the endpoint table
    // (`SessionEndpoints`, EngineCore) — the same handler closures the
    // loopback server registers. See `dispatchViaTable`.

    private func handleGrammar(_ body: Data) async throws -> Data {
        let request = try JSONDecoder().decode(GrammarCheckRequest.self, from: body)
        let outcome = await GrammarEngine.shared.check(
            text: request.text,
            categories: request.categories,
            usePOS: request.usePOS ?? true
        )
        let response = GrammarCheckResponse(
            result: outcome.result,
            correctionsApplied: outcome.correctionsApplied,
            ruleCount: GrammarEngine.shared.ruleCount
        )
        return try JSONEncoder().encode(response)
    }

    private func handleVAD(_ body: Data) async throws -> Data {
        let request = try JSONDecoder().decode(VADRequest.self, from: body)
        if await !VADEngine.shared.isLoaded {
            try await VADEngine.shared.load()
        }
        let segments = try await VADEngine.shared.detect(
            samples: request.samples,
            resetState: request.reset ?? true
        )
        let response = VADResponse(
            segments: segments.map {
                SpeechSegment(
                    startMs: $0.startMs,
                    endMs: $0.endMs,
                    probability: $0.probability
                )
            }
        )
        return try JSONEncoder().encode(response)
    }

    private func handleBatch(_ body: Data) async throws -> Data {
        let request = try JSONDecoder().decode(BatchSTTRequest.self, from: body)
        // A missing `language`/`mode` key decodes as `"en"`/`"normal"` — the
        // defaults live on the canonical `BatchSTTRequest` itself, so this
        // handler and the loopback route agree by construction (#225 review).
        // An unknown language string is still a hard 400; an unknown mode
        // string coerces to `.normal` for parity with the loopback server,
        // which also coerces rather than 400s.
        guard let language = STTModule.STTLanguage.fromCode(request.language) else {
            throw YoozEngineError.serverError(
                statusCode: 400,
                code: "invalid_language",
                message: "Unknown STT language '\(request.language)'"
            )
        }
        let mode = STTModule.AudioMode(rawValue: request.mode) ?? .normal
        let start = ContinuousClock.now
        // yooz-labs/yooz-whisper#280 forensics: log the DECODED sample count
        // (post-JSONDecoder), not the raw body byte count — a mismatch
        // against the caller's expected sample count (bodyBytes / ~10 per
        // JSON float) would pinpoint decode-time truncation vs. a downstream
        // transcription-time loss.
        let enterMessage = "stt.batch.enter samples=\(request.samples.count) mode=\(mode.rawValue) "
            + "language=\(request.language) aligned=\(request.aligned == true)"
        Self.sttLogger.log("\(enterMessage, privacy: .public)")
        switch YoozSTTEngine.shared.currentBackend {
        case .parakeet, .fastConformer:
            // Await any in-flight warmup first (engine#252, PR #255 review
            // finding C2) — see the matching comment in `openSTTStream`.
            await YoozSTTEngine.shared.awaitWarmupIfNeeded(
                deadlineSeconds: EngineConfig.modelLoadDeadlineSeconds
            )
            // Coalescing + bounded wait (engine#252): route through the same
            // enqueueLoad/awaitLoadTask primitive openSTTStream already uses
            // for these backends, so a proactive startup warmup
            // (XPCService/main.swift) racing this request for the same
            // language shares the one underlying load instead of each
            // independently paying `ParakeetModel.fromDirectory`'s full
            // cost and racing to assign `model` under `YoozSTTEngine`'s
            // lock. Previously this called `start(language:)` directly,
            // with no dedup against a concurrent load for the same
            // language and no bound on how long a wedged load could hang
            // the caller.
            let loadTask = await YoozSTTEngine.shared.enqueueLoad(language: language) {
                try await YoozSTTEngine.shared.start(language: language)
            }
            try await awaitLoadOrTypedDeadline(loadTask, deadlineSeconds: EngineConfig.modelLoadDeadlineSeconds)
        default:
            try await YoozSTTEngine.shared.start(language: language)
        }

        // `batchTranscribe` is non-throwing and returns `ParakeetResult.empty`
        // when no model is loaded — indistinguishable from genuine silence. The
        // throwing `start()` above normally guarantees a loaded model, but guard
        // explicitly so a load-state inconsistency surfaces as an error instead
        // of a misleading empty transcript.
        //
        // (When qwen3 becomes selectable in-process, this path must add the
        // qwen3 dispatch the loopback server uses; qwen3 is not reachable
        // in-process in Phase 2a because the backend picker is deferred.)
        guard YoozSTTEngine.shared.isRunning else {
            throw YoozEngineError.serverError(
                statusCode: 503,
                code: "stt_not_loaded",
                message: "STT model failed to load for language '\(request.language)'"
            )
        }

        if request.aligned == true {
            let result = try await YoozSTTEngine.shared.batchTranscribeAligned(
                samples: request.samples,
                mode: mode
            )
            let tokens = result.tokens.map {
                SDKAlignedToken(text: $0.text, start: $0.start, end: $0.end)
            }
            let response = SDKTranscriptionResult(
                text: result.text,
                finalized: result.text,
                draft: "",
                language: request.language,
                tokens: tokens
            )
            let data = try JSONEncoder().encode(response)
            let exitMessage = "stt.batch.exit chars=\(result.text.count) "
                + "elapsedMs=\(start.duration(to: .now).milliseconds) aligned=true"
            Self.sttLogger.log("\(exitMessage, privacy: .public)")
            return data
        }

        let result = await YoozSTTEngine.shared.batchTranscribe(
            samples: request.samples,
            mode: mode
        )
        let response = SDKTranscriptionResult(
            text: result.text,
            finalized: result.finalized,
            draft: result.draft,
            language: request.language,
            tokens: nil
        )
        let data = try JSONEncoder().encode(response)
        let exitMessage = "stt.batch.exit chars=\(result.text.count) "
            + "elapsedMs=\(start.duration(to: .now).milliseconds) aligned=false"
        Self.sttLogger.log("\(exitMessage, privacy: .public)")
        return data
    }

    /// Pre-load the active backend's STT model for a language and return the
    /// resulting status. Backs both `loadModel` (`/v1/stt/load?wait=true`) and
    /// `loadModelAsync` (`/v1/stt/load`, fire-and-forget).
    ///
    /// For the MLX backends the load is routed through the engine's cancellable
    /// `enqueueLoad` state machine — the same primitive the loopback
    /// `/v1/stt/load` route uses — so the load is observable via `loadState`
    /// and bounded. `wait == true` awaits completion under
    /// `EngineConfig.modelLoadDeadlineSeconds`; `wait == false` returns the
    /// current (`.loading`) status immediately and the consumer polls
    /// `/v1/stt/status` for the `.ready` / `.failed` transition. This replaces
    /// the prior synchronous, unbounded `start()` that pinned status at
    /// "Downloading 100%" until weights finished materializing.
    ///
    /// The same lazy `start()` also runs on the first `batch`/stream call, so
    /// this endpoint is a pre-warm, not a prerequisite, for transcription.
    private func handleSTTLoad(_ body: Data, wait: Bool) async throws -> Data {
        let request = try JSONDecoder().decode(STTLoadRequest.self, from: body)
        // A missing `language` key decodes as `"en"` — the default lives on
        // the canonical `STTLoadRequest` itself, so this handler and the
        // loopback route agree by construction (#225 review). An unknown
        // language string is still a hard 400.
        guard let language = STTModule.STTLanguage.fromCode(request.language) else {
            throw YoozEngineError.serverError(
                statusCode: 400,
                code: "invalid_language",
                message: "Unknown STT language '\(request.language)'"
            )
        }
        // Honored by the MLX branch below; parity with the loopback route's
        // `runSTTLoad(language:allowFetch:)`. Apple STT ignores the flag —
        // its model is supplied by the OS.
        let allowFetch = request.allowFetch ?? true
        switch YoozSTTEngine.shared.currentBackend {
        case .appleSTT:
            guard let appleLang = AppleSTTLanguage.from(rawCode: request.language) else {
                throw YoozEngineError.serverError(
                    statusCode: 400, code: "invalid_language",
                    message: "Language '\(request.language)' is not supported by Apple STT"
                )
            }
            // Apple STT has no HF download/materialize phase; load synchronously.
            try await AppleSTTEngine.shared.start(language: appleLang)
        case .qwen3ASRPreview:
            // The preview backend is loopback/dev only (unstable; engine#154).
            throw YoozEngineError.unsupportedOperation(operation: "load qwen3 preview")
        case .parakeet, .fastConformer:
            let task = await YoozSTTEngine.shared.enqueueLoad(language: language) {
                try await YoozSTTEngine.shared.start(
                    language: language, allowFetch: allowFetch
                )
            }
            if wait {
                try await awaitLoadOrTypedDeadline(
                    task, deadlineSeconds: EngineConfig.modelLoadDeadlineSeconds
                )
            } else {
                // Fire-and-forget: the task settles loadState/lastLoadError in the
                // background and a failure surfaces on the next /v1/stt/status
                // poll. Log at the dispatch site so that failure is correlatable
                // here, not only via the NSLog deep inside start().
                NSLog(
                    "InProcessTransport: STT load dispatched fire-and-forget for %@; poll /v1/stt/status for completion",
                    language.rawValue
                )
            }
        }
        return try await handleSTTStatus()
    }

    private func handleLLM(_ body: Data) async throws -> Data {
        let request = try JSONDecoder().decode(LLMGenerateRequest.self, from: body)
        // An unrecognized model name is a hard error (parity with the loopback
        // server's `invalid_model` 400) — never a silent downgrade to Light,
        // which would return a response whose `model` field lies about what ran.
        let modelType: LLMModelType
        if let name = request.model, !name.isEmpty {
            guard let resolved = LLMModelType(rawValue: name) else {
                throw YoozEngineError.serverError(
                    statusCode: 400,
                    code: "invalid_model",
                    message: "Unknown LLM model '\(name)'"
                )
            }
            modelType = resolved
        } else {
            modelType = .yoozLight
        }
        let text = try await TouchUpEngine.shared.generate(
            prompt: request.prompt,
            systemPrompt: request.systemPrompt ?? "",
            modelType: modelType,
            workloadClass: request.workloadClass ?? .background
        )
        let response = LLMGenerateResponse(
            text: text,
            model: modelType.rawValue,
            tokensGenerated: nil,
            processingTimeMs: nil
        )
        return try JSONEncoder().encode(response)
    }

    // MARK: - Status

    private func handleSTTStatus() async throws -> Data {
        let status: STTStatus
        if YoozSTTEngine.shared.currentBackend == .appleSTT {
            let loaded = await AppleSTTEngine.shared.isLoaded
            let language = await AppleSTTEngine.shared.currentLanguage.rawValue
            let streaming = await AppleSTTEngine.shared.isStreaming
            // Apple STT has no fetcher/materialize lifecycle; map loaded → ready.
            status = STTStatus(
                loaded: loaded, language: language, streaming: streaming,
                progress: nil, state: loaded ? .ready : .idle, lastError: nil
            )
        } else {
            // Surface the load lifecycle + last error (mirrors the loopback
            // /v1/stt/status, engine#125) so the consumer can distinguish a
            // download ("Downloading X%") from the synchronous materialization
            // window ("Loading model…": progress nil + state .loading) and from a
            // failed load (state .failed). The in-process path constructs the SDK
            // type directly, so the engine LoadState is bridged by rawValue.
            let engine = YoozSTTEngine.shared
            let loaded = engine.isRunning
            // Read the @Published, MainActor-written fields in a single hop so a
            // poll can't observe a torn snapshot (e.g. the stale ~1.0 progress
            // mid-reset alongside an already-advanced state).
            let (rawProgress, engineState, engineError) = await MainActor.run {
                (engine.downloadProgress, engine.loadState, engine.lastLoadError)
            }
            // When the model is resident, force progress nil + state .ready: a
            // just-finished load leaves downloadProgress at 1.0, and a redundant
            // enqueueLoad on an already-loaded engine briefly flips loadState to
            // .loading — neither should surface as "Downloading 100%" / "loading"
            // for a ready model. While not loaded, a fraction of exactly 1.0 means
            // "download done, materializing" -> nil (the STT analog of the LLM
            // `< 1` filter).
            let resolvedState: EngineCore.LoadState = loaded ? .ready : engineState
            status = STTStatus(
                loaded: loaded,
                language: engine.currentLanguage.rawValue,
                streaming: engine.isStreaming,
                progress: loaded ? nil : (rawProgress > 0 && rawProgress < 1 ? rawProgress : nil),
                state: LoadState(rawValue: resolvedState.rawValue),
                lastError: loaded ? nil : engineError
            )
        }
        return try JSONEncoder().encode(status)
    }

    private func handleLLMStatus() async throws -> Data {
        let engine = TouchUpEngine.shared
        let active = await engine.activeModel
        let loaded: Bool
        // Mirror the loopback `/v1/llm/status` progress contract so the
        // consumer-side touch-up download banner works in-process too (#214):
        // nil when the active tier is already loaded (no download to track),
        // nil for Apple Intelligence (OS-provided, no HF fetch), otherwise the
        // live fraction (>0) the `loadModelContainer` callback last reported.
        // Was hardcoded `nil`, so the in-process Light/Quality download bar
        // never moved on a first switch to a not-yet-cached tier.
        // A fraction of 1.0 means the download finished and the model is now
        // materializing (loadModelContainer hasn't returned yet): report nil so
        // the consumer renders "Loading model…" (state .loading) rather than a
        // frozen "Downloading 100%" — the LLM analog of the STT progress reset.
        let progress: Double?
        // engine#125: also surface the per-tier lifecycle state + last error so
        // consumers distinguish downloading / loading / failed. Bridged to the
        // SDK enum by rawValue (the in-process path builds the SDK type directly).
        let engineState: EngineCore.LoadState
        let lastError: String?
        switch active {
        case .yoozLight:
            loaded = await engine.isLightModelLoaded
            if loaded {
                progress = nil
            } else {
                let fraction = await engine.downloadProgress(for: .yoozLight) ?? 0
                progress = (fraction > 0 && fraction < 1) ? fraction : nil
            }
            engineState = await engine.loadState(for: .yoozLight)
            lastError = await engine.lastLoadError(for: .yoozLight)
        case .yoozQuality:
            loaded = await engine.isQualityModelLoaded
            if loaded {
                progress = nil
            } else {
                let fraction = await engine.downloadProgress(for: .yoozQuality) ?? 0
                progress = (fraction > 0 && fraction < 1) ? fraction : nil
            }
            engineState = await engine.loadState(for: .yoozQuality)
            lastError = await engine.lastLoadError(for: .yoozQuality)
        case .foundationModels:
            loaded = await engine.isFoundationModelsLoaded
            progress = nil
            engineState = loaded ? .ready : .idle
            lastError = nil
        }
        let status = LLMStatus(
            loaded: loaded, modelId: active.rawValue, progress: progress,
            state: LoadState(rawValue: engineState.rawValue),
            lastError: lastError
        )
        return try JSONEncoder().encode(status)
    }

    // MARK: - STT picker

    private func handleSTTLanguages() async throws -> Data {
        let infos = YoozSTTEngine.shared.availableLanguages.map {
            STTLanguageInfo(
                code: $0.rawValue,
                name: $0.displayName,
                implemented: $0.isImplemented,
                family: $0.modelFamily.rawValue
            )
        }
        return try JSONEncoder().encode(STTLanguagesResponse(languages: infos))
    }

    private func handleSTTEngine() async throws -> Data {
        let active = YoozSTTEngine.shared.currentBackend
        let activeLoaded = await YoozSTTEngine.shared.isCurrentBackendLoaded()
        let downloadedIds = await STTDownloadCoordinator.shared
            .downloadedBackendIds(language: YoozSTTEngine.shared.currentLanguage)
        let fractions = await STTDownloadCoordinator.shared.inFlightFractions()
        let backends = STTModule.STTBackendID.allCases.map {
            sttBackendInfo(
                $0, active: active, activeLoaded: activeLoaded,
                downloadedIds: downloadedIds, downloadFractions: fractions
            )
        }
        return try JSONEncoder().encode(
            STTBackendsResponse(backends: backends, activeId: active.rawValue)
        )
    }

    private func handleSetSTTEngine(_ body: Data) async throws -> Data {
        let request = try JSONDecoder().decode(STTSetBackendRequest.self, from: body)
        guard let backend = STTModule.STTBackendID(rawValue: request.id) else {
            throw YoozEngineError.serverError(
                statusCode: 400, code: "invalid_backend",
                message: "Unknown STT backend '\(request.id)'"
            )
        }
        await YoozSTTEngine.shared.setBackend(backend)
        let loaded = await YoozSTTEngine.shared.isCurrentBackendLoaded()
        let info = sttBackendInfo(backend, active: backend, activeLoaded: loaded)
        return try JSONEncoder().encode(info)
    }

    private func sttBackendInfo(
        _ backend: STTModule.STTBackendID,
        active: STTModule.STTBackendID,
        activeLoaded: Bool,
        downloadedIds: Set<String> = [],
        downloadFractions: [String: Double] = [:]
    ) -> STTBackendInfo {
        let isActive = backend == active
        // Same three-way state as the loopback builder (engine#291): a
        // downloaded-but-inactive backend must read `.cached`, not
        // `.available`, or a picker offers Download for what it already has.
        let loadState: ModelLoadState
        if isActive, activeLoaded {
            loadState = .loaded
        } else if downloadedIds.contains(backend.rawValue) {
            loadState = .cached
        } else {
            loadState = .available
        }
        return STTBackendInfo(
            id: backend.rawValue,
            displayName: backend.displayName,
            description: backend.pickerDescription,
            tier: ModelTier(rawValue: backend.pickerTier.rawValue) ?? .unknown,
            sizeBytes: backend.estimatedDownloadMB.map { Int64($0) * 1_000_000 },
            loadState: loadState,
            isActive: isActive,
            downloadProgress: downloadFractions[backend.rawValue],
            supportsBatch: backend.supportsBatch,
            supportsStreaming: backend.supportsStreaming,
            supportedLanguages: backend.supportedLanguages.map(\.rawValue)
        )
    }

    // MARK: - LLM / TouchUp model management

    /// Mirrors the loopback `APIServer.buildLLMModelsResponse()` (engine#303):
    /// every catalogued model, not just the TouchUp picker's three rows —
    /// `TouchUpEngine.availableModels()` is scoped to the picker
    /// (yooz-light-v3 / yooz-quality-v3 / foundation-models) and would
    /// silently drop any catalogue addition that is generate-only (e.g.
    /// yooz-instruct-4b), which is exactly the class of model this route
    /// exists to surface. `current` is the preferred-model flag
    /// (`POST /v1/llm/model`), which is independent of the picker's active
    /// selection — same source the loopback route reads.
    private func handleLLMModels() async throws -> Data {
        let info = await TouchUpEngine.shared.getModelInfo()
        let current = await TouchUpEngine.shared.preferredModel
        let available = info.map {
            SDKLLMModelInfo(
                id: $0.type.rawValue, displayName: $0.type.displayName,
                sizeBytes: $0.type.estimatedSize, loaded: $0.isLoaded,
                latencyHintMs: $0.type.latencyHintMs, purpose: $0.type.purpose
            )
        }
        return try JSONEncoder().encode(
            LLMModelsResponse(current: current.rawValue, available: available)
        )
    }

    private func handleSetLLMModel(_ body: Data) async throws -> Data {
        let request = try JSONDecoder().decode(LLMModelSelection.self, from: body)
        let modelType = try resolveLLMModel(request.model)
        await TouchUpEngine.shared.setPreferredModel(modelType)
        return try await handleLLMModels()
    }

    private func handleLLMPreload(_ body: Data) async throws -> Data {
        let request = try JSONDecoder().decode(LLMModelSelection.self, from: body)
        let modelType = try resolveLLMModel(request.model)
        try await TouchUpEngine.shared.preloadModel(modelType)
        return try JSONEncoder().encode(await llmModelInfo(modelType))
    }

    private func handleLLMUnload(_ body: Data) async throws -> Data {
        let request = try JSONDecoder().decode(LLMModelSelection.self, from: body)
        let modelType = try resolveLLMModel(request.model)
        await TouchUpEngine.shared.unload(modelType)
        return try JSONEncoder().encode(await llmModelInfo(modelType))
    }

    /// `model` is optional (engine#299): an empty body — RouteParityTests's
    /// probe, and a genuinely body-less POST from a real caller — means the
    /// same thing as a decoded `{"model": null}`, so it is special-cased
    /// ahead of the decode rather than surfacing as a decode failure.
    private func handleLLMClearCache(_ body: Data) async throws -> Data {
        let request: LLMClearCacheRequest = body.isEmpty
            ? LLMClearCacheRequest(model: nil)
            : try JSONDecoder().decode(LLMClearCacheRequest.self, from: body)
        let modelType = try request.model.map(resolveLLMModel)
        let cleared = await TouchUpEngine.shared.clearCache(modelType)
        return try JSONEncoder().encode(LLMClearCacheResponse(cleared: cleared.map(\.rawValue)))
    }

    /// Resolve the optional GPU-admission class from its raw wire string
    /// (engine#228) for the `TouchUpBody` decode shim below. Nil/omitted means
    /// today's default (`.background`); an unrecognized value is a hard 400 —
    /// parity with the loopback server (whose typed `Decodable` enum rejects
    /// unknown values as `invalid_request`) and with this transport's own mode
    /// handling ("an unknown mode is a hard error"): a declared scheduling
    /// class is explicit caller intent, and silently downgrading a mistyped
    /// `.interactive` to `.background` would make the request queue behind
    /// other work with no trace of why. (`handleLLM` needs no equivalent: it
    /// decodes the canonical `LLMGenerateRequest`, whose typed
    /// `workloadClass` field rejects unknown values at decode.)
    private static func resolveWorkloadClass(
        _ raw: String?
    ) throws -> MLXWorkloadClass {
        guard let raw, !raw.isEmpty else { return .background }
        guard let resolved = MLXWorkloadClass(rawValue: raw) else {
            throw YoozEngineError.serverError(
                statusCode: 400,
                code: "invalid_request",
                message: "Unknown workloadClass '\(raw)'"
            )
        }
        return resolved
    }

    /// Decode `body` as `TouchUpBody` and resolve it into the exact
    /// arguments `handleTouchUp` passes to
    /// `TouchUpEngine.processWithActiveModel` (engine#280 review item 4).
    /// Pulled out of `handleTouchUp` so a regression test can assert on the
    /// BUILT arguments directly: mode `"off"` (the fast/deterministic wire
    /// test choice) returns from `processWithActiveModel` before
    /// `contextVocabulary`/`contextAppName` are ever read, so a bare
    /// "the /v1/touchup call didn't throw" check cannot distinguish real
    /// forwarding from a silent `nil, nil` regression — asserting equality
    /// on this struct can.
    static func resolvedTouchUpCallArguments(from body: Data) throws -> TouchUpCallArguments {
        let request = try JSONDecoder().decode(TouchUpBody.self, from: body)
        // The requested mode is explicit caller intent (off/light/standard/full),
        // not a forward-compat wire value — an unknown mode is a hard error, never
        // a silent run at a different cleanup level.
        guard let mode = TouchUpMode(rawValue: request.mode) else {
            throw YoozEngineError.serverError(
                statusCode: 400, code: "invalid_mode",
                message: "Unknown TouchUp mode '\(request.mode)'"
            )
        }
        return TouchUpCallArguments(
            text: request.text,
            mode: mode,
            workloadClass: try resolveWorkloadClass(request.workloadClass),
            contextVocabulary: request.contextVocabulary,
            contextAppName: request.contextAppName
        )
    }

    private func handleTouchUp(_ body: Data) async throws -> Data {
        let arguments = try Self.resolvedTouchUpCallArguments(from: body)
        // Route through the active-model picker (not the legacy `process()`),
        // so the user's selection is honored in-process: Apple Intelligence
        // (FoundationModels), Yooz Light, or Yooz Quality. `process()` ignored
        // the selection (always MLX-light) and never lazy-loaded it, so
        // in-process cleanup silently passed text through. Each backend now
        // lazy-loads on first use (mirrors the STT lazy-load).
        let result = await TouchUpEngine.shared.processWithActiveModel(
            text: arguments.text,
            mode: arguments.mode,
            workloadClass: arguments.workloadClass,
            contextVocabulary: arguments.contextVocabulary,
            contextAppName: arguments.contextAppName
        )
        let response = TouchUpResponse(
            result: result.text,
            mode: arguments.mode,
            processingTimeMs: Int(result.latencyMs),
            modelUsed: result.modelUsed.rawValue,
            warnings: result.fallbackReason.map { [$0] }
        )
        return try JSONEncoder().encode(response)
    }

    // `GET /v1/touchup/models` and `POST /v1/touchup/model` dispatch through
    // the endpoint table (`TouchUpEndpoints`, LLMModule). See
    // `dispatchViaTable`.

    // MARK: - Model management (disk hygiene)
    //
    // `GET /v1/models`, `DELETE /v1/models/:id`, and `POST /v1/models/cleanup`
    // dispatch through the endpoint table (`ModelManagementEndpoints`,
    // LLMModule), with this transport injecting the STT-owned input below.

    /// Hub dir name of the active STT model, or `nil` for Apple Speech (no HF
    /// footprint). Parakeet TDT is the only HF-backed STT family. Injected
    /// into `ModelManagementEndpoints` at table construction — the endpoints
    /// live in LLMModule, which cannot depend on STTModule.
    private static func activeSTTRepoDirName() -> String? {
        let engine = YoozSTTEngine.shared
        guard engine.currentBackend != .appleSTT,
              let hfID = engine.currentLanguage.huggingFaceID
        else { return nil }
        return ModelCacheDescriptor.hubRepoDirName(forHuggingFaceID: hfID)
    }

    // MARK: - Mapping helpers

    private func resolveLLMModel(_ id: String) throws -> LLMModelType {
        guard let modelType = LLMModelType(rawValue: id) else {
            throw YoozEngineError.serverError(
                statusCode: 400, code: "invalid_model",
                message: "Unknown LLM model '\(id)'"
            )
        }
        return modelType
    }

    /// Real post-operation model info for the preload/unload responses.
    /// Reads `getModelInfo()` (catalogue-wide, engine#303) for the live
    /// `loaded` state rather than `TouchUpEngine.availableModels()`, whose
    /// three rows are scoped to the TouchUp picker — that used to leave any
    /// generate-only catalogue model (e.g. yooz-instruct-4b) falling through
    /// to a fabricated `loaded: false` row even right after a successful
    /// preload. `displayName` / `sizeBytes` / `latencyHintMs` come straight
    /// from the catalogue, so there is no "row not found" case left: every
    /// `LLMModelType` this is called with (always caller-resolved via
    /// `resolveLLMModel`) is by construction a valid catalogue entry.
    private func llmModelInfo(_ modelType: LLMModelType) async -> SDKLLMModelInfo {
        let info = await TouchUpEngine.shared.getModelInfo()
        let isLoaded = info.first(where: { $0.type == modelType })?.isLoaded ?? false
        return SDKLLMModelInfo(
            id: modelType.rawValue, displayName: modelType.displayName,
            sizeBytes: modelType.estimatedSize, loaded: isLoaded,
            latencyHintMs: modelType.latencyHintMs, purpose: modelType.purpose
        )
    }

}

/// Fully-resolved arguments for one `/v1/touchup` call (engine#280 review
/// item 4), built by `InProcessTransport.resolvedTouchUpCallArguments(from:)`.
/// `Equatable` so a regression test can assert on the exact built values —
/// see that function's doc for why a bare success/failure check on the
/// call isn't sufficient proof of forwarding.
struct TouchUpCallArguments: Equatable {
    let text: String
    let mode: TouchUpMode
    let workloadClass: MLXWorkloadClass
    let contextVocabulary: [String]?
    let contextAppName: String?
}

// MARK: - TouchUp mode decode shim
//
// `TouchUpBody` stays a local, minimal decode struct rather than the
// canonical `TouchUpRequest` (#225): `TouchUpRequest.mode` is the strict
// `TouchUpMode` enum, whose decode failure on an unrecognized string would
// surface as an opaque `DecodingError` instead of the `invalid_mode`
// `YoozEngineError.serverError` this handler has always returned. Decoding
// the raw string here and resolving it against `TouchUpMode(rawValue:)`
// explicitly preserves that error contract byte-for-byte. Every other
// former mirror struct in this section decoded a shape with no such
// custom-error behavior riding on it, so those were deleted outright in
// favor of the shared `YoozEngineWire` request types.

private struct TouchUpBody: Decodable {
    let text: String
    let mode: String
    let language: String?
    /// Raw wire value of `EngineCore.MLXWorkloadClass` (engine#228). Kept as
    /// a plain `String?` at the decode layer; `resolveWorkloadClass` maps
    /// nil/empty to `.background` and rejects unknown values with a 400.
    let workloadClass: String?
    /// Mirrors `TouchUpRequest.contextVocabulary`/`contextAppName`
    /// (engine#280 Phase 4). This decode shim is a SEPARATE type from the
    /// canonical `TouchUpRequest` (#225 — see the doc above), so adding a
    /// field to one does not add it to the other: without these two
    /// fields, `handleTouchUp` below would silently drop whisper's
    /// in-process context payload even though the loopback route decodes
    /// it fine — the "two-struct trap".
    let contextVocabulary: [String]?
    let contextAppName: String?
}
