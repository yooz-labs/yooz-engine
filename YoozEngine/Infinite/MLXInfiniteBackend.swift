// MLXInfiniteBackend.swift
// InfiniteModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore
import Foundation
import os.log

#if canImport(MLXLMCommon) && canImport(MLXHuggingFace)
import MLX
import MLXLMCommon
import MLXHuggingFace
import Tokenizers
import HuggingFace
#endif

private let mlxInfiniteLogger = Logger(
    subsystem: "live.yooz.engine",
    category: "MLXInfiniteBackend"
)

/// Outcome of one native-context generation.
public struct InfiniteGenerationResult: Sendable {
    public let text: String
    public let tokenCount: Int
    public let decodeTokensPerSecond: Double
    public let finishReason: String

    public init(
        text: String,
        tokenCount: Int,
        decodeTokensPerSecond: Double,
        finishReason: String
    ) {
        self.text = text
        self.tokenCount = tokenCount
        self.decodeTokensPerSecond = decodeTokensPerSecond
        self.finishReason = finishReason
    }
}

/// Outcome of one live-session `appendTokens` call.
public struct SessionAppendOutcome: Sendable, Equatable {
    /// `tokenizer.encode(text)`'s token count for just this call's text.
    public let appendedTokenCount: Int
    /// `tokenRecord.count` after this append (durable tokens plus the new
    /// pending token).
    public let totalTokenCount: Int

    public init(appendedTokenCount: Int, totalTokenCount: Int) {
        self.appendedTokenCount = appendedTokenCount
        self.totalTokenCount = totalTokenCount
    }
}

/// Sendable facts about one just-written checkpoint's live-session token
/// state, handed back to `InfiniteEngine` so it can write `tokens.bin` +
/// `manifest.json` — only `[Int]`/`Int?` leave the actor here, never an
/// `MLXArray`/`KVCache`.
public struct SessionCheckpointSnapshot: Sendable, Equatable {
    public let tokenRecord: [Int]
    public let pendingToken: Int?

    public init(tokenRecord: [Int], pendingToken: Int?) {
        self.tokenRecord = tokenRecord
        self.pendingToken = pendingToken
    }
}

/// Outcome of one live-session `generateSession` call.
public struct SessionGenerateOutcome: Sendable, Equatable {
    public let text: String
    public let tokenIds: [Int]
    public let finishReason: String
    /// This call's own prefill length (pending token, if any, plus the new
    /// prompt) — NOT the whole session history. See `SessionDecodeOutcome
    /// .prefillTokenCount`.
    public let prefillTokenCount: Int
    public let prefillTokensPerSecond: Double
    public let decodeTokensPerSecond: Double
    /// `tokenRecord.count` after this generate call.
    public let totalTokenCount: Int

    public init(
        text: String,
        tokenIds: [Int],
        finishReason: String,
        prefillTokenCount: Int,
        prefillTokensPerSecond: Double,
        decodeTokensPerSecond: Double,
        totalTokenCount: Int
    ) {
        self.text = text
        self.tokenIds = tokenIds
        self.finishReason = finishReason
        self.prefillTokenCount = prefillTokenCount
        self.prefillTokensPerSecond = prefillTokensPerSecond
        self.decodeTokensPerSecond = decodeTokensPerSecond
        self.totalTokenCount = totalTokenCount
    }
}

/// Turn-commit policy for one session (engine#267, epic#263 phase 4):
/// mirrors `SessionKnobs.turnPolicy`'s wire strings 1:1.
public enum InfiniteTurnPolicy: String, Sendable {
    /// Decode the whole turn (reasoning + answer) on a disposable branch of
    /// the durable cache, discard the branch, and commit only the user turn
    /// plus the stable-framed answer to the durable session — reasoning
    /// tokens never enter durable KV.
    case turnCommit = "turn_commit"
    /// Decode directly on the durable cache — reasoning tokens (if any)
    /// become a permanent part of the session's history, same as any other
    /// generated token.
    case thinkingInSession = "thinking_in_session"
}

/// Outcome of one live-session `generateTurn` call (engine#267).
public struct SessionTurnOutcome: Sendable, Equatable {
    /// The committed answer text — never includes reasoning, even for
    /// `.thinkingInSession` (the raw decoded text there has no reasoning
    /// markers to begin with unless the composer's `generationPrompt`
    /// actually opened one, and this is still the same decoded text
    /// `generateSession` would have returned).
    public let text: String
    /// `"stop"`, `"length"`, `"cancelled"`, or (turnCommit only)
    /// `"length_in_think"` — the branch hit `maxTokens` before ever closing
    /// its think block, so the reasoning is quarantined as usual but the
    /// committed answer is forced empty rather than trusting unclosed
    /// reasoning as prose.
    public let finishReason: String
    /// Reasoning-side token count, approximated by re-encoding the
    /// `reasoning` half of `composer.splitThinking`. `nil` for
    /// `.thinkingInSession` (nothing is split out; there is no separate
    /// "reasoning" bucket in that policy).
    public let thinkingTokens: Int?
    /// `(U + S).count` — the exact token count committed to the durable
    /// cache this call (user turn plus the stable-framed answer). `nil` for
    /// `.thinkingInSession` (nothing is separately "committed" beyond the
    /// ordinary session-append accounting).
    public let committedTokens: Int?
    /// Wall-clock seconds spent chunk-prefilling the commit onto the
    /// durable cache. `nil` for `.thinkingInSession` (no separate commit
    /// step — the durable cache IS what was just decoded).
    public let commitSeconds: Double?
    /// This call's own prefill length (pending token, if any, plus the new
    /// prompt/generation-prompt tokens) — see `SessionDecodeOutcome
    /// .prefillTokenCount`. For `.turnCommit` this is the BRANCH's prefill,
    /// not the durable commit's.
    public let prefillTokenCount: Int
    public let prefillTokensPerSecond: Double
    /// For `.turnCommit`, the branch's own decode throughput (thinking +
    /// answer together) — the durable commit step is a plain chunked
    /// prefill with no decode, so it has no tokens/s of its own.
    public let decodeTokensPerSecond: Double
    /// `tokenRecord.count` after this call.
    public let totalTokenCount: Int

    public init(
        text: String,
        finishReason: String,
        thinkingTokens: Int?,
        committedTokens: Int?,
        commitSeconds: Double?,
        prefillTokenCount: Int,
        prefillTokensPerSecond: Double,
        decodeTokensPerSecond: Double,
        totalTokenCount: Int
    ) {
        self.text = text
        self.finishReason = finishReason
        self.thinkingTokens = thinkingTokens
        self.committedTokens = committedTokens
        self.commitSeconds = commitSeconds
        self.prefillTokenCount = prefillTokenCount
        self.prefillTokensPerSecond = prefillTokensPerSecond
        self.decodeTokensPerSecond = decodeTokensPerSecond
        self.totalTokenCount = totalTokenCount
    }
}

#if canImport(MLXLMCommon) && canImport(MLXHuggingFace)
/// Wraps a freshly-created `[any KVCache]` so it can cross out of
/// `ModelContainer.perform`'s isolation as a `Sendable` return value.
/// `KVCache` does not declare `: Sendable` (its concrete classes hold
/// non-Sendable `MLXArray` state), so the array itself cannot satisfy a
/// generic `R: Sendable` return constraint directly. Safe here because the
/// box is unwrapped immediately after the single `perform` call that
/// creates it, with no other live reference in the meantime.
private final class KVCacheBox: @unchecked Sendable {
    let caches: [any KVCache]
    init(_ caches: [any KVCache]) {
        self.caches = caches
    }
}
#endif

/// Real MLX-Swift backend for the InfiniteModule native-context path.
///
/// Loads a model whose architecture the `mlx-swift-lm` fork supports
/// (`qwen3_5_moe` and both Gemma4 rows — 26B-A4B #184 and the E4B OptiQ-4bit
/// build #186 — verified vs Python mlx-lm) and runs generation with a fresh
/// per-call KV cache.
///
/// Two call surfaces coexist:
/// - `generate(context:prompt:...)` — the original sessionless one-shot
///   path, unchanged: re-prefills `context + prompt` from a fresh cache
///   every call.
/// - `openSession`/`appendTokens`/`generateSession`/`releaseSession` — the
///   live-session path (engine#265/epic#263 phase 2): each session keeps
///   its own durable `[any KVCache]` in `sessions`, appended to
///   incrementally instead of re-prefilled from scratch. See
///   `InfiniteSessionRuntime.swift` for the `tokenRecord`/`pendingToken`
///   invariant both operations maintain.
public actor MLXInfiniteBackend {
    public nonisolated let selection: InfiniteModelSelection

    #if canImport(MLXLMCommon) && canImport(MLXHuggingFace)
    private let container: ModelContainer
    private var sessions: [String: LiveSessionState] = [:]
    /// 2048-token chunks for the `appendTokens` chunked prefill — larger
    /// than mlx-swift-lm's own 512-token default prefill step size since
    /// append has no decode interleaved and can afford bigger batches
    /// between admission-gate checkpoints.
    private static let appendChunkSize = 2048

    private init(selection: InfiniteModelSelection, container: ModelContainer) {
        self.selection = selection
        self.container = container
    }
    #else
    private init(selection: InfiniteModelSelection) {
        self.selection = selection
    }
    #endif

    /// Load the model's weights (revision-pinned) into a `ModelContainer`.
    /// First-run downloads stream into `~/.cache/huggingface/hub/`.
    public static func load(
        _ descriptor: InfiniteBackendDescriptor
    ) async throws -> MLXInfiniteBackend {
        #if canImport(MLXLMCommon) && canImport(MLXHuggingFace)
        guard let repository = descriptor.repository else {
            throw InfiniteError.modelSetFailed(
                "model \(descriptor.selection.rawValue) has no model repository to load"
            )
        }
        let repoRef = "\(repository.id)@\(repository.revision)"
        mlxInfiniteLogger.info(
            "Loading Infinite \(descriptor.selection.rawValue, privacy: .public) from \(repoRef, privacy: .public)"
        )
        do {
            let configuration = ModelConfiguration(
                id: repository.id,
                revision: repository.revision
            )
            let container = try await loadModelContainer(
                from: #hubDownloader(),
                using: #huggingFaceTokenizerLoader(),
                configuration: configuration,
                progressHandler: { _ in }
            )
            return MLXInfiniteBackend(selection: descriptor.selection, container: container)
        } catch {
            throw InfiniteError.modelSetFailed(error.localizedDescription)
        }
        #else
        throw InfiniteError.modelSetFailed("MLX runtime is not linked into this build")
        #endif
    }

    /// Generate from the session's accumulated `context` plus a new `prompt`.
    /// Bounded to the model's native window; the 1M paging path is epic #180.
    public func generate(
        context: String,
        prompt: String,
        maxTokens: Int,
        nativeContextTokens: Int,
        temperature: Double = 0.7
    ) async throws -> InfiniteGenerationResult {
        #if canImport(MLXLMCommon) && canImport(MLXHuggingFace)
        let userText = context.isEmpty ? prompt : context + "\n\n" + prompt
        let userInput = UserInput(chat: [.user(userText)])
        let preparedInput = try await container.prepare(input: userInput)

        let promptTokens = preparedInput.text.tokens.size
        guard promptTokens <= nativeContextTokens else {
            throw InfiniteError.invalidSessionInput(
                "context (~\(promptTokens) tokens) exceeds the model's native window of \(nativeContextTokens); 1M paging is tracked in #180"
            )
        }

        // temperature 0 makes MLXLMCommon select argmax (greedy) and ignore
        // topP, giving deterministic output for the gemma4 parity test (#184).
        let params = GenerateParameters(
            maxTokens: maxTokens, temperature: Float(temperature), topP: 0.95
        )
        // GPU admission (engine#228): Infinite generation is background
        // throughput work per the issue's classification — no per-request
        // override on this path (unlike TouchUp/raw generate, Infinite has
        // no interactive use case), so it unconditionally checks the gate
        // before doing GPU work and again between chunks. (`append` is pure
        // session bookkeeping — no GPU work, so no gate there.)
        // `admissionWorkStart` shares one aging budget across all this
        // generation's checkpoints — see `MLXAdmissionGate.checkpoint`.
        let admissionWorkStart = ContinuousClock.now
        try await MLXAdmissionGate.shared.checkpoint(
            workStartedAt: admissionWorkStart
        )
        let collected = try await container.perform { (ctx: ModelContext) -> (String, GenerateCompletionInfo?) in
            let cache = ctx.model.newCache(parameters: params)
            let stream = try MLXLMCommon.generate(
                input: preparedInput,
                cache: cache,
                parameters: params,
                context: ctx
            )
            var text = ""
            var info: GenerateCompletionInfo?
            for await generation in stream {
                // Chunk-level yielding (engine#228) — see the identical
                // pattern in `MLXLLMBackend.generate` and the granularity
                // rationale on `MLXAdmissionGate.checkpoint`.
                try await MLXAdmissionGate.shared.checkpoint(
                    workStartedAt: admissionWorkStart
                )
                switch generation {
                case let .chunk(chunk):
                    text += chunk
                case let .info(completion):
                    info = completion
                default:
                    // Surface any future stream variant (e.g. a new error or
                    // tool-call frame) rather than silently dropping it into an
                    // empty/truncated result with finishReason "stop".
                    mlxInfiniteLogger.debug(
                        "MLXInfiniteBackend: unhandled generation stream variant, skipped"
                    )
                }
            }
            return (text, info)
        }
        // Prefer the engine's own decode-only stats (tokensPerSecond divides
        // generationTokenCount by generateTime, excluding prefill) and exact
        // stop reason over chunk-count proxies.
        let info = collected.1
        let finishReason: String
        switch info?.stopReason {
        case .length: finishReason = "length"
        case .cancelled: finishReason = "cancelled"
        case .stop, nil: finishReason = "stop"
        }
        return InfiniteGenerationResult(
            text: collected.0,
            tokenCount: info?.generationTokenCount ?? 0,
            decodeTokensPerSecond: info?.tokensPerSecond ?? 0,
            finishReason: finishReason
        )
        #else
        throw InfiniteError.generationUnavailable("MLX runtime is not linked into this build")
        #endif
    }

    // MARK: - Live sessions (engine#265)

    /// Opens a fresh, empty live session — idempotent, so a caller that
    /// isn't sure whether a session was already opened for `id` can call
    /// this unconditionally before `appendTokens`/`generateSession`.
    ///
    /// `kvBits` opts the session into a quantized KV cache (engine#268):
    /// each fresh `KVCacheSimple` layer is converted to `QuantizedKVCache`
    /// via `toQuantized(groupSize:bits:)` — free at this point since the
    /// cache is still empty (offset 0). Any other cache class (`MambaCache`
    /// GDN layers, `RotatingKVCache`, ...) is left untouched. Quantization
    /// happens ONLY here, once, at session open — never by threading
    /// `kvBits`/`kvScheme` into `GenerateParameters` for the decode loop
    /// (see `runSessionDecodeLoop`'s doc comment on
    /// `maybeQuantizeKVCache`'s entry-replacement hazard). The caller
    /// (`InfiniteEngine`) is responsible for only ever passing `kvBits` for
    /// a model whose attention path is quantization-safe
    /// (`InfiniteModelSelection.supportsQuantizedKVCache`) — this function
    /// has no model-family awareness of its own and applies the conversion
    /// mechanically.
    public func openSession(id: String, kvBits: Int? = nil, kvGroupSize: Int? = nil) async {
        #if canImport(MLXLMCommon) && canImport(MLXHuggingFace)
        guard sessions[id] == nil else { return }
        let groupSize = kvGroupSize ?? 64
        let box: KVCacheBox = await container.perform { ctx in
            let caches = ctx.model.newCache(parameters: nil)
            guard let kvBits else { return KVCacheBox(caches) }
            let quantized: [any KVCache] = caches.map { cache in
                (cache as? KVCacheSimple)?.toQuantized(groupSize: groupSize, bits: kvBits) ?? cache
            }
            return KVCacheBox(quantized)
        }
        sessions[id] = LiveSessionState(caches: box.caches)
        #endif
    }

    /// Drops a live session's durable KV cache. A no-op (not an error) if
    /// `id` has no open session — mirrors `InfiniteSessionStore.deleteSession`'s
    /// own no-op-on-missing convention.
    public func releaseSession(id: String) async {
        #if canImport(MLXLMCommon) && canImport(MLXHuggingFace)
        sessions[id] = nil
        #endif
    }

    /// Chunk-prefills `text`'s tokens onto session `id`'s durable cache,
    /// raw (no chat template — the session path is plain text continuation
    /// in this phase; chat framing arrives with the turn composers in a
    /// later phase). Opens the session first if it isn't already open.
    ///
    /// Withholds the last new token from the prefill (feeds
    /// `[pendingToken] + newTokens.dropLast()`), so `pendingToken` is
    /// always set afterward — see `InfiniteSessionRuntime`'s invariant doc.
    public func appendTokens(id: String, text: String) async throws -> SessionAppendOutcome {
        try InfiniteSessionRuntime.requireNonEmpty(text, operation: "append")
        #if canImport(MLXLMCommon) && canImport(MLXHuggingFace)
        await openSession(id: id)
        guard var session = sessions[id] else {
            throw InfiniteError.sessionNotFound(id)
        }

        let tokenizer = await container.tokenizer
        let newTokens = tokenizer.encode(text: text, addSpecialTokens: false)
        guard !newTokens.isEmpty else {
            throw InfiniteError.invalidSessionInput("append text tokenized to zero tokens")
        }

        let toFeed = (session.pendingToken.map { [$0] } ?? []) + newTokens.dropLast()
        let admissionWorkStart = ContinuousClock.now
        let outcome = await container.perform(nonSendable: session.caches) { ctx, caches in
            await chunkedPrefill(
                tokens: toFeed,
                caches: caches,
                model: ctx.model,
                chunkSize: Self.appendChunkSize,
                admissionWorkStart: admissionWorkStart
            )
        }

        if outcome.cancelled {
            let (tokenRecord, pendingToken) = InfiniteSessionRuntime.truncatedAppendState(
                priorTokenRecord: session.tokenRecord,
                priorPendingToken: session.pendingToken,
                toFeed: toFeed,
                chunkSize: Self.appendChunkSize,
                chunksCompleted: outcome.chunksCompleted
            )
            session.tokenRecord = tokenRecord
            session.pendingToken = pendingToken
            sessions[id] = session
            throw CancellationError()
        }

        session.tokenRecord += newTokens
        session.pendingToken = newTokens.last
        sessions[id] = session

        return SessionAppendOutcome(
            appendedTokenCount: newTokens.count,
            totalTokenCount: session.tokenRecord.count
        )
        #else
        throw InfiniteError.generationUnavailable("MLX runtime is not linked into this build")
        #endif
    }

    /// Generates from session `id`'s durable cache continued by `prompt`,
    /// raw (no chat template — see `appendTokens`'s doc). Greedy decode
    /// when `temperature == 0`. Opens the session first if it isn't
    /// already open.
    ///
    /// Never withholds a pending token: every token `TokenIterator.next()`
    /// returns has already been fed to the cache by the time it returns it
    /// (verified against `Evaluate.swift`; see `InfiniteSessionRuntime`'s
    /// invariant doc), so `pendingToken` is always `nil` afterward.
    public func generateSession(
        id: String,
        prompt: String,
        maxTokens: Int,
        temperature: Double
    ) async throws -> SessionGenerateOutcome {
        try InfiniteSessionRuntime.requireNonEmpty(prompt, operation: "generate")
        #if canImport(MLXLMCommon) && canImport(MLXHuggingFace)
        await openSession(id: id)
        guard var session = sessions[id] else {
            throw InfiniteError.sessionNotFound(id)
        }

        let tokenizer = await container.tokenizer
        let promptTokens = tokenizer.encode(text: prompt, addSpecialTokens: false)
        guard !promptTokens.isEmpty else {
            throw InfiniteError.invalidSessionInput("generate prompt tokenized to zero tokens")
        }

        let seedIds = (session.pendingToken.map { [$0] } ?? []) + promptTokens
        let admissionWorkStart = ContinuousClock.now
        try await MLXAdmissionGate.shared.checkpoint(workStartedAt: admissionWorkStart)

        let outcome = try await container.perform(nonSendable: session.caches) { ctx, caches in
            try await runSessionDecodeLoop(
                seedIds: seedIds,
                caches: caches,
                context: ctx,
                parameters: SessionDecodeParameters(maxTokens: maxTokens, temperature: temperature),
                admissionWorkStart: admissionWorkStart
            )
        }

        // Every id in outcome.fedIds (and all of promptTokens, fed inside
        // TokenIterator's own `prepare()` before the loop even starts) was
        // durably fed regardless of whether the loop finished cleanly or
        // was cancelled mid-decode — see the invariant doc — so this
        // commit always happens before rethrowing.
        session.tokenRecord += promptTokens + outcome.fedIds
        session.pendingToken = nil
        sessions[id] = session

        if outcome.finishReason == "cancelled" {
            throw CancellationError()
        }

        return SessionGenerateOutcome(
            text: outcome.text,
            tokenIds: outcome.emittedIds,
            finishReason: outcome.finishReason,
            prefillTokenCount: outcome.prefillTokenCount,
            prefillTokensPerSecond: outcome.prefillTokensPerSecond,
            decodeTokensPerSecond: outcome.decodeTokensPerSecond,
            totalTokenCount: session.tokenRecord.count
        )
        #else
        throw InfiniteError.generationUnavailable("MLX runtime is not linked into this build")
        #endif
    }

    // MARK: - Turn-commit (engine#267)

    /// Groups a turn's chat-template framing inputs so
    /// `generateTurnThinkingInSession`/`generateTurnCommit` stay under the
    /// parameter-count lint limit (mirrors `SessionDecodeParameters`'
    /// identical purpose).
    #if canImport(MLXLMCommon) && canImport(MLXHuggingFace)
    private struct TurnFraming {
        let composer: any InfiniteTurnComposer
        let tokenizer: any MLXLMCommon.Tokenizer
        /// `encode(userOpen + trimmedPrompt + userClose)` — computed once in
        /// `generateTurn` since it is identical across both policies.
        let userTokens: [Int]
    }
    #endif

    /// Generates one conversational turn from session `id`, framed by
    /// `selection.turnComposer` (Qwen3.5 ChatML or Gemma4 `<|turn>`), under
    /// `policy`. Opens the session first if it isn't already open. See
    /// `InfiniteTurnPolicy`'s doc for what each policy actually does to the
    /// durable cache.
    #if canImport(MLXLMCommon) && canImport(MLXHuggingFace)
    public func generateTurn(
        id: String,
        prompt: String,
        maxTokens: Int,
        temperature: Double,
        policy: InfiniteTurnPolicy = .turnCommit
    ) async throws -> SessionTurnOutcome {
        try InfiniteSessionRuntime.requireNonEmpty(prompt, operation: "generate")
        await openSession(id: id)
        guard var session = sessions[id] else {
            throw InfiniteError.sessionNotFound(id)
        }

        let composer = selection.turnComposer
        let tokenizer = await container.tokenizer
        // Mirrors the chat template's own content trimming (verified against
        // both pinned templates — Qwen's `render_content(...)|trim`, Gemma4's
        // `content | trim` / `strip_thinking`'s final `|trim`) so composed
        // fragments stay token-identical to `apply_chat_template`.
        let trimmedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let userTokens = tokenizer.encode(
            text: composer.userOpen + trimmedPrompt + composer.userClose, addSpecialTokens: false
        )
        guard !userTokens.isEmpty else {
            throw InfiniteError.invalidSessionInput("generate prompt tokenized to zero tokens")
        }

        let framing = TurnFraming(composer: composer, tokenizer: tokenizer, userTokens: userTokens)
        let parameters = SessionDecodeParameters(maxTokens: maxTokens, temperature: temperature)
        switch policy {
        case .thinkingInSession:
            return try await generateTurnThinkingInSession(
                id: id, session: session, framing: framing, parameters: parameters
            )
        case .turnCommit:
            return try await generateTurnCommit(
                id: id, session: session, framing: framing, parameters: parameters
            )
        }
    }

    /// `.thinkingInSession`: decodes `[pending] + U + G` directly on the
    /// durable cache via the same `runSessionDecodeLoop` `generateSession`
    /// uses — nothing is branched or quarantined. `G` requests thinking
    /// OFF: this policy's whole point is "whatever the model does, keep
    /// it," not "make it think" — a plain chat turn under the template's own
    /// default (absent an explicit override) is thinking off. `.turnCommit`
    /// forces thinking ON regardless of the template default specifically
    /// because it needs a `<think>` block to quarantine.
    private func generateTurnThinkingInSession(
        id: String,
        session: LiveSessionState,
        framing: TurnFraming,
        parameters: SessionDecodeParameters
    ) async throws -> SessionTurnOutcome {
        var session = session
        let genTokens = framing.tokenizer.encode(
            text: framing.composer.generationPrompt(thinking: false), addSpecialTokens: false
        )
        let seedIds = (session.pendingToken.map { [$0] } ?? []) + framing.userTokens + genTokens

        let admissionWorkStart = ContinuousClock.now
        try await MLXAdmissionGate.shared.checkpoint(workStartedAt: admissionWorkStart)
        let outcome = try await container.perform(nonSendable: session.caches) { ctx, caches in
            try await runSessionDecodeLoop(
                seedIds: seedIds, caches: caches, context: ctx,
                parameters: parameters,
                admissionWorkStart: admissionWorkStart
            )
        }

        // Every id in seedIds/outcome.fedIds was durably fed regardless of a
        // clean finish or mid-decode cancellation — see generateSession's
        // identical commit-before-rethrow reasoning.
        session.tokenRecord += framing.userTokens + genTokens + outcome.fedIds
        session.pendingToken = nil
        sessions[id] = session

        if outcome.finishReason == "cancelled" {
            throw CancellationError()
        }

        return SessionTurnOutcome(
            text: outcome.text,
            finishReason: outcome.finishReason,
            thinkingTokens: nil,
            committedTokens: nil,
            commitSeconds: nil,
            prefillTokenCount: outcome.prefillTokenCount,
            prefillTokensPerSecond: outcome.prefillTokensPerSecond,
            decodeTokensPerSecond: outcome.decodeTokensPerSecond,
            totalTokenCount: session.tokenRecord.count
        )
    }

    /// `.turnCommit`: branches the durable cache (`branchCaches`), decodes
    /// `[pending] + U + G` (thinking forced ON) entirely on the branch,
    /// discards the branch, splits the branch's raw text into
    /// (reasoning, answer) via `composer.splitThinking`, then commits
    /// `U + stableAssistantWrap(answer)` to the DURABLE cache via the same
    /// chunked-prefill primitive `appendTokens` uses — maintaining the
    /// identical pending-token invariant (the commit's own last token is
    /// held back as pending, never the branch's).
    private func generateTurnCommit(
        id: String,
        session: LiveSessionState,
        framing: TurnFraming,
        parameters: SessionDecodeParameters
    ) async throws -> SessionTurnOutcome {
        var session = session
        let composer = framing.composer
        let tokenizer = framing.tokenizer
        let userTokens = framing.userTokens
        let genTokens = tokenizer.encode(
            text: composer.generationPrompt(thinking: true), addSpecialTokens: false
        )
        let seedIds = (session.pendingToken.map { [$0] } ?? []) + userTokens + genTokens

        let branch = try branchCaches(session.caches)
        let admissionWorkStart = ContinuousClock.now
        try await MLXAdmissionGate.shared.checkpoint(workStartedAt: admissionWorkStart)
        let branchOutcome = try await container.perform(nonSendable: branch) { ctx, caches in
            try await runSessionDecodeLoop(
                seedIds: seedIds, caches: caches, context: ctx,
                parameters: parameters,
                admissionWorkStart: admissionWorkStart
            )
        }
        // `branch` is discarded here — it is never written back to
        // `sessions`, so the durable cache above is untouched by everything
        // the branch just decoded (its own copy-on-write arrays simply go
        // out of scope with this function).

        if branchOutcome.finishReason == "cancelled" {
            // Nothing was ever committed to the durable cache (the commit
            // step below never ran), so the session is exactly as it was.
            throw CancellationError()
        }

        let rawText = branchOutcome.text
        // Generic "does this composer support thinking at all" probe: a
        // composer whose generationPrompt ignores `thinking` (Gemma4) never
        // opens a think block to begin with, so "unclosed think" cannot
        // apply to it.
        let opensThinking = composer.generationPrompt(thinking: true) != composer.generationPrompt(thinking: false)
        let unclosedThink = opensThinking && !rawText.contains("</think>")
        let (splitReasoning, splitAnswer) = composer.splitThinking(rawText)
        // When thinking never closed, the ENTIRE branch output was reasoning
        // that never reached a trustworthy conclusion — count all of it as
        // quarantined, not the empty string `splitThinking`'s "no </think>"
        // contract would otherwise report.
        let reasoningForStats = unclosedThink ? rawText : splitReasoning
        let answer = unclosedThink ? "" : splitAnswer
        let finishReason = unclosedThink ? "length_in_think" : branchOutcome.finishReason

        let stableTokens = tokenizer.encode(
            text: composer.stableAssistantWrap(answer: answer), addSpecialTokens: false
        )
        let toCommit = userTokens + stableTokens
        let toFeed = (session.pendingToken.map { [$0] } ?? []) + toCommit.dropLast()

        let commitAdmissionStart = ContinuousClock.now
        let commitStartTime = Date.timeIntervalSinceReferenceDate
        let commitOutcome = await container.perform(nonSendable: session.caches) { ctx, caches in
            await chunkedPrefill(
                tokens: toFeed, caches: caches, model: ctx.model,
                chunkSize: Self.appendChunkSize, admissionWorkStart: commitAdmissionStart
            )
        }
        let commitSeconds = Date.timeIntervalSinceReferenceDate - commitStartTime

        if commitOutcome.cancelled {
            let (tokenRecord, pendingToken) = InfiniteSessionRuntime.truncatedAppendState(
                priorTokenRecord: session.tokenRecord,
                priorPendingToken: session.pendingToken,
                toFeed: toFeed,
                chunkSize: Self.appendChunkSize,
                chunksCompleted: commitOutcome.chunksCompleted
            )
            session.tokenRecord = tokenRecord
            session.pendingToken = pendingToken
            sessions[id] = session
            throw CancellationError()
        }

        session.tokenRecord += toCommit
        session.pendingToken = toCommit.last
        sessions[id] = session

        let thinkingTokenCount = tokenizer.encode(text: reasoningForStats, addSpecialTokens: false).count

        return SessionTurnOutcome(
            text: answer,
            finishReason: finishReason,
            thinkingTokens: thinkingTokenCount,
            committedTokens: toCommit.count,
            commitSeconds: commitSeconds,
            prefillTokenCount: branchOutcome.prefillTokenCount,
            prefillTokensPerSecond: branchOutcome.prefillTokensPerSecond,
            decodeTokensPerSecond: branchOutcome.decodeTokensPerSecond,
            totalTokenCount: session.tokenRecord.count
        )
    }
    #else
    public func generateTurn(
        id: String,
        prompt: String,
        maxTokens: Int,
        temperature: Double,
        policy: InfiniteTurnPolicy = .turnCommit
    ) async throws -> SessionTurnOutcome {
        throw InfiniteError.generationUnavailable("MLX runtime is not linked into this build")
    }
    #endif

    // MARK: - Checkpoint / resume (engine#266)

    #if canImport(MLXLMCommon) && canImport(MLXHuggingFace)
    /// Branches session `id`'s live KV caches (`branchCaches` — parity-guarded
    /// independent copies, see `KVCacheBranching.swift`), evaluates +
    /// synchronizes them, and writes `cache.safetensors` to `cacheURL` via
    /// mlx-swift-lm's `savePromptCache`. The live session itself is left
    /// untouched — safe to call while the session stays hot; releasing it
    /// from RAM ("park") is a separate `releaseSession` call the caller
    /// makes afterward if it wants that.
    public func checkpointSession(id: String, cacheURL: URL) async throws -> SessionCheckpointSnapshot {
        guard let session = sessions[id] else {
            throw InfiniteError.sessionNotFound(id)
        }
        let branched = try branchCaches(session.caches)
        eval(branched)
        Stream().synchronize()
        try savePromptCache(url: cacheURL, cache: branched)
        return SessionCheckpointSnapshot(tokenRecord: session.tokenRecord, pendingToken: session.pendingToken)
    }

    /// Loads `cacheURL` (written by `checkpointSession`) via mlx-swift-lm's
    /// `loadPromptCache` and installs it as session `id`'s live state,
    /// paired with the caller-supplied `tokenRecord`/`pendingToken` (read
    /// from the checkpoint's `tokens.bin`/manifest by
    /// `InfiniteSessionStore` — plain `[Int]`/`Int?` data, so no MLX/
    /// tokenizer round-trip is needed for those here). Overwrites any
    /// existing live state for `id` unconditionally; the caller
    /// (`InfiniteEngine`) decides whether a resume should be a no-op (an
    /// already-open session with no explicit checkpoint id).
    public func resumeSession(
        id: String,
        cacheURL: URL,
        tokenRecord: [Int],
        pendingToken: Int?
    ) async throws {
        let (caches, _) = try loadPromptCache(url: cacheURL)
        var state = LiveSessionState(caches: caches)
        state.tokenRecord = tokenRecord
        state.pendingToken = pendingToken
        sessions[id] = state
    }

    /// Hex sha256 of the tokenizer actually in use by this loaded model —
    /// pinned into `InfiniteSessionManifest.tokenizerHash` at checkpoint
    /// time and re-derived at resume time for
    /// `InfiniteSessionStore.verify`'s integrity gate. Reads
    /// `tokenizer.json` from `ModelContainer.tokenizerDirectory` (the
    /// resolved local directory for this container's tokenizer) rather
    /// than trusting the model selection label, so a resume genuinely
    /// checks the tokenizer bytes in use, not just a repo id string.
    public func tokenizerHash() async throws -> String {
        let tokenizerDirectory = try await container.tokenizerDirectory
        let tokenizerJSONURL = tokenizerDirectory.appendingPathComponent("tokenizer.json", isDirectory: false)
        let data = try Data(contentsOf: tokenizerJSONURL)
        return InfiniteSessionStore.sha256Hex(data)
    }
    #else
    public func checkpointSession(id: String, cacheURL: URL) async throws -> SessionCheckpointSnapshot {
        throw InfiniteError.generationUnavailable("MLX runtime is not linked into this build")
    }

    public func resumeSession(
        id: String,
        cacheURL: URL,
        tokenRecord: [Int],
        pendingToken: Int?
    ) async throws {
        throw InfiniteError.generationUnavailable("MLX runtime is not linked into this build")
    }

    public func tokenizerHash() async throws -> String {
        throw InfiniteError.generationUnavailable("MLX runtime is not linked into this build")
    }
    #endif

    // MARK: - Test-only escape hatches (engine#265 live equivalence gate)

    #if canImport(MLXLMCommon) && canImport(MLXHuggingFace)
    /// Raw-tokenizes `text` exactly like `appendTokens`/`generateSession`
    /// do (no chat template). Internal, not public API — lets
    /// `InfiniteSessionRuntimeTests`'s live equivalence gate build a
    /// "cold restart" reference token sequence the same way the session
    /// path itself tokenizes, without reaching into `container` (private).
    func encodeForTesting(_ text: String) async -> [Int] {
        let tokenizer = await container.tokenizer
        return tokenizer.encode(text: text, addSpecialTokens: false)
    }

    /// Decodes `seedIds` from a brand-new cache with no session state and
    /// no pending token — the "cold restart" reference the live
    /// equivalence gate compares the live session path against. Internal,
    /// not public API.
    func coldDecodeForTesting(
        seedIds: [Int],
        maxTokens: Int,
        temperature: Double
    ) async throws -> SessionDecodeOutcome {
        let admissionWorkStart = ContinuousClock.now
        return try await container.perform { ctx in
            let cache = ctx.model.newCache(parameters: nil)
            return try await runSessionDecodeLoop(
                seedIds: seedIds,
                caches: cache,
                context: ctx,
                parameters: SessionDecodeParameters(maxTokens: maxTokens, temperature: temperature),
                admissionWorkStart: admissionWorkStart
            )
        }
    }

    /// Per-cache-layer runtime class name + state-array dtypes for session
    /// `id`'s live KV cache, in layer order. Lets the engine#266 checkpoint/
    /// resume live test assert a hybrid model's `MambaCache` (GDN) layers
    /// keep their `.float32` recurrent state across a save/load round trip
    /// through `savePromptCache`/`loadPromptCache` (see `GatedDelta.swift`'s
    /// "state kept in fp32 to match Python mlx-lm" invariant) — not
    /// observable any other way, since `LiveSessionState` is private to
    /// this actor. Internal, not public API.
    func liveCacheLayerDTypesForTesting(id: String) -> [(className: String, dtypes: [DType])] {
        guard let session = sessions[id] else { return [] }
        return session.caches.map { cache in
            (String(describing: type(of: cache)), cache.state.map(\.dtype))
        }
    }

    /// `tokenRecord`/`pendingToken` for session `id`'s live state — lets the
    /// turn-commit live gate (`InfiniteTurnCommitLiveTests`) assert exactly
    /// which tokens landed in the durable cache after a `generateTurn` call,
    /// and reconstruct a "cold restart" continuation seed from it. Internal,
    /// not public API.
    func tokenRecordForTesting(id: String) -> (tokenRecord: [Int], pendingToken: Int?) {
        guard let session = sessions[id] else { return ([], nil) }
        return (session.tokenRecord, session.pendingToken)
    }

    /// Byte-exact comparison of two sessions' live KV cache state arrays
    /// (offset, layer count, per-layer state shapes and values via
    /// `MLXArray.arrayEqual`) — the turn-commit live bit-exactness gate
    /// (`testBranchDecodeLeavesDurableBitExact`) compares a session that went
    /// through a `.turnCommit` branch decode against a control session built
    /// by directly appending the same committed text with no branch
    /// involved; the raw `MLXArray`s never leave the actor (mirrors
    /// `SessionCheckpointSnapshot`'s "only `[Int]`/`Int?` leave the actor"
    /// convention) — only the `Bool` verdict crosses. Internal, not public API.
    func liveCacheStateEqualsForTesting(id: String, matches otherID: String) -> Bool {
        guard let a = sessions[id], let b = sessions[otherID] else { return false }
        guard a.caches.count == b.caches.count else { return false }
        for (cacheA, cacheB) in zip(a.caches, b.caches) {
            guard cacheA.offset == cacheB.offset else { return false }
            let stateA = cacheA.state
            let stateB = cacheB.state
            guard stateA.count == stateB.count else { return false }
            for (arrA, arrB) in zip(stateA, stateB) {
                guard arrA.shape == arrB.shape else { return false }
                guard arrA.arrayEqual(arrB).item(Bool.self) else { return false }
            }
        }
        return true
    }

    /// `tokenizer.applyChatTemplate(messages:)` (always `addGenerationPrompt:
    /// true` — that overload's fixed default on the `MLXLMCommon.Tokenizer`
    /// bridge) — the template-parity live gate
    /// (`testTemplateParityTwoTurns`) compares this against composer
    /// fragments encoded the same way `generateTurn` itself tokenizes each
    /// turn (separate `encode` calls per turn, concatenated — never one
    /// `encode` of the whole joined string). Internal, not public API.
    func applyChatTemplateForTesting(messages: [[String: any Sendable]]) async throws -> [Int] {
        let tokenizer = await container.tokenizer
        return try tokenizer.applyChatTemplate(messages: messages)
    }
    #endif
}
