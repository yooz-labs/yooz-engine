// InfiniteSessionRuntime.swift
// InfiniteModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore
import Foundation

#if canImport(MLXLMCommon) && canImport(MLXHuggingFace)
import MLX
import MLXLMCommon
#endif

// MARK: - Core invariant (infinite ADR 0007 D4)
//
// Per session, `MLXInfiniteBackend` holds `tokenRecord: [Int]` — exactly the
// token ids fed to the durable KV cache, in order — plus `pendingToken:
// Int?`. Invariant: the durable cache always holds
// `tokenRecord[..<(tokenRecord.count - (pendingToken == nil ? 0 : 1))]`;
// when `pendingToken` is non-nil it is `tokenRecord.last`, a token that is
// NOT yet in the cache.
//
// `append(newTokens)` deliberately withholds `newTokens.last` from the
// chunked prefill (feeding only `[pendingToken] + newTokens.dropLast()`),
// so `pendingToken` is always set after a successful append — this
// guarantees the next append/generate call always has a concrete,
// non-empty seed token to build its `LMInput` from.
//
// `generateSession` does NOT withhold anything: verified against
// `TokenIterator.next()` (mlx-swift-lm `Libraries/MLXLMCommon/Evaluate.swift`)
// — every token `next()` returns has ALREADY been fed to the cache by the
// time it is returned (`step(previous:)` feeds `previousY` and samples a
// new `y` in the same call, then returns the now-fed `previousY`). So every
// emitted token, and the stop token if one was hit, is durably cached by
// construction; there is nothing left to hold in reserve. `pendingToken`
// is therefore always `nil` after a completed `generateSession` call —
// this holds unconditionally for a `TokenIterator`-driven decode, not just
// in the common case, because `next()` has no code path that returns a
// token without first feeding it.
//
// (TokenIterator does keep one sampled-but-unfed token in its own private
// `y` field at all times — the "prime the pump" lookahead — but that
// field is internal to MLXLMCommon and not observable from here. This
// runtime does not need its value: since every returned/fed token is
// already accounted for, discarding TokenIterator's own private lookahead
// costs nothing observable — the next call simply starts a fresh
// TokenIterator from the last durably-fed token.)

/// Per-session live KV-cache state held by `MLXInfiniteBackend`.
#if canImport(MLXLMCommon) && canImport(MLXHuggingFace)
public struct LiveSessionState {
    public var caches: [any KVCache]
    public var tokenRecord: [Int]
    public var pendingToken: Int?

    public init(caches: [any KVCache]) {
        self.caches = caches
        self.tokenRecord = []
        self.pendingToken = nil
    }
}
#endif

/// Outcome of one chunked-prefill `append` call.
public struct ChunkedPrefillOutcome: Sendable, Equatable {
    /// Whole `chunkSize`-wide chunks that were fed and `eval`'d.
    public let chunksCompleted: Int
    /// `true` when the admission gate cancelled the prefill partway
    /// through (`chunksCompleted` may be `0`, meaning nothing happened).
    public let cancelled: Bool

    public init(chunksCompleted: Int, cancelled: Bool) {
        self.chunksCompleted = chunksCompleted
        self.cancelled = cancelled
    }
}

/// Sampling knobs for one `runSessionDecodeLoop` call — grouped into one
/// type so the function itself stays under the parameter-count limit.
public struct SessionDecodeParameters: Sendable, Equatable {
    public let maxTokens: Int
    public let temperature: Double

    public init(maxTokens: Int, temperature: Double) {
        self.maxTokens = maxTokens
        self.temperature = temperature
    }
}

/// Outcome of one `generateSession` decode loop.
public struct SessionDecodeOutcome: Sendable, Equatable {
    /// Tokens surfaced in `text` — excludes a stop token, even though a
    /// stop token (if hit) was still fed to the cache.
    public let emittedIds: [Int]
    /// Every token actually fed to the cache during the loop, in order —
    /// `emittedIds` plus the stop token if one was hit.
    public let fedIds: [Int]
    public let text: String
    /// `"stop"`, `"length"`, or `"cancelled"` (admission gate cancellation
    /// mid-decode; `fedIds` still reflects everything genuinely fed).
    public let finishReason: String
    /// `seedIds.count` — how many tokens this call's own prefill fed
    /// (the pending token, if any, plus the new prompt). Exact count, not
    /// derived from `prefillTokensPerSecond`, so a caller (the live
    /// session-reuse test) can assert it directly equals the new prompt's
    /// own length rather than the whole session history.
    public let prefillTokenCount: Int
    public let prefillTokensPerSecond: Double
    public let decodeTokensPerSecond: Double

    public init(
        emittedIds: [Int],
        fedIds: [Int],
        text: String,
        finishReason: String,
        prefillTokenCount: Int,
        prefillTokensPerSecond: Double,
        decodeTokensPerSecond: Double
    ) {
        self.emittedIds = emittedIds
        self.fedIds = fedIds
        self.text = text
        self.finishReason = finishReason
        self.prefillTokenCount = prefillTokenCount
        self.prefillTokensPerSecond = prefillTokensPerSecond
        self.decodeTokensPerSecond = decodeTokensPerSecond
    }
}

public enum InfiniteSessionRuntime {

    /// Rejects an empty `append`/`generateSession` text input. Callable
    /// with no model loaded — both entry points check this before doing
    /// any tokenization or GPU work.
    public static func requireNonEmpty(_ text: String, operation: String) throws {
        guard !text.isEmpty else {
            throw InfiniteError.invalidSessionInput("\(operation) input must not be empty")
        }
    }

    /// Recomputes `tokenRecord`/`pendingToken` after a chunked-prefill
    /// `append` was interrupted (GPU admission gate cancellation) partway
    /// through feeding `toFeed` to the durable cache.
    ///
    /// `toFeed` is the sequence `appendTokens` was chunk-feeding when
    /// cancellation hit — `[priorPendingToken] + newTokens.dropLast()`
    /// (or just `newTokens.dropLast()` for a session's first append, which
    /// has no prior pending token). `chunksCompleted` whole `chunkSize`-wide
    /// chunks of `toFeed` are durably in the cache; nothing past that is.
    ///
    /// `chunksCompleted == 0` means the append never got past its first
    /// checkpoint, so the session must come back exactly as it was —
    /// this is also what the general formula below produces on its own
    /// (`toFeed`'s own first element, when non-empty, is always
    /// `priorPendingToken`), but is spelled out as an explicit early return
    /// so a `priorPendingToken == nil` (first-append) session can't
    /// manufacture a pending token out of zero fed work.
    public static func truncatedAppendState(
        priorTokenRecord: [Int],
        priorPendingToken: Int?,
        toFeed: [Int],
        chunkSize: Int,
        chunksCompleted: Int
    ) -> (tokenRecord: [Int], pendingToken: Int?) {
        precondition(chunkSize > 0, "chunkSize must be positive")
        guard chunksCompleted > 0 else {
            return (priorTokenRecord, priorPendingToken)
        }

        let fedCount = min(chunksCompleted * chunkSize, toFeed.count)
        let durablePrefix = priorPendingToken == nil
            ? priorTokenRecord
            : Array(priorTokenRecord.dropLast())
        let pendingToken = fedCount < toFeed.count ? toFeed[fedCount] : nil

        var tokenRecord = durablePrefix + toFeed[..<fedCount]
        if let pendingToken {
            tokenRecord.append(pendingToken)
        }
        return (tokenRecord, pendingToken)
    }
}

#if canImport(MLXLMCommon) && canImport(MLXHuggingFace)

/// Chunk-prefills `tokens` through `model` into `caches`, `chunkSize` ids
/// at a time, `eval`-ing the cache and checking the GPU admission gate
/// between chunks (mirrors `LLMModel.prepare`'s own chunking loop, except
/// every chunk — including the final partial one — is fed; nothing is
/// held back for a "prime the pump" step since `append` has no sampling
/// step at all).
///
/// Never throws: a gate cancellation is captured in the returned
/// `ChunkedPrefillOutcome` rather than propagated, so the caller (running
/// on the backend actor, not inside `container.perform`'s closure) can
/// safely commit `chunksCompleted` worth of progress before rethrowing.
func chunkedPrefill(
    tokens: [Int],
    caches: [any KVCache],
    model: any LanguageModel,
    chunkSize: Int,
    admissionWorkStart: ContinuousClock.Instant
) async -> ChunkedPrefillOutcome {
    guard !tokens.isEmpty else {
        return ChunkedPrefillOutcome(chunksCompleted: 0, cancelled: false)
    }

    var remaining = LMInput.Text(tokens: MLXArray(tokens))
    var state: LMOutput.State?
    var chunksCompleted = 0

    while remaining.tokens.size > 0 {
        do {
            // MLXAdmissionGate.checkpoint only ever throws CancellationError
            // (cooperative cancellation while queued behind interactive
            // load); there is no other failure mode to distinguish here.
            try await MLXAdmissionGate.shared.checkpoint(workStartedAt: admissionWorkStart)
        } catch {
            return ChunkedPrefillOutcome(chunksCompleted: chunksCompleted, cancelled: true)
        }

        let n = min(chunkSize, remaining.tokens.size)
        let chunkInput = remaining[.newAxis, ..<n]
        let output = model(chunkInput, cache: caches.isEmpty ? nil : caches, state: state)
        state = output.state
        eval(caches)
        chunksCompleted += 1
        remaining = remaining[n...]
    }

    return ChunkedPrefillOutcome(chunksCompleted: chunksCompleted, cancelled: false)
}

/// Decodes up to `maxTokens` from `seedIds` (the session's pending token,
/// if any, followed by the new prompt tokens), driving `TokenIterator`
/// directly — the engine's own equivalent of MLXLMCommon's private
/// `runSynchronousGenerationLoop` (`Evaluate.swift`), which this mirrors:
/// same stop-token set construction, same "check `next()`'s result before
/// counting it as emitted" ordering, same final `Stream().synchronize()`.
///
/// Never throws: a gate cancellation mid-decode is captured as
/// `finishReason == "cancelled"` in the returned outcome (with `fedIds`
/// reflecting everything genuinely fed up to that point) rather than
/// propagated, so the caller — back on the backend actor, not inside
/// `container.perform`'s closure — can commit that progress to the
/// session's `tokenRecord` before rethrowing `CancellationError` itself.
func runSessionDecodeLoop(
    seedIds: [Int],
    caches: [any KVCache],
    context: ModelContext,
    parameters: SessionDecodeParameters,
    admissionWorkStart: ContinuousClock.Instant
) async throws -> SessionDecodeOutcome {
    precondition(!seedIds.isEmpty, "seedIds must not be empty")
    let temperature = parameters.temperature

    let params = GenerateParameters(
        maxTokens: parameters.maxTokens,
        // NEVER set kvBits/kvScheme here. TokenIterator applies
        // maybeQuantizeKVCache(cache: &self.cache, ...) after every step,
        // which — when kvBits/kvScheme is set — REPLACES entries of its
        // own local `cache` array (a value-type Array whose elements are
        // reference-type KVCache objects) with quantized copies. That
        // replacement only lands in the iterator's own array; the
        // session's `caches` array (this function's parameter) keeps
        // pointing at the original, now-stale, unquantized objects —
        // silently detaching the session from the cache the iterator
        // actually advanced. Quantized-KV sessions are a documented
        // follow-up (PR5); this phase never sets these fields.
        temperature: Float(temperature),
        topP: temperature == 0 ? 1.0 : 0.95
    )
    var iterator = try TokenIterator(
        input: LMInput(tokens: MLXArray(seedIds)),
        model: context.model,
        cache: caches,
        parameters: params
    )

    // Mirrors MLXLMCommon.Evaluate's private `buildStopTokenIds` (not
    // public, so the same three sources are recombined here): the
    // tokenizer's own EOS id, the configuration's extra EOS ids, and any
    // configuration EOS strings converted through the tokenizer.
    var stopIds = context.configuration.eosTokenIds
    if let eos = context.tokenizer.eosTokenId {
        stopIds.insert(eos)
    }
    for token in context.configuration.extraEOSTokens {
        if let id = context.tokenizer.convertTokenToId(token) {
            stopIds.insert(id)
        }
    }

    var emittedIds: [Int] = []
    var fedIds: [Int] = []
    var finishReason = "length"
    var tokensSinceCheckpoint = 0
    let decodeStart = Date.timeIntervalSinceReferenceDate

    decodeLoop: while let token = iterator.next() {
        // `next()` has already fed `token` into `caches` by the time it
        // returns it (see the invariant note above) — true whether or not
        // it turns out to be a stop token, so `fedIds` always gets it.
        fedIds.append(token)

        if token == context.tokenizer.unknownTokenId || stopIds.contains(token) {
            finishReason = "stop"
            break decodeLoop
        }

        emittedIds.append(token)
        tokensSinceCheckpoint += 1
        if tokensSinceCheckpoint >= 32 {
            do {
                try await MLXAdmissionGate.shared.checkpoint(workStartedAt: admissionWorkStart)
            } catch {
                finishReason = "cancelled"
                break decodeLoop
            }
            tokensSinceCheckpoint = 0
        }
    }

    let decodeSeconds = Date.timeIntervalSinceReferenceDate - decodeStart

    // TokenIterator uses asyncEval() to keep the pipeline full; synchronize
    // before returning so no async eval work is still running against
    // `caches` once the caller persists/inspects them (mirrors
    // runSynchronousGenerationLoop's own final Stream().synchronize()).
    Stream().synchronize()

    let text = context.tokenizer.decode(tokenIds: emittedIds)
    let decodeTokensPerSecond = (emittedIds.isEmpty || decodeSeconds <= 0)
        ? 0 : Double(emittedIds.count) / decodeSeconds
    let prefillTokensPerSecond = iterator.promptPrefillTime > 0
        ? Double(seedIds.count) / iterator.promptPrefillTime
        : 0

    return SessionDecodeOutcome(
        emittedIds: emittedIds,
        fedIds: fedIds,
        text: text,
        finishReason: finishReason,
        prefillTokenCount: seedIds.count,
        prefillTokensPerSecond: prefillTokensPerSecond,
        decodeTokensPerSecond: decodeTokensPerSecond
    )
}

#endif
