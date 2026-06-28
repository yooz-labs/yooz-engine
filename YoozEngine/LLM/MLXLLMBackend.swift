// MLXLLMBackend.swift
// LLMModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore
import Foundation
import os.log

#if canImport(MLXLMCommon)
import MLX
import MLXLMCommon
#endif

#if canImport(MLXHuggingFace)
import MLXHuggingFace
// `#huggingFaceTokenizerLoader()` expands into code that calls
// `Tokenizers.AutoTokenizer.from(modelFolder:)` and `HuggingFace.HubClient`
// constructors, so both modules must be in scope at the call site.
import Tokenizers
import HuggingFace
#endif

private let logger = Logger(subsystem: "live.yooz.engine", category: "MLXLLMBackend")

#if canImport(MLXLMCommon)
/// Per-generate-call result captured inside the `container.perform`
/// closure and returned to the actor for post-processing. Lifted into a
/// struct so the closure return type stays under the swiftlint
/// `large_tuple` cap.
private struct GenerateResult {
    let text: String
    let kvSnapshot: [[MLXArray]]?
}
#endif

/// MLX-Swift backend for Yooz LLM models.
/// Pulls model weights from Hugging Face on first load (cached under
/// `~/.cache/huggingface/hub/`). Implements system-prompt KV cache
/// optimisation to skip re-computing the system prompt tokens on
/// subsequent calls with the same system prompt.
actor MLXLLMBackend: LLMBackend {

    // MARK: - Properties

    let identifier: String
    let modelType: LLMModelType

    /// Whether a model is loaded and ready for generation.
    private(set) var isLoaded = false

    /// Download progress (0.0 to 1.0). First-run downloads stream from
    /// Hugging Face via `#huggingFaceLoadModelContainer`; cached snapshots
    /// jump straight to 1.0.
    private(set) var downloadProgress: Double = 0

    private let bundleIdentifier: String

    #if canImport(MLXLMCommon)
    private var modelContainer: ModelContainer?

    /// Cached KV state containing only system prompt tokens.
    /// After the first call with a given system prompt, we snapshot the KV cache
    /// at the system prompt boundary so subsequent calls skip re-computing it.
    private var cachedPromptKVState: [[MLXArray]]?

    /// Number of tokens in the cached system prompt KV state
    private var cachedPromptTokenCount: Int = 0

    /// System prompt that produced the cached KV state
    private var cachedSystemPrompt: String?
    #endif

    // MARK: - Initialization

    init(
        modelType: LLMModelType,
        bundleIdentifier: String = "live.yooz.engine"
    ) {
        self.identifier = modelType.rawValue
        self.modelType = modelType
        self.bundleIdentifier = bundleIdentifier
    }

    // MARK: - LLMBackend Protocol

    /// Apply an `MLXResidency` directive to the process-global MLX cache knobs.
    /// The only place this backend mutates `MLX.Memory`.
    private func applyResidency(_ directive: MLXResidencyDirective) {
        Memory.cacheLimit = directive.cacheLimitBytes
        if directive.flush {
            Memory.clearCache()
        }
    }

    func load() async throws {
        guard !isLoaded else { return }

        // Register this LLM/TouchUp model as resident and size MLX's global
        // buffer cache to the sum across resident categories (per-category
        // budget x category count). Replaces the prior unilateral 512 MB cap,
        // which starved a coexisting STT model's scratch. A load only grows the
        // budget, so `register` never flushes. Applied here (a real load path)
        // rather than at process start so it never touches the Metal allocator
        // in the non-GPU structural tests. Rolled back in every failure path
        // below so a failed load never leaves a phantom-resident category. See
        // `EngineCore.MLXResidency`.
        applyResidency(MLXResidency.shared.register(.touchUp))

        #if canImport(MLXLMCommon) && canImport(MLXHuggingFace)
        let hfID = modelType.huggingFaceID
        logger.info("Loading model \(self.modelType.rawValue) from HF \(hfID)...")

        do {
            // Use the explicit `loadModelContainer(from: downloader, using:
            // tokenizerLoader, configuration:)` form rather than the
            // `#huggingFaceLoadModelContainer` macro because the macro's
            // closure-arg expansion currently fails to produce a diagnostic
            // when the progress handler captures `[weak self]` (Swift compiler
            // bug observed at swift-5.9 / mlx-swift-lm 3.x). The explicit
            // form composes the same default downloader (swift-transformers
            // `Hub` via `#hubDownloader()`) and tokenizer loader macros and
            // is what the macro itself expands to internally.
            //
            // First-run downloads stream into `~/.cache/huggingface/hub/`;
            // cached snapshots reuse the same on-disk layout. No
            // `/Volumes/S1` fallback, no embedded bundle dance — packaged
            // builds and fresh installs hit the same code path.
            let configuration = ModelConfiguration(id: hfID)
            modelContainer = try await loadModelContainer(
                from: #hubDownloader(),
                using: #huggingFaceTokenizerLoader(),
                configuration: configuration,
                progressHandler: { [weak self] progress in
                    let fraction = progress.fractionCompleted
                    Task {
                        await self?.setDownloadProgress(fraction)
                    }
                }
            )

            isLoaded = true
            logger.info("Model \(self.modelType.rawValue) loaded successfully")
        } catch let error as LLMError {
            // Roll back the registration above: no model became resident, so
            // balance `register(.touchUp)` to keep the resident set (and the
            // flush-when-empty signal) accurate. The rollback flushes only if
            // no other MLX category remains, so it never evicts a coexisting
            // STT model's warm buffers.
            applyResidency(MLXResidency.shared.unregister(.touchUp))
            throw error
        } catch {
            logger.error("Failed to load model: \(error.localizedDescription)")
            applyResidency(MLXResidency.shared.unregister(.touchUp))
            throw LLMError.loadFailed(error.localizedDescription)
        }
        #else
        logger.error("MLXLMCommon / MLXHuggingFace not available")
        applyResidency(MLXResidency.shared.unregister(.touchUp))
        throw LLMError.notAvailable("MLX framework not linked. Please rebuild with mlx-swift-lm package.")
        #endif
    }

    func unload() {
        let wasLoaded = isLoaded
        #if canImport(MLXLMCommon)
        modelContainer = nil
        cachedPromptKVState = nil
        cachedPromptTokenCount = 0
        cachedSystemPrompt = nil
        #endif
        isLoaded = false
        downloadProgress = 0
        // Release this model's residency. Trims the global cache budget to the
        // categories still resident; the freed weight/KV buffers are returned
        // to the OS by the flush — but only when no MLX category is left
        // resident, so unloading an LLM tier never evicts a coexisting STT
        // model's warm buffers (the cross-category stomp this epic fixes). When
        // another category is still resident the lowered limit reclaims the
        // parked buffers on its next allocation. Guarded on `wasLoaded` so
        // unloading a tier that never loaded does not touch the Metal allocator
        // (which faults where `default.metallib` is absent, e.g. `swift test`).
        if wasLoaded {
            applyResidency(MLXResidency.shared.unregister(.touchUp))
        }
        logger.info("Model \(self.modelType.rawValue) unloaded")
    }

    /// Clear the cached system prompt KV state, forcing re-computation on the next call.
    func clearSession() {
        #if canImport(MLXLMCommon)
        cachedPromptKVState = nil
        cachedPromptTokenCount = 0
        cachedSystemPrompt = nil
        #endif
        logger.debug("Prompt cache cleared")
    }

    func generate(prompt: String, systemPrompt: String) async throws -> String {
        guard isLoaded else {
            throw LLMError.notLoaded
        }

        #if canImport(MLXLMCommon)
        guard let container = modelContainer else {
            throw LLMError.notLoaded
        }

        var phase = "setup"
        do {
            let estimatedTokens = max(100, (prompt.count / 3) + 50)

            // Invalidate cache if system prompt changed
            if systemPrompt != cachedSystemPrompt {
                cachedPromptKVState = nil
                cachedPromptTokenCount = 0
                cachedSystemPrompt = nil
            }

            // Tokenize the full [system, user] message sequence
            let messages: [Chat.Message] = [
                .system(systemPrompt),
                .user(prompt)
            ]
            let userInput = UserInput(chat: messages)
            let fullInput = try await container.prepare(input: userInput)

            let params = GenerateParameters(
                maxTokens: estimatedTokens,
                temperature: 0.1,
                topP: 0.9
            )

            let savedKVState = cachedPromptKVState
            let savedTokenCount = cachedPromptTokenCount

            // On the first call, find the system prompt token boundary by comparing
            // two full sequences with different user messages. Chat templates are not
            // additive (system-only tokenization differs from the system portion of a
            // full sequence), so we find the common prefix of two full sequences instead.
            var sysOnlyTokenCount = savedTokenCount
            if savedKVState == nil {
                let probeMessages: [Chat.Message] = [
                    .system(systemPrompt),
                    .user("_")
                ]
                let probeInput = UserInput(chat: probeMessages)
                let probeLMInput = try await container.prepare(input: probeInput)

                let fullTokens = fullInput.text.tokens
                let probeTokens = probeLMInput.text.tokens
                let minLen = min(fullTokens.size, probeTokens.size)

                // Find where the two sequences diverge; that's the user content boundary
                if minLen > 0 {
                    let match = (fullTokens[..<minLen] .== probeTokens[..<minLen])
                    let matchArray = match.asArray(Bool.self)
                    var prefixLen = 0
                    for v in matchArray {
                        if v { prefixLen += 1 } else { break }
                    }
                    sysOnlyTokenCount = prefixLen
                }
            }
            let sysCount = sysOnlyTokenCount

            // If we have a cached system prompt KV state, skip the system tokens and
            // only feed the user portion. Otherwise, feed all tokens.
            let hasCachedState = savedKVState != nil && sysCount > 0
            let inputForModel: LMInput
            if hasCachedState {
                let userTokens = fullInput.text.tokens[sysCount...]
                inputForModel = LMInput(text: .init(tokens: userTokens))
            } else {
                inputForModel = fullInput
            }

            phase = "generation"

            let result: GenerateResult = try await container.perform { (context: ModelContext) in
                // Create fresh KV cache and restore cached system prompt state if available
                var cache = context.model.newCache(parameters: params)
                if let savedState = savedKVState {
                    for i in 0..<min(cache.count, savedState.count) {
                        cache[i].state = savedState[i]
                    }
                    eval(cache)
                }

                // Generate using the lower-level API with our managed cache
                let stream = try MLXLMCommon.generate(
                    input: inputForModel,
                    cache: cache,
                    parameters: params,
                    context: context
                )

                var text = ""
                for await generation in stream {
                    // ml-explore `Generation.chunk` carries the decoded String.
                    if case let .chunk(chunk) = generation {
                        text += chunk
                    }
                }

                // Reset the cache to hold ONLY the system-prompt tokens, then
                // snapshot it for the next call's prompt-cache reuse. This is the
                // "keep the system prompt, drop this turn's transcription +
                // response" step: trimming the user + generated tokens off the
                // tail is what keeps the persistent cache from carrying one
                // recording's content into the next (engine#212 KV-cache bleed).
                //
                // The trim is VERIFIED, not trusted. A snapshot is kept only when
                // the cache provably holds exactly `sysCount` tokens afterwards.
                // `trim()` removes from the tail and positions 0..<sysCount are
                // always the system prefix (fed in order on the cold call,
                // restored from a clean snapshot on warm calls), so an offset that
                // lands back on `sysCount` proves the remaining KV is the untouched
                // system prompt. If the offset bookkeeping has drifted, the trim
                // would leave stale transcription/response KV behind — so we keep
                // no snapshot and the actor drops the whole prompt cache below.
                var snapshot: [[MLXArray]]?
                if sysCount > 0 {
                    let currentOffset = cache.first?.offset ?? 0
                    let trimAmount = currentOffset - sysCount
                    if trimAmount > 0 {
                        for layer in cache {
                            layer.trim(trimAmount)
                        }
                        eval(cache)
                    }
                    let trimmedOffset = cache.first?.offset ?? 0
                    if trimAmount >= 0 && trimmedOffset == sysCount {
                        snapshot = cache.map(\.state)
                    }
                }
                return GenerateResult(
                    text: text,
                    kvSnapshot: snapshot
                )
            }

            // Update cached state. A non-nil snapshot is provably system-prompt
            // only (verified above), so persist it for reuse. A nil snapshot means
            // either there was no system prefix to cache, or the trim could not be
            // proven clean — drop the prompt cache entirely so the next call starts
            // cold (recomputing the system prompt) instead of reusing a snapshot we
            // cannot trust. This is the self-healing "otherwise delete all cache"
            // fallback: by induction every kept snapshot is clean, so no call ever
            // restores transcription/response KV from a prior recording.
            if let snapshot = result.kvSnapshot {
                cachedPromptKVState = snapshot
                cachedPromptTokenCount = sysCount
                cachedSystemPrompt = systemPrompt
                if savedKVState == nil {
                    logger.info("Cached system prompt KV (\(sysCount) tokens)")
                }
            } else {
                // Either there was no system prefix to cache (sysCount == 0) or
                // the trim could not be proven clean. Drop any prior prompt cache
                // so the next call recomputes the system prompt instead of reusing
                // a snapshot we cannot trust. Log only when an actual cache was
                // evicted, so a benign cold call (nothing cached yet) stays quiet.
                let hadCache = cachedPromptKVState != nil
                cachedPromptKVState = nil
                cachedPromptTokenCount = 0
                cachedSystemPrompt = nil
                if hadCache {
                    logger.debug("Dropped prompt cache (unverified system-only trim); next call recomputes")
                }
            }

            logger.debug("Generation complete, got \(result.text.count) chars")
            return postProcessResponse(result.text, originalInput: prompt)
        } catch {
            cachedPromptKVState = nil
            cachedPromptTokenCount = 0
            cachedSystemPrompt = nil
            logger.error("Generation failed during \(phase): \(error.localizedDescription)")
            throw LLMError.generationFailed("[\(phase)] \(error.localizedDescription)")
        }
        #else
        throw LLMError.notAvailable("MLX framework not linked")
        #endif
    }

    // MARK: - Progress

    private func setDownloadProgress(_ value: Double) {
        downloadProgress = value
        logger.debug("Download progress: \(Int(value * 100))%")
    }

    // MARK: - Post-Processing

    private func postProcessResponse(_ response: String, originalInput: String) -> String {
        let trimmed = response.trimmingCharacters(in: .whitespacesAndNewlines)

        let commentaryPrefixes = [
            "here's", "here is", "i'm sorry", "i apologize",
            "as a transcription", "as an ai", "the corrected",
            "the cleaned", "the revised", "i think", "i believe",
            "it seems", "let me", "sure,", "certainly,", "of course,"
        ]

        let lowercased = trimmed.lowercased()
        let hasCommentary = commentaryPrefixes.contains { lowercased.hasPrefix($0) }

        if hasCommentary {
            logger.debug("Detected commentary, attempting to extract clean text...")

            if let extracted = extractQuotedText(from: trimmed) {
                return extracted
            }

            if let extracted = extractAfterColon(from: trimmed) {
                return extracted
            }

            logger.warning("Could not extract clean text, returning original input")
            return originalInput
        }

        if trimmed.count > originalInput.count * 3 && trimmed.count > 200 {
            logger.warning("Response suspiciously long, returning original input")
            return originalInput
        }

        return trimmed
    }

    private func extractQuotedText(from text: String) -> String? {
        let pattern = "\"([^\"]+)\""
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []),
              let match = regex.firstMatch(in: text, options: [], range: NSRange(text.startIndex..., in: text)),
              let range = Range(match.range(at: 1), in: text) else {
            return nil
        }
        return String(text[range])
    }

    private func extractAfterColon(from text: String) -> String? {
        guard let colonIndex = text.firstIndex(of: ":") else { return nil }
        let afterColon = text[text.index(after: colonIndex)...]
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if afterColon.count > 10 {
            return afterColon
        }
        return nil
    }

    // MARK: - Cache

    /// Whether the HF snapshot for this model is already on disk.
    ///
    /// Used by Touch-up's picker UX to decide whether selecting this
    /// model triggers a fresh download. Resolves the cache root via
    /// `swift-huggingface`'s `HubCache` so the same code path works
    /// across non-sandboxed `YoozEngine.app` (`~/.cache/huggingface/hub`),
    /// sandboxed bundled helpers
    /// (`<container>/Library/Caches/huggingface/hub`), and explicit
    /// overrides via `HF_HUB_CACHE` / `HF_HOME`.
    ///
    /// A snapshot counts as cached when it contains both `config.json`
    /// and at least one `*.safetensors` file. An empty or partial
    /// snapshot dir reports `false` so the picker doesn't claim "ready"
    /// for an interrupted download. Repo IDs without an owner segment
    /// fall back to `false` (the engine never wires such IDs today).
    var isModelCached: Bool {
        #if canImport(MLXHuggingFace)
        let id = modelType.huggingFaceID
        let parts = id.split(separator: "/", maxSplits: 1, omittingEmptySubsequences: true)
        guard parts.count == 2 else { return false }
        let repoID = HuggingFace.Repo.ID(
            namespace: String(parts[0]),
            name: String(parts[1])
        )
        let snapshotsRoot = HubCache().snapshotsDirectory(repo: repoID, kind: .model)
        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: snapshotsRoot, includingPropertiesForKeys: nil
        ) else {
            return false
        }
        for snapshot in entries {
            let config = snapshot.appendingPathComponent("config.json")
            guard FileManager.default.fileExists(atPath: config.path) else {
                continue
            }
            let contents = (try? FileManager.default.contentsOfDirectory(
                at: snapshot, includingPropertiesForKeys: nil
            )) ?? []
            let hasWeights = contents.contains { $0.pathExtension == "safetensors" }
            if hasWeights { return true }
        }
        return false
        #else
        return false
        #endif
    }
}

// MARK: - Factory

extension MLXLLMBackend {
    static func create(
        for type: LLMModelType,
        bundleIdentifier: String = "live.yooz.engine"
    ) -> MLXLLMBackend {
        return MLXLLMBackend(
            modelType: type,
            bundleIdentifier: bundleIdentifier
        )
    }

    static func createLight(
        bundleIdentifier: String = "live.yooz.engine"
    ) -> MLXLLMBackend {
        return create(
            for: .yoozLight,
            bundleIdentifier: bundleIdentifier
        )
    }

    static func createQuality(
        bundleIdentifier: String = "live.yooz.engine"
    ) -> MLXLLMBackend {
        return create(
            for: .yoozQuality,
            bundleIdentifier: bundleIdentifier
        )
    }
}

// MARK: - SessionResettable

/// Per-recording-session reset boundary (engine issue #114). Drops the
/// cached system-prompt KV state so the next recording starts cold,
/// preventing recording N's context from leaking into recording N+1's
/// touch-up output. Idempotent and weight-preserving — `unload()` is a
/// separate operation that throws away the model container.
extension MLXLLMBackend: SessionResettable {
    func resetForNewSession() async {
        clearSession()
    }
}
