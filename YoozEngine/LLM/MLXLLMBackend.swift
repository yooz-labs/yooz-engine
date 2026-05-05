// MLXLLMBackend.swift
// YoozEngine
//
// Copyright 2026 Yooz Labs. All rights reserved.

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
/// Capability protocol for KV cache layers that accept TurboQuant's per-layer
/// `turboQuantEnabled` flag. The engine depends on the *capability*, not on
/// SharpAI's concrete `KVCacheSimple` class — so a future fork rename or a
/// new cache subclass that implements this property continues to work
/// without source changes here.
///
/// `KVCacheSimple` from the SharpAI mlx-swift-lm fork conforms automatically
/// via the extension below. Other built-in cache types
/// (`QuantizedKVCache` / `RotatingKVCache` / `ChunkedKVCache`) deliberately
/// do not. When a `.turbo3` request lands on a model whose layers do not
/// conform, the engine logs an error and the run silently falls back to
/// FP16 — the cast pattern in `generate(...)` makes that case observable
/// via the `lastTurboLayersEnabled` actor property.
protocol TurboQuantCapable: AnyObject {
    var turboQuantEnabled: Bool { get set }
}

extension KVCacheSimple: TurboQuantCapable {}
#endif

#if canImport(MLXLMCommon)
/// Per-generate-call result captured inside the `container.perform`
/// closure and returned to the actor for post-processing. Lifted into a
/// struct so the closure return type stays under the swiftlint
/// `large_tuple` cap (4 fields would otherwise trigger the rule).
private struct GenerateResult {
    let text: String
    let kvSnapshot: [[MLXArray]]?
    let turboEnabled: Int
    let turboTotal: Int
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

    /// KV cache compression mode for this backend instance. Defaults to
    /// `EngineConfig.kvCompression`, which today is `.off` so there is no
    /// behavioral change for existing call sites until a caller opts in.
    private let kvCompression: KVCompressionMode

    /// Read-only accessor on the `kvCompression` mode this backend was
    /// constructed with. Exposed primarily for tests and `/v1/health`-style
    /// observability so callers can confirm the configured mode without
    /// having to introspect private state.
    var currentKVCompression: KVCompressionMode { kvCompression }

    /// Number of cache layers on which `turboQuantEnabled = true` was
    /// successfully set in the most recent `generate(...)` call. When
    /// `kvCompression == .turbo3` and this is `0`, the run silently fell
    /// back to FP16 (the model's cache type does not adopt
    /// `TurboQuantCapable`); the generate path also logs an error in that
    /// case. Reset at the start of every generate call.
    private(set) var lastTurboLayersEnabled: Int = 0

    /// Total number of cache layers seen during the most recent
    /// `generate(...)` call (irrespective of whether they accepted the
    /// turbo flag). Together with `lastTurboLayersEnabled` this lets a
    /// caller distinguish "no layers exist" from "layers exist but none
    /// were `TurboQuantCapable`".
    private(set) var lastTurboLayersTotal: Int = 0

    /// Whether the most recent `generate(...)` call's prompt token count
    /// exceeded the upstream `turboMinActivationTokens` gate (default 2048).
    /// Below the gate, even a `.turbo3` cache stays on FP16. Surfacing this
    /// makes the "I enabled turbo3 and nothing happened" triage path
    /// observable. Reset at the start of every generate call.
    private(set) var lastActivationGatePassed: Bool = false

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
        bundleIdentifier: String = "live.yooz.engine",
        kvCompression: KVCompressionMode = EngineConfig.kvCompression
    ) {
        self.identifier = modelType.rawValue
        self.modelType = modelType
        self.bundleIdentifier = bundleIdentifier
        self.kvCompression = kvCompression
    }

    // MARK: - LLMBackend Protocol

    func load() async throws {
        guard !isLoaded else { return }

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
            throw error
        } catch {
            logger.error("Failed to load model: \(error.localizedDescription)")
            throw LLMError.loadFailed(error.localizedDescription)
        }
        #else
        logger.error("MLXLMCommon / MLXHuggingFace not available")
        throw LLMError.notAvailable("MLX framework not linked. Please rebuild with mlx-swift-lm package.")
        #endif
    }

    func unload() {
        #if canImport(MLXLMCommon)
        modelContainer = nil
        cachedPromptKVState = nil
        cachedPromptTokenCount = 0
        cachedSystemPrompt = nil
        #endif
        isLoaded = false
        downloadProgress = 0
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

        // Reset per-call observability counters before doing any work.
        // If the call short-circuits early (e.g., model not available)
        // these stay at 0/false, which is the correct interpretation
        // ("no turbo activation occurred this call").
        self.lastTurboLayersEnabled = 0
        self.lastTurboLayersTotal = 0
        self.lastActivationGatePassed = false

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
            // Capture into a local: actor-isolated `self.kvCompression`
            // cannot cross the Sendable closure boundary in
            // `container.perform { context in ... }`. The local copy is
            // a `Sendable` value so the closure can read it directly.
            let kvCompressionMode = self.kvCompression
            // Activation-gate visibility (issue #10 / silent-failure-hunter
            // finding #3). The upstream `turboMinActivationTokens` default
            // is 2048; below that, even a `.turbo3` cache stays on FP16.
            // Surface this at info so triage can distinguish "compression
            // requested but gate filtered it out" from "compression
            // requested and silently no-op'd because the cache type
            // changed" (the latter is logged below as an error).
            let promptTokenCount = fullInput.text.tokens.size
            let gatePassed = promptTokenCount >= 2048
            self.lastActivationGatePassed = gatePassed
            if kvCompressionMode == .turbo3 && !gatePassed {
                logger.info(
                    "turbo3 requested but prompt tokens=\(promptTokenCount) < activation gate (2048); generation will use FP16 (expected for TouchUp / chat-turn workloads)."
                )
            }

            let result: GenerateResult = try await container.perform { context in
                // Create fresh KV cache and restore cached system prompt state if available
                var cache = context.model.newCache(parameters: params)
                if let savedState = savedKVState {
                    for i in 0..<min(cache.count, savedState.count) {
                        cache[i].state = savedState[i]
                    }
                    eval(cache)
                }

                // TurboQuant KV cache compression (issue #10).
                // When `kvCompression == .turbo3`, opt every per-layer
                // `TurboQuantCapable` cache into SharpAI's TurboKV path.
                // The upstream class gates actual compression behind
                // `turboMinActivationTokens` (default 2048), so short
                // workloads (TouchUp / chat) stay on FP16.
                //
                // All currently-supported Yooz models (Yooz-Light Qwen2.5-0.5B
                // and Yooz-Quality Qwen3-1.7B, both head_dim=128) return
                // `KVCacheSimple` for every layer, which conforms to
                // `TurboQuantCapable`. Sliding-window models (Mistral, Gemma)
                // would yield `RotatingKVCache` for some layers and turbo3
                // does not apply to those. Quantized variants
                // (`QuantizedKVCache`, when `kvBits != nil`) are also skipped.
                //
                // The upstream `KVCacheSimple` self-disables
                // `turboQuantEnabled` at runtime when `head_dim ∉ {128, 256, 512}`
                // (verified at SharpAI mlx-swift-lm KVCache.swift:395-405),
                // so we don't need a head_dim guard here — but we count and
                // surface "no layers accepted the flag" as an error so a
                // future fork rename or cache-type change is observable
                // via `lastTurboLayersEnabled`.
                var turboEnabledCount = 0
                let turboTotalCount = cache.count
                if kvCompressionMode == .turbo3 {
                    var skippedTypes: [String] = []
                    for layer in cache {
                        if let capable = layer as? TurboQuantCapable {
                            capable.turboQuantEnabled = true
                            turboEnabledCount += 1
                        } else {
                            skippedTypes.append(String(describing: type(of: layer)))
                        }
                    }
                    if turboEnabledCount == 0 {
                        let typeList = skippedTypes.isEmpty
                            ? "<no layers>"
                            : Array(Set(skippedTypes)).joined(separator: ", ")
                        logger.error(
                            "turbo3 requested but no TurboQuantCapable cache layers found (cache types=[\(typeList)]); compression will not activate, falling back to FP16."
                        )
                    } else if !skippedTypes.isEmpty {
                        let typeList = Array(Set(skippedTypes)).joined(separator: ", ")
                        logger.warning("turbo3: enabled on \(turboEnabledCount)/\(turboTotalCount) layers; skipped types=[\(typeList)]")
                    } else {
                        logger.info("turbo3: enabled on all \(turboEnabledCount) cache layers (gate triggers at >=2048 tokens)")
                    }
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
                    // mlx-swift-lm 3.x changed `.chunk(String)` to
                    // `.chunk(String, tokenId: Int)`. We don't track per-chunk
                    // token IDs in this path, so the second associated value
                    // is ignored. The wildcard `_` (rather than a named
                    // placeholder) means a future 3.x minor that adds a
                    // third associated value will fail-fast at compile time.
                    if case let .chunk(chunk, _) = generation {
                        text += chunk
                    }
                }

                // Trim the cache back to the system prompt tokens and
                // snapshot for next call's prompt-cache reuse.
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
                    // KVCacheSimple.trim() workaround for SharpAI fork.
                    //
                    // The fork's `trim(_:)` only decrements `offset`; it
                    // does NOT clear `polarKeys` / `compressedOffset`. If a
                    // generation crossed the 2048-token activation gate,
                    // user-side tokens were compressed into `polarKeys`.
                    // After trimming back to `sysCount`, those compressed
                    // user tokens still live in `polarKeys`, and the
                    // `state` getter will decode them and concatenate with
                    // the (now-trimmed) hot window — meaning the snapshot
                    // contains user-side history we explicitly tried to
                    // strip out.
                    //
                    // Defensive bail-out: if any layer has
                    // `compressedOffset > sysCount`, skip the snapshot.
                    // The next call re-computes the system prompt from
                    // scratch, which is correct (just slower). Filed
                    // upstream at SharpAI/mlx-swift-lm; remove this guard
                    // once `trim(_:)` clears `polarKeys` (or once
                    // `state` getter respects `offset` for the polar band).
                    var compressedBeyondSys = false
                    for layer in cache {
                        if let simple = layer as? KVCacheSimple,
                           simple.compressedOffset > sysCount {
                            compressedBeyondSys = true
                            break
                        }
                    }
                    if compressedBeyondSys {
                        logger.warning(
                            "Skipping prompt-cache snapshot: SharpAI trim() cannot evict user-side compressed history. Next call recomputes the system prompt."
                        )
                        snapshot = nil
                    } else {
                        // SharpAI fork's `KVCacheSimple.state` getter
                        // decodes `polarKeys` back to fp16 and
                        // concatenates with the hot window, so the
                        // snapshot is fork-agnostic — both FP16-only and
                        // turbo3-active runs return a valid fp16 pair the
                        // setter can restore. See SharpAI mlx-swift-lm
                        // KVCache.swift lines ~476-509.
                        snapshot = cache.map(\.state)
                    }
                }
                return GenerateResult(
                    text: text,
                    kvSnapshot: snapshot,
                    turboEnabled: turboEnabledCount,
                    turboTotal: turboTotalCount
                )
            }

            // Surface the TurboQuant outcome on the actor for tests /
            // health-style observability.
            self.lastTurboLayersEnabled = result.turboEnabled
            self.lastTurboLayersTotal = result.turboTotal

            // Update cached state
            if let snapshot = result.kvSnapshot {
                cachedPromptKVState = snapshot
                cachedPromptTokenCount = sysCount
                cachedSystemPrompt = systemPrompt
                if savedKVState == nil {
                    logger.info("Cached system prompt KV (\(sysCount) tokens)")
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
    /// Construct a backend, optionally overriding the engine-wide
    /// `kvCompression` default. Passing `nil` (or omitting) keeps the
    /// `EngineConfig.kvCompression` default, so existing call sites do not
    /// change behavior. Per-request callers (e.g., the
    /// `/v1/llm/generate` handler in `APIServer`) pass the user-supplied
    /// override here.
    static func create(
        for type: LLMModelType,
        bundleIdentifier: String = "live.yooz.engine",
        kvCompression: KVCompressionMode? = nil
    ) -> MLXLLMBackend {
        return MLXLLMBackend(
            modelType: type,
            bundleIdentifier: bundleIdentifier,
            kvCompression: kvCompression ?? EngineConfig.kvCompression
        )
    }

    static func createLight(
        bundleIdentifier: String = "live.yooz.engine",
        kvCompression: KVCompressionMode? = nil
    ) -> MLXLLMBackend {
        return create(
            for: .yoozLight,
            bundleIdentifier: bundleIdentifier,
            kvCompression: kvCompression
        )
    }

    static func createQuality(
        bundleIdentifier: String = "live.yooz.engine",
        kvCompression: KVCompressionMode? = nil
    ) -> MLXLLMBackend {
        return create(
            for: .yoozQuality,
            bundleIdentifier: bundleIdentifier,
            kvCompression: kvCompression
        )
    }
}
