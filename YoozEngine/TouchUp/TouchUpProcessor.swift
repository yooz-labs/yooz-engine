// TouchUpProcessor.swift
// LLMModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore
import Foundation
import os.log

private let logger = Logger(subsystem: "live.yooz.engine", category: "TouchUpProcessor")

/// Smart routing processor for AI touch-up of transcriptions.
///
/// Uses two-model routing strategy:
/// - No replacements: the Light tier proofreads only (fast path)
/// - Has replacements: the Quality tier validates replacements + proofreads
///
/// Pipeline:
/// ```
/// Whisper -> Fuzzy Dict -> Regex Voice Cmds -> Router -> Model(s) -> Output
/// ```
public enum TouchUpProcessor {
    // MARK: - Types

    /// Which model was used for processing.
    public enum ModelUsed: String, Sendable {
        case light = "yooz-light-v3"
        case quality = "yooz-quality-v3"
        case foundationModels = "foundation-models"
        case regexOnly = "regex-only"
        case fallbackRegex = "fallback-regex"
    }

    /// A proposed fuzzy/dictionary replacement to validate.
    public struct Replacement: Sendable {
        public let original: String
        public let replacement: String

        public init(original: String, replacement: String) {
            self.original = original
            self.replacement = replacement
        }
    }

    /// Result of smart processing.
    public struct ProcessResult: Sendable {
        public let text: String
        public let keepDecisions: [Bool]
        public let modelUsed: ModelUsed
        public let latencyMs: Double
        public let fallbackReason: String?

        public init(
            text: String,
            keepDecisions: [Bool],
            modelUsed: ModelUsed,
            latencyMs: Double,
            fallbackReason: String?
        ) {
            self.text = text
            self.keepDecisions = keepDecisions
            self.modelUsed = modelUsed
            self.latencyMs = latencyMs
            self.fallbackReason = fallbackReason
        }
    }

    // MARK: - Voice Commands (Regex-based)

    /// Voice commands that can be processed with regex.
    /// Model-based classification is not needed; regex achieves 95.8% accuracy.
    static let voiceCommands: [String: (patterns: [String], replacement: String)] = [
        // Line breaks
        "new_line": (["new line", "newline"], "\n"),
        "new_paragraph": (["new paragraph"], "\n\n"),
        // Punctuation
        "period": (["period", "full stop"], ". "),
        "comma": (["comma"], ", "),
        "question_mark": (["question mark"], "? "),
        "exclamation": (["exclamation mark", "exclamation point"], "! "),
        "colon": (["colon"], ": "),
        "semicolon": (["semicolon", "semi colon"], "; "),
        // Lists
        "bullet": (["bullet point", "start bullet", "start bullets"], "\n\u{2022} "),
        "number_1": (["number one"], "\n1. "),
        "number_2": (["number two"], "\n2. "),
        "number_3": (["number three"], "\n3. "),
        "number_4": (["number four"], "\n4. "),
        "number_5": (["number five"], "\n5. "),
    ]

    /// Detect which voice commands are present in text.
    static func detectCommands(in text: String) -> [String] {
        var found: [String] = []
        let textLower = text.lowercased()

        for (cmdId, cmdInfo) in voiceCommands {
            for pattern in cmdInfo.patterns {
                let escapedPattern = NSRegularExpression.escapedPattern(for: pattern)
                let regex = try? NSRegularExpression(
                    pattern: "\\b\(escapedPattern)\\b",
                    options: .caseInsensitive
                )
                if let regex,
                   regex.firstMatch(in: textLower, range: NSRange(textLower.startIndex..., in: textLower)) != nil
                {
                    found.append(cmdId)
                    break
                }
            }
        }

        return found
    }

    /// Apply voice commands using regex replacement.
    static func applyCommands(_ text: String, commands: [String]? = nil) -> String {
        var result = text
        let commandsToApply = commands ?? Array(voiceCommands.keys)

        for cmdId in commandsToApply {
            guard let cmdInfo = voiceCommands[cmdId] else { continue }

            for pattern in cmdInfo.patterns {
                let escapedPattern = NSRegularExpression.escapedPattern(for: pattern)
                if let regex = try? NSRegularExpression(
                    pattern: "\\s*\\b\(escapedPattern)\\b\\s*",
                    options: .caseInsensitive
                ) {
                    result = regex.stringByReplacingMatches(
                        in: result,
                        range: NSRange(result.startIndex..., in: result),
                        withTemplate: cmdInfo.replacement
                    )
                }
            }
        }

        // Clean up whitespace
        result = result.replacingOccurrences(of: "[ \\t]+", with: " ", options: .regularExpression)
        result = result.replacingOccurrences(of: "\n +", with: "\n", options: .regularExpression)
        result = result.replacingOccurrences(of: " +\n", with: "\n", options: .regularExpression)
        result = result.trimmingCharacters(in: .whitespacesAndNewlines)

        // Capitalize after sentence-ending punctuation
        if let regex = try? NSRegularExpression(pattern: "([.!?]\\s+)([a-z])", options: []) {
            var mutableResult = result
            let matches = regex.matches(in: result, range: NSRange(result.startIndex..., in: result))
            // Process in reverse to avoid index shifting
            for match in matches.reversed() {
                if let range = Range(match.range(at: 2), in: mutableResult),
                   let fullRange = Range(match.range, in: mutableResult)
                {
                    let char = String(mutableResult[range]).uppercased()
                    let punctuation = String(mutableResult[Range(match.range(at: 1), in: mutableResult)!])
                    mutableResult.replaceSubrange(fullRange, with: punctuation + char)
                }
            }
            result = mutableResult
        }

        // Capitalize first letter
        if let first = result.first, first.isLowercase {
            result = first.uppercased() + String(result.dropFirst())
        }

        return result
    }

    // MARK: - Smart Processing

    /// Process text with smart two-model routing.
    ///
    /// - Parameters:
    ///   - text: Text with fuzzy replacements already applied
    ///   - replacements: List of fuzzy replacements to validate
    ///   - lightModel: Fast model for proofreading (Yooz-Light)
    ///   - qualityModel: Quality model for validation (Yooz-Quality)
    ///   - proofreadPrompt: System prompt for proofreading (allows mode-specific selection)
    ///   - workloadClass: GPU admission class (engine#228). Defaults to
    ///     `.background` — TouchUp generation is throughput work per the
    ///     issue's classification, so it queues/yields behind a
    ///     concurrently-active interactive workload (a live streaming STT
    ///     session) rather than contending for the GPU.
    /// - Returns: Processed result with final text and metadata
    static func process(
        text: String,
        replacements: [Replacement],
        lightModel: any LLMBackend,
        qualityModel: any LLMBackend,
        proofreadPrompt: String = TouchUpPrompts.proofread,
        workloadClass: MLXWorkloadClass = .background
    ) async -> ProcessResult {
        let startTime = CFAbsoluteTimeGetCurrent()

        // Step 1: Apply voice commands with regex (instant)
        var processedText = text
        let commands = detectCommands(in: text)
        if !commands.isEmpty {
            processedText = applyCommands(text, commands: commands)
            logger.debug("Applied \(commands.count) voice commands")
        }

        // Step 2: Route based on whether replacements need validation
        if replacements.isEmpty {
            // Fast path: just proofread with light model
            do {
                let response = try await lightModel.generate(
                    prompt: processedText,
                    systemPrompt: proofreadPrompt,
                    workloadClass: workloadClass,
                    // Proofreading: the salvage pass belongs here (engine#312).
                    postProcess: true
                )

                let (resultText, success) = parseProofreadResponse(response, fallback: processedText)
                let latencyMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000

                logger.info("Proofread complete in \(Int(latencyMs))ms, success=\(success)")

                guard success else {
                    // The model ran and returned a response, but parsing it
                    // failed (malformed JSON, placeholder echo, etc.) —
                    // `resultText` is the unprocessed fallback, not real
                    // cleanup, so this must not read as a successful `.light`
                    // pass (engine#279 review: both branches previously
                    // hardcoded `fallbackReason: nil` here regardless of
                    // `success`, silently masking parse failures both in
                    // logs and in the wire `warnings` field, which forwards
                    // `fallbackReason` at the transport layer).
                    logger.warning("Light model response failed to parse; returning unprocessed text")
                    return ProcessResult(
                        text: resultText,
                        keepDecisions: [],
                        modelUsed: .fallbackRegex,
                        latencyMs: latencyMs,
                        fallbackReason: "Light model response failed to parse"
                    )
                }

                return ProcessResult(
                    text: resultText,
                    keepDecisions: [],
                    modelUsed: .light,
                    latencyMs: latencyMs,
                    fallbackReason: nil
                )
            } catch is CancellationError {
                // Cancelled while queued at the GPU admission gate or
                // mid-generation (engine#228) — the caller went away, so
                // nobody consumes this result. Return the regex-processed
                // text with a truthful reason instead of logging a phantom
                // "model failed" error that would pollute failure triage.
                logger.debug("Light model generation cancelled")
                let latencyMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
                return ProcessResult(
                    text: processedText,
                    keepDecisions: [],
                    modelUsed: .fallbackRegex,
                    latencyMs: latencyMs,
                    fallbackReason: "Generation cancelled"
                )
            } catch {
                logger.error("Light model failed: \(error.localizedDescription)")
                let latencyMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
                return ProcessResult(
                    text: processedText,
                    keepDecisions: [],
                    modelUsed: .fallbackRegex,
                    latencyMs: latencyMs,
                    fallbackReason: "Light model failed: \(error.localizedDescription)"
                )
            }
        } else {
            // Validation path: use quality model
            let prompt = TouchUpPrompts.buildValidationPrompt(
                text: processedText,
                replacements: replacements.map { ($0.original, $0.replacement) }
            )

            do {
                let response = try await qualityModel.generate(
                    prompt: prompt,
                    systemPrompt: TouchUpPrompts.validateAndProofread,
                    workloadClass: workloadClass,
                    // Proofreading: the salvage pass belongs here (engine#312).
                    postProcess: true
                )

                let (resultText, keepDecisions, success) = parseValidateResponse(
                    response,
                    fallback: processedText,
                    numReplacements: replacements.count
                )

                // Apply revert decisions
                var finalText = resultText
                for (replacement, keep) in zip(replacements, keepDecisions) {
                    if !keep {
                        // Revert this replacement
                        finalText = finalText.replacingOccurrences(
                            of: replacement.replacement,
                            with: replacement.original,
                            options: [],
                            range: finalText.range(of: replacement.replacement)
                        )
                        logger.debug("Reverted: '\(replacement.replacement)' -> '\(replacement.original)'")
                    }
                }

                let latencyMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
                let keptCount = keepDecisions.filter { $0 }.count
                logger.info("Validation complete in \(Int(latencyMs))ms, kept \(keptCount)/\(keepDecisions.count), success=\(success)")

                guard success else {
                    // Same masking bug as the fast path above, previously
                    // discarding `success` entirely via `_`: parsing failed,
                    // so `keepDecisions` is the all-revert default and
                    // `finalText` is not real quality-model validation —
                    // must not report `.quality` (engine#279 review).
                    logger.warning("Quality model response failed to parse; reverted all replacements")
                    return ProcessResult(
                        text: finalText,
                        keepDecisions: keepDecisions,
                        modelUsed: .fallbackRegex,
                        latencyMs: latencyMs,
                        fallbackReason: "Quality model response failed to parse"
                    )
                }

                return ProcessResult(
                    text: finalText,
                    keepDecisions: keepDecisions,
                    modelUsed: .quality,
                    latencyMs: latencyMs,
                    fallbackReason: nil
                )
            } catch is CancellationError {
                // See the light-path twin above: a cancelled request is not
                // a model fault; keep failure triage clean.
                logger.debug("Quality model generation cancelled")
                let latencyMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
                return ProcessResult(
                    text: processedText,
                    keepDecisions: replacements.map { _ in true },
                    modelUsed: .fallbackRegex,
                    latencyMs: latencyMs,
                    fallbackReason: "Generation cancelled"
                )
            } catch {
                logger.error("Quality model failed: \(error.localizedDescription)")
                let latencyMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
                // Keep all replacements on failure
                return ProcessResult(
                    text: processedText,
                    keepDecisions: replacements.map { _ in true },
                    modelUsed: .fallbackRegex,
                    latencyMs: latencyMs,
                    fallbackReason: "Quality model failed: \(error.localizedDescription)"
                )
            }
        }
    }

    /// Process text with regex only (no LLM).
    /// Use this as a fast fallback when LLM models are not available.
    static func processRegexOnly(text: String, replacements: [Replacement]) -> ProcessResult {
        let startTime = CFAbsoluteTimeGetCurrent()

        // Apply voice commands with regex (instant)
        var processedText = text
        let commands = detectCommands(in: text)
        if !commands.isEmpty {
            processedText = applyCommands(text, commands: commands)
        }

        let latencyMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000

        return ProcessResult(
            text: processedText,
            keepDecisions: replacements.map { _ in true },
            modelUsed: .regexOnly,
            latencyMs: latencyMs,
            fallbackReason: nil
        )
    }
}
