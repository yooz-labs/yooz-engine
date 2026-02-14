// FoundationModelsBackend.swift
// YoozEngine
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import os.log

#if canImport(FoundationModels)
import FoundationModels

// MARK: - Structured Output Types

/// Input structure for JSON-based proofreading
@available(macOS 26.0, *)
struct ProofreadInput: Codable {
    let text: String
    let rules: [String]
}

/// Structured response for text touch-up using Guided Generation
@available(macOS 26.0, *)
@Generable
struct ProofreadOutput {
    @Guide(description: "The corrected text with all rules applied")
    var correctedText: String

    @Guide(description: "Brief list of changes made, or empty if no changes")
    var changesSummary: String
}
#endif

private let logger = Logger(subsystem: "live.yooz.engine", category: "FoundationModelsBackend")

/// Apple Foundation Models backend using the built-in 3B on-device model.
/// Requires macOS 26+ with Apple Intelligence enabled.
/// Standalone actor; not integrated into the TouchUpEngine pipeline yet.
actor FoundationModelsBackend {

    // MARK: - Properties

    let identifier = "foundation-models"

    private(set) var isLoaded = false

    // MARK: - Lifecycle

    func load() async throws {
        guard !isLoaded else { return }

        #if canImport(FoundationModels)
        if #available(macOS 26.0, *) {
            let testSession = LanguageModelSession()
            _ = testSession
            isLoaded = true
            logger.info("Apple Intelligence ready (single-turn mode)")
        } else {
            throw LLMError.notAvailable("FoundationModels requires macOS 26+")
        }
        #else
        throw LLMError.notAvailable("FoundationModels framework not available. Requires macOS 26+")
        #endif
    }

    func unload() {
        isLoaded = false
        logger.info("Unloaded")
    }

    // MARK: - Generation

    func generate(prompt: String, systemPrompt: String?) async throws -> String {
        guard isLoaded else {
            throw LLMError.notLoaded
        }

        #if canImport(FoundationModels)
        guard #available(macOS 26.0, *) else {
            throw LLMError.notAvailable("FoundationModels requires macOS 26+")
        }

        let jsonInput = formatAsJSON(text: prompt, systemPrompt: systemPrompt)

        let instructions = """
        You are a proofreader. Read the JSON input and apply the rules to correct the text.
        Output ONLY the corrected text in the correctedText field.
        The changesSummary should briefly note what was changed, or be empty if no changes.
        """

        let session = LanguageModelSession(instructions: instructions)

        do {
            logger.debug("Starting generation for: \(prompt.prefix(50))...")

            let response = try await session.respond(to: jsonInput, generating: ProofreadOutput.self)

            var result = response.content.correctedText.trimmingCharacters(in: .whitespacesAndNewlines)
            let changes = response.content.changesSummary
            if !changes.isEmpty {
                logger.debug("Changes: \(changes.prefix(100))...")
            }

            result = sanitizeOutput(result)

            if isInstructionLeakage(result, originalInput: prompt) {
                logger.warning("Detected instruction leakage, returning original")
                return prompt
            }

            return result
        } catch {
            logger.error("Structured generation error: \(error.localizedDescription)")

            // Fallback to free-form
            logger.info("Falling back to free-form generation")
            do {
                var fallbackResult = try await generateFreeForm(prompt: prompt, systemPrompt: systemPrompt)
                fallbackResult = sanitizeOutput(fallbackResult)

                if isInstructionLeakage(fallbackResult, originalInput: prompt) {
                    logger.warning("Instruction leakage in fallback, returning original")
                    return prompt
                }

                return fallbackResult
            } catch {
                logger.error("Free-form fallback also failed: \(error.localizedDescription)")
                throw LLMError.generationFailed(error.localizedDescription)
            }
        }
        #else
        throw LLMError.notAvailable("FoundationModels framework not available")
        #endif
    }

    // MARK: - Availability

    nonisolated func isAvailable() -> Bool {
        #if canImport(FoundationModels)
        return true
        #else
        return false
        #endif
    }

    // MARK: - Private Helpers

    @available(macOS 26.0, *)
    private func formatAsJSON(text: String, systemPrompt: String?) -> String {
        let rules = extractRules(from: systemPrompt)
        let input = ProofreadInput(text: text, rules: rules)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]

        do {
            let jsonData = try encoder.encode(input)
            if let jsonString = String(data: jsonData, encoding: .utf8) {
                return jsonString
            }
        } catch {
            logger.warning("JSON encoding failed: \(error.localizedDescription)")
        }

        // Fallback: manual JSON
        let escapedText = text
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
            .replacingOccurrences(of: "\r", with: "\\r")
            .replacingOccurrences(of: "\t", with: "\\t")
        return """
        {
          "text": "\(escapedText)",
          "rules": ["proofread", "fix grammar", "convert numbers"]
        }
        """
    }

    private func extractRules(from systemPrompt: String?) -> [String] {
        guard let prompt = systemPrompt, !prompt.isEmpty else {
            return ["Fix grammar and punctuation", "Convert spoken numbers to digits"]
        }

        var rules: [String] = []

        if prompt.contains("Times:") || prompt.contains("a.m.") {
            rules.append("Convert times: 'two a.m.' to '2 a.m.', 'nine thirty p.m.' to '9:30 p.m.'")
        }
        if prompt.contains("Numbers:") || prompt.contains("percent") {
            rules.append("Convert numbers: 'seventeen percent' to '17%', 'twenty three dollars' to '$23'")
        }
        if prompt.contains("Decimals:") || prompt.contains("point") {
            rules.append("Convert decimals: 'three point five' to '3.5'")
        }
        if prompt.contains("Ordinals:") || prompt.contains("1st") {
            rules.append("Convert ordinals: 'first' to '1st', 'February twenty first' to 'February 21st'")
        }
        if prompt.contains("Years:") || prompt.contains("1984") {
            rules.append("Convert years: 'nineteen eighty four' to '1984'")
        }
        if prompt.contains("Versions:") || prompt.contains("0.7.12") {
            rules.append("Convert versions: 'zero point seven point twelve' to '0.7.12'")
        }
        if prompt.contains("Files:") || prompt.contains("dot md") {
            rules.append("Convert file names: 'claude dot md' to 'claude.md'")
        }
        if prompt.contains("grammar") {
            rules.append("Fix grammar, punctuation, and misheard words")
        }
        if prompt.contains("Preserve") {
            rules.append("Preserve names and technical terms")
        }
        if prompt.contains("Self-corrections") || prompt.contains("scratch that") {
            rules.append("Handle self-corrections and remove 'scratch that', 'never mind'")
        }

        if rules.isEmpty {
            rules = ["Fix grammar and punctuation", "Convert spoken numbers to digits", "Preserve meaning"]
        }

        return rules
    }

    private func isInstructionLeakage(_ output: String, originalInput: String) -> Bool {
        if output.contains("\u{2192}") {  // Arrow character
            return true
        }

        let leakagePatterns = [
            "two a.m is 2 a.m",
            "nine thirty p.m is 9:30",
            "seventeen percent is 17",
            "twenty three dollars is",
            "three point five is 3.5",
            "first is 1st",
            "february twenty first is",
            "nineteen eighty four is 1984",
            "zero point seven point twelve",
            "claude.md, script.py",
            "fix grammar, punctuation",
            "preserve names and technical"
        ]

        let lowercaseOutput = output.lowercased()
        for pattern in leakagePatterns {
            if lowercaseOutput.contains(pattern) {
                return true
            }
        }

        if output.count > originalInput.count * 4 {
            let commaCount = output.filter { $0 == "," }.count
            if commaCount >= 5 {
                return true
            }
        }

        return false
    }

    private func sanitizeOutput(_ text: String) -> String {
        var result = text
        result = result.replacingOccurrences(of: "\u{00BD}", with: "'")  // ½
        result = result.replacingOccurrences(of: "\u{00BC}", with: "'")  // ¼
        result = result.replacingOccurrences(of: "\u{00BE}", with: "'")  // ¾
        result = result.replacingOccurrences(of: "\u{2018}", with: "'")  // Left single quote
        result = result.replacingOccurrences(of: "\u{2019}", with: "'")  // Right single quote
        result = result.replacingOccurrences(of: "\u{201C}", with: "\"") // Left double quote
        result = result.replacingOccurrences(of: "\u{201D}", with: "\"") // Right double quote
        return result
    }

    #if canImport(FoundationModels)
    @available(macOS 26.0, *)
    private func generateFreeForm(prompt: String, systemPrompt: String?) async throws -> String {
        let instructions = (systemPrompt ?? "Proofread this text.") + "\nOutput ONLY the corrected text, nothing else."
        let session = LanguageModelSession(instructions: instructions)

        let response = try await session.respond(to: prompt)
        var result = response.content.trimmingCharacters(in: .whitespacesAndNewlines)

        let preamblePatterns = [
            "Certainly, here is the corrected text:",
            "Here is the corrected text:",
            "Corrected text:",
            "Output:",
            "Result:"
        ]
        for pattern in preamblePatterns {
            if result.hasPrefix(pattern) {
                result = String(result.dropFirst(pattern.count)).trimmingCharacters(in: .whitespacesAndNewlines)
                if result.hasPrefix("\"") && result.hasSuffix("\"") && result.count > 2 {
                    result = String(result.dropFirst().dropLast())
                }
                break
            }
        }
        return result
    }
    #endif
}
