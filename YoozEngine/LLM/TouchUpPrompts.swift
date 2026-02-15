// TouchUpPrompts.swift
// YoozEngine
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// System prompts for LLM touch-up processing
enum TouchUpPrompts {

    // MARK: - Proofreading (Fast Model - Standard Mode)

    /// Prompt for fast proofreading with Yooz-Light (Qwen2.5-0.5B)
    /// Used for Standard mode when there are no vocabulary replacements to validate
    static let proofread = """
        Fix grammar, capitalize properly, and convert spoken numbers to digits. Convert spoken version numbers like "zero point four point zero" to "0.4.0". Keep ALL sentences. Return the fixed text as JSON.

        Input: the meeting is at two pm on march fifteenth
        {"result": "The meeting is at 2 PM on March 15th."}

        Input: we need about fifty units ready by friday and I think we should prepare
        {"result": "We need about 50 units ready by Friday and I think we should prepare."}

        Input: we are releasing version zero point four point zero next week
        {"result": "We are releasing version 0.4.0 next week."}

        Input: update it to version one point six point three and test it
        {"result": "Update it to version 1.6.3 and test it."}

        Input: he said it would cost around one hundred and fifty dollars but we can negotiate
        {"result": "He said it would cost around $150 but we can negotiate."}

        Always respond with ONLY a JSON object. Never remove sentences. Never include explanations.
        """

    // MARK: - Rewriting (Full Mode)

    /// Prompt for aggressive rewriting with clarity improvements
    /// Used for Full mode - matches Apple Intelligence level of processing
    static let rewrite = """
        Rewrite voice transcription for clarity and conciseness. Fix grammar, convert numbers, fix misheard words, remove filler words (um, uh, like, you know), handle self-corrections. Return the fixed text as JSON.

        Input: um so like the meeting is at two pm on march fifteenth you know
        {"result": "The meeting is at 2 PM on March 15th."}

        Input: we need about fifty no wait I meant sixty units ready by friday
        {"result": "We need about 60 units ready by Friday."}

        Input: we are releasing version zero point four point zero next week scratch that make it zero point five
        {"result": "We are releasing version 0.5.0 next week."}

        Input: update it to version one point six point three and uh test it thoroughly
        {"result": "Update it to version 1.6.3 and test it thoroughly."}

        Input: he said it would cost around one hundred and fifty dollars but um we can negotiate
        {"result": "He said it would cost around $150 but we can negotiate."}

        Remove: "scratch that", "never mind", "delete that" and preceding phrase. Convert spoken numbers and version numbers. Fix grammar and misheard words. Always respond with ONLY a JSON object. Never include explanations.
        """

    // MARK: - Validation + Proofreading (Quality Model)

    /// Prompt for validation + proofreading with Yooz-Quality (Qwen3-1.7B)
    /// Used when vocabulary replacements need contextual validation
    static let validateAndProofread = """
        Do TWO tasks: (1) For each replacement, decide if it fits the context. (2) Fix grammar, capitalize, convert spoken numbers to digits.

        Input: {"text": "I talked to Claude about fifty dollars", "replacements": [{"orig": "cloud", "repl": "Claude"}]}
        {"result": "I talked to Claude about $50.", "keep": [true]}

        Input: {"text": "The Claude is fluffy", "replacements": [{"orig": "cloud", "repl": "Claude"}]}
        {"result": "The cloud is fluffy.", "keep": [false]}

        Input: {"text": "Ask Siri to set a timer for thirty minutes", "replacements": [{"orig": "series", "repl": "Siri"}]}
        {"result": "Ask Siri to set a timer for 30 minutes.", "keep": [true]}

        Input: {"text": "The Siri is on Netflix", "replacements": [{"orig": "series", "repl": "Siri"}]}
        {"result": "The series is on Netflix.", "keep": [false]}

        Always respond with ONLY a JSON object. Never include explanations.
        """

    // MARK: - Prompt Building

    /// Build a validation prompt with replacements
    /// - Parameters:
    ///   - text: The text with replacements already applied
    ///   - replacements: Array of (original, replacement) tuples
    /// - Returns: JSON prompt for the validation model
    static func buildValidationPrompt(
        text: String,
        replacements: [(original: String, replacement: String)]
    ) -> String {
        let replacementsJSON = replacements.map { r in
            "{\"orig\": \"\(escapeJSON(r.original))\", \"repl\": \"\(escapeJSON(r.replacement))\"}"
        }.joined(separator: ", ")

        return "{\"text\": \"\(escapeJSON(text))\", \"replacements\": [\(replacementsJSON)]}"
    }

    /// Escape special characters for JSON
    private static func escapeJSON(_ string: String) -> String {
        var result = string
        result = result.replacingOccurrences(of: "\\", with: "\\\\")
        result = result.replacingOccurrences(of: "\"", with: "\\\"")
        result = result.replacingOccurrences(of: "\n", with: "\\n")
        result = result.replacingOccurrences(of: "\r", with: "\\r")
        result = result.replacingOccurrences(of: "\t", with: "\\t")
        return result
    }
}
