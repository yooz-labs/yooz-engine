// JSONParsing.swift
// YoozEngine
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import os.log

private let logger = Logger(subsystem: "live.yooz.engine", category: "JSONParsing")

// MARK: - Response Structures

/// Response from proofreading (fast model)
struct ProofreadResponse: Codable, Sendable {
    let result: String

    init(result: String) {
        self.result = result
    }
}

/// Response from validation + proofreading (quality model)
struct ValidateResponse: Codable, Sendable {
    let result: String
    let keep: [Bool]

    init(result: String, keep: [Bool]) {
        self.result = result
        self.keep = keep
    }
}

// MARK: - Parsing Functions

/// Parse a proofreading response from the LLM
/// - Parameters:
///   - response: Raw LLM response text
///   - fallback: Fallback text if parsing fails
/// - Returns: Tuple of (result text, success flag)
func parseProofreadResponse(_ response: String, fallback: String) -> (text: String, success: Bool) {
    let trimmed = response.trimmingCharacters(in: .whitespacesAndNewlines)

    // Try direct parse
    if let data = trimmed.data(using: .utf8),
       let decoded = try? JSONDecoder().decode(ProofreadResponse.self, from: data) {
        return (decoded.result, true)
    }

    // Try to find JSON in the response (model may add extra text)
    if let jsonText = extractJSON(from: trimmed),
       let data = jsonText.data(using: .utf8),
       let decoded = try? JSONDecoder().decode(ProofreadResponse.self, from: data) {
        return (decoded.result, true)
    }

    // Parsing failed
    logger.warning("Failed to parse proofread response. Preview: \(trimmed.prefix(200))")
    return (fallback, false)
}

/// Parse a validation response from the LLM
/// - Parameters:
///   - response: Raw LLM response text
///   - fallback: Fallback text if parsing fails
///   - numReplacements: Expected number of keep decisions
/// - Returns: Tuple of (result text, keep decisions, success flag)
func parseValidateResponse(
    _ response: String,
    fallback: String,
    numReplacements: Int
) -> (text: String, keepDecisions: [Bool], success: Bool) {
    let trimmed = response.trimmingCharacters(in: .whitespacesAndNewlines)
    // Default to reverting all replacements when parsing fails (safer)
    let defaultKeep = Array(repeating: false, count: numReplacements)

    // Try direct parse
    if let data = trimmed.data(using: .utf8),
       let decoded = try? JSONDecoder().decode(ValidateResponse.self, from: data) {
        let keep = normalizeKeepDecisions(decoded.keep, expected: numReplacements)
        return (decoded.result, keep, true)
    }

    // Try to find JSON in the response
    if let jsonText = extractJSON(from: trimmed),
       let data = jsonText.data(using: .utf8),
       let decoded = try? JSONDecoder().decode(ValidateResponse.self, from: data) {
        let keep = normalizeKeepDecisions(decoded.keep, expected: numReplacements)
        return (decoded.result, keep, true)
    }

    // Parsing failed - log and return safe defaults (revert all)
    logger.warning("Failed to parse validation response, reverting all. Preview: \(trimmed.prefix(200))")
    return (fallback, defaultKeep, false)
}

// MARK: - Helper Functions

/// Extract JSON object from a string that may contain extra text
private func extractJSON(from text: String) -> String? {
    guard let startIndex = text.firstIndex(of: "{"),
          let endIndex = text.lastIndex(of: "}"),
          startIndex < endIndex else {
        return nil
    }
    return String(text[startIndex...endIndex])
}

/// Normalize keep decisions to match expected count
private func normalizeKeepDecisions(_ keep: [Bool], expected: Int) -> [Bool] {
    if keep.count == expected {
        return keep
    } else if keep.count > expected {
        return Array(keep.prefix(expected))
    } else {
        // Pad with true (keep by default if not specified)
        return keep + Array(repeating: true, count: expected - keep.count)
    }
}
