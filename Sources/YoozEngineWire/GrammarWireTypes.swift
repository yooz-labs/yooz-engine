// GrammarWireTypes.swift
// YoozEngineWire
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

public struct GrammarCheckRequest: Codable, Sendable {
    public let text: String
    public let categories: [String]?
    /// Use NLTagger POS tagging for more accurate correction.
    /// Defaults to true server-side when nil.
    public let usePOS: Bool?

    public init(text: String, categories: [String]? = nil, usePOS: Bool? = nil) {
        self.text = text
        self.categories = categories
        self.usePOS = usePOS
    }
}

public struct GrammarCheckResponse: Codable, Sendable {
    public let result: String
    public let correctionsApplied: Int
    /// Total rule count used for this check (nil if server does not report it).
    public let ruleCount: Int?

    public init(result: String, correctionsApplied: Int, ruleCount: Int?) {
        self.result = result
        self.correctionsApplied = correctionsApplied
        self.ruleCount = ruleCount
    }
}
