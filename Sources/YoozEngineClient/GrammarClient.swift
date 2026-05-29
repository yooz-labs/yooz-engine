import Foundation

/// Client for the grammar check API endpoint.
public struct GrammarClient: Sendable {
    private let engine: YoozEngineClient

    init(engine: YoozEngineClient) {
        self.engine = engine
    }

    /// Check and correct grammar in text.
    public func check(_ request: GrammarCheckRequest) async throws -> GrammarCheckResponse {
        let body = try JSONEncoder().encode(request)
        let data = try await engine.post("/v1/grammar/check", body: body)
        return try JSONDecoder().decode(GrammarCheckResponse.self, from: data)
    }

    /// Convenience: correct text with all grammar categories (server defaults to POS tagging).
    public func correct(text: String) async throws -> String {
        let request = GrammarCheckRequest(text: text)
        let response = try await check(request)
        return response.result
    }

    /// Check text and return the structured per-match data for rendering
    /// targeted underlines, alongside the corrected text.
    ///
    /// Match `offset` / `length` are UTF-16 code units in the ORIGINAL `text`.
    /// `matches` is `nil` when talking to an older server that predates the
    /// field; treat that as "no structured matches available".
    ///
    /// - Parameters:
    ///   - text: Text to check.
    ///   - categories: Optional category names restricting which rules apply.
    ///   - usePOS: Use NLTagger POS tagging. Server defaults to true when nil.
    /// - Returns: Corrected text plus optional structured matches.
    public func checkWithMatches(
        text: String,
        categories: [String]? = nil,
        usePOS: Bool? = nil
    ) async throws -> (result: String, matches: [GrammarMatch]?) {
        let request = GrammarCheckRequest(text: text, categories: categories, usePOS: usePOS)
        let response = try await check(request)
        return (response.result, response.matches)
    }

    /// Correct text using tier-appropriate categories.
    ///
    /// - Parameters:
    ///   - text: Text to correct.
    ///   - tier: Subscription tier controlling rule selection.
    /// - Returns: Corrected text.
    public func correct(text: String, tier: GrammarTier) async throws -> String {
        let categories: [String]
        let usePOS: Bool

        switch tier {
        case .free:
            categories = grammarFreeCategories
            usePOS = false
        case .pro, .premium:
            categories = grammarAllCategories
            usePOS = true
        }

        let request = GrammarCheckRequest(text: text, categories: categories, usePOS: usePOS)
        let response = try await check(request)
        return response.result
    }
}
