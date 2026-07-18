// TouchUpContextInjectionTests.swift
// LLMModuleTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Pure, ungated unit tests for the Phase 4 context-injection helpers
// (engine#280 / whisper#317): `TouchUpEngine.cappedContextVocabulary` (the
// server-side defensive cap, applied on receipt regardless of transport or
// mode) and `TouchUpEngine.withContext` (the eval-gated prompt composition
// step). Both are synchronous, model-free transforms — no LLM weights are
// loaded to exercise them, matching the project's real-tests-only policy
// without paying the model-load cost that gates the rest of this file's
// deeper eval-gate coverage (`TouchUpFidelityEvalTests`).
//
// `withContext` is deliberately tested as a pure function rather than
// through `TouchUpEngine.process(...)`/`processWithActiveModel(...)`:
// those two only reach `selectPrompt`/`withContext` once a real light or
// quality backend has finished loading (mode `.off` and a load failure both
// return before that point), so exercising the composition through the
// full call would itself require `YOOZ_LLM_LOAD_MODELS=1` real weights —
// that end-to-end proof lives in `TouchUpFidelityEvalTests`'s context-block
// fixtures instead.

import XCTest
@testable import EngineCore
@testable import LLMModule

final class TouchUpContextInjectionTests: XCTestCase {

    private func withEnvVar(
        _ name: String,
        value: String?,
        _ body: () throws -> Void
    ) rethrows {
        let prior = ProcessInfo.processInfo.environment[name]
        defer {
            if let prior {
                setenv(name, prior, 1)
            } else {
                unsetenv(name)
            }
        }
        if let value {
            setenv(name, value, 1)
        } else {
            unsetenv(name)
        }
        try body()
    }

    // MARK: - cappedContextVocabulary

    func testCapConsumesAtMostThirtyTerms() {
        let overCap = (0..<31).map { "term\($0)" }
        let capped = TouchUpEngine.cappedContextVocabulary(overCap)
        XCTAssertEqual(capped?.count, 30)
        XCTAssertEqual(capped, Array(overCap.prefix(30)))
    }

    func testCapLeavesUnderCapListUntouched() {
        let underCap = ["Robinhood", "Cloudflare", "AWS S3"]
        XCTAssertEqual(TouchUpEngine.cappedContextVocabulary(underCap), underCap)
    }

    func testCapLeavesExactlyThirtyUntouched() {
        let exactlyCap = (0..<30).map { "term\($0)" }
        XCTAssertEqual(TouchUpEngine.cappedContextVocabulary(exactlyCap), exactlyCap)
    }

    func testCapPassesNilThrough() {
        XCTAssertNil(TouchUpEngine.cappedContextVocabulary(nil))
    }

    func testCapPassesThroughEmptyArrayAsEmpty() {
        XCTAssertEqual(TouchUpEngine.cappedContextVocabulary([]), [])
    }

    // MARK: - cappedContextVocabulary: sanitization (engine#280 review item 2)

    func testCapCollapsesEmbeddedNewlineToSpace() {
        let capped = TouchUpEngine.cappedContextVocabulary(["Ac\nme"])
        XCTAssertEqual(capped, ["Ac me"])
    }

    func testCapCollapsesEmbeddedTabAndControlCharactersToSpace() {
        let capped = TouchUpEngine.cappedContextVocabulary(["Ac\t\u{0007}me"])
        XCTAssertEqual(capped, ["Ac me"])
    }

    func testCapDropsWhitespaceOnlyTermEntirely() {
        XCTAssertEqual(TouchUpEngine.cappedContextVocabulary(["   ", "Robinhood"]), ["Robinhood"])
    }

    func testCapDropsCommaFromTerm() {
        // "Acme, Inc." must not survive with its comma intact: the composed
        // block joins terms with ", ", so an embedded comma would read as
        // two separate terms in that list.
        let capped = TouchUpEngine.cappedContextVocabulary(["Acme, Inc."])
        XCTAssertEqual(capped, ["Acme Inc."])
    }

    func testCapDedupesCaseInsensitivelyKeepingFirstOccurrence() {
        let capped = TouchUpEngine.cappedContextVocabulary(["Robinhood", "robinhood", "ROBINHOOD", "Cloudflare"])
        XCTAssertEqual(capped, ["Robinhood", "Cloudflare"])
    }

    func testCapDoesNotLetBlanksOrDuplicatesWasteCapSlots() {
        // 40 blanks/duplicates followed by 29 genuinely distinct terms
        // (30 total unique candidates once "Robinhood" dedupes to one
        // entry — exactly at the cap): if the cap were applied BEFORE
        // filtering, the noise would consume slots and some real terms
        // would never survive. The correct order (sanitize -> filter ->
        // dedupe -> cap) must let all 29 real terms through alongside the
        // single deduped "Robinhood".
        let noise = Array(repeating: "  ", count: 20) + Array(repeating: "Robinhood", count: 20)
        let real = (0..<29).map { "term\($0)" }
        let capped = TouchUpEngine.cappedContextVocabulary(noise + real)
        XCTAssertEqual(capped, ["Robinhood"] + real)
        XCTAssertEqual(capped?.count, 30)
    }

    /// The literal placeholder string `YoozPrompts.resultPlaceholder`
    /// ("corrected text") must survive sanitization unchanged — it is not
    /// special-cased at this layer (engine#280 review item 2's "verify"
    /// ask). See `testVocabularyTermCorrectedTextDoesNotAffectPlaceholderEchoGuard`
    /// below for why this term is safe to compose into the prompt at all.
    func testCapLeavesResultPlaceholderLiteralUnchanged() {
        let capped = TouchUpEngine.cappedContextVocabulary([YoozPrompts.resultPlaceholder])
        XCTAssertEqual(capped, [YoozPrompts.resultPlaceholder])
    }

    // MARK: - cappedContextVocabulary: character cap (engine#280 review item 1)

    func testCapDropsOverLengthTermEntirelyRatherThanTruncating() {
        let tooLong = String(repeating: "x", count: TouchUpEngine.touchUpContextTermCharacterCap + 1)
        let capped = TouchUpEngine.cappedContextVocabulary([tooLong, "Robinhood"])
        XCTAssertEqual(capped, ["Robinhood"], "an over-cap term must be dropped whole, never truncated mid-word")
    }

    func testCapKeepsTermExactlyAtCharacterCap() {
        let exactlyCap = String(repeating: "x", count: TouchUpEngine.touchUpContextTermCharacterCap)
        XCTAssertEqual(TouchUpEngine.cappedContextVocabulary([exactlyCap]), [exactlyCap])
    }

    // MARK: - sanitizedContextAppName (engine#280 review items 1-2)

    func testSanitizedAppNameCollapsesEmbeddedNewlineToSpace() {
        XCTAssertEqual(TouchUpEngine.sanitizedContextAppName("Sla\nck"), "Sla ck")
    }

    func testSanitizedAppNameNilForWhitespaceOnlyInput() {
        XCTAssertNil(TouchUpEngine.sanitizedContextAppName("   "))
    }

    func testSanitizedAppNameNilForNilInput() {
        XCTAssertNil(TouchUpEngine.sanitizedContextAppName(nil))
    }

    func testSanitizedAppNameKeepsCommaUnlikeVocabularyTerms() {
        // Unlike vocabulary terms, the app name is never joined into a
        // comma-separated list, so a comma is unambiguous and kept as-is.
        XCTAssertEqual(TouchUpEngine.sanitizedContextAppName("Acme, Inc."), "Acme, Inc.")
    }

    func testSanitizedAppNameTruncatesOverCapRatherThanDropping() {
        // Unlike vocabulary terms, an over-cap app name is TRUNCATED, not
        // dropped: there is only ever one app name, so dropping it loses
        // the feature entirely for any app with a long display name.
        let tooLong = String(repeating: "y", count: TouchUpEngine.touchUpContextAppNameCharacterCap + 10)
        let result = TouchUpEngine.sanitizedContextAppName(tooLong)
        XCTAssertEqual(result?.count, TouchUpEngine.touchUpContextAppNameCharacterCap)
        XCTAssertEqual(result, String(tooLong.prefix(TouchUpEngine.touchUpContextAppNameCharacterCap)))
    }

    func testSanitizedAppNameKeepsNameExactlyAtCharacterCap() {
        let exactlyCap = String(repeating: "y", count: TouchUpEngine.touchUpContextAppNameCharacterCap)
        XCTAssertEqual(TouchUpEngine.sanitizedContextAppName(exactlyCap), exactlyCap)
    }

    // MARK: - withContext (flag off / fields nil -> byte-identical)

    func testFlagOffReproducesPromptByteIdentical() {
        withEnvVar("YOOZ_TOUCHUP_CONTEXT_ENABLED", value: "0") {
            let prompt = YoozPrompts.qualityFull
            let composed = TouchUpEngine.withContext(
                prompt: prompt, vocabulary: ["Robinhood"], appName: "Slack"
            )
            XCTAssertEqual(composed, prompt)
        }
    }

    func testFlagOnBothFieldsNilReproducesPromptByteIdentical() {
        withEnvVar("YOOZ_TOUCHUP_CONTEXT_ENABLED", value: "1") {
            let prompt = YoozPrompts.qualityFull
            let composed = TouchUpEngine.withContext(prompt: prompt, vocabulary: nil, appName: nil)
            XCTAssertEqual(composed, prompt)
        }
    }

    func testFlagOnEmptyVocabularyAndNilAppNameReproducesPromptByteIdentical() {
        withEnvVar("YOOZ_TOUCHUP_CONTEXT_ENABLED", value: "1") {
            let prompt = YoozPrompts.lightStandard
            let composed = TouchUpEngine.withContext(prompt: prompt, vocabulary: [], appName: nil)
            XCTAssertEqual(composed, prompt)
        }
    }

    // MARK: - withContext (flag on + fields -> prompt gains block)

    func testFlagOnWithVocabularyAndAppNameAppendsBothLines() {
        withEnvVar("YOOZ_TOUCHUP_CONTEXT_ENABLED", value: "1") {
            let prompt = YoozPrompts.qualityFull
            let composed = TouchUpEngine.withContext(
                prompt: prompt,
                vocabulary: ["Robinhood", "Cloudflare", "AWS S3"],
                appName: "Slack"
            )
            let expected = prompt + "\n\n"
                + "Known terms the speaker may use: Robinhood, Cloudflare, AWS S3.\n"
                + "Text will be pasted into: Slack."
            XCTAssertEqual(composed, expected)
        }
    }

    func testFlagOnWithVocabularyOnlyOmitsAppNameLine() {
        withEnvVar("YOOZ_TOUCHUP_CONTEXT_ENABLED", value: "1") {
            let prompt = YoozPrompts.qualityFull
            let composed = TouchUpEngine.withContext(
                prompt: prompt, vocabulary: ["Robinhood"], appName: nil
            )
            let expected = prompt + "\n\nKnown terms the speaker may use: Robinhood."
            XCTAssertEqual(composed, expected)
        }
    }

    func testFlagOnWithAppNameOnlyOmitsVocabularyLine() {
        withEnvVar("YOOZ_TOUCHUP_CONTEXT_ENABLED", value: "1") {
            let prompt = YoozPrompts.qualityFull
            let composed = TouchUpEngine.withContext(prompt: prompt, vocabulary: nil, appName: "Slack")
            let expected = prompt + "\n\nText will be pasted into: Slack."
            XCTAssertEqual(composed, expected)
        }
    }

    func testFlagOnWithBlankAppNameOmitsAppNameLine() {
        withEnvVar("YOOZ_TOUCHUP_CONTEXT_ENABLED", value: "1") {
            let prompt = YoozPrompts.qualityFull
            let composed = TouchUpEngine.withContext(
                prompt: prompt, vocabulary: nil, appName: "   "
            )
            XCTAssertEqual(composed, prompt)
        }
    }

    func testFlagOnDoesNotMutateYoozPromptsConstant() {
        // The composition step must never write back into the trained
        // constant `YoozPromptsParityTests` locks — it only builds a new
        // string. Sampling the constant before/after composing pins that.
        withEnvVar("YOOZ_TOUCHUP_CONTEXT_ENABLED", value: "1") {
            let before = YoozPrompts.qualityFull
            _ = TouchUpEngine.withContext(prompt: YoozPrompts.qualityFull, vocabulary: ["X"], appName: "Y")
            XCTAssertEqual(YoozPrompts.qualityFull, before)
        }
    }

    // MARK: - withContext: sanitization applies even to direct, un-sanitized
    // callers (engine#280 review item 2) — defense in depth alongside
    // `cappedContextVocabulary`/`sanitizedContextAppName`, which the
    // production call sites already run before reaching `withContext`.

    func testWithContextSanitizesRawVocabularyDirectly() {
        withEnvVar("YOOZ_TOUCHUP_CONTEXT_ENABLED", value: "1") {
            let prompt = YoozPrompts.qualityFull
            let composed = TouchUpEngine.withContext(
                prompt: prompt, vocabulary: ["Acme,\nInc.", "  "], appName: nil
            )
            let expected = prompt + "\n\nKnown terms the speaker may use: Acme Inc.."
            XCTAssertEqual(composed, expected)
        }
    }

    func testWithContextTruncatesRawOverLengthAppNameDirectly() {
        withEnvVar("YOOZ_TOUCHUP_CONTEXT_ENABLED", value: "1") {
            let prompt = YoozPrompts.qualityFull
            let tooLong = String(repeating: "z", count: TouchUpEngine.touchUpContextAppNameCharacterCap + 5)
            let composed = TouchUpEngine.withContext(prompt: prompt, vocabulary: nil, appName: tooLong)
            let expected = prompt + "\n\nText will be pasted into: "
                + String(tooLong.prefix(TouchUpEngine.touchUpContextAppNameCharacterCap)) + "."
            XCTAssertEqual(composed, expected)
        }
    }

    func testWithContextDropsRawOverLengthTermDirectly() {
        withEnvVar("YOOZ_TOUCHUP_CONTEXT_ENABLED", value: "1") {
            let prompt = YoozPrompts.qualityFull
            let tooLong = String(repeating: "x", count: TouchUpEngine.touchUpContextTermCharacterCap + 1)
            let composed = TouchUpEngine.withContext(
                prompt: prompt, vocabulary: [tooLong, "Robinhood"], appName: nil
            )
            let expected = prompt + "\n\nKnown terms the speaker may use: Robinhood."
            XCTAssertEqual(composed, expected)
        }
    }

    // MARK: - "corrected text" placeholder-echo guard interaction (engine#280 review item 2 "verify")

    /// `YoozPrompts.resultPlaceholder` ("corrected text") surviving into the
    /// composed context block must not interact with `JSONParsing`'s
    /// placeholder-echo guard (`isPlaceholderEcho`, exercised here via the
    /// internal `parseProofreadResponse`). That guard only ever inspects the
    /// MODEL'S RAW OUTPUT — its signature has no prompt/vocabulary
    /// parameter at all — so there is no code path by which a context-block
    /// term could influence it. This test proves that empirically: a
    /// composed prompt containing the literal term is built, then
    /// `parseProofreadResponse` is exercised directly against both a
    /// genuine placeholder echo (must still fail) and a legitimate response
    /// that happens to contain the same words as real content (must still
    /// succeed) — neither outcome depends on what is in the prompt.
    func testVocabularyTermCorrectedTextDoesNotAffectPlaceholderEchoGuard() {
        withEnvVar("YOOZ_TOUCHUP_CONTEXT_ENABLED", value: "1") {
            let composed = TouchUpEngine.withContext(
                prompt: YoozPrompts.qualityFull,
                vocabulary: ["corrected text", "Robinhood"],
                appName: nil
            )
            XCTAssertTrue(
                composed.contains("Known terms the speaker may use: corrected text, Robinhood.")
            )

            // A genuine placeholder echo must still be rejected regardless
            // of what the composed prompt contained.
            let placeholderResponse = #"{"result": "corrected text"}"#
            let (_, placeholderSuccess) = parseProofreadResponse(placeholderResponse, fallback: "fallback text")
            XCTAssertFalse(
                placeholderSuccess,
                "a genuine placeholder echo must still fail even though the prompt referenced the same literal"
            )

            // A legitimate response that merely CONTAINS the same words as
            // real content must still parse and succeed.
            let legitimateResponse = #"{"result": "Please return the corrected text field only."}"#
            let (parsedText, legitimateSuccess) = parseProofreadResponse(legitimateResponse, fallback: "fallback text")
            XCTAssertTrue(legitimateSuccess)
            XCTAssertEqual(parsedText, "Please return the corrected text field only.")
        }
    }
}
