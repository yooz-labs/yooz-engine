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
}
