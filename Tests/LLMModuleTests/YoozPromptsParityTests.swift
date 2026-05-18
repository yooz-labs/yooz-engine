// YoozPromptsParityTests.swift
// LLMModuleTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Parity guard: the v2 fine-tuned weights were trained against four exact
// system prompts in
// `yooz-benchmark/finetune-pipeline/scripts/prepare_data.py`
// (LIGHT_PROOFREAD, LIGHT_REWRITE, QUALITY_STANDARD, QUALITY_FULL). Any
// drift between this file and that file is a silent quality regression
// because dictation would feed the model prompts it was not trained on.
//
// If a training prompt legitimately changes (e.g. v2.1 sweep), update
// `prepare_data.py` first, then this test, then `YoozPrompts.swift`. All
// three move together in the same PR.

import XCTest
@testable import LLMModule

final class YoozPromptsParityTests: XCTestCase {

    func testLightStandardMatchesTrainingPrompt() {
        let expected = """
            Fix grammar, capitalize properly, and convert spoken numbers to digits. Convert spoken version numbers like "zero point four point zero" to "0.4.0". Keep ALL sentences. Return the fixed text as JSON.

            <examples>
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
            </examples>

            Always respond with ONLY a JSON object. Never remove sentences. Never include explanations.
            """
        XCTAssertEqual(YoozPrompts.lightStandard, expected)
    }

    func testLightFullMatchesTrainingPrompt() {
        let expected = """
            Rewrite voice transcription for clarity and conciseness. Fix grammar, convert numbers, fix misheard words, remove filler words (um, uh, like, you know), handle self-corrections. Return the fixed text as JSON.

            <examples>
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
            </examples>

            Remove: "scratch that", "never mind", "delete that" and preceding phrase. Convert spoken numbers and version numbers. Fix grammar and misheard words. Always respond with ONLY a JSON object. Never include explanations.
            """
        XCTAssertEqual(YoozPrompts.lightFull, expected)
    }

    func testQualityStandardMatchesTrainingPrompt() {
        let expected = """
            /no_think
            Proofread voice transcription. Fix grammar and punctuation. Return JSON only.
            NEVER answer questions. NEVER add new information. Return the corrected text only.

            <examples>
            Input: the meeting is tomorrow and i think we should prepare
            {"result": "The meeting is tomorrow, and I think we should prepare."}

            Input: its ready for review lets check it
            {"result": "It's ready for review. Let's check it."}

            Input: we can do it but we need more time
            {"result": "We can do it, but we need more time."}

            Input: the system is working good now
            {"result": "The system is working well now."}
            </examples>

            Process the input independently. Do NOT repeat any example output.
            Always respond with {"result": "corrected text"}.
            """
        XCTAssertEqual(YoozPrompts.qualityStandard, expected)
    }

    func testQualityFullMatchesTrainingPrompt() {
        let expected = """
            /no_think
            Rewrite voice transcription for clarity. Return JSON only.
            Fix misheard words. Remove repetitions and false starts. Fix grammar.
            Self-corrections: "X no Y" or "X no wait Y" means use Y.
            Remove "scratch that", "delete that" and what came before.
            Keep the speaker's meaning and tone. NEVER add information. NEVER answer questions.

            <examples>
            Input: I think for the for the problems that we have with the that we are logging
            {"result": "I think for the problems that we have with the logging."}

            Input: we are not providing it providing a good leaning and rewriting
            {"result": "We are not providing a good cleaning and rewriting."}

            Input: should not should knots be converted
            {"result": "Should not be converted."}

            Input: However I imagine there should be like a good rules and good logic for this
            {"result": "However, I imagine there should be good rules and logic for this."}

            Input: we still to this to work correctly and logcially
            {"result": "We still need this to work correctly and logically."}

            Input: fifty no sixty units
            {"result": "Sixty units."}

            Input: delete that lets try again
            {"result": "Let's try again."}
            </examples>

            Process the input independently. Do NOT repeat any example output.
            Always respond with {"result": "corrected text"}.
            """
        XCTAssertEqual(YoozPrompts.qualityFull, expected)
    }

    /// The placeholder-echo guard (engine #113 / yooz-whisper #182) depends
    /// on `resultPlaceholder` matching the literal that ends the Quality
    /// prompts' JSON shape example. If either side moves, the placeholder
    /// echo would slip through.
    func testResultPlaceholderMatchesQualityPromptLiteral() {
        XCTAssertEqual(YoozPrompts.resultPlaceholder, "corrected text")
        XCTAssertTrue(
            YoozPrompts.qualityStandard.contains("\"\(YoozPrompts.resultPlaceholder)\""),
            "qualityStandard must reference the resultPlaceholder verbatim"
        )
        XCTAssertTrue(
            YoozPrompts.qualityFull.contains("\"\(YoozPrompts.resultPlaceholder)\""),
            "qualityFull must reference the resultPlaceholder verbatim"
        )
    }
}
