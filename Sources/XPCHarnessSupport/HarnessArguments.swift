// HarnessArguments.swift
// XPCHarnessSupport
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Pure, dependency-free helpers for `YoozEngineXPCHarness`'s `--batch-wav`
/// mode (yooz-labs/yooz-whisper#280 PR #251 review). Split out of
/// `HarnessMain.swift` into their own SwiftPM target so they are unit
/// testable under a headless `swift test` — the harness app target itself is
/// deliberately NOT an XCTest target (see `HarnessMain.swift`'s header
/// comment: XPC services are launchd-managed with no GUI test-runner, and
/// this repo's headless build environment cannot attach an app-hosted
/// XCTest runner).
public enum HarnessArguments {
    /// Parse an integer-valued flag out of `arguments` (e.g. `--warm-runs
    /// 3`). Returns `nil` if the flag is absent, has no following value, or
    /// the value doesn't parse as an `Int` — the caller decides whether
    /// "absent" and "unparseable" should be treated differently. Does NOT
    /// reject negative values; callers whose flag has a non-negative
    /// contract (e.g. `--warm-runs`, `--idle-seconds`) validate that
    /// themselves and print a usage error, since "negative" is a distinct
    /// failure mode from "missing" worth its own message.
    public static func intArgument(_ arguments: [String], flag: String) -> Int? {
        guard let idx = arguments.firstIndex(of: flag), idx + 1 < arguments.count else { return nil }
        return Int(arguments[idx + 1])
    }

    /// Coarse coverage metric for the synthesized numbered-sentence test
    /// corpus: counts how many of the "number N" markers (digit or spelled
    /// ordinal form) appear in `text`, out of `expected`. Deliberately loose
    /// (substring, not exact ordinal spelling match) since Parakeet's own
    /// ordinal transcription varies ("11" vs "eleven") — see the corpus's
    /// own `script.txt` ground truth.
    ///
    /// `expected <= 0` returns `0` without touching the `1...expected`
    /// range (which traps for `expected < 1`) — a caller that legitimately
    /// wants "don't check coverage" passes `expected == 0` and should skip
    /// logging the result entirely rather than treat this as an error.
    public static func sentenceCoverage(_ text: String, upTo expected: Int) -> Int {
        guard expected >= 1 else { return 0 }
        let lowered = text.lowercased()
        var covered = 0
        for n in 1...expected {
            let digitForm = "number \(n)."
            let digitFormComma = "number \(n),"
            if lowered.contains(digitForm) || lowered.contains(digitFormComma) {
                covered += 1
                continue
            }
            if let word = ordinalWords[n], lowered.contains("number \(word)") {
                covered += 1
            }
        }
        return covered
    }

    static let ordinalWords: [Int: String] = [
        1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
        8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
        14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
        19: "nineteen", 20: "twenty", 21: "twenty one", 22: "twenty two", 23: "twenty three",
        24: "twenty four",
    ]
}
