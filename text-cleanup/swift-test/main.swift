// Swift integration test for YoozTextCleanup XCFramework
// This validates that the Rust library can be called from Swift

import Foundation
import NaturalLanguage

print("=== YoozTextCleanup Swift Integration Test ===\n")

// Test 1: Version check
let version = getVersion()
print("[OK] Library version: \(version)")

// Test 2: Grammar correction - subject-verb agreement
let test1 = "I are happy"
let result1 = correctGrammar(text: test1)
print("[OK] '\(test1)' -> '\(result1)'")
assert(result1 == "I am happy", "Expected 'I am happy' but got '\(result1)'")

// Test 3: Article correction
let test2 = "I want a apple"
let result2 = correctGrammar(text: test2)
print("[OK] '\(test2)' -> '\(result2)'")
assert(result2 == "I want an apple", "Expected 'I want an apple' but got '\(result2)'")

// Test 4: Informal to formal
let test3 = "I gonna go now"
let result3 = correctGrammar(text: test3)
print("[OK] '\(test3)' -> '\(result3)'")
assert(result3 == "I going to go now", "Expected 'I going to go now' but got '\(result3)'")

// Test 5: Common misspelling
let test4 = "I need alot of help"
let result4 = correctGrammar(text: test4)
print("[OK] '\(test4)' -> '\(result4)'")
assert(result4 == "I need a lot of help", "Expected 'I need a lot of help' but got '\(result4)'")

// Test 6: He/She/It don't -> doesn't
let test5 = "He don't know"
let result5 = correctGrammar(text: test5)
print("[OK] '\(test5)' -> '\(result5)'")
assert(result5 == "He doesn't know", "Expected 'He doesn't know' but got '\(result5)'")

// Test 7: Irregular verb
let test6 = "I seen it yesterday"
let result6 = correctGrammar(text: test6)
print("[OK] '\(test6)' -> '\(result6)'")
assert(result6 == "I saw it yesterday", "Expected 'I saw it yesterday' but got '\(result6)'")

print("\n--- Phase 2: Category and Language API ---\n")

// Test 8: Language API
let allLanguages = getAvailableLanguages()
print("[OK] Available languages: \(allLanguages.count)")
assert(allLanguages.count == 9, "Expected 9 languages")

let implementedLanguages = getImplementedLanguages()
print("[OK] Implemented languages: \(implementedLanguages.count)")
assert(implementedLanguages.count == 3, "Expected 3 implemented languages (English variants)")

// Test 9: Category API
let allCategories = getAllCategories()
print("[OK] All categories: \(allCategories.count)")
assert(allCategories.count == 9, "Expected 9 categories")

let freeCategories = getFreeCategories()
print("[OK] Free tier categories: \(freeCategories.count)")
assert(freeCategories.count == 1, "Expected 1 free tier category (Grammar only)")

let proCategories = getProCategories()
print("[OK] Pro tier categories: \(proCategories.count)")
assert(proCategories.count == 9, "Expected 9 pro tier categories")

let premiumCategories = getPremiumCategories()
print("[OK] Premium tier categories: \(premiumCategories.count)")
assert(premiumCategories.count == 9, "Expected 9 premium tier categories")

// Test 10: Rule count API
let ruleCount = getRuleCount(language: .english)
print("[OK] English rule count: \(ruleCount)")
assert(ruleCount > 0, "Expected at least 1 rule")

let grammarRules = getRuleCountForCategory(language: .english, category: .grammar)
print("[OK] Grammar category rules: \(grammarRules)")

let articleRules = getRuleCountForCategory(language: .english, category: .articles)
print("[OK] Articles category rules: \(articleRules)")

// Test 11: Category filtering
let grammarOnly = correctGrammarWithCategories(
    text: "I are eating a apple",
    language: .english,
    categories: [.grammar]
)
print("[OK] Grammar only: 'I are eating a apple' -> '\(grammarOnly)'")
assert(grammarOnly == "I am eating a apple", "Grammar-only should fix 'I are' but not 'a apple'")

let articlesOnly = correctGrammarWithCategories(
    text: "I are eating a apple",
    language: .english,
    categories: [.articles]
)
print("[OK] Articles only: 'I are eating a apple' -> '\(articlesOnly)'")
assert(articlesOnly == "I are eating an apple", "Articles-only should fix 'a apple' but not 'I are'")

let bothCategories = correctGrammarWithCategories(
    text: "I are eating a apple",
    language: .english,
    categories: [.grammar, .articles]
)
print("[OK] Both categories: 'I are eating a apple' -> '\(bothCategories)'")
assert(bothCategories == "I am eating an apple", "Both categories should fix everything")

// Test 12: Rule info API
let allRules = getAllRules(language: .english)
print("[OK] All rules info: \(allRules.count) rules")

let grammarRulesInfo = getRulesForCategory(language: .english, category: .grammar)
print("[OK] Grammar rules info: \(grammarRulesInfo.count) rules")

// Test 13: Available categories for language
let availableCategories = getAvailableCategoriesForLanguage(language: .english)
print("[OK] Available categories for English: \(availableCategories.count)")
assert(availableCategories.count > 0, "Expected at least 1 available category")

// Test 14: Language-specific correction
let resultUS = correctGrammarForLanguage(text: "I are happy", language: .englishUs)
print("[OK] US English: 'I are happy' -> '\(resultUS)'")
assert(resultUS == "I am happy", "US English should work")

print("\n--- Phase 3: POS-Based Correction with NLTagger ---\n")

// Initialize NLTagger bridge
let taggerBridge = NLTaggerBridge()

// Test cases - some need POS tagging to work
let posTestCases = [
    // These work with simple rules
    "Its going to rain today",
    "He should of known better",
    "I seen him yesterday",
    "They was at the party",
    "He don't know the answer",
    // These REQUIRE POS tagging (VBG detection)
    "She is good in swimming",        // good in VBG -> good at VBG
    "He is good in cooking",          // good in VBG -> good at VBG
    "Your not going anywhere",        // your not VBG -> you're not VBG
    "Your not even trying",           // your not RB VBG -> you're not RB VBG
]

print("POS vs Simple Correction Comparison:")
print(String(repeating: "-", count: 100))

for input in posTestCases {
    let simpleResult = correctGrammar(text: input)

    // Use NLTagger for POS tagging
    let tokens = taggerBridge.tokenize(input)
    let posResult = correctGrammarWithPosCategories(
        tokens: tokens,
        language: .english,
        categories: getAllCategories()
    )

    print("Input:  '\(input)'")
    print("Simple: '\(simpleResult)'\(simpleResult != input ? " [changed]" : "")")
    print("POS:    '\(posResult)'\(posResult != input ? " [changed]" : "")")
    print("")
}

// Show NLTagger tagging for samples
let sampleTexts = ["good in swimming", "Your not going"]
for sampleText in sampleTexts {
    let sampleTokens = taggerBridge.tokenize(sampleText)
    print("NLTagger POS tags for '\(sampleText)':")
    for token in sampleTokens {
        if !token.text.isEmpty {
            print("  '\(token.text)' -> \(token.tag)")
        }
    }
    print("")
}

print("\n=== All tests passed! ===")
print("Phase 3: POS-based correction with NLTagger completed.")
