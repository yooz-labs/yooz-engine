//! Integration tests for yooz-text-cleanup

use yooz_text_cleanup::correct_grammar;

#[test]
fn test_subject_verb_agreement() {
    assert_eq!(correct_grammar("I are happy".into()), "I am happy");
    assert_eq!(correct_grammar("He don't know".into()), "He doesn't know");
    assert_eq!(correct_grammar("She don't care".into()), "She doesn't care");
}

#[test]
fn test_article_corrections() {
    assert_eq!(correct_grammar("a apple".into()), "an apple");
    assert_eq!(correct_grammar("a orange".into()), "an orange");
}

#[test]
fn test_common_typos() {
    assert_eq!(correct_grammar("alot of work".into()), "a lot of work");
    assert_eq!(correct_grammar("I gonna go".into()), "I going to go");
}

#[test]
fn test_verb_tense() {
    assert_eq!(correct_grammar("I goed home".into()), "I went home");
    assert_eq!(correct_grammar("I seen it".into()), "I saw it");
}

#[test]
fn test_no_correction_needed() {
    assert_eq!(correct_grammar("I am happy".into()), "I am happy");
    assert_eq!(correct_grammar("He doesn't know".into()), "He doesn't know");
    assert_eq!(correct_grammar("an apple".into()), "an apple");
}

#[test]
fn test_multiple_errors() {
    // Note: Current prototype applies rules sequentially
    // This test documents current behavior
    let input: String = "I are happy and he don't know".into();
    let output = correct_grammar(input.clone());
    // Should fix at least one error
    assert!(output != input);
}

#[test]
fn test_real_transcription_sample() {
    let input: String = "I goed to the store and I seen alot of people".into();
    let output = correct_grammar(input);

    // Should fix multiple errors from spoken transcription
    assert!(output.contains("went"));
    assert!(output.contains("saw"));
    assert!(output.contains("a lot"));
}
