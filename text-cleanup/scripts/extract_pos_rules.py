#!/usr/bin/env python3
"""
Extract LanguageTool rules that use POS tagging.
These rules require NLTagger integration to work.
"""

import xml.etree.ElementTree as ET
from pathlib import Path


# POS tags we can support with NLTagger (coarse-grained mapping)
SUPPORTED_POS_PATTERNS = {
    # Verbs (VB.* pattern)
    "VB",
    "VBD",
    "VBG",
    "VBN",
    "VBP",
    "VBZ",
    "V.*",
    # Nouns (NN.* pattern)
    "NN",
    "NNS",
    "NNP",
    "NNPS",
    "NN:UN",
    "N.*",
    # Adjectives
    "JJ",
    "JJR",
    "JJS",
    "J.*",
    # Adverbs
    "RB",
    "RBR",
    "RBS",
    "R.*",
    # Pronouns
    "PRP",
    "PRP$",
    "WP",
    "P.*",
    # Determiners
    "DT",
    "PDT",
    "WDT",
    # Others
    "IN",
    "TO",
    "RP",
    "CC",
    "CD",
    "MD",
    "EX",
    "UH",
    # Special
    "SENT_START",
    "SENT_END",
    "PCT",
    "POS",
    "UNKNOWN",
}


def can_support_postag(postag_value, is_regexp=False):
    """Check if we can support this POS tag pattern."""
    if not postag_value:
        return True

    if is_regexp:
        # Split by | for alternation patterns
        for part in postag_value.split("|"):
            # Remove regex suffix
            base = part.strip().rstrip(".*?+")
            if base and base not in SUPPORTED_POS_PATTERNS:
                # Check if any prefix matches
                if not any(
                    base.startswith(p.rstrip(".*")) for p in SUPPORTED_POS_PATTERNS
                ):
                    return False
    else:
        if postag_value not in SUPPORTED_POS_PATTERNS:
            return False

    return True


def is_pos_rule(rule_element):
    """Check if rule uses POS tagging and if we can support it."""
    xml_str = ET.tostring(rule_element, encoding="unicode")

    # Must have postag to be a POS rule
    has_postag = "postag=" in xml_str or "postag_regexp=" in xml_str

    if not has_postag:
        return False, "no_postag"

    # Check if we can support all the postags in this rule
    pattern = rule_element.find(".//pattern")
    if pattern is None:
        return False, "no_pattern"

    # Collect all tokens (including inside markers)
    all_tokens = []
    for elem in pattern.iter():
        if elem.tag == "token":
            all_tokens.append(elem)

    # Reject patterns that are too short (need at least 2 meaningful tokens)
    if len(all_tokens) < 2:
        return False, "too_short"

    # Count how many tokens have constraints (text or postag)
    constrained_tokens = 0
    for token in all_tokens:
        text = token.text.strip() if token.text else ""
        postag = token.get("postag")
        regexp = token.get("regexp")

        # A token is constrained if it has text, postag, or regexp
        if text or postag or regexp:
            constrained_tokens += 1

    # Reject patterns with wildcards (tokens without any constraint)
    if constrained_tokens < len(all_tokens):
        return False, "wildcard_token"

    # Require at least 2 constrained tokens for specificity
    if constrained_tokens < 2:
        return False, "too_few_constraints"

    for token in all_tokens:
        postag = token.get("postag")
        postag_regexp = token.get("postag_regexp") == "yes"

        if postag and not can_support_postag(postag, postag_regexp):
            return False, f"unsupported_postag:{postag}"

        # Check exceptions too
        for exc in token.findall(".//exception"):
            exc_postag = exc.get("postag")
            exc_regexp = exc.get("postag_regexp") == "yes"
            if exc_postag and not can_support_postag(exc_postag, exc_regexp):
                return False, f"unsupported_exc_postag:{exc_postag}"

    # Skip rules with features we don't support yet
    if "<antipattern>" in xml_str:
        return False, "antipattern"
    if "chunk_re=" in xml_str or "chunk=" in xml_str:
        return False, "chunking"
    if "skip=" in xml_str:
        return False, "skip"
    if "negate_pos=" in xml_str:
        return False, "negate_pos"

    # Check for proper suggestion
    suggestion = rule_element.find(".//suggestion")
    if suggestion is None:
        return False, "no_suggestion"

    # Check for <match> elements in suggestion (dynamic replacement)
    if "<match " in xml_str or "<match>" in xml_str:
        return False, "dynamic_match"

    # Check for backreferences in suggestion (e.g., \1, \2)
    suggestion_text = suggestion.text if suggestion.text else ""
    if "\\" in suggestion_text:
        import re

        if re.search(r"\\[0-9]", suggestion_text):
            return False, "backreference"

    return True, "ok"


def extract_pos_token(token):
    """Extract token info including POS tag."""
    info = {
        "text": token.text.strip() if token.text else "",
        "postag": token.get("postag"),
        "postag_regexp": token.get("postag_regexp") == "yes",
        "regexp": token.get("regexp"),
        "min": token.get("min", "1"),
        "max": token.get("max", "1"),
    }
    return info


def simplify_pos_rule(rule_element):
    """Create a simplified version of a POS rule."""
    new_rule = ET.Element("rule")
    new_rule.set("id", rule_element.get("id", "UNKNOWN"))
    new_rule.set("name", rule_element.get("name", rule_element.get("id", "Unknown")))

    # Copy pattern with POS info
    pattern = rule_element.find(".//pattern")
    new_pattern = ET.SubElement(new_rule, "pattern")

    # Handle markers by flattening
    all_tokens = []
    marked_index = -1

    def process_element(elem):
        nonlocal marked_index
        for child in elem:
            if child.tag == "marker":
                marked_index = len(all_tokens)
                for token in child.findall(".//token"):
                    all_tokens.append(token)
            elif child.tag == "token":
                all_tokens.append(child)

    process_element(pattern)

    # Add tokens with POS info
    for token in all_tokens:
        new_token = ET.SubElement(new_pattern, "token")
        if token.text:
            new_token.text = token.text.strip()
        # Copy supported attributes
        if token.get("postag"):
            new_token.set("postag", token.get("postag"))
        if token.get("postag_regexp"):
            new_token.set("postag_regexp", token.get("postag_regexp"))
        if token.get("regexp"):
            new_token.set("regexp", token.get("regexp"))
        if token.get("case_sensitive"):
            new_token.set("case_sensitive", token.get("case_sensitive"))
        if token.get("min"):
            new_token.set("min", token.get("min"))
        if token.get("max"):
            new_token.set("max", token.get("max"))

    # Copy message
    message = rule_element.find(".//message")
    new_message = ET.SubElement(new_rule, "message")
    if message is not None and message.text:
        new_message.text = message.text.strip()
    else:
        new_message.text = "Grammar correction"

    # Build suggestion
    suggestion = rule_element.find(".//suggestion")
    suggestion_text = (
        suggestion.text.strip() if suggestion is not None and suggestion.text else ""
    )

    if marked_index >= 0 and len(all_tokens) > 1:
        parts = []
        for i, token in enumerate(all_tokens):
            if i == marked_index:
                parts.append(suggestion_text)
            else:
                parts.append(token.text.strip() if token.text else "")
        suggestion_text = " ".join(parts)

    new_suggestion = ET.SubElement(new_rule, "suggestion")
    new_suggestion.text = suggestion_text

    return new_rule


def extract_pos_rules(input_file, output_file):
    """Extract POS-based rules."""
    tree = ET.parse(input_file)
    root = tree.getroot()

    new_root = ET.Element("rules")
    stats = {"total": 0, "extracted": 0, "skipped": {}, "by_category": {}}

    for category in root.findall(".//category"):
        cat_id = category.get("id", "UNKNOWN")
        cat_name = category.get("name", cat_id)

        new_category = ET.SubElement(new_root, "category")
        new_category.set("id", cat_id)
        new_category.set("name", cat_name)

        rules_added = 0

        for rule in category.findall(".//rule"):
            stats["total"] += 1

            is_pos, reason = is_pos_rule(rule)
            if not is_pos:
                stats["skipped"][reason] = stats["skipped"].get(reason, 0) + 1
                continue

            # Simplify and add rule
            try:
                simplified = simplify_pos_rule(rule)
                new_category.append(simplified)
                rules_added += 1
                stats["extracted"] += 1
            except Exception as e:
                stats["skipped"]["error"] = stats["skipped"].get("error", 0) + 1
                print(f"Error processing rule {rule.get('id')}: {e}")

        if rules_added > 0:
            stats["by_category"][cat_id] = rules_added
        else:
            new_root.remove(new_category)

    # Write output
    tree = ET.ElementTree(new_root)
    ET.indent(tree, space="    ")
    tree.write(output_file, encoding="unicode", xml_declaration=True)

    return stats


def main():
    ref_dir = Path(__file__).parent.parent / "reference" / "languagetool"
    output_dir = Path(__file__).parent.parent / "rules" / "en"

    print("Extracting POS-based rules from LanguageTool...\n")

    # Extract grammar rules with POS
    grammar_stats = extract_pos_rules(
        ref_dir / "en-grammar.xml", output_dir / "lt-grammar-pos.xml"
    )

    print("Grammar rules (POS-based):")
    print(f"  Total scanned: {grammar_stats['total']}")
    print(f"  Extracted: {grammar_stats['extracted']}")
    print("\n  Skipped by reason:")
    for reason, count in sorted(grammar_stats["skipped"].items(), key=lambda x: -x[1]):
        print(f"    {reason}: {count}")
    print("\n  By category:")
    for cat, count in sorted(grammar_stats["by_category"].items()):
        print(f"    {cat}: {count}")

    # Extract style rules with POS
    style_stats = extract_pos_rules(
        ref_dir / "en-style.xml", output_dir / "lt-style-pos.xml"
    )

    print("\nStyle rules (POS-based):")
    print(f"  Total scanned: {style_stats['total']}")
    print(f"  Extracted: {style_stats['extracted']}")
    print("\n  Skipped by reason:")
    for reason, count in sorted(style_stats["skipped"].items(), key=lambda x: -x[1]):
        print(f"    {reason}: {count}")
    print("\n  By category:")
    for cat, count in sorted(style_stats["by_category"].items()):
        print(f"    {cat}: {count}")

    total_extracted = grammar_stats["extracted"] + style_stats["extracted"]
    print(f"\n=== TOTAL POS RULES EXTRACTED: {total_extracted} ===")
    print("\nOutput files:")
    print(f"  {output_dir / 'lt-grammar-pos.xml'}")
    print(f"  {output_dir / 'lt-style-pos.xml'}")


if __name__ == "__main__":
    main()
