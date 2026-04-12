#!/bin/bash
# Run POS-based correction test

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO_DIR="$(dirname "$SCRIPT_DIR")"
LIB_DIR="$PROTO_DIR/target/aarch64-apple-darwin/release"

echo "Building POS test..."

swiftc \
    -O \
    -target arm64-apple-macosx14.0 \
    -I "$SCRIPT_DIR" \
    -L "$LIB_DIR" \
    -lyooz_text_cleanup \
    -import-objc-header "$SCRIPT_DIR/yooz_text_cleanupFFI.h" \
    "$SCRIPT_DIR/yooz_text_cleanup.swift" \
    "$SCRIPT_DIR/NLTaggerBridge.swift" \
    "$SCRIPT_DIR/main.swift" \
    -o "$SCRIPT_DIR/pos_test_runner"

echo "Running POS test..."
"$SCRIPT_DIR/pos_test_runner"

rm -f "$SCRIPT_DIR/pos_test_runner"
