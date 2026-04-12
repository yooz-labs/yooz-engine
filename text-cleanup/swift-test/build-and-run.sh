#!/bin/bash
# Build and run Swift integration test for YoozTextCleanup

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO_DIR="$(dirname "$SCRIPT_DIR")"
BINDINGS_DIR="$PROTO_DIR/bindings/swift"
LIB_DIR="$PROTO_DIR/target/aarch64-apple-darwin/release"

echo "Building Swift integration test..."

# Compile and link
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
    -o "$SCRIPT_DIR/test_runner"

echo "Running test..."
"$SCRIPT_DIR/test_runner"

echo ""
echo "Cleanup..."
rm -f "$SCRIPT_DIR/test_runner"
echo "Done!"
