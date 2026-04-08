#!/bin/bash
# Run NLTagger benchmark

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO_DIR="$(dirname "$SCRIPT_DIR")"
LIB_DIR="$PROTO_DIR/target/aarch64-apple-darwin/release"

echo "Building NLTagger benchmark..."

# Check if library exists
if [ ! -f "$LIB_DIR/libyooz_text_cleanup.a" ]; then
    echo "Building Rust library first..."
    cd "$PROTO_DIR"
    cargo build --release --target aarch64-apple-darwin
    cd "$SCRIPT_DIR"
fi

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
    "$SCRIPT_DIR/nltagger_benchmark.swift" \
    -o "$SCRIPT_DIR/nltagger_benchmark"

echo "Running benchmark..."
cd "$PROTO_DIR"  # Run from proto dir so relative paths work
"$SCRIPT_DIR/nltagger_benchmark"

echo ""
echo "Cleanup..."
rm -f "$SCRIPT_DIR/nltagger_benchmark"
echo "Done!"
