#!/bin/bash
#
# Build XCFramework for Yooz Text Cleanup
#
# This script:
# 1. Builds Rust library for all Apple platforms (universal: arm64 + x86_64)
# 2. Generates Swift bindings via UniFFI
# 3. Creates XCFramework for easy Xcode integration
#
# Prerequisites:
# - Rust installed via rustup (with x86_64-apple-darwin target)
# - Xcode command line tools
#
# Note: If you have both Homebrew rust and rustup installed, this script
# will temporarily use rustup's toolchain for cross-compilation.

set -e

echo "=== Yooz Text Cleanup XCFramework Builder ==="
echo ""

# Configuration
LIB_NAME="libyooz_text_cleanup"
FRAMEWORK_NAME="YoozTextCleanup"
OUTPUT_DIR="./build"
SWIFT_OUTPUT="./generated"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Detect rustup toolchain path
RUSTUP_TOOLCHAIN="$HOME/.rustup/toolchains/stable-aarch64-apple-darwin"
USE_RUSTUP=false

# Check prerequisites
check_prereqs() {
    echo "Checking prerequisites..."

    if ! command -v xcodebuild &> /dev/null; then
        echo -e "${RED}Error: xcodebuild not found. Install Xcode command line tools.${NC}"
        exit 1
    fi

    # Check if rustup toolchain exists (needed for cross-compilation)
    if [ -d "$RUSTUP_TOOLCHAIN/bin" ]; then
        echo "Found rustup toolchain at $RUSTUP_TOOLCHAIN"
        USE_RUSTUP=true

        # Check if x86_64 target is installed
        if [ ! -d "$RUSTUP_TOOLCHAIN/lib/rustlib/x86_64-apple-darwin" ]; then
            echo -e "${YELLOW}Adding x86_64-apple-darwin target...${NC}"
            rustup target add x86_64-apple-darwin
        fi
    elif command -v cargo &> /dev/null; then
        echo "Using system cargo (arm64 only)"
        USE_RUSTUP=false
    else
        echo -e "${RED}Error: Neither rustup nor cargo found. Install Rust first.${NC}"
        echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.sh | sh"
        exit 1
    fi

    echo -e "${GREEN}Prerequisites OK${NC}"
}

# Add required Rust targets
add_targets() {
    echo ""
    echo "Adding Rust targets for Apple platforms..."

    if [ "$USE_RUSTUP" = true ]; then
        # Use rustup to add targets
        rustup target add aarch64-apple-darwin 2>/dev/null || true
        rustup target add x86_64-apple-darwin 2>/dev/null || true
    fi

    echo -e "${GREEN}Targets configured${NC}"
}

# Run cargo with the correct toolchain
run_cargo() {
    if [ "$USE_RUSTUP" = true ]; then
        # Use rustup toolchain directly to avoid Homebrew rust conflicts
        env PATH="$RUSTUP_TOOLCHAIN/bin:/usr/bin:/bin" \
            "$RUSTUP_TOOLCHAIN/bin/cargo" "$@"
    else
        cargo "$@"
    fi
}

# Build for a specific target
build_target() {
    local target=$1
    echo "  Building for $target..."
    run_cargo build --release --target "$target" 2>&1 | grep -E "(Compiling|Finished|error)" || true
}

# Build all platforms
build_all() {
    echo ""
    echo "Building Rust library for macOS (universal binary)..."

    # macOS arm64 (Apple Silicon)
    build_target "aarch64-apple-darwin"

    # macOS x86_64 (Intel) - only if rustup is available
    if [ "$USE_RUSTUP" = true ]; then
        build_target "x86_64-apple-darwin"
    else
        echo -e "${YELLOW}Skipping x86_64 build (rustup not available)${NC}"
    fi

    echo -e "${GREEN}Build complete${NC}"
}

# Create fat binaries using lipo
create_fat_binaries() {
    echo ""
    echo "Creating universal binaries..."

    mkdir -p "$OUTPUT_DIR/macos"

    # macOS Universal (arm64 + x86_64) or arm64-only
    if [ "$USE_RUSTUP" = true ] && [ -f "target/x86_64-apple-darwin/release/${LIB_NAME}.a" ]; then
        echo "Creating universal binary (arm64 + x86_64)..."
        lipo -create \
            "target/aarch64-apple-darwin/release/${LIB_NAME}.a" \
            "target/x86_64-apple-darwin/release/${LIB_NAME}.a" \
            -output "$OUTPUT_DIR/macos/${LIB_NAME}.a"
    else
        echo "Creating arm64-only binary..."
        cp "target/aarch64-apple-darwin/release/${LIB_NAME}.a" \
            "$OUTPUT_DIR/macos/${LIB_NAME}.a"
    fi

    # Show architecture info
    lipo -info "$OUTPUT_DIR/macos/${LIB_NAME}.a"

    echo -e "${GREEN}Binary created${NC}"
}

# Generate Swift bindings using UniFFI
generate_bindings() {
    echo ""
    echo "Generating Swift bindings..."

    mkdir -p "$SWIFT_OUTPUT"

    # Generate Swift bindings using the built dylib
    run_cargo run --bin uniffi-bindgen generate \
        --library "target/aarch64-apple-darwin/release/${LIB_NAME}.dylib" \
        --language swift \
        --out-dir "$SWIFT_OUTPUT" 2>/dev/null || {

        echo -e "${YELLOW}Warning: Could not generate Swift bindings automatically.${NC}"
        echo "  This may require running: cargo run --bin uniffi-bindgen generate ..."
    }

    if [ -f "$SWIFT_OUTPUT/yooz_text_cleanup.swift" ]; then
        echo -e "${GREEN}Swift bindings generated in $SWIFT_OUTPUT${NC}"
    fi
}

# Create headers for XCFramework
create_headers() {
    echo ""
    echo "Creating module headers..."

    mkdir -p "$OUTPUT_DIR/macos/Headers"

    # Copy generated header if exists
    if [ -f "$SWIFT_OUTPUT/yooz_text_cleanupFFI.h" ]; then
        cp "$SWIFT_OUTPUT/yooz_text_cleanupFFI.h" "$OUTPUT_DIR/macos/Headers/"
    fi

    # Create module.modulemap
    cat > "$OUTPUT_DIR/macos/Headers/module.modulemap" << EOF
module yooz_text_cleanupFFI {
    header "yooz_text_cleanupFFI.h"
    export *
}
EOF

    echo -e "${GREEN}Headers created${NC}"
}

# Create XCFramework
create_xcframework() {
    echo ""
    echo "Creating XCFramework..."

    # Remove existing framework
    rm -rf "$OUTPUT_DIR/${FRAMEWORK_NAME}.xcframework"

    xcodebuild -create-xcframework \
        -library "$OUTPUT_DIR/macos/${LIB_NAME}.a" \
        -headers "$OUTPUT_DIR/macos/Headers" \
        -output "$OUTPUT_DIR/${FRAMEWORK_NAME}.xcframework"

    echo -e "${GREEN}XCFramework created: $OUTPUT_DIR/${FRAMEWORK_NAME}.xcframework${NC}"

    # Show size
    du -sh "$OUTPUT_DIR/${FRAMEWORK_NAME}.xcframework"
}

# Show summary
show_summary() {
    echo ""
    echo "=== Build Summary ==="
    echo ""
    echo "XCFramework: $OUTPUT_DIR/${FRAMEWORK_NAME}.xcframework"
    echo "Swift bindings: $SWIFT_OUTPUT/yooz_text_cleanup.swift"
    echo ""
    echo "To use in Xcode:"
    echo "  1. Drag ${FRAMEWORK_NAME}.xcframework into your project"
    echo "  2. Add yooz_text_cleanup.swift to your project"
    echo "  3. Import and use:"
    echo ""
    echo "     import YoozTextCleanupFFI"
    echo ""
    echo "     let corrected = correctGrammar(\"I are happy\")"
    echo "     // Returns: \"I am happy\""
    echo ""
}

# Main
main() {
    check_prereqs
    add_targets
    build_all
    create_fat_binaries
    generate_bindings
    create_headers
    create_xcframework
    show_summary

    echo -e "${GREEN}Done!${NC}"
}

# Run with optional argument for specific step
case "${1:-all}" in
    check)
        check_prereqs
        ;;
    build)
        build_all
        ;;
    all)
        main
        ;;
    *)
        echo "Usage: $0 [check|build|all]"
        exit 1
        ;;
esac
