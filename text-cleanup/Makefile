# Makefile for Yooz Text Cleanup Prototype
#
# Quick commands for development and testing

.PHONY: all build test clean xcframework check benchmark

# Default: run tests
all: test

# Build library (debug)
build:
	cargo build

# Build release
release:
	cargo build --release

# Run all tests
test:
	cargo test

# Run tests with output
test-verbose:
	cargo test -- --nocapture

# Build XCFramework for app integration
xcframework:
	./build-xcframework.sh

# Check prerequisites
check:
	./build-xcframework.sh check

# Clean build artifacts
clean:
	cargo clean
	rm -rf build/
	rm -rf generated/

# Run benchmark (requires criterion)
benchmark:
	cargo bench

# Format code
fmt:
	cargo fmt

# Lint code
lint:
	cargo clippy

# Quick validation: build + test
validate: build test
	@echo "Validation passed!"

# Install Rust targets for Apple platforms
setup-targets:
	rustup target add aarch64-apple-darwin
	rustup target add x86_64-apple-darwin
	rustup target add aarch64-apple-ios
	rustup target add aarch64-apple-ios-sim
	rustup target add x86_64-apple-ios

# Help
help:
	@echo "Yooz Text Cleanup - Makefile Commands"
	@echo ""
	@echo "  make build       - Build debug version"
	@echo "  make release     - Build release version"
	@echo "  make test        - Run all tests"
	@echo "  make xcframework - Build XCFramework for Xcode"
	@echo "  make clean       - Clean build artifacts"
	@echo "  make validate    - Quick build + test"
	@echo "  make setup-targets - Install Rust targets for Apple"
	@echo ""
