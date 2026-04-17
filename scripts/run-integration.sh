#!/bin/bash
#
# run-integration.sh — end-to-end harness runner for YoozEngine.
#
# Flow:
#   1. Regenerate the Xcode project (needed after project.yml edits).
#   2. Build the `YoozEngine` scheme; locate the resulting `Yooz Engine.app`
#      inside DerivedData.
#   3. Invoke `xcodebuild test` with `YOOZ_INTEGRATION=1` and
#      `YOOZ_ENGINE_APP_PATH` pointing at the freshly-built bundle. The
#      harness itself launches the subprocess — this script does NOT start
#      the engine ahead of time.
#   4. Print per-endpoint timings (grep `[timing] ...` lines emitted by
#      IntegrationTestCase) and an overall pass/fail summary.
#
# Usage:
#   scripts/run-integration.sh                  # full run
#   YOOZ_TEST_ENGINE_URL=http://127.0.0.1:19920 \
#       scripts/run-integration.sh              # reuse an already-running engine
#
# Requirements: xcodegen, xcodebuild (Xcode command-line tools).

set -eo pipefail
# Deliberately omit `-u` (nounset). The macOS default /bin/bash is 3.2 and
# expanding an empty array with "${arr[@]}" under `-u` is treated as
# unbound, which would kill the test invocation when YOOZ_TEST_ENGINE_URL
# short-circuits the build and DERIVED_DATA_FLAGS stays empty. We still
# get -e and pipefail safety below.

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

SCHEME_APP="YoozEngine"
SCHEME_TESTS="IntegrationTests"
PROJECT="YoozEngine.xcodeproj"
CONFIGURATION="Debug"
LOG_DIR="$ROOT/.build/integration-logs"

mkdir -p "$LOG_DIR"

echo "[run-integration] root=$ROOT"

echo "[run-integration] step 1: xcodegen generate"
if ! command -v xcodegen >/dev/null 2>&1; then
    echo "error: xcodegen not on PATH; install via 'brew install xcodegen'" >&2
    exit 1
fi
xcodegen generate

DERIVED_DATA_FLAGS=()
if [[ -n "${YOOZ_TEST_ENGINE_URL:-}" ]]; then
    echo "[run-integration] YOOZ_TEST_ENGINE_URL=$YOOZ_TEST_ENGINE_URL — skipping engine build"
else
    echo "[run-integration] step 2: build $SCHEME_APP ($CONFIGURATION)"
    DERIVED_DATA="$ROOT/.build/DerivedData"
    DERIVED_DATA_FLAGS=(-derivedDataPath "$DERIVED_DATA")
    mkdir -p "$DERIVED_DATA"
    xcodebuild \
        -project "$PROJECT" \
        -scheme "$SCHEME_APP" \
        -configuration "$CONFIGURATION" \
        "${DERIVED_DATA_FLAGS[@]}" \
        CODE_SIGNING_ALLOWED=NO \
        build \
        > "$LOG_DIR/build.log" 2>&1 || {
            echo "error: build failed; see $LOG_DIR/build.log" >&2
            tail -40 "$LOG_DIR/build.log" >&2
            exit 1
        }

    APP_PATH="$DERIVED_DATA/Build/Products/$CONFIGURATION/Yooz Engine.app"
    if [[ ! -d "$APP_PATH" ]]; then
        echo "error: built app not found at: $APP_PATH" >&2
        exit 1
    fi
    export YOOZ_ENGINE_APP_PATH="$APP_PATH"
    echo "[run-integration] YOOZ_ENGINE_APP_PATH=$APP_PATH"
fi

echo "[run-integration] step 3: run $SCHEME_TESTS with YOOZ_INTEGRATION=1"
TEST_LOG="$LOG_DIR/test.log"

# xcodebuild forwards env vars to the xctest host only via -testEnvironment.
# Plain shell env prefix won't reach XCTest (xcodebuild spawns the test host
# through xctestrun, which filters parent env by default). Available since
# Xcode 13; required for our XCTSkipUnless gate to see YOOZ_INTEGRATION.
TEST_ENV_FLAGS=(
    -testEnvironment "YOOZ_INTEGRATION=1"
)
if [[ -n "${YOOZ_ENGINE_APP_PATH:-}" ]]; then
    TEST_ENV_FLAGS+=(-testEnvironment "YOOZ_ENGINE_APP_PATH=$YOOZ_ENGINE_APP_PATH")
fi
if [[ -n "${YOOZ_TEST_ENGINE_URL:-}" ]]; then
    TEST_ENV_FLAGS+=(-testEnvironment "YOOZ_TEST_ENGINE_URL=$YOOZ_TEST_ENGINE_URL")
fi

xcodebuild \
    -project "$PROJECT" \
    -scheme "$SCHEME_TESTS" \
    -configuration "$CONFIGURATION" \
    "${DERIVED_DATA_FLAGS[@]}" \
    CODE_SIGNING_ALLOWED=NO \
    "${TEST_ENV_FLAGS[@]}" \
    test \
    > "$TEST_LOG" 2>&1 && TEST_EXIT=0 || TEST_EXIT=$?

echo "[run-integration] step 4: summary"
echo "--- timings ---"
grep -E "^\[timing\] " "$TEST_LOG" || echo "(no timings captured)"
echo "--- xctest summary ---"
grep -E "Test Suite '.*' (passed|failed)|Executed [0-9]+ tests?" "$TEST_LOG" | tail -5 || true

if [[ "$TEST_EXIT" -ne 0 ]]; then
    echo "[run-integration] FAILED (exit $TEST_EXIT); tail of test log:"
    tail -40 "$TEST_LOG" >&2
    exit "$TEST_EXIT"
fi

echo "[run-integration] PASSED"
