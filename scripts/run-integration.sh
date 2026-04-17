#!/bin/bash
#
# run-integration.sh — end-to-end harness runner for YoozEngine.
#
# Flow:
#   1. Regenerate the Xcode project (needed after project.yml edits).
#   2. Build the `YoozEngine` scheme; locate the resulting `Yooz Engine.app`
#      inside DerivedData.
#   3. Invoke `xcodebuild test -testPlan IntegrationTests`. The plan sets
#      `YOOZ_INTEGRATION=1` unconditionally and expands
#      `YOOZ_ENGINE_APP_PATH` / `YOOZ_TEST_ENGINE_URL` from build settings
#      passed on the command line. The harness itself launches the
#      subprocess — this script does NOT start the engine ahead of time.
#   4. Print per-endpoint timings (grep `[timing] ...` lines emitted by
#      IntegrationTestCase) and an overall pass/fail summary.
#
# Usage:
#   scripts/run-integration.sh                  # full run
#   YOOZ_TEST_ENGINE_URL=http://127.0.0.1:19920 \
#       scripts/run-integration.sh              # reuse an already-running engine
#
# Requirements: xcodegen, xcodebuild (Xcode command-line tools).
#
# History: Xcode's `-testEnvironment` flag is unsupported on recent
# toolchains, so `YOOZ_INTEGRATION=1` never reached the xctest process and
# every end-to-end test silently skipped. The `.xctestplan` route is the
# supported way to pin environment variables on the test host.

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
TEST_PLAN="IntegrationTests"
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
BUILD_SETTING_FLAGS=()
if [[ -n "${YOOZ_TEST_ENGINE_URL:-}" ]]; then
    echo "[run-integration] YOOZ_TEST_ENGINE_URL=$YOOZ_TEST_ENGINE_URL — skipping engine build"
    BUILD_SETTING_FLAGS+=("YOOZ_TEST_ENGINE_URL=$YOOZ_TEST_ENGINE_URL")
    # Still provide APP_PATH as an empty string so the xctestplan
    # $(YOOZ_ENGINE_APP_PATH) expansion resolves; IntegrationTestCase only
    # reads it when YOOZ_TEST_ENGINE_URL is unset.
    BUILD_SETTING_FLAGS+=("YOOZ_ENGINE_APP_PATH=")
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
    BUILD_SETTING_FLAGS+=("YOOZ_ENGINE_APP_PATH=$APP_PATH")
    BUILD_SETTING_FLAGS+=("YOOZ_TEST_ENGINE_URL=")
    echo "[run-integration] YOOZ_ENGINE_APP_PATH=$APP_PATH"
fi

echo "[run-integration] step 3: run $SCHEME_TESTS via -testPlan $TEST_PLAN"
TEST_LOG="$LOG_DIR/test.log"

# xcodebuild forwards the IntegrationTests.xctestplan environment to the
# xctest host. Command-line build settings (KEY=VALUE after the action) are
# expanded into `$(KEY)` references inside the plan's
# `environmentVariableEntries`, which is how we parameterize the app path
# without shipping a developer-specific plan.
xcodebuild \
    -project "$PROJECT" \
    -scheme "$SCHEME_TESTS" \
    -configuration "$CONFIGURATION" \
    -testPlan "$TEST_PLAN" \
    "${DERIVED_DATA_FLAGS[@]}" \
    CODE_SIGNING_ALLOWED=NO \
    "${BUILD_SETTING_FLAGS[@]}" \
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

echo "[run-integration] PASSED (log: $TEST_LOG)"
