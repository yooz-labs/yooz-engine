#!/bin/bash
#
# run-live-sdk-tests.sh — run the YoozEngineClient SDK tests against a real
# engine.
#
# These tests live under Tests/YoozEngineClientTests and exercise the
# live-engine code paths that the `.xctestplan` (LiveSDK) enumerates. The
# SPM test target isn't an xcodeproj scheme, so `xcodebuild -testPlan LiveSDK`
# will NOT resolve — we use `swift test` instead, which picks up the shell
# environment verbatim (no `-testEnvironment` filtering).
#
# Flow:
#   1. Build YoozEngine, locate the `.app`.
#   2. Launch the engine subprocess from the built bundle.
#   3. Poll /v1/health until 200 OK (15s max).
#   4. `swift test --filter YoozEngineClientTests` with the env vars from
#      TestPlans/LiveSDK.xctestplan injected via the shell.
#   5. Terminate the engine and report.
#
# Usage:
#   scripts/run-live-sdk-tests.sh
#
# Requirements: xcodegen, xcodebuild, swift (toolchain).

set -eo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

SCHEME_APP="YoozEngine"
PROJECT="YoozEngine.xcodeproj"
CONFIGURATION="Debug"
LOG_DIR="$ROOT/.build/live-sdk-logs"
DERIVED_DATA="$ROOT/.build/DerivedData"
ENGINE_URL="http://127.0.0.1:19920"
HEALTH_URL="$ENGINE_URL/v1/health"

mkdir -p "$LOG_DIR" "$DERIVED_DATA"

ENGINE_PID=""
cleanup() {
    if [[ -n "$ENGINE_PID" ]] && kill -0 "$ENGINE_PID" 2>/dev/null; then
        echo "[run-live-sdk] stopping engine pid=$ENGINE_PID"
        kill "$ENGINE_PID" 2>/dev/null || true
        # Give it up to 5s, then SIGKILL.
        for _ in 1 2 3 4 5; do
            kill -0 "$ENGINE_PID" 2>/dev/null || break
            sleep 1
        done
        kill -9 "$ENGINE_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "[run-live-sdk] root=$ROOT"
echo "[run-live-sdk] step 1: xcodegen + build $SCHEME_APP"
command -v xcodegen >/dev/null 2>&1 || {
    echo "error: xcodegen not on PATH" >&2; exit 1
}
xcodegen generate
xcodebuild \
    -project "$PROJECT" \
    -scheme "$SCHEME_APP" \
    -configuration "$CONFIGURATION" \
    -derivedDataPath "$DERIVED_DATA" \
    CODE_SIGNING_ALLOWED=NO \
    build \
    > "$LOG_DIR/build.log" 2>&1 || {
        echo "error: build failed; see $LOG_DIR/build.log" >&2
        tail -40 "$LOG_DIR/build.log" >&2
        exit 1
    }

APP_PATH="$DERIVED_DATA/Build/Products/$CONFIGURATION/Yooz Engine.app"
EXE_PATH="$APP_PATH/Contents/MacOS/Yooz Engine"
[[ -x "$EXE_PATH" ]] || {
    echo "error: engine executable missing: $EXE_PATH" >&2; exit 1
}

echo "[run-live-sdk] step 2: launch engine"
"$EXE_PATH" > "$LOG_DIR/engine.log" 2>&1 &
ENGINE_PID=$!
echo "[run-live-sdk] engine pid=$ENGINE_PID"

echo "[run-live-sdk] step 3: wait for $HEALTH_URL"
for attempt in $(seq 1 30); do
    if curl -sf -o /dev/null "$HEALTH_URL"; then
        echo "[run-live-sdk] engine ready after ${attempt}*0.5s"
        break
    fi
    if ! kill -0 "$ENGINE_PID" 2>/dev/null; then
        echo "error: engine died before ready; see $LOG_DIR/engine.log" >&2
        tail -40 "$LOG_DIR/engine.log" >&2
        exit 1
    fi
    sleep 0.5
done
if ! curl -sf -o /dev/null "$HEALTH_URL"; then
    echo "error: engine never became ready; see $LOG_DIR/engine.log" >&2
    tail -40 "$LOG_DIR/engine.log" >&2
    exit 1
fi

echo "[run-live-sdk] step 4: swift test (env from LiveSDK.xctestplan)"
# The `.xctestplan` at TestPlans/LiveSDK.xctestplan documents the intended
# environment in one place; we mirror it here for `swift test`.
export YOOZ_ENGINE_LIVE_TESTS=1
export YOOZ_TEST_ENGINE_URL="$ENGINE_URL"

TEST_LOG="$LOG_DIR/test.log"
swift test --filter YoozEngineClientTests \
    > "$TEST_LOG" 2>&1 && TEST_EXIT=0 || TEST_EXIT=$?

echo "[run-live-sdk] step 5: summary"
grep -E "Test Suite '.*' (passed|failed)|Executed [0-9]+ tests?" "$TEST_LOG" | tail -5 || true

if [[ "$TEST_EXIT" -ne 0 ]]; then
    echo "[run-live-sdk] FAILED (exit $TEST_EXIT); tail of test log:"
    tail -40 "$TEST_LOG" >&2
    exit "$TEST_EXIT"
fi

echo "[run-live-sdk] PASSED (log: $TEST_LOG)"
