#!/usr/bin/env bash
#
# smoke-test-release.sh — A6 (#30) post-release verification.
# Launches each signed `.app` under `dist/`, polls `/v1/health` on the
# fixed engine port (19920), and prints ALIVE/DEAD per variant. All three
# variants bind the same port, so launches are strictly serial: the
# previous process is terminated and the port is confirmed free before
# the next one starts.
#
# Exit: non-zero if any variant fails to respond, or if the port is held
# by a stale process at startup.

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

DIST_DIR="$ROOT/dist"
PORT="${YOOZ_ENGINE_PORT:-19920}"
HOST="127.0.0.1"
HEALTH_URL="http://$HOST:$PORT/v1/health"
LOG="$ROOT/.build/smoke-test-release.log"

# Variant spec: dist-name|binary-name
VARIANTS=(
    "YoozEngine.app|Yooz Engine"
    "YoozEngineLite.app|Yooz Engine (Lite)"
    "YoozEngineWhisper.app|Yooz Engine (Whisper)"
)

mkdir -p "$ROOT/.build"
: > "$LOG"

log() { printf "[smoke-test] %s\n" "$*"; }
warn() { printf "[smoke-test] WARN: %s\n" "$*" >&2; }

# -----------------------------------------------------------------------------
# 0. Preflight: port must be free (don't smoke-test against a foreign engine)
# -----------------------------------------------------------------------------
port_busy() {
    # lsof exit: 0 = in use, 1 = free. Suppress stderr for the "not running" case.
    lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1
}

if port_busy; then
    warn "port $PORT is held by another process at startup:"
    lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >&2 || true
    printf "[smoke-test] ERROR: refusing to proceed; kill the holder and re-run.\n" >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# 1. Per-variant launch helper
# -----------------------------------------------------------------------------
declare -A RESULTS=()
OVERALL_EXIT=0

smoke_one() {
    local name="$1" bin_name="$2"
    local app_path="$DIST_DIR/$name"
    local bin_path="$app_path/Contents/MacOS/$bin_name"

    log "==> $name"

    if [[ ! -d "$app_path" ]]; then
        warn "missing bundle: $app_path"
        RESULTS["$name"]="DEAD (missing bundle)"
        OVERALL_EXIT=1
        return
    fi
    if [[ ! -x "$bin_path" ]]; then
        warn "missing/non-exec binary: $bin_path"
        RESULTS["$name"]="DEAD (no binary)"
        OVERALL_EXIT=1
        return
    fi

    # Launch in the background. Use a plain exec so we own the PID; `open -a`
    # would return before ready and make cleanup racy.
    "$bin_path" >>"$LOG" 2>&1 &
    local pid=$!
    log "  launched pid=$pid"

    # Poll /v1/health for up to ~20s.
    local status="DEAD (timeout)"
    local body=""
    for i in $(seq 1 40); do
        if ! kill -0 "$pid" 2>/dev/null; then
            status="DEAD (exited early)"
            break
        fi
        body="$(curl -fsS --max-time 1 "$HEALTH_URL" 2>/dev/null || true)"
        if [[ -n "$body" ]]; then
            status="ALIVE"
            break
        fi
        sleep 0.5
    done

    if [[ "$status" == "ALIVE" ]]; then
        log "  $HEALTH_URL -> OK"
        log "  body: $(printf '%s' "$body" | head -c 200)"
    else
        warn "  $name did not respond on $HEALTH_URL within 20s"
        OVERALL_EXIT=1
    fi

    # Terminate cleanly.
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        for _ in $(seq 1 20); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.25
        done
        if kill -0 "$pid" 2>/dev/null; then
            warn "  pid $pid did not exit on SIGTERM; sending SIGKILL"
            kill -9 "$pid" 2>/dev/null || true
        fi
    fi

    # Wait for port to free up so the next variant can bind.
    for _ in $(seq 1 40); do
        port_busy || break
        sleep 0.25
    done
    if port_busy; then
        warn "  port $PORT still held after termination"
        status="${status} (port stuck)"
        OVERALL_EXIT=1
    fi

    RESULTS["$name"]="$status"
}

for spec in "${VARIANTS[@]}"; do
    name="${spec%%|*}"
    bin_name="${spec##*|}"
    smoke_one "$name" "$bin_name"
done

# -----------------------------------------------------------------------------
# 2. Summary table
# -----------------------------------------------------------------------------
printf "\n[smoke-test] RESULTS\n"
for spec in "${VARIANTS[@]}"; do
    name="${spec%%|*}"
    printf "  %-28s  %s\n" "$name" "${RESULTS[$name]:-UNKNOWN}"
done
printf "\n  log: %s\n" "$LOG"

exit "$OVERALL_EXIT"
