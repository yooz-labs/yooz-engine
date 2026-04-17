#!/usr/bin/env bash
#
# verify-whisper-helper.sh — validation-only companion to
# build-whisper-helper.sh. Re-checks the signed artifact that already
# lives at dist/YoozEngineWhisper.app (or at the path passed as $1).
#
# Safe to run in CI-ish contexts (no builds, no keychain writes). Used
# by the yooz-whisper B1 team as a pre-embed gate: fail the whisper
# build if the engine helper artifact ever drifts out of spec.

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

TARGET="${1:-$ROOT/dist/YoozEngineWhisper.app}"
if [[ ! -d "$TARGET" ]]; then
    printf "[verify-whisper-helper] ERROR: bundle not found at %s\n" "$TARGET" >&2
    printf "  run scripts/build-whisper-helper.sh first.\n" >&2
    exit 1
fi

log() { printf "[verify-whisper-helper] %s\n" "$*"; }

log "target: $TARGET"

BIN="$TARGET/Contents/MacOS/Yooz Engine (Whisper)"
[[ -x "$BIN" ]] || { log "ERROR: main executable missing at $BIN"; exit 1; }

# 1. Codesign integrity (strict + deep catches nested framework drift).
log "codesign --verify --deep --strict --verbose=2"
codesign --verify --deep --strict --verbose=2 "$TARGET"

# 2. Identity + entitlements display — so CI logs show who signed it.
log "codesign display"
codesign --display --verbose=4 --entitlements :- "$TARGET" 2>&1 \
    | grep -E '^(Identifier|Authority|TeamIdentifier|Signature|Sealed|\[Dict\]|\[Key\])' \
    || true

# 3. Relocatability — the bundle must not depend on its absolute install
#    path. A `@executable_path/../Frameworks` rpath is expected; anything
#    like `/Users/...` or `/private/...` breaks when embedded.
log "relocatability check (LC_RPATH)"
if otool -l "$BIN" | awk '/LC_RPATH/{show=3} show-->0' | grep -E ' path (/|/Users/|/private/|/tmp/)' >/dev/null; then
    log "FAIL: executable has absolute LC_RPATH entries"
    otool -l "$BIN" | awk '/LC_RPATH/{show=3} show-->0' | grep 'path ' >&2
    exit 1
fi
log "  OK: no absolute rpaths"

# 4. Bundle + binary hashes for release-note integrity.
BUNDLE_HASH="$(find "$TARGET" -type f -not -path '*/_CodeSignature/*' -print0 \
    | sort -z \
    | xargs -0 shasum -a 256 \
    | shasum -a 256 \
    | awk '{print $1}')"
BIN_HASH="$(shasum -a 256 "$BIN" | awk '{print $1}')"

cat <<EOF

[verify-whisper-helper] OK
  bundle:         $TARGET
  binary sha256:  $BIN_HASH
  bundle sha256:  $BUNDLE_HASH
EOF
