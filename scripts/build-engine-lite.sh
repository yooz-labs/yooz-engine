#!/usr/bin/env bash
#
# build-engine-lite.sh — A6 (#30) one-command build + sign for the
# YoozEngineLite variant (Apple STT only, no MLX runtime).
#
# Produces:   dist/YoozEngineLite.app   (== `Yooz Engine (Lite).app`)
# Scheme:     YoozEngineLite            (defined in project.yml)
# Config:     Debug by default (see #38); override with YOOZ_ENGINE_CONFIG
# Signing:    Developer ID Application if present on keychain; otherwise
#             ad-hoc (`-s -`). Notarization deferred per phase5_epic.md.
#
# Sub-GB variant targeted at Remi-class apps and future iOS ports. STTModule
# and VADModule are not linked; AppleSTTModule is the sole speech backend.

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

PROJECT="YoozEngine.xcodeproj"
SCHEME="YoozEngineLite"
CONFIG="${YOOZ_ENGINE_CONFIG:-Debug}"
APP_NAME="Yooz Engine (Lite).app"
BIN_NAME="Yooz Engine (Lite)"
DIST_DIR="$ROOT/dist"
DIST_APP="$DIST_DIR/YoozEngineLite.app"
DERIVED_DATA="$ROOT/.build/DerivedData-EngineLite"
BUILD_LOG="$ROOT/.build/engine-lite-build.log"
IDENTITY="${YOOZ_SIGNING_IDENTITY:-}"
# Prefer a Lite-specific entitlements file if one ever lands; fall back to
# the full engine's entitlements (current state per project.yml). Mirrors
# the pattern in build-whisper-helper.sh:147-149.
ENTITLEMENTS="$ROOT/YoozEngine/YoozEngineLite.entitlements"
[[ -f "$ENTITLEMENTS" ]] || ENTITLEMENTS="$ROOT/YoozEngine/YoozEngine.entitlements"

log() { printf "[build-engine-lite] %s\n" "$*"; }
fail() { printf "[build-engine-lite] ERROR: %s\n" "$*" >&2; exit 1; }

# -----------------------------------------------------------------------------
# 0. Config warning
# -----------------------------------------------------------------------------
if [[ "$CONFIG" == "Debug" ]]; then
    log "WARNING: building with CONFIG=Debug (see #38 workaround)."
    log "         Debug binaries carry -Onone, debug symbols, and -enable-testing."
    log "         Do NOT ship this to users; override once #38 lands with:"
    log "         export YOOZ_ENGINE_CONFIG=Release"
fi

# -----------------------------------------------------------------------------
# 1. Preflight
# -----------------------------------------------------------------------------
command -v xcodegen >/dev/null 2>&1 || fail "xcodegen not on PATH (brew install xcodegen)"
command -v xcodebuild >/dev/null 2>&1 || fail "xcodebuild not found; install Xcode CLTs"
command -v codesign >/dev/null 2>&1 || fail "codesign missing from system"

mkdir -p "$ROOT/.build" "$DIST_DIR"

[[ -f "$ENTITLEMENTS" ]] || fail "entitlements file missing: $ENTITLEMENTS"

if [[ -z "$IDENTITY" ]]; then
    IDENTITY_LINE="$(security find-identity -v -p codesigning 2>/dev/null \
        | grep 'Developer ID Application' \
        | head -1 || true)"
    if [[ -n "$IDENTITY_LINE" ]]; then
        IDENTITY="$(printf '%s\n' "$IDENTITY_LINE" \
            | sed -E 's/^[[:space:]]*[0-9]+\)[[:space:]]+([A-F0-9]+).*/\1/')"
        log "Using Developer ID Application identity: $IDENTITY"
    else
        IDENTITY="-"
        log "WARNING: no Developer ID Application cert in keychain; ad-hoc signing."
    fi
else
    log "Using pinned signing identity from \$YOOZ_SIGNING_IDENTITY: $IDENTITY"
fi

# -----------------------------------------------------------------------------
# 2. Clean slate
# -----------------------------------------------------------------------------
log "cleaning stale outputs"
rm -rf "$DIST_APP" "$DERIVED_DATA"
: > "$BUILD_LOG"

# -----------------------------------------------------------------------------
# 3. xcodegen
# -----------------------------------------------------------------------------
log "xcodegen generate"
xcodegen generate >>"$BUILD_LOG" 2>&1 || { tail -40 "$BUILD_LOG" >&2; fail "xcodegen failed"; }

# -----------------------------------------------------------------------------
# 4. Build unsigned
# -----------------------------------------------------------------------------
log "xcodebuild -scheme $SCHEME -configuration $CONFIG"
xcodebuild \
    -project "$PROJECT" \
    -scheme "$SCHEME" \
    -configuration "$CONFIG" \
    -derivedDataPath "$DERIVED_DATA" \
    CODE_SIGN_IDENTITY="" \
    CODE_SIGNING_REQUIRED=NO \
    CODE_SIGNING_ALLOWED=NO \
    build \
    >>"$BUILD_LOG" 2>&1 || {
        tail -80 "$BUILD_LOG" >&2
        fail "build failed; full log at $BUILD_LOG"
    }

BUILT_APP="$DERIVED_DATA/Build/Products/$CONFIG/$APP_NAME"
[[ -d "$BUILT_APP" ]] || fail "built app not found at: $BUILT_APP"

# -----------------------------------------------------------------------------
# 5. Stage + PackageFrameworks post-copy
# -----------------------------------------------------------------------------
log "staging -> $DIST_APP"
rm -rf "$DIST_APP"
cp -R "$BUILT_APP" "$DIST_APP"

PKG_FRAMEWORKS_SRC="$DERIVED_DATA/Build/Products/$CONFIG/PackageFrameworks"
DIST_FRAMEWORKS="$DIST_APP/Contents/Frameworks"
if [[ -d "$PKG_FRAMEWORKS_SRC" ]]; then
    log "copying PackageFrameworks -> Contents/Frameworks"
    mkdir -p "$DIST_FRAMEWORKS"
    for fw in "$PKG_FRAMEWORKS_SRC"/*.framework; do
        [[ -d "$fw" ]] || continue
        name="$(basename "$fw")"
        [[ -d "$DIST_FRAMEWORKS/$name" ]] && continue
        log "  copy: $name"
        cp -R "$fw" "$DIST_FRAMEWORKS/"
    done
fi

# -----------------------------------------------------------------------------
# 6. Sign
# -----------------------------------------------------------------------------
log "entitlements: $ENTITLEMENTS"

SIGN_OPTS=(--force --timestamp --options runtime --sign "$IDENTITY")
if [[ "$IDENTITY" == "-" ]]; then
    SIGN_OPTS=(--force --sign "$IDENTITY")
fi

log "signing nested frameworks + dylibs"
while IFS= read -r -d '' artifact; do
    log "  sign: ${artifact#$DIST_APP/}"
    codesign "${SIGN_OPTS[@]}" "$artifact" \
        >>"$BUILD_LOG" 2>&1 || {
            tail -40 "$BUILD_LOG" >&2
            fail "codesign failed on $artifact"
        }
done < <(find "$DIST_APP/Contents" \
    \( -name '*.dylib' -o -name '*.framework' -o -name '*.bundle' -o -name '*.xpc' \) \
    -print0 | sort -rz)

log "signing outer .app with entitlements"
codesign "${SIGN_OPTS[@]}" --entitlements "$ENTITLEMENTS" "$DIST_APP" \
    >>"$BUILD_LOG" 2>&1 || {
        tail -40 "$BUILD_LOG" >&2
        fail "codesign of outer bundle failed"
    }

# -----------------------------------------------------------------------------
# 7. Verify
# -----------------------------------------------------------------------------
log "codesign --verify --deep --strict"
codesign --verify --deep --strict --verbose=2 "$DIST_APP" \
    >>"$BUILD_LOG" 2>&1 || {
        tail -40 "$BUILD_LOG" >&2
        fail "codesign verify failed"
    }
codesign --display --verbose=2 "$DIST_APP" 2>&1 | tee -a "$BUILD_LOG" \
    | grep -E '^(Identifier|Authority|TeamIdentifier|Signature|Sealed)' || true

BIN="$DIST_APP/Contents/MacOS/$BIN_NAME"
[[ -x "$BIN" ]] || fail "main executable missing at $BIN"
HASH="$(shasum -a 256 "$BIN" | awk '{print $1}')"

cat <<EOF

[build-engine-lite] DONE
  bundle:      $DIST_APP
  identity:    $IDENTITY
  config:      $CONFIG
  binary sha:  $HASH
  log:         $BUILD_LOG
EOF
