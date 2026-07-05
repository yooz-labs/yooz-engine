#!/usr/bin/env bash
#
# build-engine-release.sh — A6 (#30) one-command build + sign for the
# full YoozEngine.app variant (STT + LLM + VAD + Grammar + AppleSTT).
#
# Produces:   dist/YoozEngine.app   (== `Yooz Engine.app`)
# Scheme:     YoozEngine            (defined in project.yml)
# Config:     Debug by default (see #38); override with YOOZ_ENGINE_CONFIG
# Signing:    Developer ID Application if present on keychain; otherwise
#             ad-hoc (`-s -`). Notarization deferred per phase5_epic.md.
#
# Idempotent: wipes dist/YoozEngine.app and the DerivedData tree every run
# so stale artifacts never leak into the signed output.

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

PROJECT="YoozEngine.xcodeproj"
SCHEME="YoozEngine"
CONFIG="${YOOZ_ENGINE_CONFIG:-Debug}"  # #38: Release SPM embed transitive deps broken
APP_NAME="Yooz Engine.app"
BIN_NAME="Yooz Engine"
DIST_DIR="$ROOT/dist"
DIST_APP="$DIST_DIR/YoozEngine.app"
DERIVED_DATA="$ROOT/.build/DerivedData-Engine"
BUILD_LOG="$ROOT/.build/engine-release-build.log"
IDENTITY="${YOOZ_SIGNING_IDENTITY:-}"
ENTITLEMENTS="$ROOT/YoozEngine/YoozEngine.entitlements"

log() { printf "[build-engine-release] %s\n" "$*"; }
fail() { printf "[build-engine-release] ERROR: %s\n" "$*" >&2; exit 1; }

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

# Resolve signing identity. Prefer Developer ID Application; fall back to
# ad-hoc so the artifact still passes a strict codesign verify.
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
        log "         Install a Developer ID cert to produce distributable artifacts."
    fi
else
    log "Using pinned signing identity from \$YOOZ_SIGNING_IDENTITY: $IDENTITY"
fi

# -----------------------------------------------------------------------------
# 2. Clean slate (scoped; never rm outside dist/ or .build/)
# -----------------------------------------------------------------------------
log "cleaning stale outputs"
rm -rf "$DIST_APP" "$DERIVED_DATA"
: > "$BUILD_LOG"

# -----------------------------------------------------------------------------
# 3. Regenerate Xcode project (project.yml is the source of truth)
# -----------------------------------------------------------------------------
log "xcodegen generate"
xcodegen generate >>"$BUILD_LOG" 2>&1 || { tail -40 "$BUILD_LOG" >&2; fail "xcodegen failed"; }

# -----------------------------------------------------------------------------
# 4. Build unsigned; we re-sign below to control identity explicitly
# -----------------------------------------------------------------------------
log "xcodebuild -scheme $SCHEME -configuration $CONFIG"
xcodebuild \
    -project "$PROJECT" \
    -scheme "$SCHEME" \
    -configuration "$CONFIG" \
    -derivedDataPath "$DERIVED_DATA" \
    -skipMacroValidation \
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
# 5. Stage into dist/ before signing
# -----------------------------------------------------------------------------
log "staging -> $DIST_APP"
rm -rf "$DIST_APP"
cp -R "$BUILT_APP" "$DIST_APP"

# -----------------------------------------------------------------------------
# 5b. Copy SPM PackageFrameworks that xcodebuild linked but did not embed.
#     Transitive deps from STTModule/LLMModule (MLX, MLXNN, MLXLLM, etc.)
#     aren't auto-embedded even when the products are direct app-target deps.
#     Post-build copy is the pragmatic workaround until #38 lands.
# -----------------------------------------------------------------------------
PKG_FRAMEWORKS_SRC="$DERIVED_DATA/Build/Products/$CONFIG/PackageFrameworks"
DIST_FRAMEWORKS="$DIST_APP/Contents/Frameworks"
if [[ -d "$PKG_FRAMEWORKS_SRC" ]]; then
    log "copying PackageFrameworks -> Contents/Frameworks"
    mkdir -p "$DIST_FRAMEWORKS"
    for fw in "$PKG_FRAMEWORKS_SRC"/*.framework; do
        [[ -d "$fw" ]] || continue
        name="$(basename "$fw")"
        [[ -d "$DIST_FRAMEWORKS/$name" ]] && continue  # already embedded by Xcode
        log "  copy: $name"
        cp -R "$fw" "$DIST_FRAMEWORKS/"
    done
fi

# -----------------------------------------------------------------------------
# 6. Sign nested dylibs + frameworks (deepest-first), then the outer app
# -----------------------------------------------------------------------------
log "entitlements: $ENTITLEMENTS"

SIGN_OPTS=(--force --timestamp --options runtime --sign "$IDENTITY")
if [[ "$IDENTITY" == "-" ]]; then
    # Ad-hoc cannot carry a secure timestamp; hardened runtime + ad-hoc is
    # not honoured by Gatekeeper regardless. Drop both flags to avoid
    # "unknown option -o runtime" on some SDKs.
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

# Hash for downstream integrity checks / release notes.
HASH="$(shasum -a 256 "$BIN" | awk '{print $1}')"

# -----------------------------------------------------------------------------
# 8. Summary
# -----------------------------------------------------------------------------
cat <<EOF

[build-engine-release] DONE
  bundle:      $DIST_APP
  identity:    $IDENTITY
  config:      $CONFIG
  binary sha:  $HASH
  log:         $BUILD_LOG

Next steps:
  - Smoke-launch:   "$DIST_APP/Contents/MacOS/$BIN_NAME" &
                    curl http://127.0.0.1:19920/v1/health
  - Zip:            ditto -c -k --keepParent "$DIST_APP" dist/YoozEngine.app.zip
EOF
