#!/usr/bin/env bash
#
# build-whisper-helper.sh — A5 (#29) one-command build + sign for the
# YoozEngineWhisper helper variant.
#
# Produces:   dist/YoozEngineWhisper.app   (== `Yooz Engine (Whisper).app`)
# Scheme:     YoozEngineWhisper           (defined in project.yml)
# Config:     Release
# Signing:    Developer ID Application if one is on the keychain; otherwise
#             ad-hoc (`-s -`) with a clear warning. Notarization deferred
#             per phase5_epic.md (see Stream D / issue #24).
# Embedding:  The helper is relocatable — `LC_RPATH` stays as
#             `@executable_path/../Frameworks`, so dropping the bundle into
#             a host app's `Contents/Helpers/` still resolves Frameworks
#             inside the helper's own `Contents/Frameworks/`.
#
# Idempotent: cleans `dist/` + the DerivedData tree before every run so
# stale artifacts never leak into the signed output.

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

PROJECT="YoozEngine.xcodeproj"
SCHEME="YoozEngineWhisper"
CONFIG="Release"
APP_NAME="Yooz Engine (Whisper).app"
DIST_DIR="$ROOT/dist"
DIST_APP="$DIST_DIR/YoozEngineWhisper.app"
DERIVED_DATA="$ROOT/.build/DerivedData-WhisperHelper"
BUILD_LOG="$ROOT/.build/whisper-helper-build.log"
IDENTITY="${YOOZ_SIGNING_IDENTITY:-}"  # allow override via env

log() { printf "[build-whisper-helper] %s\n" "$*"; }
fail() { printf "[build-whisper-helper] ERROR: %s\n" "$*" >&2; exit 1; }

# -----------------------------------------------------------------------------
# 1. Preflight
# -----------------------------------------------------------------------------
command -v xcodegen >/dev/null 2>&1 || fail "xcodegen not on PATH (brew install xcodegen)"
command -v xcodebuild >/dev/null 2>&1 || fail "xcodebuild not found; install Xcode command-line tools"
command -v codesign >/dev/null 2>&1 || fail "codesign missing from system"

mkdir -p "$ROOT/.build" "$DIST_DIR"

# Resolve signing identity. Prefer Developer ID Application; fall back to
# ad-hoc (`-`) so the artifact still passes a strict codesign verify.
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
        log "WARNING: no Developer ID Application cert in keychain; falling back to ad-hoc signing."
        log "         Install a Developer ID cert to produce distributable artifacts."
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
# 3. Regenerate Xcode project (project.yml is source of truth)
# -----------------------------------------------------------------------------
log "xcodegen generate"
xcodegen generate >>"$BUILD_LOG" 2>&1 || { tail -40 "$BUILD_LOG" >&2; fail "xcodegen failed"; }

# -----------------------------------------------------------------------------
# 4. Build Release (unsigned; we re-sign below to control identity explicitly)
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
# 5. Stage into dist/ before signing so the signature sticks to the shipped
#    artifact, not the DerivedData tree (which gets clobbered on next build).
# -----------------------------------------------------------------------------
log "staging → $DIST_APP"
rm -rf "$DIST_APP"
cp -R "$BUILT_APP" "$DIST_APP"

# -----------------------------------------------------------------------------
# 6. Sign every embedded framework and dylib, then the app itself.
#    `--deep` is intentionally avoided — Apple deprecates it and it can
#    skip nested bundles that need distinct entitlements. We walk the
#    tree explicitly.
# -----------------------------------------------------------------------------
ENTITLEMENTS="$ROOT/YoozEngine/YoozEngineWhisper.entitlements"
[[ -f "$ENTITLEMENTS" ]] || ENTITLEMENTS="$ROOT/YoozEngine/YoozEngine.entitlements"
[[ -f "$ENTITLEMENTS" ]] || fail "no entitlements file found"
log "entitlements: $ENTITLEMENTS"

SIGN_OPTS=(--force --timestamp --options runtime --sign "$IDENTITY")
if [[ "$IDENTITY" == "-" ]]; then
    # Ad-hoc signatures cannot carry a secure timestamp. Drop --timestamp
    # and --options runtime because hardened runtime with ad-hoc is not
    # honoured by Gatekeeper anyway; keeping the flags would make
    # codesign error out with "unknown option -o runtime" on some SDKs.
    SIGN_OPTS=(--force --sign "$IDENTITY")
fi

log "signing nested frameworks + dylibs"
# Sign deepest-first: dylibs, then frameworks, then the outer .app. Any
# `.app` bundled inside (e.g. Sparkle-style helpers) would be handled here
# too; none today but the walk is future-proof.
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
# 7. Verify. `--deep --strict --verbose=2` catches mismatched embedded
#    signatures and broken resource sealing; the separate --display pass
#    prints the identity + team prefix for sanity.
# -----------------------------------------------------------------------------
log "codesign --verify --deep --strict"
codesign --verify --deep --strict --verbose=2 "$DIST_APP" \
    >>"$BUILD_LOG" 2>&1 || {
        tail -40 "$BUILD_LOG" >&2
        fail "codesign verify failed"
    }
codesign --display --verbose=2 "$DIST_APP" 2>&1 | tee -a "$BUILD_LOG" \
    | grep -E '^(Identifier|Authority|TeamIdentifier|Signature|Sealed)' || true

# Sanity check that the bundle is relocatable — no embedded absolute
# rpaths that would break when dropped into a host .app/Contents/Helpers.
BIN="$DIST_APP/Contents/MacOS/Yooz Engine (Whisper)"
[[ -x "$BIN" ]] || fail "main executable missing at $BIN"
if otool -l "$BIN" 2>/dev/null | grep -A2 LC_RPATH | grep -E ' path (/|/Users/)' >/dev/null; then
    log "WARNING: executable contains absolute LC_RPATH entries — may break when embedded"
    otool -l "$BIN" | grep -A2 LC_RPATH | grep -E ' path ' >&2 || true
fi

# Hash for downstream integrity checks / release notes.
HASH="$(shasum -a 256 "$BIN" | awk '{print $1}')"

# -----------------------------------------------------------------------------
# 8. Summary
# -----------------------------------------------------------------------------
cat <<EOF

[build-whisper-helper] DONE
  bundle:      $DIST_APP
  identity:    $IDENTITY
  config:      $CONFIG
  binary sha:  $HASH
  log:         $BUILD_LOG

Next steps:
  - Verify anytime with:  scripts/verify-whisper-helper.sh
  - Smoke-launch:         "$DIST_APP/Contents/MacOS/Yooz Engine (Whisper)" &
                          curl http://127.0.0.1:19920/v1/health
  - Embed into Whisper:   cp -R "$DIST_APP" Yooz\\ Whisper.app/Contents/Helpers/
EOF
