#!/usr/bin/env bash
#
# verify-xpc-portability.sh — engine#248 acceptance gate.
#
# Proves `YoozEngineXPC.xpc` is self-contained in RELEASE, not just Debug
# (Debug is masked by an absolute, machine-local DerivedData LC_RPATH — see
# scripts/embed-xpc-package-frameworks.sh's header comment). Two checks:
#
#   1. Runtime: build `YoozEngineXPCHarness` in Release with an isolated
#      -derivedDataPath, copy the built app OUTSIDE any DerivedData tree,
#      re-sign it there, and run it. Requires the HEALTH_OK round trip
#      (STREAM/EVENTS lines are printed too but aren't gated on — they can
#      legitimately fail without weights/permissions on a fresh machine;
#      HEALTH_OK + a clean exit is the packaging proof per engine#248's
#      acceptance criteria).
#   2. Static: for every framework embedded in the `.xpc`, `otool -L` must
#      show no @rpath dependency that isn't resolvable either inside the
#      bundle or via an allowed system search path (/usr/lib, /System) —
#      catches a regression back to "SPM frameworks reference @rpath but
#      never get embedded" without needing a full runtime spawn.
#
# Uses scripts/dev-adhoc-xpc.entitlements (adds
# com.apple.security.cs.disable-library-validation on top of the real
# XPCService/YoozEngineXPC.entitlements baseline) because this repo's own
# dev-only build signs ad-hoc (CODE_SIGN_IDENTITY: "-", see YoozEngineXPC's
# project.yml comment) and hardened-runtime library validation rejects
# loading ANY ad-hoc-signed framework that doesn't share the loading
# binary's Team ID — which no two independently ad-hoc-signed artifacts
# ever do, regardless of engine#248's fix. A real Developer ID / Apple
# Distribution build (what every shipping consumer app uses) does not hit
# this: scripts/embed-xpc-package-frameworks.sh and
# scripts/resign-embedded-xpc.sh already sign every nested framework with
# $EXPANDED_CODE_SIGN_IDENTITY, so a real build's nested content shares the
# same Team ID as its host and satisfies library validation without this
# entitlement. Override YOOZ_XPC_SIGN_IDENTITY / YOOZ_XPC_ENTITLEMENTS to
# exercise a real identity instead (e.g. in a CI environment with a
# provisioned Developer ID cert available).
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

PROJECT="YoozEngine.xcodeproj"
SCHEME="YoozEngineXPCHarness"
CONFIG="Release"
DERIVED_DATA="$ROOT/.build/DerivedData-XPCPortabilityCheck"
VERIFY_DIR="$ROOT/.build/xpc-portability-check"
BUILD_LOG="$ROOT/.build/xpc-portability-check-build.log"

IDENTITY="${YOOZ_XPC_SIGN_IDENTITY:--}"
ENTITLEMENTS="${YOOZ_XPC_ENTITLEMENTS:-$ROOT/scripts/dev-adhoc-xpc.entitlements}"

log() { printf "[verify-xpc-portability] %s\n" "$*"; }
fail() { printf "[verify-xpc-portability] ERROR: %s\n" "$*" >&2; exit 1; }

command -v xcodegen >/dev/null 2>&1 || fail "xcodegen not on PATH"
command -v xcodebuild >/dev/null 2>&1 || fail "xcodebuild not found"
command -v codesign >/dev/null 2>&1 || fail "codesign missing"
[[ -f "$ENTITLEMENTS" ]] || fail "entitlements override not found: $ENTITLEMENTS"

mkdir -p "$ROOT/.build"
rm -rf "$DERIVED_DATA" "$VERIFY_DIR"
: > "$BUILD_LOG"

# -----------------------------------------------------------------------------
# 1. Regenerate the project (project.yml is source of truth) and build
#    Release. DEBUG_INFORMATION_FORMAT=dwarf skips dSYM generation -- this
#    is a throwaway verification build, and dSYM bundles for 27 embedded
#    frameworks cost real disk for no benefit here.
# -----------------------------------------------------------------------------
log "xcodegen generate"
xcodegen generate >>"$BUILD_LOG" 2>&1 || { tail -40 "$BUILD_LOG" >&2; fail "xcodegen failed"; }

log "xcodebuild -scheme $SCHEME -configuration $CONFIG (identity: $IDENTITY)"
# Deliberately does NOT pass CODE_SIGN_IDENTITY on the command line: doing
# so applies it GLOBALLY to every target in the build, including SPM
# plugin executable targets like MLXHuggingFaceMacros. That forces a
# different signing pass/timing onto the macro plugin binary than its own
# default build gets, and was observed to make swift-frontend's
# macro-plugin handshake fail deterministically ("produced malformed
# response"). CODE_SIGN_ENTITLEMENTS alone (only meaningful to the two
# signable bundle targets) doesn't have this problem. project.yml already
# pins YoozEngineXPC/YoozEngineXPCHarness to CODE_SIGN_IDENTITY: "-", so
# the default $IDENTITY ("-") already matches the build's actual identity
# without needing the command-line override. A caller who overrides
# YOOZ_XPC_SIGN_IDENTITY (e.g. a real Developer ID cert) explicitly wants
# that identity applied everywhere, so it's threaded through in that case.
XCODEBUILD_IDENTITY_ARGS=()
[[ "$IDENTITY" != "-" ]] && XCODEBUILD_IDENTITY_ARGS=(CODE_SIGN_IDENTITY="$IDENTITY")
xcodebuild \
    -project "$PROJECT" \
    -scheme "$SCHEME" \
    -configuration "$CONFIG" \
    -skipMacroValidation \
    -derivedDataPath "$DERIVED_DATA" \
    "${XCODEBUILD_IDENTITY_ARGS[@]}" \
    CODE_SIGN_ENTITLEMENTS="$ENTITLEMENTS" \
    DEBUG_INFORMATION_FORMAT=dwarf \
    build \
    >>"$BUILD_LOG" 2>&1 || {
        tail -80 "$BUILD_LOG" >&2
        fail "build failed; full log at $BUILD_LOG"
    }

BUILT_APP="$DERIVED_DATA/Build/Products/$CONFIG/YoozEngineXPCHarness.app"
[[ -d "$BUILT_APP" ]] || fail "built app not found at: $BUILT_APP"
XPC_IN_APP="$BUILT_APP/Contents/XPCServices/YoozEngineXPC.xpc"
[[ -d "$XPC_IN_APP" ]] || fail "embedded XPC service not found at: $XPC_IN_APP"

# -----------------------------------------------------------------------------
# 2. Relocate OUTSIDE DerivedData -- the whole point of this proof is that
#    the artifact does not depend on its build-machine-local absolute path.
# -----------------------------------------------------------------------------
mkdir -p "$VERIFY_DIR"
RELOCATED_APP="$VERIFY_DIR/YoozEngineXPCHarness.app"
log "relocating → $RELOCATED_APP"
ditto "$BUILT_APP" "$RELOCATED_APP" || fail "ditto copy failed"

RELOCATED_XPC="$RELOCATED_APP/Contents/XPCServices/YoozEngineXPC.xpc"

# Re-sign if needed: `ditto` preserves signatures byte-for-byte, so this is
# normally a no-op, but re-verify (and re-sign on failure) rather than
# assume -- a moved bundle is exactly the scenario a stale/incomplete
# signature would surface in.
if ! codesign --verify --deep --strict "$RELOCATED_APP" >/dev/null 2>&1; then
    log "signature invalid after relocation, re-signing"
    shopt -s nullglob
    for fw in "$RELOCATED_XPC"/Contents/Frameworks/*.framework; do
        codesign --force --sign "$IDENTITY" "$fw" || fail "re-sign failed: $fw"
    done
    shopt -u nullglob
    codesign --force --sign "$IDENTITY" --entitlements "$ENTITLEMENTS" "$RELOCATED_XPC" \
        || fail "re-sign failed: $RELOCATED_XPC"
    codesign --force --sign "$IDENTITY" --entitlements "$ENTITLEMENTS" "$RELOCATED_APP" \
        || fail "re-sign failed: $RELOCATED_APP"
fi

log "codesign --verify --deep --strict (post-relocation)"
codesign --verify --deep --strict --verbose=2 "$RELOCATED_APP" >>"$BUILD_LOG" 2>&1 \
    || { tail -40 "$BUILD_LOG" >&2; fail "codesign verify failed on relocated bundle"; }

# -----------------------------------------------------------------------------
# 3. Runtime proof: HEALTH_OK round trip through the relocated harness.
# -----------------------------------------------------------------------------
log "running relocated harness"
HARNESS_BIN="$RELOCATED_APP/Contents/MacOS/YoozEngineXPCHarness"
[[ -x "$HARNESS_BIN" ]] || fail "harness executable missing: $HARNESS_BIN"

HARNESS_OUT="$("$HARNESS_BIN" 2>&1)" || {
    printf '%s\n' "$HARNESS_OUT" >&2
    fail "harness exited non-zero"
}
printf '%s\n' "$HARNESS_OUT"
echo "$HARNESS_OUT" | grep -q '^HEALTH_OK' \
    || fail "harness did not print HEALTH_OK -- packaging round trip failed"
log "HEALTH_OK confirmed from a Release build running outside DerivedData"

# -----------------------------------------------------------------------------
# 4. Static check: no unresolvable @rpath dependency in any embedded
#    framework. Allowed resolutions: a matching framework inside the same
#    Contents/Frameworks directory, or a declared LC_RPATH pointing at
#    /usr/lib or /System (the Swift runtime's OS-provided location).
# -----------------------------------------------------------------------------
log "static check: otool -L on every embedded framework"
FRAMEWORKS_DIR="$RELOCATED_XPC/Contents/Frameworks"
[[ -d "$FRAMEWORKS_DIR" ]] || fail "no Contents/Frameworks in relocated .xpc: $FRAMEWORKS_DIR"

STATIC_FAIL=0
shopt -s nullglob
for fw in "$FRAMEWORKS_DIR"/*.framework; do
    name="$(basename "$fw" .framework)"
    exe="$(find "$fw" -type f -perm -u+x \( -path '*/Versions/*' -o -path "$fw/$name" \) 2>/dev/null | head -1)"
    [[ -n "$exe" ]] || { log "  WARN: no executable found under $fw"; continue; }

    # LC_RPATH search paths declared by this binary, in order.
    rpaths=()
    while IFS= read -r p; do rpaths+=("$p"); done < <(
        otool -l "$exe" | awk '/LC_RPATH/{f=1} f && /^ *path /{print $2; f=0}'
    )

    deps="$(otool -L "$exe" | tail -n +2 | awk '{print $1}')"
    while IFS= read -r dep; do
        [[ -z "$dep" ]] && continue
        case "$dep" in
            /usr/lib/*|/System/*) continue ;;
            @executable_path/*|@loader_path/*) continue ;;
            @rpath/*)
                suffix="${dep#@rpath/}"
                resolved=0
                for rp in "${rpaths[@]}"; do
                    case "$rp" in
                        /usr/lib*|/System/*)
                            # OS-provided search path (e.g. /usr/lib/swift) --
                            # trust it without checking the filesystem, since
                            # the Swift runtime lives in the dyld shared
                            # cache, not on disk as a plain file.
                            resolved=1
                            ;;
                        @executable_path/../Frameworks|@loader_path/../*)
                            candidate="$FRAMEWORKS_DIR/${suffix%%/*}"
                            [[ -d "$candidate" ]] && resolved=1
                            ;;
                    esac
                    [[ $resolved -eq 1 ]] && break
                done
                if [[ $resolved -eq 0 && -d "$FRAMEWORKS_DIR/${suffix%%/*}" ]]; then
                    resolved=1  # bundled alongside even without a matching LC_RPATH entry
                fi
                if [[ $resolved -eq 0 ]]; then
                    log "  UNRESOLVED: $(basename "$exe") -> $dep"
                    STATIC_FAIL=1
                fi
                ;;
        esac
    done <<<"$deps"
done
shopt -u nullglob

[[ $STATIC_FAIL -eq 0 ]] || fail "one or more embedded frameworks have unresolvable @rpath dependencies"
log "static check OK: every embedded framework resolves within the bundle or via an allowed system path"

log "PASS -- YoozEngineXPC.xpc is self-contained in Release (engine#248)"
