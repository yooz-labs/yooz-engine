#!/usr/bin/env bash
#
# resign-embedded-xpc.sh — engine#248 postbuild phase for any target that
# embeds `YoozEngineXPC.xpc` (or a renamed copy of it) via a native
# xcodegen/Xcode `embed: true, codeSign: true` target dependency.
#
# Xcode's own embed step for a NESTED bundle (a `.xpc` that itself carries a
# `Contents/Frameworks`) strips each nested framework's
# `Modules/*.swiftmodule` interface content as part of copying it into the
# outer app -- a size optimization for embedded frameworks -- WITHOUT
# re-signing the now-smaller framework. The framework's existing
# CodeDirectory still lists the stripped files, so `codesign --verify`
# reports "a sealed resource is missing or invalid" and the OS refuses to
# load it at spawn. This is downstream of, and distinct from, engine#248's
# primary fix (`scripts/embed-xpc-package-frameworks.sh`, which embeds the
# SPM package frameworks into the `.xpc` in the first place): it only
# surfaces once something ELSE embeds that already-built, already-signed
# `.xpc` into a further bundle -- this repo's own `YoozEngineXPCHarness`, or
# a consumer app following docs/CONSUMER_INTEGRATION.md's "XPC service
# embed recipe" (`embed: true, codeSign: true`).
#
# Walks deepest-first (nested frameworks, then the `.xpc` itself) and
# re-signs each, mirroring scripts/build-whisper-helper.sh's proven
# bottom-up pattern (that script hit the same class of problem for the
# `.app` variants -- see issue #38).
#
# PR #256 review (C1): the final `.xpc` re-sign MUST pass `--entitlements`.
# `codesign --force --sign X` with no `--entitlements` flag does not carry
# the bundle's existing entitlements forward into the new signature -- it
# produces a validly-signed but entitlement-LESS `.xpc`, silently dropping
# `app-sandbox`, `network.client`, and `application-groups` on every build.
# `codesign --verify` and a plain health round trip both stay green in that
# state (the signature itself is valid), which is exactly why this needs a
# capability check, not just a signature check -- see
# scripts/verify-xpc-portability.sh's entitlements assertion.
set -euo pipefail

log() { printf "[resign-embedded-xpc] %s\n" "$*"; }
fail() { printf "[resign-embedded-xpc] ERROR: %s\n" "$*" >&2; exit 1; }

: "${BUILT_PRODUCTS_DIR:?must run from an Xcode build phase}"
: "${CONTENTS_FOLDER_PATH:?must run from an Xcode build phase}"
: "${SRCROOT:?must run from an Xcode build phase}"

# The embedded .xpc's own bundle name. This repo's own harness always
# embeds it under its own product name; a consumer app that renames the
# copy (whisper does, to YoozWhisperXPC.xpc -- see
# scripts/embed-engine-xpc.sh in yooz-whisper) overrides XPC_BUNDLE_NAME.
XPC_BUNDLE_NAME="${XPC_BUNDLE_NAME:-YoozEngineXPC.xpc}"
XPC_PATH="$BUILT_PRODUCTS_DIR/$CONTENTS_FOLDER_PATH/XPCServices/$XPC_BUNDLE_NAME"

# The entitlements to re-apply to the .xpc's own signature. Defaults to
# this repo's own XPCService/YoozEngineXPC.entitlements (the same file
# CODE_SIGN_ENTITLEMENTS already points the target's own build at); a
# consumer app that renames the copy overrides XPC_ENTITLEMENTS the same
# way it overrides XPC_BUNDLE_NAME.
XPC_ENTITLEMENTS="${XPC_ENTITLEMENTS:-$SRCROOT/XPCService/YoozEngineXPC.entitlements}"

[[ -d "$XPC_PATH" ]] || fail "embedded XPC service not found at $XPC_PATH"
[[ -f "$XPC_ENTITLEMENTS" ]] || fail "entitlements file not found: $XPC_ENTITLEMENTS"

IDENTITY="${EXPANDED_CODE_SIGN_IDENTITY:--}"
log "re-signing $XPC_PATH (identity: $IDENTITY, entitlements: $XPC_ENTITLEMENTS)"

# Deepest-first: nested frameworks before the .xpc bundle itself, matching
# build-whisper-helper.sh's "sign deepest-first" convention. Frameworks
# carry no entitlements of their own (only a spawned process -- the .xpc
# itself -- does), so they're re-signed without --entitlements.
shopt -s nullglob
frameworks=("$XPC_PATH"/Contents/Frameworks/*.framework)
shopt -u nullglob
for fw in "${frameworks[@]}"; do
    log "  sign: ${fw#"$XPC_PATH"/}"
    codesign --force --sign "$IDENTITY" "$fw" \
        || fail "codesign failed for $fw"
done

log "  sign: $(basename "$XPC_PATH")"
codesign --force --sign "$IDENTITY" --entitlements "$XPC_ENTITLEMENTS" "$XPC_PATH" \
    || fail "codesign failed for $XPC_PATH"

log "DONE"
