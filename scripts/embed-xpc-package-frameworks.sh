#!/usr/bin/env bash
#
# embed-xpc-package-frameworks.sh — engine#248 postbuild script phase for
# the `YoozEngineXPC` xcodegen target (see that target's project.yml
# comment; invoked via `postbuildScripts`, basedOnDependencyAnalysis: false
# so it runs on every build, not just when Xcode's dependency analysis
# thinks something changed).
#
# Root cause (engine#248): `.xpc` bundles do not get Xcode's automatic
# "Embed Frameworks" treatment for SPM dynamic package products the way
# `.app` targets do (the same underlying limitation whisper#38 already hit
# for the `.app` variants — Xcode's implicit embedding only walks NATIVE
# target dependencies, and `LLMModule`/`STTModule`/etc. link `mlx-swift`,
# `mlx-swift-lm`, `swift-transformers`, `swift-huggingface` as `package:`
# dependencies of THEMSELVES, one hop away from `YoozEngineXPC`). Those
# package products land in `$BUILT_PRODUCTS_DIR/PackageFrameworks` and stay
# there — never copied into the `.xpc`'s own `Contents/Frameworks`. Debug
# builds are masked by an absolute, machine-local DerivedData `LC_RPATH`
# Xcode also writes (see the `YoozEngineClient` target's project.yml
# comment); Release strips that masking rpath and dyld fails at spawn with
# "Library not loaded: @rpath/MLX.framework/...".
#
# This script copies every `PackageFrameworks/*.framework` into the
# `.xpc`'s `Contents/Frameworks` and code-signs each copy individually.
# Signing here (not left to Xcode) matters because the service binary runs
# with hardened runtime + no `disable-library-validation` entitlement in
# Release: library validation rejects any loaded framework that isn't
# signed consistently with the loading binary, and content a plain script
# `cp -R`s into a bundle carries no signature at all until something signs
# it. Runs BEFORE Xcode's own implicit code-signing step for this same
# target (`postbuildScripts` always precede the implicit "sign product"
# step Xcode inserts after every explicit build phase), so Xcode's own
# sign step, which runs next, seals the outer `.xpc` (Info.plist,
# executable, entitlements) over a bundle that already contains
# properly-signed nested frameworks — the same pattern Xcode's native
# "Embed Frameworks" Copy Files phase uses when `Code Sign on Copy` is
# checked, just done by hand because xcodegen has no embed support for
# `package:` dependencies at all (established in PR #240).
set -euo pipefail

log() { printf "[embed-xpc-package-frameworks] %s\n" "$*"; }
fail() { printf "[embed-xpc-package-frameworks] ERROR: %s\n" "$*" >&2; exit 1; }

: "${BUILT_PRODUCTS_DIR:?must run from an Xcode build phase}"
: "${CONTENTS_FOLDER_PATH:?must run from an Xcode build phase}"

PACKAGE_FRAMEWORKS_DIR="$BUILT_PRODUCTS_DIR/PackageFrameworks"
XPC_FRAMEWORKS_DIR="$BUILT_PRODUCTS_DIR/$CONTENTS_FOLDER_PATH/Frameworks"

if [[ ! -d "$PACKAGE_FRAMEWORKS_DIR" ]]; then
    fail "PackageFrameworks not found at $PACKAGE_FRAMEWORKS_DIR -- did LLMModule/STTModule build before this script phase ran?"
fi

shopt -s nullglob
frameworks=("$PACKAGE_FRAMEWORKS_DIR"/*.framework)
shopt -u nullglob
[[ ${#frameworks[@]} -gt 0 ]] || fail "no *.framework found under $PACKAGE_FRAMEWORKS_DIR"

mkdir -p "$XPC_FRAMEWORKS_DIR"

# Falls back to ad-hoc ("-") for this repo's own dev-only build (project.yml
# pins CODE_SIGN_IDENTITY: "-" on YoozEngineXPC — see that target's comment
# on why); a consumer app building this same script phase with a real
# DEVELOPMENT_TEAM gets EXPANDED_CODE_SIGN_IDENTITY resolved to that real
# identity instead, so the nested frameworks stay consistent with whatever
# identity actually signs the outer .xpc.
IDENTITY="${EXPANDED_CODE_SIGN_IDENTITY:--}"
log "embedding ${#frameworks[@]} package frameworks into $XPC_FRAMEWORKS_DIR (identity: $IDENTITY)"

for fw in "${frameworks[@]}"; do
    name="$(basename "$fw")"
    dest="$XPC_FRAMEWORKS_DIR/$name"
    rm -rf "$dest"
    cp -R "$fw" "$dest"
    codesign --force --sign "$IDENTITY" "$dest" \
        || fail "codesign failed for $dest"
    log "  embedded + signed: $name"
done

log "DONE"
