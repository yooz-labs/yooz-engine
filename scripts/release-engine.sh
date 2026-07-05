#!/usr/bin/env bash
#
# release-engine.sh — A6 (#30) orchestrator for the local-only release
# pipeline. Runs the three build scripts (full, lite, whisper helper), zips
# each signed .app with `ditto` (preserves codesign on nested bundles),
# computes per-artifact SHA256s, and emits a release manifest at
# `dist/RELEASE.md` that operators paste into the GitHub release body.
#
# No macOS runners in GitHub Actions (see engine#23) so this is the local
# source of truth for release artifacts. `gh release upload` wires them up.
#
# Idempotent: each child script wipes its own bundle before rebuilding.
# This orchestrator only touches paths under $ROOT/dist and $ROOT/.build.

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

DIST_DIR="$ROOT/dist"
BUILD_DIR="$ROOT/.build"
MANIFEST="$DIST_DIR/RELEASE.md"
LOG="$BUILD_DIR/release-engine.log"

log() { printf "[release-engine] %s\n" "$*"; }
fail() { printf "[release-engine] ERROR: %s\n" "$*" >&2; exit 1; }

mkdir -p "$DIST_DIR" "$BUILD_DIR"
: > "$LOG"

# -----------------------------------------------------------------------------
# 0. Read engine version from EngineConfig (single source of truth)
# -----------------------------------------------------------------------------
CONFIG_FILE="$ROOT/Sources/EngineCore/EngineConfig.swift"
[[ -f "$CONFIG_FILE" ]] || fail "EngineConfig.swift missing: $CONFIG_FILE"
VERSION="$(grep -E 'static let version' "$CONFIG_FILE" \
    | head -1 \
    | sed -E 's/.*"([^"]+)".*/\1/')"
[[ -n "$VERSION" ]] || fail "could not parse version from $CONFIG_FILE"
log "engine version: $VERSION"

TIMESTAMP="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
CONFIG="${YOOZ_ENGINE_CONFIG:-Debug}"

# Force the whisper-helper build to the same config as the engine/lite
# builds. The A5 script reads its own `YOOZ_HELPER_CONFIG` env var;
# exporting `$CONFIG` here unconditionally prevents the manifest's
# `- **Config:** <value>` row from drifting from the config each sub-build
# actually used (I1 in the A6 review).
export YOOZ_HELPER_CONFIG="$CONFIG"

# -----------------------------------------------------------------------------
# 1. Run the three build scripts. Each is independent; fail fast on error.
# -----------------------------------------------------------------------------
run_build() {
    local name="$1" script="$2"
    log "==> building: $name ($script)"
    if ! bash "$ROOT/scripts/$script" 2>&1 | tee -a "$LOG"; then
        fail "$name build failed; see $LOG"
    fi
}

run_build "whisper-helper" "build-whisper-helper.sh"
run_build "engine-release" "build-engine-release.sh"
run_build "engine-lite" "build-engine-lite.sh"

# -----------------------------------------------------------------------------
# 2. Declare the three artifacts. Keep bundle-name + binary-name aligned
#    with project.yml PRODUCT_NAME.
# -----------------------------------------------------------------------------
# Format: dist-name|binary-name-under-Contents/MacOS
ARTIFACTS=(
    "YoozEngine.app|Yooz Engine"
    "YoozEngineLite.app|Yooz Engine (Lite)"
    "YoozEngineWhisper.app|Yooz Engine (Whisper)"
)

for spec in "${ARTIFACTS[@]}"; do
    name="${spec%%|*}"
    [[ -d "$DIST_DIR/$name" ]] || fail "expected artifact missing: $DIST_DIR/$name"
done

# -----------------------------------------------------------------------------
# 3. Zip each signed .app with `ditto`. `zip` corrupts codesign on nested
#    bundles; ditto is the Apple-sanctioned archiver.
# -----------------------------------------------------------------------------
log "zipping signed bundles"
for spec in "${ARTIFACTS[@]}"; do
    name="${spec%%|*}"
    zip_path="$DIST_DIR/${name}.zip"
    rm -f "$zip_path"
    ditto -c -k --keepParent "$DIST_DIR/$name" "$zip_path" \
        || fail "ditto zip failed for $name"
    log "  zipped: ${name}.zip"
done

# -----------------------------------------------------------------------------
# 4. Emit the release manifest
# -----------------------------------------------------------------------------
log "generating manifest: $MANIFEST"

# Track any variant that ended up unsigned (codesign failure upstream would
# already have aborted, but an unsigned bundle reaching this point means
# something slipped through). Populated inside the manifest block below.
UNSIGNED_VARIANTS=()

{
    printf "# Yooz Engine Release %s\n\n" "$VERSION"
    printf -- "- **Built:** %s\n" "$TIMESTAMP"
    printf -- "- **Config:** %s\n" "$CONFIG"
    printf -- "- **Host:** %s (%s)\n" "$(hostname -s)" "$(uname -sm)"
    printf -- "- **Git:** \`%s\` @ \`%s\`\n" \
        "$(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)" \
        "$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    printf "\n"
    printf "## Artifacts\n\n"
    printf "| Variant | File | Size | SHA256 | Signing identity |\n"
    printf "|---|---|---:|---|---|\n"

    for spec in "${ARTIFACTS[@]}"; do
        name="${spec%%|*}"
        bin_name="${spec##*|}"
        app_path="$DIST_DIR/$name"
        zip_path="$DIST_DIR/${name}.zip"
        bin_path="$app_path/Contents/MacOS/$bin_name"

        app_size="$(du -sh "$app_path" 2>/dev/null | awk '{print $1}')"
        zip_size="$(du -sh "$zip_path" 2>/dev/null | awk '{print $1}')"
        bin_sha="$(shasum -a 256 "$bin_path" | awk '{print $1}')"
        zip_sha="$(shasum -a 256 "$zip_path" | awk '{print $1}')"
        codesign_out="$(codesign --display --verbose=2 "$app_path" 2>&1 || true)"
        identity="$(printf '%s\n' "$codesign_out" \
            | grep -E '^Authority=' | head -1 \
            | sed -E 's/^Authority=//' || true)"
        sig_line="$(printf '%s\n' "$codesign_out" \
            | grep -E '^Signature=' | head -1 || true)"
        if [[ -n "$identity" ]]; then
            :  # Developer ID / Apple Development / etc. — use identity as-is.
        elif [[ "$sig_line" == "Signature=adhoc" ]]; then
            identity="ad-hoc"
        else
            identity="unsigned (WARN)"
            UNSIGNED_VARIANTS+=("$name")
        fi

        printf "| \`%s\` | \`%s\` | %s | \`%s\` | %s |\n" \
            "$name" "$name" "$app_size" "$bin_sha" "$identity"
        printf "| \`%s\` (zip) | \`%s.zip\` | %s | \`%s\` | - |\n" \
            "$name" "$name" "$zip_size" "$zip_sha"
    done

    printf "\n"
    printf "## Upload\n\n"
    printf '```bash\n'
    # In bash double-quotes `\\` -> `\`; printf then sees `\` + `n` and
    # emits a newline. To emit a literal `\` (shell line-continuation)
    # followed by a real newline, we need four backslashes in the source:
    # bash strips to two (`\\`), printf emits one backslash + newline.
    printf "gh release upload v%s \\\\\n" "$VERSION"
    printf "  dist/YoozEngine.app.zip \\\\\n"
    printf "  dist/YoozEngineLite.app.zip \\\\\n"
    printf "  dist/YoozEngineWhisper.app.zip \\\\\n"
    printf "  dist/RELEASE.md\n"
    printf '```\n'
    printf "\n"
    printf "## Verify a downloaded artifact\n\n"
    printf '```bash\n'
    printf "unzip -q YoozEngine.app.zip\n"
    printf "codesign --verify --deep --strict --verbose=2 YoozEngine.app\n"
    printf "shasum -a 256 \"YoozEngine.app/Contents/MacOS/Yooz Engine\"\n"
    printf '```\n'
} > "$MANIFEST"

# -----------------------------------------------------------------------------
# 4b. Hard-fail if any variant reached the manifest unsigned. The manifest
#     is still written so operators can see which variant slipped; a
#     non-zero exit prevents the release from being published (I2 in the
#     A6 review).
# -----------------------------------------------------------------------------
if (( ${#UNSIGNED_VARIANTS[@]} > 0 )); then
    printf "\n"
    printf "[release-engine] WARN: unsigned bundle(s) detected in manifest:\n" >&2
    for v in "${UNSIGNED_VARIANTS[@]}"; do
        printf "[release-engine] WARN:   - %s\n" "$v" >&2
    done
    printf "[release-engine] WARN: refusing to exit clean; inspect build logs under %s\n" "$BUILD_DIR" >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# 5. Summary
# -----------------------------------------------------------------------------
cat <<EOF

[release-engine] DONE
  version:     $VERSION
  config:      $CONFIG
  manifest:    $MANIFEST
  artifacts:
EOF
for spec in "${ARTIFACTS[@]}"; do
    name="${spec%%|*}"
    printf "    - %s\n" "$DIST_DIR/$name"
    printf "    - %s\n" "$DIST_DIR/${name}.zip"
done

cat <<'EOF'

Next steps:
  - Smoke-test: bash scripts/smoke-test-release.sh
  - Tag + release (see docs/RELEASE.md):
      git tag vX.Y.Z
      git push origin vX.Y.Z
      gh release create vX.Y.Z --draft --notes-file dist/RELEASE.md
      gh release upload vX.Y.Z dist/*.app.zip dist/RELEASE.md
EOF
