# Gemma4 native-context parity (issues #184, #186, #187)

**Status:** Three Gemma4-family rows — **26B-A4B** (#184), **E4B** (#186), and the
dense **12B** (`gemma4_unified`, #187) — load and generate at native context
through the engine's `MLXInfiniteBackend`. The `gemma4` rows are verified against
`mlx-lm==0.31.3` (the version Infinite validated on); the 12B `gemma4_unified_text`
row has no mlx-lm implementation, so it is verified against the `mlx-vlm` text path.
All three `swiftRuntimeSupported` gates are flipped. Only retrieval mode has no MLX
backend.

**Machine:** Apple M4 Pro, 64 GiB (full tier), macOS 26. Runtime pinned to
Apple's ml-explore upstream: `mlx-swift@e23ae6b` (main) + `mlx-swift-lm@f4fd39e`
(`yooz-labs/mlx-swift-lm` main, a clean ml-explore fork = ml-explore main + the
gemma4_unified `vision_embedder` sanitize fix + the MLXLLM Gemma4 MoE/KV-sharing
port; both have upstream PRs in flight). The SharpAI lineage is retired.
Reference engines: `mlx-lm==0.31.3` (`gemma4` rows) and `mlx-vlm==0.6.3`
(`gemma4_unified`).

## Results

| Model | repo | model_type | Swift load+generate | parity assertion |
| --- | --- | --- | --- | --- |
| **26B-A4B** | `mlx-community/gemma-4-26b-a4b-it-4bit` | `gemma4` | **PASS** — `2, 3, 5, 7, 11` (finish=stop) | contains correct answer |
| **E4B** | `mlx-community/gemma-4-e4b-it-qat-OptiQ-4bit` | `gemma4` | **PASS** — exact greedy match vs Python | exact greedy parity |
| **12B** | `mlx-community/gemma-4-12B-it-4bit` | `gemma4_unified` | **PASS** — exact greedy match vs mlx-vlm | exact greedy parity |

Decode rates on a short prompt are consistent with Infinite `research/18`.

## 12B `gemma4_unified` onboarding (#187)

`gemma4_unified_text` is *not* a new architecture: the multimodal model reuses
gemma4's `LanguageModel`, and the 12B text config is a dense gemma4 —
`num_kv_shared_layers: 0`, `hidden_size_per_layer_input: 0`, `attention_k_eq_v:
true`, `num_global_key_value_heads: 1`, dense MLP, proportional RoPE
(`partial_rotary_factor: 0.25`), `layer_types` explicit. So the fork registers
`gemma4_unified` / `gemma4_unified_text` against the existing
`Gemma4Configuration`/`Gemma4Model` + `Gemma4TextModel` rather than duplicating a
port; the multimodal checkpoint's `model.language_model.` prefixes and
vision/audio tensors are handled by the existing `Gemma4Model.sanitize`.

The reference engine for this row is `mlx-vlm`, not `mlx-lm`. mlx-vlm reads the
same HF `tokenizer_config` chat template the engine's swift-transformers stack
does (including the `<|channel>thought` reasoning preamble), so the template
matches token-for-token — which is why 12B can use the strict **exact-greedy**
assertion where 26B (referenced via mlx-lm) could only assert "contains answer".

### K-eq-V value-path fix

Onboarding 12B surfaced a latent numerics bug in the fork's `Gemma4Text.swift`
K-eq-V attention path (used by every full-attention layer when `attention_k_eq_v`
is set — 12B *and* 26B). The Python reference takes `values = keys` from the raw
`k_proj` output **before** `k_norm` and **before** RoPE, applying only `v_norm`
(`mlx_vlm/models/gemma4/language.py`). The Swift port instead derived values from
the already-normed + RoPE'd keys (`v = vNorm(k)`), incorrectly applying both
`k_norm` and a rotation to the values. The fix captures the raw `k_proj` output
and applies only `v_norm` + transpose, matching the reference. 26B's prior
"contains answer" assertion was too weak to catch this; the 12B exact-greedy test
pins it.

## Why the two models assert differently

- **26B-A4B is a reasoning model.** Under greedy decoding, Python `mlx-lm` opens a
  `<|channel>thought` preamble while the engine's chat-template path emits the
  direct answer. That is a **chat-template difference, not a model-numerics one**
  — both produce the correct `2, 3, 5, 7, 11`. So the 26B test asserts a correct
  on-task answer (`testGemma4_26B_A4B_LoadsAndGeneratesNativeContext`), the honest
  proof that the fork loads the weights and computes correctly.
- **E4B is non-reasoning**, so exact greedy token parity is the right (stricter)
  assertion (`testGemma4E4BGreedyParityVsPython`, `.exactGreedy`): the Swift
  output matches the Python reference token-for-token.

## E4B fix (#186)

The OptiQ-4bit E4B originally failed to load. Two layered bugs in the fork's
`Gemma4Text.swift`, both fixed:

1. **`per_layer_model_projection` was not quantizable.** It used a custom
   `ScaledLinear` (a bare `Module` holding a raw `MLXArray`), which the quantize
   pass skips — but the OptiQ checkpoint ships that weight quantized, so the shape
   mismatched. Fixed by using a standard (quantizable) `Linear` and applying the
   scalar at the call site, matching the Python reference.
2. **KV-shared layers demanded weights the checkpoint omits.** Attention created
   `k_proj`/`k_norm`/`v_norm` for every layer, but E4B's KV-shared layers (their
   KV comes from an earlier layer) carry none — `mlx_lm.convert` drops them. Fixed
   by making those projections + norms `has_kv`-conditional, matching Python.

26B-A4B (`num_kv_shared_layers: 0`, no per-layer-input block) was unaffected by
both. The same fix was arrived at independently upstream in
[ml-explore PR #330](https://github.com/ml-explore/mlx-swift-lm/pull/330); when
that merges and our fork rebases, the custom patch is dropped.

## How to reproduce

The reference fixtures are produced by `scripts/gemma4_parity_reference.py`. The
`gemma4` rows use mlx-lm; the 12B `gemma4_unified` row uses mlx-vlm:

```bash
# gemma4 rows (e4b, 26b-a4b)
uv run --with mlx-lm==0.31.3 python scripts/gemma4_parity_reference.py --model e4b
uv run --with mlx-lm==0.31.3 python scripts/gemma4_parity_reference.py --model 26b-a4b
# gemma4_unified row (12b) — mlx-vlm reference
uv run --with mlx-vlm==0.6.3 python scripts/gemma4_parity_reference.py --model 12b
```

The Swift side runs through a dedicated scheme + test plan (`InfiniteLive`) that
sets `INFINITE_LIVE=1` for the hosted xctest process — `xcodebuild` does not
propagate CLI env to the test runner (see `scripts/run-integration.sh`):

```bash
xcodegen generate
xcodebuild -project YoozEngine.xcodeproj -scheme InfiniteLive \
  -skipMacroValidation -derivedDataPath build -enableCodeCoverage NO \
  -destination 'platform=macOS' test
```

The tests `XCTSkip` unless `INFINITE_LIVE=1` and the weights are in the HF cache,
so CI (neither) skips cleanly instead of downloading multi-GB weights.

## Reference fixtures

- `gemma4_26b-a4b_parity_reference.json` — greedy reference, 26B-A4B (`mlx-lm==0.31.3`)
- `gemma4_e4b_parity_reference.json` — greedy reference, E4B (`mlx-lm==0.31.3`)
- `gemma4_12b_parity_reference.json` — greedy reference, 12B (`mlx-vlm==0.6.3`)

The mlx-lm fixtures carry `prompt_token_ids`, `generated_token_ids`,
`generated_text`, `first_step_top5_token_ids`, and `finish_reason`. The mlx-vlm
fixture carries `prompt_formatted`, `generated_token_ids`, `generated_text`, and
`finish_reason` (mlx-vlm's streaming API exposes per-step tokens but not the
first-step top-5 logits the mlx-lm path captures).
