# Gemma4 native-context parity (issues #184, #186)

**Status:** Both Gemma4 rows — **26B-A4B** (#184) and **E4B** (#186) — load and
generate at native context through the engine's `MLXInfiniteBackend`, verified
against the Python `mlx-lm==0.31.3` reference (the version Infinite validated on).
Both `swiftRuntimeSupported` gates are flipped. Only retrieval mode has no MLX
backend.

**Machine:** Apple M4 Pro, 64 GiB (full tier), macOS 26. mlx-swift-lm pinned at
`yooz-labs/mlx-swift-lm@9c73df9` (our fork; SharpAI `38d7ff2` lineage + the #186
Gemma4 E4B fix). Reference engine: `mlx-lm==0.31.3` (same as Infinite `research/18`).

## Results

| Model | repo | model_type | Swift load+generate | decode | parity assertion |
| --- | --- | --- | --- | --- | --- |
| **26B-A4B** | `mlx-community/gemma-4-26b-a4b-it-4bit` | `gemma4` | **PASS** — `2, 3, 5, 7, 11` (finish=stop, 14 tok) | 32.9 tok/s | contains correct answer |
| **E4B** | `mlx-community/gemma-4-e4b-it-qat-OptiQ-4bit` | `gemma4` | **PASS** — exact greedy match vs Python | 37.2 tok/s | exact greedy parity |

Decode rates on a short prompt are consistent with Infinite `research/18`.

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

The reference fixtures are produced by `scripts/gemma4_parity_reference.py`:

```bash
uv run --with mlx-lm==0.31.3 python scripts/gemma4_parity_reference.py --model all
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

- `gemma4_26b-a4b_parity_reference.json` — greedy reference, 26B-A4B
- `gemma4_e4b_parity_reference.json` — greedy reference, E4B

Each carries `prompt_token_ids`, `generated_token_ids`, `generated_text`,
`first_step_top5_token_ids`, and `finish_reason` under `mlx-lm==0.31.3`.
