"""Pick top-2 quality + speed-feasible models per tier from Phase 1+2 results.

Reads:
  /Volumes/S1/yooz/research/issue-9/results/phase1_quality/all_summaries.json
  /Volumes/S1/yooz/research/issue-9/results/phase2_speed/all_summaries.json

Writes:
  /Volumes/S1/yooz/research/issue-9/results/winners.json
  /Volumes/S1/yooz/research/issue-9/results/winners.md

Tier targets (from issue #9):
  Light   : <600 MB on disk, <150 ms median latency
  Quality : <2 GB on disk,  <400 ms median latency
"""

from __future__ import annotations

import json
from pathlib import Path

QUALITY_FILE = Path("/Volumes/S1/yooz/research/issue-9/results/phase1_quality/all_summaries.json")
SPEED_FILE = Path("/Volumes/S1/yooz/research/issue-9/results/phase2_speed/all_summaries.json")
OUT_JSON = Path("/Volumes/S1/yooz/research/issue-9/results/winners.json")
OUT_MD = Path("/Volumes/S1/yooz/research/issue-9/results/winners.md")

# Approx on-disk sizes from HF (model.safetensors) in MB.
DISK_MB = {
    "qwen2.5-0.5B-baseline": 278,
    "qwen3-1.7B-baseline": 1037,
    "qwen3.5-0.8B": 625,
    "qwen3.5-2B-optiq": 1431,
    "gemma-4-e2b-text": 2671,
}

LIGHT_MAX_MB = 600
LIGHT_MAX_LAT_MS = 150
QUALITY_MAX_MB = 2048
QUALITY_MAX_LAT_MS = 400


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def composite_score(qual: dict, speed: dict | None) -> float:
    """Higher is better. 60% similarity + 40% (1 - CER), penalised by latency."""
    if not qual or "error" in qual:
        return 0.0
    t = qual.get("total") or {}
    sim = t.get("avg_semantic_similarity", 0.0)
    cer = t.get("avg_cer", 1.0)
    json_ok = t.get("json_ok", 0.0)
    base = 0.6 * sim + 0.4 * (1.0 - min(cer, 1.0))
    base *= json_ok  # heavy penalty if model can't produce JSON
    return base


def main():
    import sys

    quality = {s["slug"]: s for s in load(QUALITY_FILE) if "slug" in s}
    speed = {s["slug"]: s for s in load(SPEED_FILE) if "slug" in s}

    rows = []
    for slug, q in quality.items():
        s = speed.get(slug)
        if slug not in DISK_MB:
            print(
                f"warning: pick_winners has no DISK_MB entry for '{slug}'; "
                f"it will be excluded from tier eligibility. Add it to "
                f"scripts/pick_winners.py.",
                file=sys.stderr,
            )
        disk_mb = DISK_MB.get(slug, 0)
        if s and "error" not in s:
            lat_ms = (s.get("ttft_ms") or {}).get("median")
            tps = (s.get("tokens_per_second") or {}).get("median", 0.0)
            rss_mb = s.get("rss_peak_bytes", 0) / (1024 * 1024)
        else:
            print(
                f"warning: pick_winners has no Phase 2 speed data for "
                f"'{slug}'; it will be excluded from tier eligibility.",
                file=sys.stderr,
            )
            lat_ms = None
            tps = 0.0
            rss_mb = 0.0
        score = composite_score(q, s)
        rows.append({
            "slug": slug,
            "role": q.get("role"),
            "disk_mb": disk_mb,
            "score": score,
            # ttft_ms is None when Phase 2 data is missing — keeps the JSON
            # spec-compliant (no Infinity) and lets downstream filter on
            # `is None` instead of float comparisons.
            "ttft_ms": lat_ms,
            "tokens_per_second": tps,
            "rss_mb": rss_mb,
            "quality": q.get("total"),
        })

    def _eligible(rows, max_mb, max_lat):
        out = []
        for r in rows:
            if not r["disk_mb"] or r["disk_mb"] >= max_mb:
                continue
            if r["ttft_ms"] is None or r["ttft_ms"] >= max_lat:
                continue
            out.append(r)
        return out

    light_eligible = _eligible(rows, LIGHT_MAX_MB, LIGHT_MAX_LAT_MS)
    quality_eligible = _eligible(rows, QUALITY_MAX_MB, QUALITY_MAX_LAT_MS)
    light_winner = max(light_eligible, key=lambda r: r["score"], default=None)
    quality_winner = max(quality_eligible, key=lambda r: r["score"], default=None)

    payload = {
        "rows": sorted(rows, key=lambda r: -r["score"]),
        "light_tier": {
            "max_disk_mb": LIGHT_MAX_MB,
            "max_latency_ms": LIGHT_MAX_LAT_MS,
            "eligible": [r["slug"] for r in light_eligible],
            "winner": (light_winner or {}).get("slug"),
        },
        "quality_tier": {
            "max_disk_mb": QUALITY_MAX_MB,
            "max_latency_ms": QUALITY_MAX_LAT_MS,
            "eligible": [r["slug"] for r in quality_eligible],
            "winner": (quality_winner or {}).get("slug"),
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    # Markdown summary.
    lines = ["# Winners — issue #9\n"]
    lines.append(
        f"Light tier: target <{LIGHT_MAX_MB} MB disk, <{LIGHT_MAX_LAT_MS} ms TTFT.\n"
        f"Winner: **{payload['light_tier']['winner'] or 'none eligible'}**\n"
    )
    lines.append(
        f"Quality tier: target <{QUALITY_MAX_MB} MB disk, <{QUALITY_MAX_LAT_MS} ms TTFT.\n"
        f"Winner: **{payload['quality_tier']['winner'] or 'none eligible'}**\n"
    )
    lines.append("\n## Ranking (composite score)\n")
    lines.append("| Rank | Model | Score | Disk MB | TTFT ms | tok/s | EM | CER | Sim |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(payload["rows"], 1):
        q = r.get("quality") or {}
        em = q.get("exact_match", 0) * 100 if q else float("nan")
        cer = q.get("avg_cer", float("nan")) if q else float("nan")
        sim = q.get("avg_semantic_similarity", float("nan")) if q else float("nan")
        ttft = r["ttft_ms"]
        tps = r["tokens_per_second"]
        ttft_s = "-" if ttft is None else f"{ttft:.0f}"
        lines.append(
            f"| {i} | {r['slug']} | {r['score']:.4f} | {r['disk_mb']} | "
            f"{ttft_s} | {tps:.1f} | {em:.1f}% | {cer:.4f} | {sim:.4f} |"
        )
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    print(OUT_MD.read_text())


if __name__ == "__main__":
    main()
