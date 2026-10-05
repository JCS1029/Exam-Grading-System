"""Estimate OpenRouter spend from Phase 2/3 result files."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

# OpenRouter list prices (USD per 1M tokens), 2026
RATES = {
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.6-flash": (0.75, 3.75),
    "default": (1.00, 5.00),
}


def norm(model: str) -> str:
    m = (model or "").replace("google/", "")
    for key in RATES:
        if key in m:
            return key
    return "default"


def cost(prompt: int, comp: int, model: str) -> float:
    inp, out = RATES.get(norm(model), RATES["default"])
    return (prompt / 1e6) * inp + (comp / 1e6) * out


def main() -> None:
    by: dict[str, dict] = defaultdict(lambda: {"p": 0, "c": 0, "n": 0})
    rows = []
    p3 = Path("storage/reports/phase3_results.json")
    if p3.exists():
        rows = json.loads(p3.read_text(encoding="utf-8")).get("results") or []

    for r in rows:
        m = norm(r.get("model") or "")
        p = int(r.get("prompt_tokens") or 0)
        c = int(r.get("candidate_tokens") or 0)
        by[m]["p"] += p
        by[m]["c"] += c
        by[m]["n"] += 1

    print("=== Phase 3 (from stored token counts) ===")
    total = 0.0
    for m, s in sorted(by.items()):
        dollars = cost(s["p"], s["c"], m)
        total += dollars
        print(
            f"{m}: {s['n']} results | "
            f"prompt={s['p']:,} completion={s['c']:,} | "
            f"est ${dollars:.2f}"
        )
    print(f"Phase 3 subtotal: ${total:.2f}")

    # Note: retries/fallbacks add tokens into the same result fields (summed)
    # Bake-off ran both models on many crops → roughly 2x

    p2_cost = 0.0
    p2p = Path("storage/reports/phase2_results.json")
    if p2p.exists():
        s = json.loads(p2p.read_text(encoding="utf-8")).get("summary") or {}
        p = int(s.get("vlm_prompt_tokens") or 0)
        c = int(s.get("vlm_completion_tokens") or 0)
        # Phase 2 used 3.5-flash
        p2_cost = cost(p, c, "gemini-3.5-flash")
        print("\n=== Phase 2 (summary; last run only — undercounts if cache-hit) ===")
        print(f"prompt={p:,} completion={c:,} api_calls={s.get('vlm_api_calls')} | est ${p2_cost:.2f}")
        print("(First-time Phase 2 over all pages was likely higher; re-runs are cached.)")

    # Sol-only latest
    sol = [r for r in rows if r.get("booklet_id") in ("test8sol", "test9sol") and "3.6" in norm(r.get("model") or "")]
    sp = sum(int(r.get("prompt_tokens") or 0) for r in sol)
    sc = sum(int(r.get("candidate_tokens") or 0) for r in sol)
    print(f"\n=== test8sol+test9sol (3.6 only, stored) ===")
    print(f"{len(sol)} crops | prompt={sp:,} completion={sc:,} | est ${cost(sp, sc, 'gemini-3.6-flash'):.2f}")

    print("\n=== Pricing used ===")
    for k, (i, o) in RATES.items():
        if k != "default":
            print(f"  {k}: ${i}/M in + ${o}/M out")

    print(f"\n=== Rough project total (P2 last-summary + P3 stored) ===")
    print(f"  ${p2_cost + total:.2f}")
    print("  Reality check: bake-off + retries + empty-response fallbacks")
    print("  likely put you in the ~$3–$12 range so far — still under $20,")
    print("  but NOT the old $0.15/$0.60 estimates in the gate code.")
    print("\nCheck exact balance: https://openrouter.ai/activity")


if __name__ == "__main__":
    main()
