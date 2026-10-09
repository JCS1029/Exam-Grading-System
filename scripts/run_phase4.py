"""
Phase 4 runner — grading engine.

Workflow (each step is explicit; grading never runs on an unapproved rubric):

    python scripts/run_phase4.py propose  [--sol docs/Dataset/sol.pdf]
        master key from sol.pdf (red text) + suit per booklet + per-version keys
        derived from each booklet's printed options → stored as a *proposed* rubric
    python scripts/run_phase4.py approve  --version N --by "<name>"
    python scripts/run_phase4.py grade    [--booklets 02,05] [--stale-only]
    python scripts/run_phase4.py evaluate [--grades-csv docs/Dataset/grades.csv]
        gate metrics vs the recorded grades (+ hand-verified sheet readings if present)
    python scripts/run_phase4.py curated  [--heldout]
        alternative-method / consequential-error curated sets (LLM, cached)

Paid calls are cached by content hash and logged to storage/cache/spend_ledger.jsonl;
VLM_RUN_BUDGET_USD caps a single run.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.grading.answer_key import extract_mcq_key
from core.grading.curated import run_curated
from core.grading.grader import grade_mcq_booklet
from core.grading.metrics import (
    GATE_ALT_FULL_MIN,
    GATE_CONSEQ_ONCE_MIN,
    GATE_MAE_MAX,
    GATE_PEARSON_MIN,
    GATE_SIGNED_ABS_MAX,
    GATE_WITHIN1_MIN,
    question_gate,
    total_gate,
)
from core.grading.omr import read_answer_grid
from core.grading.open_grader import grade_open_question
from core.grading.rubric import mcq_rubric
from core.grading.store import GradingStore
from core.grading.versions import build_version_keys, detect_suit, map_to_master, read_printed_options
from core.vlm import run_spend_usd

DATASET_DIR = Path("docs/Dataset")
DEFAULT_SOL = DATASET_DIR / "sol.pdf"
DEFAULT_GRADES = DATASET_DIR / "grades.csv"
ANON_DIR = Path("storage/anonymized")
HAND_LABELS = Path("storage/eval/mc_hand_labels.json")
REPORT_JSON = Path("storage/reports/phase4_results.json")
REPORT_MD = Path("storage/reports/phase4_report.md")
KEYS_JSON = Path("storage/reports/phase4_version_keys.json")
DEFAULT_EXAM = "203.2480"


def _rubric_id(exam_id: str) -> str:
    return f"mcq:{exam_id}"


def _booklets(selection: str | None) -> list[Path]:
    dirs = sorted(d for d in ANON_DIR.iterdir() if (d / "page_001.png").exists())
    if selection:
        wanted = {b.strip() for b in selection.split(",")}
        dirs = [d for d in dirs if d.name in wanted]
    return dirs


def cmd_propose(args) -> int:
    master = extract_mcq_key(args.sol)
    master.source = Path(args.sol).name
    if not master.ok:
        print("Master key extraction failed:", master.errors)
        return 1
    print(f"Master key ({args.sol}): {master.letters()}")
    per_booklet, details = {}, {}
    for d in _booklets(args.booklets):
        cover = d / "page_001.png"
        grid = read_answer_grid(cover)
        suit = detect_suit(cover, grid.grid_bbox) if grid.grid_bbox else None
        pages = sorted(p for p in d.glob("page_*.png") if p.name != cover.name)
        printed = read_printed_options(d.name, pages)
        letters, issues = map_to_master(printed.questions, master)
        per_booklet[d.name] = (suit.suit if suit and not suit.needs_review else None, letters)
        details[d.name] = {"suit": suit.to_dict() if suit else None, "letters": letters,
                           "issues": issues, "errors": printed.errors}
        print(f"  {d.name}: suit={per_booklet[d.name][0]} mapped={len(letters)}/10"
              + (f" issues={list(issues)}" if issues else ""))
    keys, dissent = build_version_keys(per_booklet, [q.question_id for q in master.questions])
    for suit, vk in keys.items():
        print(f"  {suit:8s} {[vk.letters.get(q.question_id) for q in master.questions]} "
              f"support={min(vk.support.values(), default=0)}-{max(vk.support.values(), default=0)} "
              f"booklets={vk.booklets}" + (f" UNRESOLVED={vk.unresolved}" if vk.unresolved else ""))
    if dissent:
        print("  DISSENT (booklet pages disagree with consensus):", dissent)

    rubric = mcq_rubric(_rubric_id(args.exam_id), args.exam_id, master,
                        {s: vk.letters for s, vk in keys.items()},
                        source=f"{Path(args.sol).name} red text + printed options of {len(per_booklet)} booklets")
    store = GradingStore()
    version = store.propose_rubric(rubric)
    KEYS_JSON.parent.mkdir(parents=True, exist_ok=True)
    KEYS_JSON.write_text(json.dumps({
        "rubric_id": rubric.rubric_id, "version": version, "master": master.to_dict(),
        "version_keys": {s: vk.to_dict() for s, vk in keys.items()},
        "dissent": dissent, "booklets": details,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nProposed {rubric.rubric_id} v{version} (hash {rubric.content_hash()}); details in {KEYS_JSON}")
    print(f"Review it, then: python scripts/run_phase4.py approve --version {version} --by \"<name>\"")
    print(f"Spend this run: ${run_spend_usd():.4f}")
    return 0


def cmd_approve(args) -> int:
    store = GradingStore()
    store.approve_rubric(_rubric_id(args.exam_id), args.version, args.by)
    print(f"Approved {_rubric_id(args.exam_id)} v{args.version} by {args.by}")
    stale = store.stale_grades(_rubric_id(args.exam_id))
    if stale:
        print(f"{len(stale)} existing grades are now stale; run: python scripts/run_phase4.py grade --stale-only")
    return 0


def cmd_grade(args) -> int:
    store = GradingStore()
    rid = _rubric_id(args.exam_id)
    approved = store.approved_rubric(rid)
    if approved is None:
        print(f"No approved rubric for {rid}. Run `propose`, review, then `approve`.")
        return 1
    rubric, version = approved
    master = extract_mcq_key(args.sol) if args.sol and Path(args.sol).exists() else None
    booklets = _booklets(args.booklets)
    if args.stale_only:
        stale = {g.booklet_id for g in store.stale_grades(rid)}
        booklets = [d for d in booklets if d.name in stale]
        print(f"Re-grading {len(booklets)} booklets with stale grades")
    for d in booklets:
        res = grade_mcq_booklet(d, rubric, version, master=master, use_vlm=not args.no_vlm)
        for g in res.grades:
            store.record_grade(g)
        flags = [f"Q{g.question_id}" for g in res.grades if g.needs_review]
        print(f"  {d.name}: suit={res.suit} total={res.total} review={flags or '-'}"
              + (f" errors={res.errors}" if res.errors else ""))
    print(f"Spend this run: ${run_spend_usd():.4f}")
    return 0


def _read_grades_csv(path: Path) -> dict[str, float]:
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].strip().isdigit() and row[1].strip():
                out[f"{int(row[0]):02d}"] = float(row[1])
    return out


def cmd_evaluate(args) -> int:
    store = GradingStore()
    rid = _rubric_id(args.exam_id)
    approved = store.approved_rubric(rid)
    if approved is None:
        print("No approved rubric.")
        return 1
    rubric, version = approved
    grades = store.current_grades(rid)
    system: dict[str, dict[str, float | None]] = {}
    review: dict[str, list[str]] = {}
    for g in grades:
        system.setdefault(g.booklet_id, {})[g.question_id] = g.score
        if g.needs_review:
            review.setdefault(g.booklet_id, []).append(g.question_id)
    totals = {b: (None if any(v is None for v in qs.values()) else sum(qs.values()))
              for b, qs in system.items()}
    max_points = {q.question_id: q.max_points for q in rubric.questions}

    recorded = {b: v for b, v in _read_grades_csv(Path(args.grades_csv)).items() if b in totals}
    vs_recorded = total_gate(totals, recorded)

    hand_q = None
    hand_totals = {}
    if HAND_LABELS.exists():
        labels = json.loads(HAND_LABELS.read_text(encoding="utf-8"))["booklets"]
        hand_q = {}
        for b, lab in labels.items():
            q_rub = {q.question_id: q for q in rubric.questions}
            hand_q[b] = {
                str(i + 1): (q_rub[str(i + 1)].max_points
                             if mark == q_rub[str(i + 1)].mcq_keys.get(lab["version"]) else 0.0)
                for i, mark in enumerate(lab["marks"])
            }
            hand_totals[b] = sum(hand_q[b].values())
    q_vs_hand = question_gate(system, hand_q, max_points) if hand_q else None
    t_vs_hand = total_gate(totals, hand_totals) if hand_totals else None

    # Per-question vs recorded: only the total is recorded, so |Δtotal| bounds the
    # per-question disagreement: a 10-point total gap on 10-point questions = one question.
    n_q = len(rubric.questions)
    q_from_totals_mae = vs_recorded.mae_total / n_q if vs_recorded.n else float("nan")

    alt = conseq = None
    curated = {}
    for name, f in (("development", "curated_cases.json"), ("heldout", "curated_cases_heldout.json")):
        rep = run_curated(lambda q, t: grade_open_question(q, t, use_cache=True), cases_file=f)
        curated[name] = {
            "alternative": rep.rate("alternative"), "consequential": rep.rate("consequential"),
            "control": rep.rate("control"),
            "failures": [o.__dict__ for o in rep.outcomes if not o.passed],
        }
    alt, conseq = curated["heldout"]["alternative"], curated["heldout"]["consequential"]

    results = {
        "rubric_id": rid, "rubric_version": version,
        "booklets": len(totals),
        "totals": totals, "recorded": recorded, "hand_totals": hand_totals,
        "review_flags": review,
        "vs_recorded_totals": vs_recorded.__dict__,
        "per_question_from_totals_mae": q_from_totals_mae,
        "per_question_vs_hand": q_vs_hand.__dict__ if q_vs_hand else None,
        "totals_vs_hand": t_vs_hand.__dict__ if t_vs_hand else None,
        "curated": curated,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def mark(ok):
        return "PASS" if ok else "FAIL"

    lines = [
        f"# Phase 4 results — {rid} v{version}",
        "",
        f"{len(totals)} booklets graded; {sum(len(v) for v in review.values())} questions flagged for review.",
        "",
        "## Gate",
        "",
        "| Metric | Threshold | Result | |",
        "| :--- | :--- | :--- | :--- |",
    ]
    if q_vs_hand:
        lines += [
            f"| Per-question MAE (vs hand-verified sheets) | ≤ {GATE_MAE_MAX} | {q_vs_hand.mae:.2f} | {mark(q_vs_hand.mae <= GATE_MAE_MAX)} |",
            f"| Questions within ±1 (vs hand-verified) | ≥ {GATE_WITHIN1_MIN:.0%} | {q_vs_hand.within1:.1%} | {mark(q_vs_hand.within1 >= GATE_WITHIN1_MIN)} |",
            f"| Mean signed error (vs hand-verified) | ±{GATE_SIGNED_ABS_MAX} | {q_vs_hand.signed:+.2f} | {mark(abs(q_vs_hand.signed) <= GATE_SIGNED_ABS_MAX)} |",
        ]
    lines += [
        f"| Per-question MAE implied by totals (vs recorded) | ≤ {GATE_MAE_MAX} | {q_from_totals_mae:.2f} | {mark(q_from_totals_mae <= GATE_MAE_MAX)} |",
        f"| Mean signed error per question (vs recorded) | ±{GATE_SIGNED_ABS_MAX} | {vs_recorded.signed_total / n_q:+.2f} | {mark(abs(vs_recorded.signed_total / n_q) <= GATE_SIGNED_ABS_MAX)} |",
        f"| Pearson r, total score (vs recorded) | ≥ {GATE_PEARSON_MIN} | {vs_recorded.pearson:.3f} | {mark(vs_recorded.passed)} |",
        f"| Alternative methods → full credit (held-out) | ≥ {GATE_ALT_FULL_MIN:.0%} | {alt:.0%} | {mark(alt >= GATE_ALT_FULL_MIN)} |",
        f"| Consequential error deducted once (held-out) | ≥ {GATE_CONSEQ_ONCE_MIN:.0%} | {conseq:.0%} | {mark(conseq >= GATE_CONSEQ_ONCE_MIN)} |",
        "",
        f"Totals equal to the recorded grade: {vs_recorded.exact_matches}/{vs_recorded.n}.",
        "",
        "## Disagreements with recorded grades",
        "",
        "| Booklet | System | Recorded | Hand-verified sheet |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for b, (s, r) in sorted(vs_recorded.disagreements.items()):
        lines.append(f"| {b} | {s:g} | {r:g} | {hand_totals.get(b, float('nan')):g} |")
    lines += ["", "## Curated sets", ""]
    for name, c in curated.items():
        lines.append(f"- **{name}**: alternative {c['alternative']:.0%}, consequential {c['consequential']:.0%}, "
                     f"controls {c['control']:.0%}; failures: {[f['case_id'] for f in c['failures']] or 'none'}")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWritten {REPORT_MD} and {REPORT_JSON}")
    return 0


def cmd_curated(args) -> int:
    f = "curated_cases_heldout.json" if args.heldout else "curated_cases.json"
    rep = run_curated(lambda q, t: grade_open_question(q, t, use_cache=not args.no_cache), cases_file=f)
    for o in rep.outcomes:
        print(f"  {o.case_id:6s} {'PASS' if o.passed else 'FAIL'} {o.score:g}/{o.max_points:g} "
              f"path={o.best_path} deductions={o.deductions}{' review' if o.needs_review else ''} {o.detail}")
    print({s: rep.rate(s) for s in ("alternative", "consequential", "control")},
          f"spend ${run_spend_usd():.4f}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 4 grading engine")
    p.add_argument("--exam-id", default=DEFAULT_EXAM)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("propose")
    sp.add_argument("--sol", default=str(DEFAULT_SOL))
    sp.add_argument("--booklets")
    sp.set_defaults(fn=cmd_propose)

    sp = sub.add_parser("approve")
    sp.add_argument("--version", type=int, required=True)
    sp.add_argument("--by", required=True)
    sp.set_defaults(fn=cmd_approve)

    sp = sub.add_parser("grade")
    sp.add_argument("--sol", default=str(DEFAULT_SOL), help="master key, for the per-booklet printed-options check")
    sp.add_argument("--booklets")
    sp.add_argument("--stale-only", action="store_true")
    sp.add_argument("--no-vlm", action="store_true", help="CV only: no cross-check, no paid calls")
    sp.set_defaults(fn=cmd_grade)

    sp = sub.add_parser("evaluate")
    sp.add_argument("--grades-csv", default=str(DEFAULT_GRADES))
    sp.set_defaults(fn=cmd_evaluate)

    sp = sub.add_parser("curated")
    sp.add_argument("--heldout", action="store_true")
    sp.add_argument("--no-cache", action="store_true")
    sp.set_defaults(fn=cmd_curated)

    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s", datefmt="%H:%M:%S")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
