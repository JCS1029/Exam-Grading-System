"""
Phase 4 tests — grading engine. Offline: no API calls.

Tests that need the (gitignored) exam PDFs or anonymised pilot pages skip when
those files are absent.
"""

import json
from pathlib import Path

import pytest

from core.grading.answer_key import McqAnswerKey, McqQuestion, extract_mcq_key
from core.grading.consequential import score_path, student_number, value_matches
from core.grading.curated import evaluate_case, load_curated
from core.grading.equivalence import evaluate_with, expressions_equivalent, to_sympy
from core.grading.metrics import pearson, question_gate, total_gate
from core.grading.omr import (
    CellReading,
    GridReading,
    RowReading,
    RowStatus,
    VlmGridReading,
    VlmRowReading,
    _parse_vlm_grid,
    _resolve_row,
    read_answer_grid,
    reconcile,
)
from core.grading.open_grader import score_extraction
from core.grading.rubric import Milestone, QuestionRubric, Rubric, SolutionPath, mcq_rubric
from core.grading.store import GradeRecord, GradingStore
from core.grading.versions import build_version_keys, detect_suit, map_to_master

ROOT = Path(__file__).resolve().parent.parent
SOL = ROOT / "docs" / "Dataset" / "sol.pdf"
HAND = ROOT / "storage" / "eval" / "mc_hand_labels.json"
ANON = ROOT / "storage" / "anonymized"


# ---------------------------------------------------------------------------
# Equivalence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a,b", [
    (r"\frac{2}{7}", "2/7"),
    (r"\sqrt{8}", r"2\sqrt{2}"),
    ("2e^(-2)", "2*exp(-2)"),
    ("x^2+2x+1", "(x+1)^2"),
    ("42.7%", "0.427"),
    ("אף אחד מהנ״ל", "none of the above"),
])
def test_equivalent(a, b):
    assert expressions_equivalent(a, b).equal is True


@pytest.mark.parametrize("a,b", [("84/100", "84/101"), ("0.427", "38/89"), ("x+1", "x+2")])
def test_not_equivalent(a, b):
    assert expressions_equivalent(a, b).equal is False


def test_to_sympy_rejects_non_values():
    assert to_sympy("2, 3") is None


def test_evaluate_with_substitution():
    assert evaluate_with("m1/(m1+m2)", {"m1": 0.36, "m2": 0.51}) == pytest.approx(0.36 / 0.87)


# ---------------------------------------------------------------------------
# Consequential error
# ---------------------------------------------------------------------------

def _bayes_path() -> SolutionPath:
    return SolutionPath("bayes", "", [
        Milestone("m1", "", 3, "0.4*0.95"),
        Milestone("m2", "", 2, "0.6*0.85"),
        Milestone("m3", "", 2, "m1+m2", ["m1", "m2"]),
        Milestone("m4", "", 3, "m1/m3", ["m1", "m3"]),
    ])


def test_student_number_takes_final_result():
    assert student_number("0.4\\cdot0.95=0.36") == pytest.approx(0.36)
    assert student_number("A (0.4) → 0.95 → 0.38") == pytest.approx(0.38)
    assert student_number(r"\frac{0.36}{0.87}\approx0.4138") == pytest.approx(0.4138)
    assert student_number("270.7, כלומר 271") == pytest.approx(271)
    assert student_number(None) is None


def test_value_matches_rounding():
    assert value_matches("0.427", 0.427, 38 / 89)
    assert value_matches("0.43", 0.43, 38 / 89)
    assert not value_matches("0.4", 0.4, 38 / 89)          # too coarse
    assert value_matches("271", 271, 270.67, abs_tol=0.5)
    assert not value_matches("271", 271, 270.67)


def test_consequential_deducts_once():
    s = score_path(_bayes_path(), {"m1": "0.36", "m2": "0.51", "m3": "0.87", "m4": "0.4138"}, {})
    assert s.score == 7 and s.deductions == ["m1"]
    assert [o.status for o in s.outcomes] == ["wrong", "correct", "consequential", "consequential"]


def test_consequential_middle_slip():
    s = score_path(_bayes_path(), {"m1": "0.38", "m2": "0.51", "m3": "0.79", "m4": "0.481"}, {})
    assert s.score == 8 and s.deductions == ["m3"]


def test_wrong_downstream_after_slip_is_not_credited():
    s = score_path(_bayes_path(), {"m1": "0.36", "m2": "0.51", "m3": "0.87", "m4": "0.5"}, {})
    assert s.deductions == ["m1", "m4"]


def test_skipped_intermediate_does_not_poison_later_steps():
    s = score_path(_bayes_path(), {"m1": "0.38", "m2": None, "m3": "0.89", "m4": "0.427"}, {})
    assert s.deductions == ["m2"] and s.score == 8


def test_all_correct_full_credit():
    s = score_path(_bayes_path(), {"m1": "0.38", "m2": "0.51", "m3": "0.89", "m4": "38/89"}, {})
    assert s.score == 10 and not s.deductions


# ---------------------------------------------------------------------------
# Open-question scoring (alternative paths, open validity)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def curated():
    return load_curated()


def test_curated_rubric_validates_and_matches_official_answers(curated):
    rubric, cases = curated
    assert rubric.validate() == []
    finals = {q.question_id: {p.path_id: list(p.resolve_expected().values())[-1] for p in q.paths}
              for q in rubric.questions}
    assert finals["2"]["bayes"] == pytest.approx(0.427, abs=5e-4)
    assert finals["3"]["series"] == pytest.approx(0.5)
    assert round(finals["6"]["exact"]) == 271 and round(finals["6"]["poisson"]) == 271
    assert {c["set"] for c in cases} == {"alternative", "consequential", "control"}


def test_best_declared_path_counts(curated):
    rubric, _ = curated
    q = rubric.question("2")
    extraction = {"path_followed": "complement",
                  "values": {"complement.c1": "0.11", "complement.c2": "0.89",
                             "complement.c3": "0.38", "complement.c4": "0.427"}}
    r = score_extraction(q, extraction)
    assert r.score == 10 and r.best_path == "complement" and not r.needs_review


def test_undeclared_valid_method_proposes_full_but_needs_review(curated):
    rubric, _ = curated
    q = rubric.question("3")
    extraction = {"path_followed": "other", "values": {}, "final_answer": "1/2",
                  "undeclared_method": {"used": True, "reasoning_valid": True}}
    r = score_extraction(q, extraction)
    assert r.score == 10 and r.needs_review


def test_undeclared_invalid_method_with_right_answer_gets_no_credit(curated):
    rubric, _ = curated
    q = rubric.question("3")
    extraction = {"path_followed": "other", "values": {}, "final_answer": "1/2",
                  "undeclared_method": {"used": True, "reasoning_valid": False}}
    r = score_extraction(q, extraction)
    assert r.score == 0 and r.needs_review


def test_undeclared_valid_method_with_wrong_answer_not_full(curated):
    rubric, _ = curated
    q = rubric.question("3")
    extraction = {"path_followed": "other", "values": {}, "final_answer": "2/3",
                  "undeclared_method": {"used": True, "reasoning_valid": True}}
    assert score_extraction(q, extraction).score < 10


def test_evaluate_case_requires_review_for_undeclared(curated):
    rubric, _ = curated
    q = rubric.question("3")
    r = score_extraction(q, {"path_followed": "series", "values": {
        "series.s1": "1/3", "series.s2": "3/2", "series.s3": "1/2"}})
    case = {"id": "x", "set": "alternative", "declared": False}
    assert not evaluate_case(case, q, r).passed            # full credit but silently auto-accepted


# ---------------------------------------------------------------------------
# Rubric + store (approval, versioning, stale grades)
# ---------------------------------------------------------------------------

def test_rubric_validation_catches_errors():
    q = QuestionRubric("1", 10, paths=[SolutionPath("p", "", [
        Milestone("m1", "", 4, "2*m2", ["m2"]),     # forward reference
        Milestone("m2", "", 4, "0.5"),               # sums to 8, not 10
    ])])
    errors = q.validate()
    assert any("not an earlier milestone" in e for e in errors)
    assert any("sum to 8" in e for e in errors)


def _toy_mcq(letter_heart: str) -> Rubric:
    master = McqAnswerKey("sol.pdf", questions=[
        McqQuestion("1", {"א": "1", "ב": "2", "ג": "3", "ד": "4"}, "ב"),
    ])
    return mcq_rubric("mcq:test", "test", master, {"heart": {"1": letter_heart}})


def test_store_approval_versioning_and_stale(tmp_path):
    store = GradingStore(tmp_path / "g.db")
    r1 = _toy_mcq("ג")
    v1 = store.propose_rubric(r1)
    assert store.propose_rubric(r1) == v1                 # identical content → same version
    assert store.approved_rubric("mcq:test") is None      # proposed is not gradeable
    store.approve_rubric("mcq:test", v1, "tester")
    rubric, version = store.approved_rubric("mcq:test")
    store.record_grade(GradeRecord("b1", "1", 10, 10, "mcq:test", version, rubric.question_hash("1"),
                                   raw_response="{}", temperature=0.0))
    assert store.stale_grades("mcq:test") == []

    v2 = store.propose_rubric(_toy_mcq("ד"))              # instructor edits the key
    assert v2 == v1 + 1
    assert store.stale_grades("mcq:test") == []           # still graded under the approved v1
    store.approve_rubric("mcq:test", v2, "tester")
    stale = store.stale_grades("mcq:test")
    assert [(g.booklet_id, g.question_id) for g in stale] == [("b1", "1")]
    versions = store.rubric_versions("mcq:test")
    assert [v["status"] for v in versions] == ["superseded", "approved"]

    rubric2, _ = store.approved_rubric("mcq:test")
    store.record_grade(GradeRecord("b1", "1", 0, 10, "mcq:test", v2, rubric2.question_hash("1")))
    assert store.stale_grades("mcq:test") == []
    assert len(store.current_grades("mcq:test")) == 1     # old grade superseded, kept for audit
    store.close()


def test_store_refuses_invalid_rubric(tmp_path):
    store = GradingStore(tmp_path / "g.db")
    bad = Rubric("open:test", "t", [QuestionRubric("1", 10, paths=[
        SolutionPath("p", "", [Milestone("m1", "", 5, "0.5")])])])
    v = store.propose_rubric(bad)
    with pytest.raises(ValueError):
        store.approve_rubric("open:test", v, "tester")
    store.close()


# ---------------------------------------------------------------------------
# Answer grid
# ---------------------------------------------------------------------------

def _row(*cells):
    return RowReading("1", [CellReading(l, 0, e, c, marked=m) for l, e, c, m in cells])


def test_resolve_single_mark():
    r = _row(("ד", .05, .2, True), ("ג", 0, 0, False), ("ב", 0, 0, False), ("א", 0, 0, False))
    _resolve_row(r)
    assert r.status == RowStatus.ANSWERED and r.chosen == "ד"


def test_resolve_cancelled_box():
    r = _row(("ד", .2, .9, True), ("ג", 0, 0, False), ("ב", .05, .3, True), ("א", 0, 0, False))
    _resolve_row(r)
    assert r.chosen == "ב" and r.cancelled == ["ד"]


def test_resolve_two_similar_marks_is_ambiguous():
    r = _row(("ד", .05, .3, True), ("ג", 0, 0, False), ("ב", .05, .32, True), ("א", 0, 0, False))
    _resolve_row(r)
    assert r.status == RowStatus.AMBIGUOUS and r.chosen is None


def test_parse_vlm_grid_accepts_list_and_dict():
    item = {"question": 1, "marks": [{"option": "א", "type": "x"}], "final_answer": "א"}
    rows, errors = _parse_vlm_grid({"rows": [item]}, 1)
    assert rows[0].final_answer == "א" and not errors
    rows, _ = _parse_vlm_grid([item], 1)
    assert rows[0].final_answer == "א"
    rows, errors = _parse_vlm_grid([], 1)
    assert errors and rows[0].final_answer is None


def test_reconcile_flags_disagreement():
    r = _row(("ד", .05, .2, True), ("ג", 0, 0, False), ("ב", 0, 0, False), ("א", 0, 0, False))
    _resolve_row(r)
    cv = GridReading("x", rows=[r])
    agree = VlmGridReading("m", "v", rows=[VlmRowReading("1", [("ד", "x")], "ד")])
    differ = VlmGridReading("m", "v", rows=[VlmRowReading("1", [("ג", "x")], "ג")])
    assert not reconcile(cv, agree)[0].needs_review
    out = reconcile(cv, differ)[0]
    assert out.needs_review and out.answer == "ד"
    assert reconcile(cv, None)[0].needs_review


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------

def test_map_to_master_permutation():
    master = McqAnswerKey("s", questions=[
        McqQuestion("1", {"א": "0.39", "ב": "0.47", "ג": "0.51", "ד": "0.55"}, "ד"),
        McqQuestion("2", {"א": "1/2", "ב": "1/3", "ג": "1/6", "ד": "אף אחד מהנ״ל"}, "ד"),
    ])
    printed = {
        "1": {"א": "0.55", "ב": "0.47", "ג": "0.51", "ד": "0.39"},
        "2": {"א": "1/3", "ב": "0.5", "ג": "1/6", "ד": "NONE"},
    }
    letters, issues = map_to_master(printed, master)
    assert letters == {"1": "א", "2": "ד"} and not issues
    letters, issues = map_to_master({"1": {"א": "0.55", "ב": "0.47", "ג": "0.52", "ד": "0.39"}}, master)
    assert "1" in issues and "1" not in letters               # misread → not a permutation


def test_build_version_keys_consensus_and_dissent():
    per = {
        "a": ("heart", {"1": "א", "2": "ב"}),
        "b": ("heart", {"1": "א", "2": "ב"}),
        "c": ("heart", {"1": "א", "2": "ג"}),
        "d": ("club", {"1": "ד"}),
    }
    keys, dissent = build_version_keys(per, ["1", "2"])
    assert keys["heart"].letters == {"1": "א", "2": "ב"}
    assert dissent == {"c": ["2"]}
    assert "2" in keys["club"].unresolved


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_metrics():
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    g = question_gate({"b": {"1": 5.0, "2": 10.0}}, {"b": {"1": 4.0, "2": 10.0}}, {"1": 5.0, "2": 10.0})
    assert g.mae == pytest.approx(1.0)            # 1 point on a 5-point question = 2 on the 10 scale
    t = total_gate({"a": 50, "b": 60, "c": 70}, {"a": 50, "b": 70, "c": 70})
    assert t.exact_matches == 2 and t.disagreements == {"b": (60, 70)}


# ---------------------------------------------------------------------------
# Pilot data (skipped when the private exam files are absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SOL.exists(), reason="sol.pdf not present")
def test_master_key_from_sol_pdf():
    key = extract_mcq_key(SOL)
    assert key.ok
    assert key.letters() == {"1": "ד", "2": "ב", "3": "ג", "4": "ד", "5": "ב",
                             "6": "ב", "7": "ד", "8": "ב", "9": "ב", "10": "ג"}


@pytest.mark.skipif(not (HAND.exists() and ANON.exists()), reason="pilot sheets not present")
def test_pilot_grids_and_suits_match_hand_labels():
    labels = json.loads(HAND.read_text(encoding="utf-8"))["booklets"]
    for bid, lab in labels.items():
        cover = ANON / bid / "page_001.png"
        if not cover.exists():
            continue
        grid = read_answer_grid(cover)
        assert grid.ok, (bid, grid.errors)
        assert [grid.answers()[str(i + 1)] for i in range(10)] == lab["marks"], bid
        assert detect_suit(cover, grid.grid_bbox).suit == lab["version"], bid
