"""
Phase 4 — Grading engine.

Modules:
    answer_key    — MCQ key from a digital solution PDF (red-marked options)
    equivalence   — SymPy symbolic equivalence with numeric fallback
    rubric        — versioned rubric model; proposed → instructor-approved
    store         — SQLite persistence for rubrics and grades
    omr           — answer-grid reader for multiple-choice cover sheets
    versions      — exam-version detection and per-version answer keys
    consequential — milestone grading with consequential-error substitution
    grader        — orchestration and reproducible grade records
"""
