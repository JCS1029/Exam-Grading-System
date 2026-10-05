"""SymPy-based LaTeX validation for objective quality scoring."""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence

from sympy.parsing.latex import parse_latex


_LATEX_INLINE = re.compile(r"\$([^$]+)\$")
_LATEX_DISPLAY = re.compile(r"\$\$([^$]+)\$\$", re.DOTALL)
_HEBREW = re.compile(r"[\u0590-\u05FF]")
_HAS_MATH_SIGNAL = re.compile(
    r"(\\([a-zA-Z]+)|[=^_{}\\]|\\frac|\\sum|\\int|\\begin|[0-9]+|[+*/]|\\cdot|\\times)"
)


def extract_latex_from_markdown(text: str) -> List[str]:
    """Pull inline and display LaTeX spans from interleaved markdown."""
    if not text:
        return []
    found: List[str] = []
    for pattern in (_LATEX_DISPLAY, _LATEX_INLINE):
        for match in pattern.finditer(text):
            body = match.group(1).strip()
            if body:
                found.append(body)
    return found


def _clean_formula(raw: str) -> str:
    s = raw.strip().strip("$").strip()
    # Common VLM / TeX normalizations SymPy accepts better
    s = s.replace(r"\dots", r"\ldots")
    s = s.replace(r"\dfrac", r"\frac")
    s = s.replace(r"\tfrac", r"\frac")
    # Trailing punctuation / incomplete RHS
    s = re.sub(r"[=+\-*/,]+\s*$", "", s).strip()
    return s


def _is_mathish(formula: str) -> bool:
    """Drop prose / MCQ junk that models sometimes put in math_latex[]."""
    if not formula or len(formula) > 500:
        return False
    if _HEBREW.search(formula):
        return False
    # Pure percentage like 40\% — treat as math-ish number
    if re.fullmatch(r"\d+(\.\d+)?\\?%", formula):
        return True
    if not _HAS_MATH_SIGNAL.search(formula):
        # Allow simple identifiers / numbers
        if re.fullmatch(r"[A-Za-z](_\{?[A-Za-z0-9]+\}?)?", formula):
            return True
        if re.fullmatch(r"\d+(\.\d+)?", formula):
            return True
        return False
    return True


def _try_parse(formula: str) -> bool:
    variants = [formula]
    # Percent
    if r"\%" in formula or formula.endswith("%"):
        variants.append(re.sub(r"\\?%$", "", formula).strip() + "/100")
    # Bare tuple / pair notation
    if re.fullmatch(r"\([^()]+,[^()]+\)", formula):
        inner = formula[1:-1]
        variants.append(inner.replace(",", "+"))  # weak but parseable signal
    # pmatrix / bmatrix often choke antlr — count structural validity separately
    if r"\begin{" in formula and r"\end{" in formula:
        if formula.count(r"\begin{") == formula.count(r"\end{"):
            return True
    for v in variants:
        try:
            parse_latex(v)
            return True
        except Exception:
            continue
    return False


def validate_latex_formulas(formulas: Iterable[str]) -> dict:
    """
    Attempt SymPy parse on each formula.

    Returns parse_rate in [0, 1] plus per-formula errors (sample capped).
    Non-math junk (Hebrew prose, empty) is excluded from the denominator.
    """
    cleaned = []
    for f in formulas:
        c = _clean_formula(f)
        if c and _is_mathish(c):
            cleaned.append(c)

    if not cleaned:
        return {
            "total": 0,
            "parsed": 0,
            "parse_rate": 1.0,
            "errors": [],
            "skipped_non_math": True,
        }

    parsed = 0
    errors: list[dict] = []
    for formula in cleaned:
        if _try_parse(formula):
            parsed += 1
        else:
            errors.append({"formula": formula[:120], "error": "parse_failed"})

    total = len(cleaned)
    return {
        "total": total,
        "parsed": parsed,
        "parse_rate": round(parsed / total, 4),
        "errors": errors[:5],
    }


def collect_all_formulas(
    math_latex: Sequence[str],
    interleaved_markdown: str,
) -> List[str]:
    """Merge explicit math_latex[] with spans extracted from markdown."""
    seen: set[str] = set()
    out: List[str] = []
    for src in list(math_latex) + extract_latex_from_markdown(interleaved_markdown):
        key = _clean_formula(src)
        if key and key not in seen and _is_mathish(key):
            seen.add(key)
            out.append(key)
    return out
