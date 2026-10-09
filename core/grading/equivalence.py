"""
Symbolic equivalence: ``simplify(student - solution) == 0`` via SymPy, with
numeric spot-checking where simplification cannot close.

Deterministic — and therefore one of the trustworthy inputs to grading and
confidence. Never penalise a correct answer for being unsimplified.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from typing import Optional

import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication,
    parse_expr,
    standard_transformations,
)

logger = logging.getLogger(__name__)

NUMERIC_REL_TOL: float = 1e-6
NUMERIC_ABS_TOL: float = 1e-9
SPOT_CHECK_POINTS: int = 8
SPOT_CHECK_SEED: int = 1729

NONE_OF_THE_ABOVE = "__NONE_OF_THE_ABOVE__"
_NONE_RE = re.compile(r"(אף\s*אחד\s*מהנ[\"״׳'`]?\s*ל|none\s+of\s+the\s+above)", re.IGNORECASE)
# implicit_multiplication (not ..._application): the latter splits names like
# ``m1`` into ``m*1``, which breaks milestone-variable substitution.
_TRANSFORMS = standard_transformations + (implicit_multiplication, convert_xor)
# ``i`` stays a symbol: in exam work it is far more often an index than sqrt(-1).
_CONSTANTS = {"e": sympy.E, "pi": sympy.pi}


@dataclass
class EquivalenceResult:
    equal: Optional[bool]          # None = could not determine
    method: str                    # symbolic | numeric | string | undetermined
    detail: str = ""


def normalize_option_text(text: str) -> str:
    """Canonical form for MCQ option comparison across exam versions."""
    t = (text or "").strip().strip("$").strip()
    if _NONE_RE.search(t):
        return NONE_OF_THE_ABOVE
    t = t.replace("\\%", "%").replace(" ", "")
    return t


def to_sympy(text: str) -> Optional[sympy.Expr]:
    """Parse plain math (``3/7``, ``x^2+1``) or LaTeX (``\\frac{3}{7}``)."""
    s = (text or "").strip().strip("$").strip()
    if not s or s == NONE_OF_THE_ABOVE:
        return None
    pct = s.endswith("%")
    if pct:
        s = s[:-1].rstrip("\\")
    expr: Optional[sympy.Expr] = None
    if "\\" in s:
        try:
            from sympy.parsing.latex import parse_latex
            expr = parse_latex(s)
        except Exception:
            expr = None
    if expr is None:
        plain = (
            s.replace("\\cdot", "*").replace("\\times", "*").replace("·", "*")
            .replace("×", "*").replace("−", "-").replace("{", "(").replace("}", ")")
        )
        try:
            expr = parse_expr(
                plain, local_dict=dict(_CONSTANTS), transformations=_TRANSFORMS, evaluate=True
            )
        except Exception:
            return None
    if isinstance(expr, sympy.Equality):
        expr = expr.lhs - expr.rhs
    if not isinstance(expr, sympy.Expr):     # tuples, booleans, sets: not a single value
        return None
    if pct:
        expr = expr / 100
    return expr


def _numeric_close(a: complex, b: complex) -> bool:
    return abs(a - b) <= max(NUMERIC_ABS_TOL, NUMERIC_REL_TOL * max(abs(a), abs(b)))


def _spot_check(diff: sympy.Expr) -> Optional[bool]:
    symbols = sorted(diff.free_symbols, key=lambda s: s.name)
    rng = random.Random(SPOT_CHECK_SEED)
    evaluated = 0
    for _ in range(SPOT_CHECK_POINTS * 3):
        point = {s: sympy.Float(rng.uniform(0.1, 3.0)) for s in symbols}
        try:
            val = complex(diff.subs(point).evalf())
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if val != val:  # NaN
            continue
        evaluated += 1
        if abs(val) > NUMERIC_ABS_TOL * 1e3:
            return False
        if evaluated >= SPOT_CHECK_POINTS:
            return True
    return True if evaluated else None


def expressions_equivalent(student: str, solution: str) -> EquivalenceResult:
    """Decide whether two expressions are mathematically equal."""
    a_norm, b_norm = normalize_option_text(student), normalize_option_text(solution)
    if a_norm == b_norm:
        return EquivalenceResult(True, "string", "identical after normalisation")
    if NONE_OF_THE_ABOVE in (a_norm, b_norm):
        return EquivalenceResult(False, "string", "only one side is 'none of the above'")

    a, b = to_sympy(a_norm), to_sympy(b_norm)
    if a is None or b is None:
        return EquivalenceResult(None, "undetermined", "could not parse one side")

    diff = a - b
    if not diff.free_symbols:
        try:
            return EquivalenceResult(
                _numeric_close(complex(a.evalf()), complex(b.evalf())),
                "numeric",
                f"{a} vs {b}",
            )
        except (TypeError, ValueError):
            pass
    try:
        if sympy.simplify(diff) == 0:
            return EquivalenceResult(True, "symbolic", "simplify(a - b) == 0")
    except Exception as exc:  # SymPy can raise many things on odd input
        logger.debug("simplify failed: %s", exc)
    spot = _spot_check(diff)
    if spot is None:
        return EquivalenceResult(None, "undetermined", "symbolic and numeric checks inconclusive")
    return EquivalenceResult(spot, "numeric", "random spot-check over free symbols")


def evaluate_with(expr_text: str, values: dict[str, float]) -> Optional[float]:
    """Evaluate an expression after substituting named values (consequential error)."""
    expr = to_sympy(expr_text)
    if expr is None:
        return None
    subs = {sympy.Symbol(k): v for k, v in values.items()}
    try:
        out = complex(expr.subs(subs).evalf())
    except (TypeError, ValueError):
        return None
    if abs(out.imag) > NUMERIC_ABS_TOL:
        return None
    return out.real


def values_equal(a: float, b: float) -> bool:
    return _numeric_close(complex(a), complex(b))
