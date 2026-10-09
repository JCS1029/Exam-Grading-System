"""
Exam-version (טור) handling for shuffled multiple-choice exams.

Every version prints the same questions in the same order but shuffles the
options. The version is identified by a printed card-suit symbol after
"!בהצלחה" on the cover sheet, and each version's key is derived — never typed
by hand — by mapping its printed options onto the master solution key:

    1. ``detect_suit``           local CV on the cover sheet (free, deterministic)
    2. ``read_printed_options``  VLM reads the printed options on the question
                                 pages (handwriting ignored; no identity data)
    3. ``map_to_master``         SymPy/string equivalence; the four printed
                                 options must be a permutation of the master's
    4. ``build_version_keys``    consensus across all booklets of a version;
                                 a booklet whose own pages disagree is flagged

The resulting keys are *proposed*; grading uses them only after approval
(see ``core.grading.store``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.grading.answer_key import OPTION_LETTERS, McqAnswerKey
from core.grading.equivalence import expressions_equivalent

logger = logging.getLogger(__name__)

SUITS: Tuple[str, ...] = ("heart", "diamond", "spade", "club")

# Measured on the 13 pilot cover sheets (4 heart, 3 diamond, 2 spade, 4 club):
#   hollow glyphs (heart/diamond): ink/filled 0.34–0.41, one hole
#   solid glyphs (spade/club):     ink/filled 1.00, no hole
#   top-sixth width: heart 0.70, diamond 0.17–0.18, club 0.31–0.35, spade 0.14–0.16
#   solidity:        club 0.78–0.79, spade 0.84–0.85
HOLLOW_INK_MAX: float = 0.70
HEART_TOP_MIN: float = 0.45
CLUB_TOP_MIN: float = 0.24
CLUB_SOLIDITY_MAX: float = 0.82

STORAGE_ROOT = Path(os.getenv("STORAGE_ROOT", "storage"))
OPTIONS_CACHE_DIR = STORAGE_ROOT / "cache" / "printed_options"
OPTIONS_PROMPT_VERSION = "printed_options_v1"
OPTIONS_MAX_DIM: int = int(os.getenv("VERSIONS_VLM_MAX_DIM", "1600"))


# ---------------------------------------------------------------------------
# Suit detection
# ---------------------------------------------------------------------------

@dataclass
class SuitDetection:
    suit: Optional[str]
    needs_review: bool
    reason: str
    features: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "suit": self.suit,
            "needs_review": self.needs_review,
            "reason": self.reason,
            "features": {k: round(v, 3) for k, v in self.features.items()},
        }


def _symbol_features(band: np.ndarray) -> Optional[Dict[str, float]]:
    """Features of the leftmost glyph on the "!בהצלחה" line inside ``band``."""
    blur = cv2.GaussianBlur(band, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    comps = [(i, *stats[i]) for i in range(1, n) if stats[i][cv2.CC_STAT_AREA] > 30]
    if not comps:
        return None
    # The greeting is the tallest text on the band; the suit is its leftmost glyph.
    tallest = max(comps, key=lambda c: c[4])
    mid = tallest[2] + tallest[4] / 2
    line = [c for c in comps if abs((c[2] + c[4] / 2) - mid) < tallest[4]]
    if len(line) < 3:
        return None
    i, sx, sy, sw, sh, area = min(line, key=lambda c: c[1])
    mask = (labels[sy : sy + sh, sx : sx + sw] == i).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    holes = sum(1 for h in hierarchy[0] if h[3] >= 0) if hierarchy is not None else 0
    outer = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [outer], -1, 255, -1)
    hull_area = cv2.contourArea(cv2.convexHull(outer))
    profile = (filled > 0).mean(axis=1)
    return {
        "ink_ratio": area / float(max(1, np.count_nonzero(filled))),
        "holes": float(holes),
        "top_width": float(profile[: max(1, sh // 6)].mean()),
        "solidity": cv2.contourArea(outer) / max(1.0, hull_area),
        "aspect": sw / float(sh),
    }


def detect_suit(image_path: str | Path, grid_bbox: Tuple[int, int, int, int]) -> SuitDetection:
    """Classify the cover sheet's suit symbol from the band below the answer grid."""
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return SuitDetection(None, True, f"cannot read image: {image_path}")
    x, y, w, h = grid_bbox
    band = gray[y + h : y + h + int(0.35 * h), max(0, x - w // 10) : x + w + w // 10]
    feats = _symbol_features(band) if band.size else None
    if feats is None:
        return SuitDetection(None, True, "suit symbol not found below the answer grid")

    hollow = feats["ink_ratio"] < HOLLOW_INK_MAX
    if hollow != (feats["holes"] >= 1):
        return SuitDetection(None, True, "hollow/solid cues disagree", feats)
    if hollow:
        suit = "heart" if feats["top_width"] >= HEART_TOP_MIN else "diamond"
        return SuitDetection(suit, False, "hollow glyph", feats)
    by_top = "club" if feats["top_width"] >= CLUB_TOP_MIN else "spade"
    by_solidity = "club" if feats["solidity"] <= CLUB_SOLIDITY_MAX else "spade"
    if by_top != by_solidity:
        return SuitDetection(None, True, f"spade/club cues disagree ({by_top} vs {by_solidity})", feats)
    return SuitDetection(by_top, False, "solid glyph", feats)


# ---------------------------------------------------------------------------
# Printed options (VLM)
# ---------------------------------------------------------------------------

OPTIONS_PROMPT = """\
These are question pages of a printed Hebrew multiple-choice exam. Each
question "שאלה N" is followed by four printed options labelled (א) (ב) (ג) (ד).
Read ONLY the printed text of each option. Ignore all handwriting: underlines,
circles, check marks, scribbles and written notes are not part of the options.

Write numbers exactly as printed (e.g. 0.427, 1/2, 4/19). Write the Hebrew
option "אף אחד מהנ\"ל" as NONE.

Return ONLY JSON:
{"questions": [{"question": 1, "options": {"א": "...", "ב": "...", "ג": "...", "ד": "..."}}, ...]}
Include every question that appears on these pages.
"""


@dataclass
class PrintedOptions:
    booklet_id: str
    questions: Dict[str, Dict[str, str]] = field(default_factory=dict)
    cost_usd: float = 0.0
    model: str = ""
    errors: List[str] = field(default_factory=list)


def _encode_page(path: Path) -> bytes:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    if max(h, w) > OPTIONS_MAX_DIM:
        s = OPTIONS_MAX_DIM / float(max(h, w))
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise ValueError(f"JPEG encode failed: {path}")
    return buf.tobytes()


def read_printed_options(
    booklet_id: str,
    page_paths: Sequence[str | Path],
    *,
    model: Optional[str] = None,
    use_cache: bool = True,
) -> PrintedOptions:
    """Read the printed option text of every question on the given pages."""
    from core.vlm import VLMError, generate_with_image, get_default_model, parse_json_response

    model = model or get_default_model()
    out = PrintedOptions(booklet_id=booklet_id, model=model)
    for path in page_paths:
        path = Path(path)
        try:
            jpeg = _encode_page(path)
        except (FileNotFoundError, ValueError) as exc:
            out.errors.append(str(exc))
            continue
        key = hashlib.sha256(jpeg).hexdigest()[:24]
        cache = OPTIONS_CACHE_DIR / f"{OPTIONS_PROMPT_VERSION}_{model.replace('/', '_')}_{key}.json"
        if use_cache and cache.exists():
            text = json.loads(cache.read_text(encoding="utf-8"))["raw_text"]
        else:
            try:
                resp = generate_with_image(
                    OPTIONS_PROMPT, jpeg, model=model, json_mode=True, purpose="printed_options"
                )
            except VLMError as exc:
                out.errors.append(f"{path.name}: VLM call failed: {exc}")
                continue
            text, out.cost_usd = resp.text, out.cost_usd + resp.cost_usd
            if use_cache:
                OPTIONS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps({"raw_text": text}, ensure_ascii=False), encoding="utf-8")
        try:
            payload = parse_json_response(text)
        except Exception as exc:
            out.errors.append(f"{path.name}: unparseable VLM output: {exc}")
            continue
        items = payload if isinstance(payload, list) else (payload or {}).get("questions") or []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                qid = str(int(item.get("question")))
            except (TypeError, ValueError):
                continue
            opts = {
                str(k).strip("() "): str(v).strip()
                for k, v in (item.get("options") or {}).items()
            }
            out.questions[qid] = {l: opts.get(l, "") for l in OPTION_LETTERS}
    return out


# ---------------------------------------------------------------------------
# Mapping to the master key
# ---------------------------------------------------------------------------

def _canon(text: str) -> str:
    return "אף אחד מהנ״ל" if text.strip().upper() == "NONE" else text


def map_to_master(
    printed: Dict[str, Dict[str, str]], master: McqAnswerKey
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    For each question, find the printed letter whose option equals the master's
    correct option. Returns (letters, issues) keyed by question id. A question
    is only mapped when the printed options are a permutation of the master's.
    """
    letters: Dict[str, str] = {}
    issues: Dict[str, str] = {}
    for mq in master.questions:
        qid = mq.question_id
        opts = printed.get(qid)
        if not opts:
            issues[qid] = "question not found on printed pages"
            continue
        perm: Dict[str, str] = {}   # printed letter → master letter
        for pl in OPTION_LETTERS:
            matches = [
                ml for ml in OPTION_LETTERS
                if expressions_equivalent(_canon(opts.get(pl, "")), mq.options.get(ml, "")).equal
            ]
            if len(matches) == 1:
                perm[pl] = matches[0]
        if len(perm) != len(OPTION_LETTERS) or len(set(perm.values())) != len(OPTION_LETTERS):
            issues[qid] = f"printed options {opts} are not a permutation of master {mq.options}"
            continue
        letters[qid] = next(pl for pl, ml in perm.items() if ml == mq.correct_letter)
    return letters, issues


@dataclass
class VersionKey:
    suit: str
    letters: Dict[str, str]                       # question id → correct letter
    support: Dict[str, int]                       # question id → booklets agreeing
    booklets: List[str]
    unresolved: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "suit": self.suit,
            "letters": self.letters,
            "support": self.support,
            "booklets": self.booklets,
            "unresolved": self.unresolved,
        }


def build_version_keys(
    per_booklet: Dict[str, Tuple[Optional[str], Dict[str, str]]],
    question_ids: Sequence[str],
) -> Tuple[Dict[str, VersionKey], Dict[str, List[str]]]:
    """
    ``per_booklet``: booklet id → (suit, mapped letters from its own pages).
    Returns (version keys, dissent) where dissent lists, per booklet, the
    questions on which its own printed pages disagree with the consensus.
    """
    by_suit: Dict[str, List[str]] = defaultdict(list)
    for bid, (suit, _) in per_booklet.items():
        if suit:
            by_suit[suit].append(bid)

    keys: Dict[str, VersionKey] = {}
    dissent: Dict[str, List[str]] = defaultdict(list)
    for suit, bids in sorted(by_suit.items()):
        vk = VersionKey(suit=suit, letters={}, support={}, booklets=sorted(bids))
        for qid in question_ids:
            votes = Counter(per_booklet[b][1][qid] for b in bids if qid in per_booklet[b][1])
            if not votes:
                vk.unresolved[qid] = "no booklet of this version had a readable option set"
                continue
            (top, n), *rest = votes.most_common()
            if rest and rest[0][1] == n:
                vk.unresolved[qid] = f"tied votes {dict(votes)}"
                continue
            vk.letters[qid], vk.support[qid] = top, n
            for b in bids:
                got = per_booklet[b][1].get(qid)
                if got is not None and got != top:
                    dissent[b].append(qid)
        keys[suit] = vk
    return keys, dict(dissent)
