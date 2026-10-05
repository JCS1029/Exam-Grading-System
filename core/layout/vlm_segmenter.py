"""
VLM-backed question / layout segmentation (planned Phase 2 core).

Takes a full page image + classical-CV gutter prior, asks the VLM for:
  - real printed question labels (שאלה 7, סעיף א, …)
  - normalized bounding boxes for each full question region
  - answered vs blank (student ink, not printed text)
  - column count / reading order

Coordinates are returned normalized to [0, 1] and converted to page pixels here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.layout.gutter import ColumnLayout
from core.layout.questions import QuestionRegion
from core.vlm import VLMError, generate_with_image, get_default_model, parse_json_response

logger = logging.getLogger(__name__)

STORAGE_ROOT: Path = Path(os.getenv("STORAGE_ROOT", "storage"))
CACHE_DIR: Path = STORAGE_ROOT / "cache" / "phase2_vlm"
PROMPT_VERSION: str = "phase2_segment_v1"
MAX_SEND_DIM: int = int(os.getenv("PHASE2_VLM_MAX_DIM", "1200"))
JPEG_QUALITY: int = int(os.getenv("PHASE2_VLM_JPEG_QUALITY", "75"))
USE_VLM_CACHE: bool = os.getenv("PHASE2_VLM_USE_CACHE", "true").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)


SEGMENTATION_PROMPT = """You are a layout analyst for scanned Hebrew university exam pages.

Your ONLY job is layout segmentation — NOT grading, NOT full transcription.

You are given:
1. A scanned exam page image.
2. A classical computer-vision gutter prior (may be wrong — override if the page is clearly single-column).

Rules:
- Identify EVERY printed question on this page by its printed label (e.g. שאלה 7, שאלה 8, 7., Q7).
- One question = ONE bounding box covering the FULL question: stem + multiple-choice options + diagrams + student work belonging to that question.
- Do NOT split one question into stem/options/diagram as separate questions.
- Do NOT invent questions that are not printed on the page.
- Do NOT renumber questions. If the page has questions 7–10, return ids "7","8","9","10".
- "answered" means the student left visible handwriting/circles/underlines for that question. Printed text alone is NOT answered — use "blank" when there is no student mark.
- Ignore CamScanner watermarks and bleed-through ghosts.
- bbox_norm is [x0, y0, x1, y1] in fractions of page width/height, origin top-left, each value in [0,1].
- If the page is a single vertical column of questions, set n_columns=1 even if there is whitespace on one side.
- For Hebrew multi-column pages, reading_order_hint should be "rtl" (right column first).

Return ONLY a JSON object with this schema:
{{
  "page_summary": "short English summary",
  "n_columns": 1,
  "reading_order_hint": "rtl" | "ltr" | "single",
  "override_gutter_prior": true,
  "questions": [
    {{
      "question_id": "7",
      "label": "שאלה 7",
      "bbox_norm": [0.05, 0.02, 0.95, 0.22],
      "state": "answered" | "blank",
      "has_student_ink": true,
      "contains_diagram": false,
      "is_subquestion": false,
      "parent_question_id": null,
      "confidence": 0.0
    }}
  ],
  "tables": [
    {{
      "kind": "score_roster" | "grid" | "unknown",
      "bbox_norm": [0.1, 0.1, 0.9, 0.4],
      "confidence": 0.0
    }}
  ]
}}

Gutter prior from OpenCV (advisory only):
{gutter_prior_json}
"""


@dataclass
class VLMSegmentResult:
    questions: List[QuestionRegion] = field(default_factory=list)
    tables: List[dict] = field(default_factory=list)
    n_columns: int = 1
    reading_order_hint: str = "single"
    override_gutter_prior: bool = False
    page_summary: str = ""
    model: str = ""
    prompt_version: str = PROMPT_VERSION
    latency_sec: float = 0.0
    prompt_tokens: int = 0
    candidate_tokens: int = 0
    cache_hit: bool = False
    raw: dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        # Zero questions is valid for cover / table-only pages.
        return len(self.errors) == 0


def _image_hash(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()[:24]


def _encode_for_vlm(image: np.ndarray) -> Tuple[bytes, float, float]:
    """
    Downscale large pages for VLM send; return JPEG bytes and scale factors
    (page_px / send_px) so normalized boxes map back to full-resolution pixels.
    """
    h, w = image.shape[:2]
    scale = 1.0
    if max(h, w) > MAX_SEND_DIM:
        scale = MAX_SEND_DIM / float(max(h, w))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        send = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        send = image

    ok, buf = cv2.imencode(".jpg", send, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise VLMError("Failed to JPEG-encode page for VLM")
    # Normalized coords are relative to the *sent* image; they map 1:1 to
    # full-page fractions, so pixel conversion uses full page W/H directly.
    return buf.tobytes(), float(w), float(h)


def _norm_to_bbox(
    bbox_norm: Sequence[float],
    page_w: float,
    page_h: float,
) -> Tuple[int, int, int, int]:
    if len(bbox_norm) != 4:
        raise ValueError(f"bbox_norm must have 4 values, got {bbox_norm!r}")
    x0, y0, x1, y1 = [float(v) for v in bbox_norm]
    # Accept [x,y,w,h] mistakes if values look like wh form
    if x1 <= 1.0 and y1 <= 1.0 and x1 > x0 and y1 > y0:
        pass
    elif x1 <= 1.0 and y1 <= 1.0 and (x1 + y1) < 1.5 and x1 > 0 and y1 > 0:
        # likely x,y,w,h normalized
        x1 = x0 + x1
        y1 = y0 + y1
    x0 = min(max(x0, 0.0), 1.0)
    y0 = min(max(y0, 0.0), 1.0)
    x1 = min(max(x1, 0.0), 1.0)
    y1 = min(max(y1, 0.0), 1.0)
    if x1 <= x0:
        x1 = min(1.0, x0 + 0.05)
    if y1 <= y0:
        y1 = min(1.0, y0 + 0.05)

    px0 = int(round(x0 * page_w))
    py0 = int(round(y0 * page_h))
    px1 = int(round(x1 * page_w))
    py1 = int(round(y1 * page_h))
    return (px0, py0, max(1, px1 - px0), max(1, py1 - py0))


def _normalize_question_id(raw: str, label: str = "") -> str:
    text = f"{raw} {label}"
    # Prefer explicit Hebrew "שאלה N"
    m = re.search(r"שאלה\s*(\d+[a-zA-Zא-ת]?)", text)
    if m:
        return m.group(1)
    m = re.search(r"(?:^|[^\d])(\d{1,2}[a-zA-Z]?)(?:[^\d]|$)", str(raw))
    if m:
        return m.group(1)
    cleaned = re.sub(r"[^0-9A-Za-zא-ת\-]", "", str(raw))
    return cleaned or str(raw).strip() or "unknown"


def _gutter_prior_payload(layout: ColumnLayout, page_w: int, page_h: int) -> dict:
    return {
        "n_columns_opencv": layout.n_columns,
        "is_multi_column_opencv": layout.is_multi_column,
        "gutter_confidences": layout.confidences,
        "gutters_px": [list(g) for g in layout.gutters],
        "columns": [
            {
                "index": c.index,
                "reading_order": c.reading_order,
                "bbox_norm": [
                    round(c.x0 / page_w, 4),
                    0.0,
                    round(c.x1 / page_w, 4),
                    1.0,
                ],
            }
            for c in layout.columns
        ],
        "note": "OpenCV prior only — trust the image over this if they disagree.",
    }


def _cache_path(image_bytes: bytes, model: str) -> Path:
    key = f"{PROMPT_VERSION}_{model}_{_image_hash(image_bytes)}"
    return CACHE_DIR / f"{key}.json"


def segment_page_with_vlm(
    image: np.ndarray,
    layout: ColumnLayout,
    *,
    model: Optional[str] = None,
    use_cache: bool = True,
) -> VLMSegmentResult:
    """
    Run VLM layout segmentation on a BGR page image.
    """
    model_name = model or get_default_model(primary=False)
    result = VLMSegmentResult(model=model_name, prompt_version=PROMPT_VERSION)

    h, w = image.shape[:2]
    image_bytes, page_w, page_h = _encode_for_vlm(image)
    prior = _gutter_prior_payload(layout, w, h)
    prompt = SEGMENTATION_PROMPT.format(
        gutter_prior_json=json.dumps(prior, ensure_ascii=False, indent=2)
    )

    cache_file = _cache_path(image_bytes, model_name)
    payload: dict
    if use_cache and cache_file.exists():
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            result.cache_hit = True
            result.latency_sec = 0.0
            result.raw = payload.get("raw_parsed") or payload
            logger.info("Phase 2 VLM cache hit: %s", cache_file.name)
        except (OSError, json.JSONDecodeError):
            payload = {}
            result.cache_hit = False
    else:
        payload = {}

    if not payload:
        try:
            vlm = generate_with_image(
                prompt,
                image_bytes,
                mime_type="image/jpeg",
                model=model_name,
                json_mode=True,
            )
        except Exception as exc:
            result.errors.append(f"VLM call failed: {exc}")
            logger.exception("VLM segmentation failed")
            return result

        result.latency_sec = vlm.latency_sec
        result.prompt_tokens = vlm.prompt_tokens
        result.candidate_tokens = vlm.candidate_tokens
        result.model = vlm.model
        try:
            parsed = parse_json_response(vlm.text)
        except Exception as exc:
            result.errors.append(f"VLM JSON parse failed: {exc}")
            result.raw = {"raw_text": vlm.text}
            return result

        payload = {
            "raw_parsed": parsed,
            "model": vlm.model,
            "prompt_version": PROMPT_VERSION,
            "latency_sec": vlm.latency_sec,
            "prompt_tokens": vlm.prompt_tokens,
            "candidate_tokens": vlm.candidate_tokens,
        }
        if use_cache:
            try:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except OSError as exc:
                logger.warning("Could not write VLM cache: %s", exc)

    parsed = payload.get("raw_parsed") or payload
    result.raw = parsed
    result.page_summary = str(parsed.get("page_summary") or "")
    result.n_columns = int(parsed.get("n_columns") or 1)
    result.reading_order_hint = str(parsed.get("reading_order_hint") or "single")
    result.override_gutter_prior = bool(parsed.get("override_gutter_prior"))
    if not result.cache_hit:
        result.latency_sec = float(payload.get("latency_sec") or result.latency_sec)
        result.prompt_tokens = int(payload.get("prompt_tokens") or 0)
        result.candidate_tokens = int(payload.get("candidate_tokens") or 0)

    questions: List[QuestionRegion] = []
    for idx, q in enumerate(parsed.get("questions") or []):
        try:
            qid = _normalize_question_id(
                str(q.get("question_id") or ""),
                str(q.get("label") or ""),
            )
            label = str(q.get("label") or f"שאלה {qid}")
            bbox = _norm_to_bbox(q.get("bbox_norm") or [0, 0, 1, 1], page_w, page_h)
            state = str(q.get("state") or "answered").lower()
            if state not in ("answered", "blank"):
                state = "answered" if q.get("has_student_ink", True) else "blank"
            conf = float(q.get("confidence") or 0.8)
            # Assign column index from bbox center vs layout (best effort)
            cx = bbox[0] + bbox[2] / 2.0
            col_index = 0
            for c in layout.columns:
                if c.x0 <= cx <= c.x1:
                    col_index = c.index
                    break
            questions.append(
                QuestionRegion(
                    question_id=qid,
                    label=label,
                    bbox=bbox,
                    column_index=col_index,
                    reading_order=idx,
                    ink_fraction=0.0,  # VLM path does not use CV ink fraction
                    state=state,
                    confidence=round(min(max(conf, 0.0), 1.0), 3),
                    is_subquestion=bool(q.get("is_subquestion")),
                )
            )
        except Exception as exc:
            result.errors.append(f"Bad question entry {idx}: {exc}")

    # De-dupe by question_id keeping highest confidence
    by_id: dict[str, QuestionRegion] = {}
    for q in questions:
        prev = by_id.get(q.question_id)
        if prev is None or q.confidence >= prev.confidence:
            by_id[q.question_id] = q
    result.questions = sorted(by_id.values(), key=lambda q: q.reading_order)
    # Re-number reading order after sort
    for i, q in enumerate(result.questions):
        q.reading_order = i

    result.tables = list(parsed.get("tables") or [])
    return result
