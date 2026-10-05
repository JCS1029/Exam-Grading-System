"""
Transcribe Phase 2 question crops via VLM with caching and objective scoring.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import cv2

from core.transcription.confidence import compute_objective_confidence
from core.transcription.discovery import CropRef, discover_crops
from core.transcription.latex import collect_all_formulas, validate_latex_formulas
from core.transcription.prompt import PROMPT_VERSION, TRANSCRIPTION_PROMPT
from core.transcription.schema import TranscriptResult
from core.vlm import VLMError, generate_with_image, get_default_model, parse_json_response

logger = logging.getLogger(__name__)

STORAGE_ROOT = Path(os.getenv("STORAGE_ROOT", "storage"))
CACHE_DIR = STORAGE_ROOT / "cache" / "phase3_vlm"
TRANSCRIPTS_DIR = STORAGE_ROOT / "transcripts"
MAX_SEND_DIM = int(os.getenv("PHASE3_VLM_MAX_DIM", "1000"))
JPEG_QUALITY = int(os.getenv("PHASE3_VLM_JPEG_QUALITY", "75"))
USE_CACHE = os.getenv("PHASE3_VLM_USE_CACHE", "true").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
# Conservative default: one cheap model only (no dual bake-off)
DEFAULT_BAKEOFF_MODELS = [
    "gemini-3.6-flash",
]
RETRY_PLAINTEXT = os.getenv("PHASE3_RETRY_PLAINTEXT", "false").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def _image_hash(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()[:24]


def _encode_crop(image_path: Path) -> bytes:
    img = cv2.imread(str(image_path))
    if img is None:
        raise VLMError(f"Cannot read crop image: {image_path}")
    h, w = img.shape[:2]
    if max(h, w) > MAX_SEND_DIM:
        scale = MAX_SEND_DIM / float(max(h, w))
        img = cv2.resize(
            img,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise VLMError(f"JPEG encode failed: {image_path}")
    return buf.tobytes()


def _cache_path(image_bytes: bytes, model: str) -> Path:
    key = f"{PROMPT_VERSION}_{model}_{_image_hash(image_bytes)}"
    return CACHE_DIR / f"{key}.json"


def _build_prompt(crop: CropRef) -> str:
    return TRANSCRIPTION_PROMPT.format(
        booklet_id=crop.booklet_id,
        question_id=crop.question_id,
        question_label=crop.question_label,
    )


def _parse_transcript_payload(parsed: dict, crop: CropRef) -> TranscriptResult:
    is_blank = bool(parsed.get("is_blank"))
    hebrew = str(parsed.get("hebrew_text") or "").strip()
    math_latex = [str(x) for x in (parsed.get("math_latex") or []) if str(x).strip()]
    interleaved = str(parsed.get("interleaved_markdown") or "").strip()
    flagged = [str(x) for x in (parsed.get("flagged_tokens") or []) if str(x).strip()]
    strike = [str(x) for x in (parsed.get("strikethrough_regions") or []) if str(x).strip()]

    if not interleaved and (hebrew or math_latex):
        interleaved = hebrew
        if math_latex:
            interleaved = f"{hebrew}\n\n" + "\n".join(f"${m}$" for m in math_latex)

    formulas = collect_all_formulas(math_latex, interleaved)
    latex_eval = validate_latex_formulas(formulas)

    self_conf = parsed.get("transcription_confidence")
    try:
        self_conf_f = float(self_conf) if self_conf is not None else None
    except (TypeError, ValueError):
        self_conf_f = None

    has_content = bool(interleaved or hebrew or math_latex)
    obj_conf, signals = compute_objective_confidence(
        latex_eval=latex_eval,
        flagged_tokens=flagged,
        has_content=has_content,
        is_blank=is_blank,
        self_reported=self_conf_f,
    )

    return TranscriptResult(
        crop_id=crop.crop_id,
        booklet_id=crop.booklet_id,
        page_name=crop.page_name,
        question_id=crop.question_id,
        question_label=str(parsed.get("question_label") or crop.question_label),
        crop_path=str(crop.crop_path),
        state=crop.state,
        hebrew_text=hebrew,
        math_latex=math_latex,
        interleaved_markdown=interleaved,
        flagged_tokens=flagged,
        strikethrough_regions=strike,
        self_reported_confidence=self_conf_f,
        legibility=str(parsed.get("legibility") or ""),
        latex_eval=latex_eval,
        objective_confidence=obj_conf,
        confidence_signals=signals,
        raw=parsed,
    )


def transcribe_crop(
    crop: CropRef,
    *,
    model: Optional[str] = None,
    use_cache: bool = USE_CACHE,
) -> TranscriptResult:
    """Transcribe one question crop; uses tenacity retries inside core.vlm."""
    model_name = model or get_default_model(primary=False)
    result = TranscriptResult(
        crop_id=crop.crop_id,
        booklet_id=crop.booklet_id,
        page_name=crop.page_name,
        question_id=crop.question_id,
        question_label=crop.question_label,
        crop_path=str(crop.crop_path),
        state=crop.state,
        model=model_name,
        prompt_version=PROMPT_VERSION,
    )

    try:
        image_bytes = _encode_crop(crop.crop_path)
    except VLMError as exc:
        result.errors.append(str(exc))
        return result

    prompt = _build_prompt(crop)
    cache_file = _cache_path(image_bytes, model_name)
    payload: dict = {}

    if use_cache and cache_file.exists():
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            result.cache_hit = True
            logger.info("Phase 3 cache hit: %s", cache_file.name)
        except (OSError, json.JSONDecodeError):
            payload = {}

    if not payload:
        last_raw = ""
        parsed = None
        total_latency = 0.0
        total_prompt = 0
        total_completion = 0
        used_model = model_name

        # Conservative retries: 1 JSON call by default.
        # Opt-in: PHASE3_RETRY_PLAINTEXT=true, PHASE3_FALLBACK_MODEL=...
        fallback_model = os.getenv("PHASE3_FALLBACK_MODEL", "").strip()
        attempts: list[tuple[str, bool]] = [(model_name, True)]
        if RETRY_PLAINTEXT:
            attempts.append((model_name, False))
        if fallback_model and fallback_model != model_name:
            attempts.append((fallback_model, True))

        for attempt, (attempt_model, use_json) in enumerate(attempts):
            try:
                vlm = generate_with_image(
                    prompt,
                    image_bytes,
                    mime_type="image/jpeg",
                    model=attempt_model,
                    json_mode=use_json,
                )
            except Exception as exc:
                result.errors.append(f"VLM call failed after retries: {exc}")
                logger.exception("Transcription failed for %s", crop.crop_id)
                return result

            total_latency += vlm.latency_sec
            total_prompt += vlm.prompt_tokens
            total_completion += vlm.candidate_tokens
            used_model = vlm.model
            last_raw = vlm.text or ""

            if not last_raw.strip():
                logger.warning(
                    "Empty VLM response for %s (attempt %d, model=%s, json=%s)",
                    crop.crop_id,
                    attempt + 1,
                    attempt_model,
                    use_json,
                )
                continue
            try:
                parsed = parse_json_response(last_raw)
                break
            except Exception as exc:
                logger.warning(
                    "JSON parse failed for %s (attempt %d): %s",
                    crop.crop_id,
                    attempt + 1,
                    exc,
                )
                continue

        result.latency_sec = round(total_latency, 2)
        result.prompt_tokens = total_prompt
        result.candidate_tokens = total_completion
        # Keep the *requested* model for bake-off grouping; record actual provider id in raw.
        result.model = model_name

        if parsed is None:
            result.errors.append(
                "JSON parse failed after retries: empty or invalid response"
            )
            result.raw = {"raw_text": last_raw, "actual_model": used_model}
            return result

        payload = {
            "raw_parsed": parsed,
            "model": model_name,
            "actual_model": used_model,
            "prompt_version": PROMPT_VERSION,
            "latency_sec": result.latency_sec,
            "prompt_tokens": total_prompt,
            "candidate_tokens": total_completion,
        }
        if use_cache:
            try:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except OSError as exc:
                logger.warning("Could not write Phase 3 cache: %s", exc)
    else:
        result.latency_sec = float(payload.get("latency_sec") or 0.0)
        result.prompt_tokens = int(payload.get("prompt_tokens") or 0)
        result.candidate_tokens = int(payload.get("candidate_tokens") or 0)
        result.model = str(payload.get("model") or model_name)

    parsed = payload.get("raw_parsed") or payload
    filled = _parse_transcript_payload(parsed, crop)
    for field in (
        "hebrew_text",
        "math_latex",
        "interleaved_markdown",
        "flagged_tokens",
        "strikethrough_regions",
        "self_reported_confidence",
        "legibility",
        "latex_eval",
        "objective_confidence",
        "confidence_signals",
        "raw",
    ):
        setattr(result, field, getattr(filled, field))

    result.prompt_version = PROMPT_VERSION
    return result


def transcribe_from_manifest(
    manifest_path: Path,
    *,
    model: Optional[str] = None,
    states: Optional[List[str]] = None,
    use_cache: bool = USE_CACHE,
) -> List[TranscriptResult]:
    """Transcribe all eligible crops for one page manifest."""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    bid = str(data.get("booklet_id") or manifest_path.parent.parent.name)
    page_name = str(data.get("page_name") or "")
    page_stem = page_name.replace(".png", "") or manifest_path.parent.name
    allowed = set(states or ["answered"])
    results: List[TranscriptResult] = []

    for q in data.get("questions") or []:
        state = str(q.get("state") or "answered")
        if state not in allowed:
            continue
        crop_file = q.get("crop")
        if not crop_file:
            continue
        crop_path = manifest_path.parent / str(crop_file)
        if not crop_path.exists():
            continue
        qid = str(q.get("question_id") or "unknown")
        crop = CropRef(
            crop_id=f"{bid}/{page_stem}/q{qid}",
            booklet_id=bid,
            page_name=page_name,
            question_id=qid,
            question_label=str(q.get("label") or f"שאלה {qid}"),
            crop_path=crop_path,
            manifest_path=manifest_path,
            state=state,
        )
        results.append(transcribe_crop(crop, model=model, use_cache=use_cache))
    return results


def save_transcript(result: TranscriptResult, *, model_slug: str) -> Path:
    """Persist transcript JSON under storage/transcripts/{booklet}/{page}/."""
    out_dir = TRANSCRIPTS_DIR / result.booklet_id / result.page_name.replace(".png", "")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model_slug.replace("/", "_").replace(":", "_")
    out_path = out_dir / f"transcript_q{result.question_id}_{safe_model}.json"
    out_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def bakeoff_models() -> List[str]:
    """Models for Phase 3 bake-off from env or defaults."""
    raw = os.getenv("PHASE3_BAKEOFF_MODELS", "").strip()
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    return list(DEFAULT_BAKEOFF_MODELS)
