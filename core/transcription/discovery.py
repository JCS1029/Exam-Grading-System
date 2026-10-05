"""Discover Phase 2 question crops eligible for Phase 3 transcription."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

STORAGE_ROOT = Path("storage")
CROPS_ROOT = STORAGE_ROOT / "crops"


@dataclass(frozen=True)
class CropRef:
    crop_id: str
    booklet_id: str
    page_name: str
    question_id: str
    question_label: str
    crop_path: Path
    manifest_path: Path
    state: str

    @property
    def page_stem(self) -> str:
        return self.page_name.replace(".png", "")


def _make_crop_id(booklet_id: str, page_stem: str, question_id: str) -> str:
    return f"{booklet_id}/{page_stem}/q{question_id}"


def discover_crops(
    *,
    booklet_ids: Optional[List[str]] = None,
    states: Optional[List[str]] = None,
    crops_root: Path = CROPS_ROOT,
) -> List[CropRef]:
    """
    Walk Phase 2 manifests and return crop references.

    Default states: answered only (blank crops rarely need transcription).
    """
    allowed_states = set(states or ["answered"])
    out: List[CropRef] = []

    if not crops_root.exists():
        return out

    for manifest_path in sorted(crops_root.glob("*/*/manifest.json")):
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        bid = str(data.get("booklet_id") or manifest_path.parent.parent.name)
        if booklet_ids and bid not in booklet_ids:
            continue

        page_name = str(data.get("page_name") or "")
        page_stem = page_name.replace(".png", "") or manifest_path.parent.name

        for q in data.get("questions") or []:
            state = str(q.get("state") or "answered")
            if state not in allowed_states:
                continue
            crop_file = q.get("crop")
            if not crop_file:
                continue
            crop_path = manifest_path.parent / str(crop_file)
            if not crop_path.exists():
                continue
            qid = str(q.get("question_id") or "unknown")
            label = str(q.get("label") or f"שאלה {qid}")
            out.append(
                CropRef(
                    crop_id=_make_crop_id(bid, page_stem, qid),
                    booklet_id=bid,
                    page_name=page_name,
                    question_id=qid,
                    question_label=label,
                    crop_path=crop_path,
                    manifest_path=manifest_path,
                    state=state,
                )
            )
    return out


def iter_manifests(crops_root: Path = CROPS_ROOT) -> Iterator[Path]:
    yield from sorted(crops_root.glob("*/*/manifest.json"))
