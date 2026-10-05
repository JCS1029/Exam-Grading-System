"""
Phase 3 — VLM transcription on bounded question crops.

Public API:
  transcribe_crop          — single crop image → canonical transcript
  transcribe_from_manifest — all answered questions on a page manifest
  discover_crops           — enumerate crops from Phase 2 manifests
"""

from core.transcription.confidence import compute_objective_confidence
from core.transcription.discovery import CropRef, discover_crops
from core.transcription.latex import validate_latex_formulas
from core.transcription.metrics import aggregate_gate_metrics, hebrew_cer
from core.transcription.schema import TranscriptResult
from core.transcription.transcriber import transcribe_crop, transcribe_from_manifest

__all__ = [
    "CropRef",
    "TranscriptResult",
    "aggregate_gate_metrics",
    "compute_objective_confidence",
    "discover_crops",
    "hebrew_cer",
    "transcribe_crop",
    "transcribe_from_manifest",
    "validate_latex_formulas",
]
