"""
Shared VLM client for layout / transcription.

Supports:
  - Google GenAI (native GEMINI_API_KEY from aistudio)
  - OpenRouter (keys starting with ``sk-or-``) via OpenAI-compatible chat API

Model IDs in ``.env`` may be written as ``gemini-3.5-flash``; OpenRouter
automatically maps them to ``google/gemini-3.5-flash``.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv()
logger = logging.getLogger(__name__)

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
FAST_VLM_MODEL: str = os.getenv("FAST_VLM_MODEL", "gemini-3.5-flash").strip()
PRIMARY_VLM_MODEL: str = os.getenv("PRIMARY_VLM_MODEL", "gemini-3.1-pro-preview").strip()
VLM_TEMPERATURE: float = float(os.getenv("VLM_TEMPERATURE", "0"))
OPENROUTER_BASE: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")


@dataclass
class VLMResponse:
    text: str
    model: str
    latency_sec: float
    prompt_tokens: int = 0
    candidate_tokens: int = 0
    raw: Any = None


class VLMError(RuntimeError):
    """Raised when a VLM call fails after retries."""


def _is_openrouter_key(key: str) -> bool:
    return key.startswith("sk-or-")


def _openrouter_model_id(model: str) -> str:
    if "/" in model:
        return model
    if model.startswith("gemini-"):
        return f"google/{model}"
    return model


def _strip_json_fences(text: str) -> str:
    clean = text.strip()
    if clean.startswith("```json"):
        clean = clean[7:]
    elif clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    return clean.strip()


def parse_json_response(text: str) -> dict:
    """Parse model JSON, with a light repair pass for common fence / trailing junk."""
    clean = _strip_json_fences(text)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", clean)
        if not match:
            raise
        return json.loads(match.group(0))


def get_default_model(*, primary: bool = False) -> str:
    return PRIMARY_VLM_MODEL if primary else FAST_VLM_MODEL


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1.5, min=1, max=20),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError, VLMError)),
)
def generate_with_image(
    prompt: str,
    image_bytes: bytes,
    *,
    mime_type: str = "image/png",
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    json_mode: bool = True,
) -> VLMResponse:
    """Send an image + text prompt to the configured VLM."""
    if not GEMINI_API_KEY:
        raise VLMError("GEMINI_API_KEY is not set in .env")

    model_name = model or get_default_model(primary=False)
    temp = VLM_TEMPERATURE if temperature is None else temperature
    start = time.time()

    if _is_openrouter_key(GEMINI_API_KEY):
        return _call_openrouter(
            prompt=prompt,
            image_bytes=image_bytes,
            mime_type=mime_type,
            model=_openrouter_model_id(model_name),
            temperature=temp,
            json_mode=json_mode,
            start=start,
        )

    return _call_google_genai(
        prompt=prompt,
        image_bytes=image_bytes,
        mime_type=mime_type,
        model=model_name,
        temperature=temp,
        json_mode=json_mode,
        start=start,
    )


def _call_openrouter(
    *,
    prompt: str,
    image_bytes: bytes,
    mime_type: str,
    model: str,
    temperature: float,
    json_mode: bool,
    start: float,
) -> VLMResponse:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{mime_type};base64,{b64}"
    payload: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    with httpx.Client(timeout=180.0) as client:
        resp = client.post(
            f"{OPENROUTER_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {GEMINI_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://localhost/exam-grading-system",
                "X-Title": "Exam Grading System Phase 2",
            },
            json=payload,
        )
        if resp.status_code in (429, 500, 502, 503, 504):
            raise VLMError(f"OpenRouter {resp.status_code}: {resp.text[:200]}")
        resp.raise_for_status()
        data = resp.json()

    try:
        text = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise VLMError(f"Malformed OpenRouter response: {data!r}") from exc

    usage = data.get("usage") or {}
    return VLMResponse(
        text=text,
        model=model,
        latency_sec=round(time.time() - start, 2),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        candidate_tokens=int(usage.get("completion_tokens") or 0),
        raw=data,
    )


def _call_google_genai(
    *,
    prompt: str,
    image_bytes: bytes,
    mime_type: str,
    model: str,
    temperature: float,
    json_mode: bool,
    start: float,
) -> VLMResponse:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    config_kwargs: dict[str, Any] = {"temperature": temperature}
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"

    try:
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                prompt,
            ],
            config=types.GenerateContentConfig(**config_kwargs),
        )
    except Exception as exc:
        msg = str(exc)
        if any(code in msg for code in ("429", "503", "500", "UNAVAILABLE", "RESOURCE_EXHAUSTED")):
            raise VLMError(msg) from exc
        raise

    text = (response.text or "").strip()
    usage = getattr(response, "usage_metadata", None)
    return VLMResponse(
        text=text,
        model=model,
        latency_sec=round(time.time() - start, 2),
        prompt_tokens=int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0,
        candidate_tokens=int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0,
        raw=response,
    )
