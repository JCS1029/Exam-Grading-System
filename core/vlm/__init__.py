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
from pathlib import Path
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv()
logger = logging.getLogger(__name__)

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
FAST_VLM_MODEL: str = os.getenv("FAST_VLM_MODEL", "gemini-3.6-flash").strip()
PRIMARY_VLM_MODEL: str = os.getenv("PRIMARY_VLM_MODEL", "gemini-3.1-pro-preview").strip()
VLM_TEMPERATURE: float = float(os.getenv("VLM_TEMPERATURE", "0"))
OPENROUTER_BASE: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
# Cap completion length — long sol-page dumps were burning budget
VLM_MAX_TOKENS: int = int(os.getenv("VLM_MAX_TOKENS", "4096"))
# Gemini-on-OpenRouter thinking: minimal | low | medium | high (minimal = cheapest)
VLM_THINKING_LEVEL: str = os.getenv("VLM_THINKING_LEVEL", "minimal").strip().lower()


@dataclass
class VLMResponse:
    text: str
    model: str
    latency_sec: float
    prompt_tokens: int = 0
    candidate_tokens: int = 0
    raw: Any = None
    cost_usd: float = 0.0          # provider-reported, when available
    provider_model: str = ""       # exact model id the provider actually served


class VLMError(RuntimeError):
    """Raised when a VLM call fails after retries."""


class VLMRequestError(RuntimeError):
    """Non-retryable request failure (bad request, auth, out of credits)."""


class BudgetExceeded(RuntimeError):
    """This process has spent its VLM_RUN_BUDGET_USD — stop, do not retry."""


# Spend control: the account has a hard credit cap and no live dashboard for
# collaborators, so every paid call is logged locally and capped per process.
SPEND_LEDGER: Path = Path(os.getenv("STORAGE_ROOT", "storage")) / "cache" / "spend_ledger.jsonl"
VLM_RUN_BUDGET_USD: float = float(os.getenv("VLM_RUN_BUDGET_USD", "1.0"))
_run_spend_usd: float = 0.0


def run_spend_usd() -> float:
    return _run_spend_usd


def _check_budget() -> None:
    if _run_spend_usd >= VLM_RUN_BUDGET_USD:
        raise BudgetExceeded(
            f"run spend ${_run_spend_usd:.4f} reached VLM_RUN_BUDGET_USD=${VLM_RUN_BUDGET_USD:.2f}"
        )


def _record_spend(resp: VLMResponse, purpose: str) -> None:
    global _run_spend_usd
    _run_spend_usd += resp.cost_usd
    entry = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "purpose": purpose,
        "model": resp.model,
        "provider_model": resp.provider_model,
        "prompt_tokens": resp.prompt_tokens,
        "completion_tokens": resp.candidate_tokens,
        "cost_usd": resp.cost_usd,
        "run_total_usd": round(_run_spend_usd, 6),
    }
    try:
        SPEND_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with SPEND_LEDGER.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except OSError as exc:
        logger.warning("Could not write spend ledger: %s", exc)


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


# Escapes that are always valid JSON (never LaTeX)
_JSON_ALWAYS = set('"\\/')


def _repair_invalid_json_escapes(text: str) -> str:
    """
    Fix LaTeX-style backslashes that break JSON (e.g. ``\\square`` → ``\\\\square``).

    VLMs often emit raw TeX inside JSON strings without doubling backslashes.
    Heuristic: keep ``\\"``, ``\\\\``, ``\\/``, ``\\uXXXX``; keep ``\\n``/``\\r``/``\\t``
    only when not starting a TeX command (next char after the escape letter is
    non-alphabetic). Everything else is doubled — including ``\\b`` in ``\\boxtimes``.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    in_string = False
    while i < n:
        ch = text[i]
        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
            i += 1
            continue

        if ch == '"':
            out.append(ch)
            in_string = False
            i += 1
            continue

        if ch == "\\":
            nxt = text[i + 1] if i + 1 < n else ""
            if nxt == "u" and i + 5 < n and all(
                c in "0123456789abcdefABCDEF" for c in text[i + 2 : i + 6]
            ):
                out.append(text[i : i + 6])
                i += 6
                continue
            if nxt in _JSON_ALWAYS:
                out.append(ch)
                out.append(nxt)
                i += 2
                continue
            # \n / \r / \t only when not a TeX command like \neq, \rightarrow
            if nxt in "nrt":
                after = text[i + 2] if i + 2 < n else ""
                if not after.isalpha():
                    out.append(ch)
                    out.append(nxt)
                    i += 2
                    continue
            # Invalid escape or TeX command — double the backslash
            out.append("\\\\")
            i += 1
            continue

        out.append(ch)
        i += 1
    return "".join(out)


def _close_truncated_json(text: str) -> str:
    """Best-effort close of truncated JSON objects/arrays/strings."""
    if not text or text[0] != "{":
        return text
    in_string = False
    escape = False
    stack: list[str] = []
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack and stack[-1] == ch:
            stack.pop()
    fixed = text
    if in_string:
        fixed += '"'
    while stack:
        fixed += stack.pop()
    return fixed


def _repair_unescaped_quotes(text: str) -> str:
    """
    Escape bare double-quotes inside JSON string values (e.g. Hebrew מהנ\"ל).

    A quote ends a string only when the next non-space char is , } ] or end.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    in_string = False
    escape = False
    while i < n:
        ch = text[i]
        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
            i += 1
            continue

        if escape:
            out.append(ch)
            escape = False
            i += 1
            continue

        if ch == "\\":
            out.append(ch)
            escape = True
            i += 1
            continue

        if ch == '"':
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            nxt = text[j] if j < n else ""
            if nxt in ",}]:" or nxt == "":
                out.append(ch)
                in_string = False
            else:
                out.append('\\"')
            i += 1
            continue

        out.append(ch)
        i += 1
    return "".join(out)


def parse_json_response(text: str) -> dict:
    """Parse model JSON, repairing fences, LaTeX escapes, quotes, truncation."""
    if not (text or "").strip():
        raise json.JSONDecodeError("Expecting value", text or "", 0)

    candidates = [_strip_json_fences(text)]
    match = re.search(r"\{[\s\S]*\}", candidates[0])
    if match:
        candidates.append(match.group(0))

    last_err: Exception | None = None
    tried: set[str] = set()
    for base in candidates:
        variants = [
            base,
            _repair_invalid_json_escapes(base),
            _repair_unescaped_quotes(base),
            _repair_invalid_json_escapes(_repair_unescaped_quotes(base)),
            _repair_unescaped_quotes(_repair_invalid_json_escapes(base)),
            _close_truncated_json(
                _repair_invalid_json_escapes(_repair_unescaped_quotes(base))
            ),
        ]
        for candidate in variants:
            if candidate in tried:
                continue
            tried.add(candidate)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as exc:
                last_err = exc
                continue

    assert last_err is not None
    raise last_err


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
    purpose: str = "",
) -> VLMResponse:
    """Send an image + text prompt to the configured VLM."""
    return _generate(
        prompt, image_bytes, mime_type=mime_type, model=model,
        temperature=temperature, json_mode=json_mode, purpose=purpose,
    )


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1.5, min=1, max=20),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError, VLMError)),
)
def generate_text(
    prompt: str,
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    json_mode: bool = True,
    purpose: str = "",
) -> VLMResponse:
    """Text-only call (grading transcripts against a rubric)."""
    return _generate(
        prompt, None, mime_type="", model=model,
        temperature=temperature, json_mode=json_mode, purpose=purpose,
    )


def _generate(
    prompt: str,
    image_bytes: Optional[bytes],
    *,
    mime_type: str,
    model: Optional[str],
    temperature: Optional[float],
    json_mode: bool,
    purpose: str,
) -> VLMResponse:
    if not GEMINI_API_KEY:
        raise VLMError("GEMINI_API_KEY is not set in .env")
    _check_budget()

    model_name = model or get_default_model(primary=False)
    temp = VLM_TEMPERATURE if temperature is None else temperature
    start = time.time()

    if _is_openrouter_key(GEMINI_API_KEY):
        resp = _call_openrouter(
            prompt=prompt,
            image_bytes=image_bytes,
            mime_type=mime_type,
            model=_openrouter_model_id(model_name),
            temperature=temp,
            json_mode=json_mode,
            start=start,
        )
    else:
        resp = _call_google_genai(
            prompt=prompt,
            image_bytes=image_bytes,
            mime_type=mime_type,
            model=model_name,
            temperature=temp,
            json_mode=json_mode,
            start=start,
        )
    _record_spend(resp, purpose)
    return resp


def _call_openrouter(
    *,
    prompt: str,
    image_bytes: Optional[bytes],
    mime_type: str,
    model: str,
    temperature: float,
    json_mode: bool,
    start: float,
) -> VLMResponse:
    content: list[dict[str, Any]] = []
    if image_bytes is not None:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}}
        )
    content.append({"type": "text", "text": prompt})
    payload: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": VLM_MAX_TOKENS,
        "messages": [{"role": "user", "content": content}],
        # Ask OpenRouter to report the billed cost on every response
        "usage": {"include": True},
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    # Reduce billed "thinking" tokens when the provider supports it
    if VLM_THINKING_LEVEL and VLM_THINKING_LEVEL != "off":
        payload["reasoning"] = {"effort": VLM_THINKING_LEVEL}

    with httpx.Client(timeout=180.0) as client:
        resp = client.post(
            f"{OPENROUTER_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {GEMINI_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://localhost/exam-grading-system",
                "X-Title": "Exam Grading System",
            },
            json=payload,
        )
        if resp.status_code in (429, 500, 502, 503, 504):
            raise VLMError(f"OpenRouter {resp.status_code}: {resp.text[:200]}")
        if 400 <= resp.status_code < 500:
            # 400 bad request / 401 auth / 402 out of credits: retrying cannot help
            raise VLMRequestError(f"OpenRouter {resp.status_code}: {resp.text[:300]}")
        resp.raise_for_status()
        data = resp.json()

    try:
        message = data["choices"][0]["message"]
        text = message.get("content") or ""
        # Some providers put JSON in refusal / reasoning fields when content is empty
        if not str(text).strip():
            text = message.get("reasoning") or message.get("refusal") or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise VLMError(f"Malformed OpenRouter response: {data!r}") from exc

    usage = data.get("usage") or {}
    return VLMResponse(
        text=str(text),
        model=model,
        latency_sec=round(time.time() - start, 2),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        candidate_tokens=int(usage.get("completion_tokens") or 0),
        raw=data,
        cost_usd=float(usage.get("cost") or 0.0),
        provider_model=str(data.get("model") or model),
    )


def _call_google_genai(
    *,
    prompt: str,
    image_bytes: Optional[bytes],
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

    contents: list[Any] = []
    if image_bytes is not None:
        contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
    contents.append(prompt)
    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
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
        provider_model=str(getattr(response, "model_version", "") or model),
    )
