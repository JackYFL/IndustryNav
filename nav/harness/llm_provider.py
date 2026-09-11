"""LLM provider client.

Supports OpenRouter, direct OpenAI Responses, and direct Google Gemini.
OpenRouter and OpenAI can share a persistent request budget across workers;
every HTTP attempt (including retries) must reserve a slot before sending.

Provider entry points:

- :func:`call_openrouter` — the primitive. Returns raw response text on
  success, or a JSON error blob (``{"reasoning": "...", "error": True}``)
  on transport / decode failures. Use this when the caller does its own
  parsing (e.g. multi-step decision agents).
- :func:`llm_openrouter` — convenience wrapper for callers that want a
  parsed ``(action, reasoning)`` tuple. Equivalent to
  ``parse_json_action(call_openrouter(...), allowed_actions)``.
- :func:`call_gemini` — direct Google Gemini REST transport using
  ``GEMINI_API_KEY``.
- :func:`call_openai` — direct OpenAI Responses API transport using
  ``OPENAI_API_KEY``.
- :func:`llm_generate` — provider-neutral parsed action wrapper used by the
  benchmark harness.

The error-on-failure flow is by design: returning a JSON blob with an
``error`` flag means downstream :func:`parse_json_action` falls through
cleanly to ``("stop", error_message)`` — same observable behavior as the
pre-refactor code, with half the lines of duplication.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import anthropic
import numpy as np
import requests

from nav.config import LLM_DEFAULT_MAX_TOKENS, LLM_REQUEST_TIMEOUT_SEC
from nav.utils import data_url_png_from_rgb, parse_json_action


#: OpenRouter's OpenAI-compatible chat completions endpoint.
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
GEMINI_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent"
)
GEMINI_TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
GEMINI_MAX_REQUEST_ATTEMPTS = 4
_gemini_last_request_at = 0.0
OPENAI_TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
OPENAI_MAX_REQUEST_ATTEMPTS = 4

ANTHROPIC_MAX_REQUEST_ATTEMPTS = 4
_anthropic_last_request_at = 0.0
_openai_last_request_at = 0.0


def _check_api_key() -> str:
    """Return ``OPENROUTER_API_KEY`` or raise with a developer-friendly hint."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. Copy tmp/secrets.sh.example to "
            "tmp/secrets.sh, fill in your key, and source it before invoking "
            "the benchmark."
        )
    return api_key


def _check_gemini_api_key() -> str:
    """Return ``GEMINI_API_KEY`` or raise with a safe setup hint."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Export it before using the Gemini provider."
        )
    return api_key


def _check_openai_api_key() -> str:
    """Return ``OPENAI_API_KEY`` or raise with a safe setup hint."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Export it before using the OpenAI provider."
        )
    return api_key


def _build_image_parts(images: List[Tuple[str, np.ndarray]]) -> List[dict]:
    """Convert ``[(name, rgb_uint8), ...]`` to OpenRouter image content parts.

    The ``name`` argument is currently unused (OpenRouter doesn't surface
    a per-image label) but is preserved in the signature for parity with
    debug-printing callers that key off it.
    """
    return [
        {
            "type": "image_url",
            "image_url": {"url": data_url_png_from_rgb(rgb)},
        }
        for _name, rgb in images
    ]


def _error_blob(message: str) -> str:
    """Wrap an error message in the JSON shape ``parse_json_action`` understands."""
    return json.dumps({"reasoning": message, "error": True})


def _reserve_shared_request(provider: str) -> bool:
    """Reserve a request, optionally pacing all workers sharing the budget."""
    raw_limit = os.getenv(f"{provider}_MAX_REQUESTS", "").strip()
    interval = float(os.getenv(f"{provider}_MIN_REQUEST_INTERVAL_SEC", "0"))
    if not raw_limit:
        if interval > 0:
            raise ValueError(f"{provider} shared pacing requires a shared request budget")
        return True
    limit = int(raw_limit)
    if limit <= 0:
        return False

    counter_path = os.getenv(f"{provider}_REQUEST_COUNTER_FILE", "").strip()
    if not counter_path:
        raise ValueError(
            f"{provider}_REQUEST_COUNTER_FILE is required when "
            f"{provider}_MAX_REQUESTS is set"
        )
    path = Path(counter_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        with closing(sqlite3.connect(path, timeout=30, isolation_level=None)) as conn:
            conn.execute("PRAGMA busy_timeout = 30000")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS request_counter "
                "(id INTEGER PRIMARY KEY CHECK (id = 1), value INTEGER NOT NULL)"
            )
            if interval > 0:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS request_pacing "
                    "(id INTEGER PRIMARY KEY CHECK (id = 1), next_at REAL NOT NULL)"
                )
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("INSERT OR IGNORE INTO request_counter VALUES (1, 0)")
            current = int(conn.execute(
                "SELECT value FROM request_counter WHERE id = 1"
            ).fetchone()[0])
            if current >= limit:
                conn.execute("ROLLBACK")
                return False
            delay = 0.0
            now = time.time()
            if interval > 0:
                pacing = conn.execute("SELECT next_at FROM request_pacing WHERE id=1").fetchone()
                delay = max(0.0, pacing[0] - now) if pacing else 0.0
            if delay > 0:
                conn.execute("ROLLBACK")
            else:
                conn.execute("UPDATE request_counter SET value=? WHERE id=1", (current + 1,))
                if interval > 0:
                    conn.execute("INSERT OR REPLACE INTO request_pacing VALUES (1, ?)", (now + interval,))
                conn.execute("COMMIT")
                return True
        # Release the SQLite write lock while other workers wait for a slot.
        time.sleep(min(delay, 30.0))


def _reserve_openrouter_request() -> bool:
    return _reserve_shared_request("OPENROUTER")


def _reserve_openai_request() -> bool:
    return _reserve_shared_request("OPENAI")


def call_openrouter(
    prompt: str,
    images: List[Tuple[str, np.ndarray]],
    model: str,
    max_tokens: int = LLM_DEFAULT_MAX_TOKENS,
) -> str:
    """Call the OpenRouter chat API and return the raw assistant text.

    On any transport, HTTP, or decode failure, returns a JSON error blob
    (``{"reasoning": "<details>", "error": True}``) rather than raising —
    this keeps the synchronous benchmark loop from crashing mid-cell and
    lets :func:`nav.utils.parse_json_action` route it to a clean ``"stop"``.
    """
    api_key = _check_api_key()
    content: List[dict] = [{"type": "text", "text": prompt}]
    if images:
        content.extend(_build_image_parts(images))

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
    }
    reasoning_enabled = os.getenv("OPENROUTER_REASONING_ENABLED", "").strip().lower()
    if reasoning_enabled in {"0", "false", "no", "off"}:
        payload["reasoning"] = {"enabled": False}
    elif reasoning_enabled in {"1", "true", "yes", "on"}:
        payload["reasoning"] = {"enabled": True}
    json_mode = os.getenv("OPENROUTER_JSON_MODE", "").strip().lower()
    if json_mode in {"1", "true", "yes", "on"}:
        payload["response_format"] = {"type": "json_object"}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        attempts = max(1, min(3, int(os.getenv("OPENROUTER_MAX_REQUEST_ATTEMPTS", "1"))))
        for attempt in range(attempts):
            if not _reserve_openrouter_request():
                return _error_blob(
                    "API error: OpenRouter request limit reached before sending"
                )
            resp = requests.post(
                OPENROUTER_URL, headers=headers, json=payload, timeout=LLM_REQUEST_TIMEOUT_SEC
            )
            if resp.status_code not in {429, 500, 502, 503, 504} or attempt + 1 >= attempts:
                break
            delay = float(60 if resp.status_code == 429 else 2**attempt)
            retry_after = resp.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    delay = max(1.0, float(retry_after))
                except (TypeError, ValueError):
                    # An unsupported cooldown format should not trigger an early retry.
                    break
            if delay > 60:
                break
            logging.getLogger(__name__).warning(
                "OpenRouter HTTP %s; retry %s/%s in %.1fs (counted against request limit)",
                resp.status_code, attempt + 1, attempts - 1, delay,
            )
            time.sleep(delay)
        resp.raise_for_status()
        response_payload = resp.json()
        choice = response_payload["choices"][0]
        text = choice["message"].get("content")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "OpenRouter response contained no assistant content "
                f"(finish_reason={choice.get('finish_reason')})"
            )
        return text
    except requests.exceptions.RequestException as e:
        return _error_blob(f"API error: OpenRouter request failed: {e}")
    except (KeyError, IndexError, ValueError) as e:
        # KeyError: response shape mismatch. ValueError: JSON decode failure.
        return _error_blob(f"API error: unexpected OpenRouter response: {e}")
    except Exception as e:  # noqa: BLE001 — last-line defense for the benchmark loop
        return _error_blob(f"API error: OpenRouter failure: {e}")


def _gemini_model_id(model: str) -> str:
    """Convert an OpenRouter-style Google slug to the direct Gemini model id."""
    return model.split("/", 1)[1] if model.startswith("google/") else model


def _gemini_image_parts(images: List[Tuple[str, np.ndarray]]) -> List[dict]:
    """Encode RGB arrays as Gemini ``inline_data`` PNG parts."""
    parts = []
    for _name, rgb in images:
        data_url = data_url_png_from_rgb(rgb)
        header, encoded = data_url.split(",", 1)
        mime_type = header.removeprefix("data:").split(";", 1)[0]
        parts.append(
            {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": encoded,
                }
            }
        )
    return parts


def _wait_for_gemini_request_interval(min_interval_sec: float) -> None:
    """Rate-limit consecutive direct-Gemini calls within one cell process."""
    global _gemini_last_request_at
    interval = max(0.0, float(min_interval_sec))
    if interval > 0.0 and _gemini_last_request_at > 0.0:
        remaining = interval - (time.monotonic() - _gemini_last_request_at)
        if remaining > 0.0:
            time.sleep(remaining)
    _gemini_last_request_at = time.monotonic()


def _gemini_retry_delay(resp: requests.Response, attempt: int) -> float:
    """Return a bounded retry delay, honoring Google's retry metadata."""
    retry_after = resp.headers.get("Retry-After")
    if retry_after is not None:
        try:
            return max(0.0, min(float(retry_after), 60.0))
        except ValueError:
            pass
    if resp.status_code == 429:
        try:
            details = resp.json().get("error", {}).get("details", [])
            for detail in details:
                retry_delay = str(detail.get("retryDelay", ""))
                if retry_delay.endswith("s"):
                    return max(0.0, min(float(retry_delay[:-1]), 60.0))
        except (AttributeError, TypeError, ValueError):
            pass
        return float(15 * (attempt + 1))
    return float(2**attempt)


def call_gemini(
    prompt: str,
    images: List[Tuple[str, np.ndarray]],
    model: str,
    max_tokens: int = LLM_DEFAULT_MAX_TOKENS,
    min_request_interval_sec: float = 0.0,
) -> str:
    """Call Google Gemini directly and return the final response text."""
    api_key = _check_gemini_api_key()
    parts: List[dict] = [{"text": prompt}]
    parts.extend(_gemini_image_parts(images))
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"maxOutputTokens": max_tokens},
    }
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    url = GEMINI_URL_TEMPLATE.format(model=_gemini_model_id(model))

    try:
        resp = None
        for attempt in range(GEMINI_MAX_REQUEST_ATTEMPTS):
            try:
                _wait_for_gemini_request_interval(min_request_interval_sec)
                resp = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=LLM_REQUEST_TIMEOUT_SEC,
                )
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                if attempt + 1 >= GEMINI_MAX_REQUEST_ATTEMPTS:
                    raise
                time.sleep(2**attempt)
                continue

            if (
                resp.status_code in GEMINI_TRANSIENT_STATUS_CODES
                and attempt + 1 < GEMINI_MAX_REQUEST_ATTEMPTS
            ):
                time.sleep(_gemini_retry_delay(resp, attempt))
                continue
            break

        if resp is None:  # Defensive; the loop always assigns or raises.
            raise RuntimeError("Gemini request loop produced no response")
        resp.raise_for_status()
        response_parts = resp.json()["candidates"][0]["content"]["parts"]
        visible = [
            part["text"]
            for part in response_parts
            if part.get("text") and not part.get("thought", False)
        ]
        if not visible:
            visible = [part["text"] for part in response_parts if part.get("text")]
        if not visible:
            raise ValueError("Gemini response contained no text")
        return "".join(visible)
    except requests.exceptions.RequestException as e:
        return _error_blob(f"API error: Gemini request failed: {e}")
    except (KeyError, IndexError, ValueError) as e:
        return _error_blob(f"API error: unexpected Gemini response: {e}")
    except Exception as e:  # noqa: BLE001 — last-line defense for benchmark runs
        return _error_blob(f"API error: Gemini failure: {e}")


def _wait_for_openai_request_interval(min_interval_sec: float) -> None:
    """Rate-limit consecutive direct-OpenAI calls within one cell process."""
    global _openai_last_request_at
    interval = max(0.0, float(min_interval_sec))
    if interval > 0.0 and _openai_last_request_at > 0.0:
        remaining = interval - (time.monotonic() - _openai_last_request_at)
        if remaining > 0.0:
            time.sleep(remaining)
    _openai_last_request_at = time.monotonic()


def _openai_retry_delay(resp: requests.Response, attempt: int) -> float:
    """Return a bounded retry delay, honoring OpenAI's Retry-After header."""
    retry_after = resp.headers.get("Retry-After")
    if retry_after is not None:
        try:
            return max(0.0, min(float(retry_after), 60.0))
        except ValueError:
            pass
    return float(min(2**attempt, 60))


def _openai_response_text(payload: dict) -> str:
    """Extract visible assistant text from a Responses API payload."""
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text

    visible = []
    for item in payload.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                visible.append(content["text"])
    if not visible:
        raise ValueError("OpenAI response contained no output text")
    return "".join(visible)


def call_openai(
    prompt: str,
    images: List[Tuple[str, np.ndarray]],
    model: str,
    max_tokens: int = LLM_DEFAULT_MAX_TOKENS,
    min_request_interval_sec: float = 0.0,
) -> str:
    """Call the OpenAI Responses API and return the final response text."""
    api_key = _check_openai_api_key()
    content: List[dict] = [{"type": "input_text", "text": prompt}]
    content.extend(
        {
            "type": "input_image",
            "image_url": data_url_png_from_rgb(rgb),
        }
        for _name, rgb in images
    )
    payload = {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": max_tokens,
        "store": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = None
        attempts = max(1, min(OPENAI_MAX_REQUEST_ATTEMPTS, int(
            os.getenv("OPENAI_MAX_REQUEST_ATTEMPTS", str(OPENAI_MAX_REQUEST_ATTEMPTS))
        )))
        for attempt in range(attempts):
            try:
                _wait_for_openai_request_interval(min_request_interval_sec)
                if not _reserve_openai_request():
                    return _error_blob(
                        "API error: OpenAI request limit reached before sending"
                    )
                resp = requests.post(
                    OPENAI_RESPONSES_URL,
                    headers=headers,
                    json=payload,
                    timeout=LLM_REQUEST_TIMEOUT_SEC,
                    allow_redirects=False,
                )
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                if attempt + 1 >= attempts:
                    raise
                time.sleep(2**attempt)
                continue

            if (
                resp.status_code in OPENAI_TRANSIENT_STATUS_CODES
                and attempt + 1 < attempts
            ):
                time.sleep(_openai_retry_delay(resp, attempt))
                continue
            break

        if resp is None:  # Defensive; the loop always assigns or raises.
            raise RuntimeError("OpenAI request loop produced no response")
        resp.raise_for_status()
        return _openai_response_text(resp.json())
    except requests.exceptions.RequestException as e:
        return _error_blob(f"API error: OpenAI request failed: {e}")
    except (KeyError, IndexError, ValueError) as e:
        return _error_blob(f"API error: unexpected OpenAI response: {e}")
    except Exception as e:  # noqa: BLE001 — last-line defense for benchmark runs
        return _error_blob(f"API error: OpenAI failure: {e}")


def _check_anthropic_api_key() -> str:
    """Return ``ANTHROPIC_API_KEY`` or raise with a safe setup hint."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. Export it before using the Anthropic provider."
        )
    return api_key


def _reserve_anthropic_request() -> bool:
    return _reserve_shared_request("ANTHROPIC")


def _wait_for_anthropic_request_interval(min_interval_sec: float) -> None:
    """Pace direct Anthropic calls within one process (shared pacing is separate)."""
    global _anthropic_last_request_at
    if min_interval_sec <= 0:
        return
    remaining = _anthropic_last_request_at + min_interval_sec - time.time()
    if remaining > 0:
        time.sleep(remaining)
    _anthropic_last_request_at = time.time()


def _anthropic_image_blocks(images: List[Tuple[str, np.ndarray]]) -> List[dict]:
    """Convert ``[(name, rgb_uint8), ...]`` to Messages API base64 image blocks."""
    prefix = "data:image/png;base64,"
    blocks = []
    for _name, rgb in images:
        data_url = data_url_png_from_rgb(rgb)
        blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": data_url[len(prefix):],
            },
        })
    return blocks


def _anthropic_retry_delay(exc: "anthropic.APIStatusError", attempt: int) -> float:
    """Honor ``retry-after`` when present; otherwise back off exponentially."""
    try:
        header = exc.response.headers.get("retry-after")
    except AttributeError:
        header = None
    if header:
        try:
            return max(1.0, min(float(header), 60.0))
        except (TypeError, ValueError):
            pass
    return float(2**attempt)


def _anthropic_response_text(message) -> str:
    """Return the concatenated text blocks; surface refusals and empty replies."""
    if message.stop_reason == "refusal":
        details = getattr(message, "stop_details", None)
        category = getattr(details, "category", None) if details else None
        raise ValueError(f"Anthropic refusal (category={category})")
    text = "".join(
        block.text for block in message.content if getattr(block, "type", "") == "text"
    )
    if not text.strip():
        raise ValueError(
            f"Anthropic response contained no text (stop_reason={message.stop_reason})"
        )
    return text


def call_anthropic(
    prompt: str,
    images: List[Tuple[str, np.ndarray]],
    model: str,
    max_tokens: int = LLM_DEFAULT_MAX_TOKENS,
    min_request_interval_sec: float = 0.0,
) -> str:
    """Call the Anthropic Messages API directly and return the response text.

    Uses the official SDK with adaptive thinking (the default on current
    models) and streaming, which the SDK requires for large ``max_tokens``.
    Transport errors, rate limits, and 5xx responses are retried up to
    ``ANTHROPIC_MAX_REQUEST_ATTEMPTS`` times; every attempt consumes one slot
    of the optional shared request budget, mirroring the OpenAI provider.
    Failures return an error blob so the benchmark loop can decide whether to
    re-query or stop.
    """
    api_key = _check_anthropic_api_key()
    content: List[dict] = _anthropic_image_blocks(images)
    content.append({"type": "text", "text": prompt})
    attempts = max(1, min(ANTHROPIC_MAX_REQUEST_ATTEMPTS, int(
        os.getenv("ANTHROPIC_MAX_REQUEST_ATTEMPTS", str(ANTHROPIC_MAX_REQUEST_ATTEMPTS))
    )))
    # The SDK's own retries are disabled so that the shared budget and the
    # per-decision attempt count stay accurate.
    client = anthropic.Anthropic(
        api_key=api_key, max_retries=0, timeout=float(LLM_REQUEST_TIMEOUT_SEC),
    )
    try:
        for attempt in range(attempts):
            _wait_for_anthropic_request_interval(min_request_interval_sec)
            if not _reserve_anthropic_request():
                return _error_blob(
                    "API error: Anthropic request limit reached before sending"
                )
            try:
                with client.messages.stream(
                    model=model,
                    max_tokens=max_tokens,
                    thinking={"type": "adaptive"},
                    messages=[{"role": "user", "content": content}],
                ) as stream:
                    message = stream.get_final_message()
                usage = getattr(message, "usage", None)
                if usage is not None:
                    logging.getLogger(__name__).info(
                        "Anthropic usage: input=%s cache_read=%s output=%s stop=%s",
                        getattr(usage, "input_tokens", None),
                        getattr(usage, "cache_read_input_tokens", None),
                        getattr(usage, "output_tokens", None),
                        message.stop_reason,
                    )
                return _anthropic_response_text(message)
            except (anthropic.RateLimitError, anthropic.InternalServerError) as e:
                if attempt + 1 >= attempts:
                    return _error_blob(f"API error: Anthropic request failed: {e}")
                time.sleep(_anthropic_retry_delay(e, attempt))
            except anthropic.APIConnectionError as e:  # includes APITimeoutError
                if attempt + 1 >= attempts:
                    return _error_blob(f"API error: Anthropic request failed: {e}")
                time.sleep(float(2**attempt))
            except anthropic.APIStatusError as e:
                return _error_blob(f"API error: Anthropic request failed: {e}")
        raise RuntimeError("Anthropic request loop produced no response")
    except ValueError as e:
        return _error_blob(f"API error: unexpected Anthropic response: {e}")
    except Exception as e:  # noqa: BLE001 — last-line defense for benchmark runs
        return _error_blob(f"API error: Anthropic failure: {e}")


def llm_openrouter(
    prompt: str,
    images: List[Tuple[str, np.ndarray]],
    model: str,
    max_tokens: int = LLM_DEFAULT_MAX_TOKENS,
    allowed_actions: Optional[Iterable[str]] = None,
) -> Tuple[str, str]:
    """Call OpenRouter and parse the response into ``(action, reasoning)``.

    Convenience wrapper around :func:`call_openrouter` +
    :func:`nav.utils.parse_json_action`. Returns ``("stop", error_msg)`` on
    failure (preserving pre-refactor behavior).
    """
    text = call_openrouter(prompt, images, model, max_tokens=max_tokens)
    return parse_json_action(text, allowed_actions=allowed_actions or [])


def llm_generate(
    prompt: str,
    images: List[Tuple[str, np.ndarray]],
    model: str,
    provider: str = "openrouter",
    max_tokens: int = LLM_DEFAULT_MAX_TOKENS,
    min_request_interval_sec: float = 0.0,
    allowed_actions: Optional[Iterable[str]] = None,
) -> Tuple[str, str]:
    """Generate and parse one action through the selected LLM provider."""
    decision = llm_generate_decision(
        prompt, images, model, provider=provider, max_tokens=max_tokens,
        min_request_interval_sec=min_request_interval_sec,
        allowed_actions=allowed_actions,
    )
    return decision["action"], decision["reasoning"]


def parse_navigation_decision(text: str, allowed_actions: Iterable[str]) -> dict:
    """Parse the navigation contract without dropping visual memory or errors."""
    try:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise ValueError("missing JSON object")
        obj = json.loads(match.group(0))
        if not isinstance(obj, dict):
            raise ValueError("response must be an object")
        if obj.get("error"):
            return {"action": "stop", "reasoning": str(obj.get("reasoning", "API error")),
                    "observation": "", "error": True}
        action = obj.get("action")
        if not isinstance(action, str) or action.strip().lower() not in allowed_actions:
            raise ValueError("invalid navigation action")
        reasoning = obj.get("reasoning", "")
        observation = obj.get("observation", "")
        if not isinstance(reasoning, str) or not isinstance(observation, str):
            raise ValueError("reasoning and observation must be strings")
        return {"action": action.strip().lower(), "reasoning": reasoning.strip(),
                "observation": observation.strip(), "error": False}
    except (TypeError, ValueError) as exc:
        return {"action": "stop", "reasoning": f"Decision error: invalid response: {exc}",
                "observation": "", "error": True}


def llm_generate_decision(
    prompt: str,
    images: List[Tuple[str, np.ndarray]],
    model: str,
    provider: str = "openrouter",
    max_tokens: int = LLM_DEFAULT_MAX_TOKENS,
    min_request_interval_sec: float = 0.0,
    allowed_actions: Optional[Iterable[str]] = None,
) -> dict:
    """Return action, reasoning, observation, and explicit failure status."""
    if provider == "openrouter":
        text = call_openrouter(prompt, images, model, max_tokens=max_tokens)
    elif provider == "gemini":
        text = call_gemini(
            prompt,
            images,
            model,
            max_tokens=max_tokens,
            min_request_interval_sec=min_request_interval_sec,
        )
    elif provider == "openai":
        text = call_openai(
            prompt,
            images,
            model,
            max_tokens=max_tokens,
            min_request_interval_sec=min_request_interval_sec,
        )
    elif provider == "anthropic":
        text = call_anthropic(
            prompt,
            images,
            model,
            max_tokens=max_tokens,
            min_request_interval_sec=min_request_interval_sec,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
    decision = parse_navigation_decision(text, allowed_actions=allowed_actions or [])
    if not images:
        decision["observation"] = ""
    return decision
