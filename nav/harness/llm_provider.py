"""LLM provider client.

Only OpenRouter is wired up today — it proxies OpenAI, Anthropic, and Google
models behind a single API and a single key, so a separate provider per
model vendor is unnecessary for now. If/when we add a direct-API path
(e.g. anthropic-sdk for ``claude-...``), introduce a ``Protocol`` and have
each backend implement it; for one concrete impl, a Protocol would be
premature abstraction.

Two functions:

- :func:`call_openrouter` — the primitive. Returns raw response text on
  success, or a JSON error blob (``{"reasoning": "...", "error": True}``)
  on transport / decode failures. Use this when the caller does its own
  parsing (e.g. multi-step decision agents).
- :func:`llm_openrouter` — convenience wrapper for callers that want a
  parsed ``(action, reasoning)`` tuple. Equivalent to
  ``parse_json_action(call_openrouter(...), allowed_actions)``.

The error-on-failure flow is by design: returning a JSON blob with an
``error`` flag means downstream :func:`parse_json_action` falls through
cleanly to ``("stop", error_message)`` — same observable behavior as the
pre-refactor code, with half the lines of duplication.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, List, Optional, Tuple

import numpy as np
import requests

from nav.config import LLM_DEFAULT_MAX_TOKENS, LLM_REQUEST_TIMEOUT_SEC
from nav.utils import data_url_png_from_rgb, parse_json_action


#: OpenRouter's OpenAI-compatible chat completions endpoint.
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _check_api_key() -> str:
    """Return ``OPENROUTER_API_KEY`` or raise with a developer-friendly hint."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. Source tmp/secrets.sh "
            "(see tmp/secrets.sh.example) before invoking the benchmark."
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
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(
            OPENROUTER_URL, headers=headers, json=payload, timeout=LLM_REQUEST_TIMEOUT_SEC
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return _error_blob(f"API request failed: {e}")
    except (KeyError, ValueError) as e:
        # KeyError: response shape mismatch. ValueError: JSON decode failure.
        return _error_blob(f"Unexpected response format: {e}")
    except Exception as e:  # noqa: BLE001 — last-line defense for the benchmark loop
        return _error_blob(f"Error: {e}")


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
