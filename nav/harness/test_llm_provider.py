"""Regression tests for direct Gemini provider routing."""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import requests

from nav.harness.llm_provider import (
    _reserve_openai_request,
    _reserve_openrouter_request,
    call_anthropic,
    call_gemini,
    call_openai,
    call_openrouter,
    llm_generate,
)


class OpenRouterProviderTest(unittest.TestCase):
    @patch("nav.harness.llm_provider.time.sleep")
    @patch("nav.harness.llm_provider.requests.post")
    def test_retry_is_counted_and_cannot_exceed_shared_limit(self, post, sleep):
        limited = Mock(status_code=429, headers={"Retry-After": "12"})
        limited.raise_for_status.side_effect = requests.HTTPError("429 Too Many Requests")
        success = Mock(status_code=200)
        success.json.return_value = {"choices": [{"message": {"content": '{"action":"forward"}'}}]}
        for limit in (1, 2):
            with self.subTest(limit=limit), tempfile.TemporaryDirectory() as tmpdir:
                post.reset_mock()
                sleep.reset_mock()
                post.side_effect = [limited, success]
                counter = str(Path(tmpdir) / "requests.sqlite3")
                with patch.dict(os.environ, {
                    "OPENROUTER_API_KEY": "test-key",
                    "OPENROUTER_MAX_REQUESTS": str(limit),
                    "OPENROUTER_REQUEST_COUNTER_FILE": counter,
                    "OPENROUTER_MAX_REQUEST_ATTEMPTS": "3",
                    "OPENROUTER_MIN_REQUEST_INTERVAL_SEC": "0",
                }):
                    text = call_openrouter("navigate", [], "qwen/test")
                self.assertEqual(post.call_count, limit)
                sleep.assert_called_once_with(12.0)
                with sqlite3.connect(counter) as conn:
                    self.assertEqual(conn.execute("SELECT value FROM request_counter").fetchone()[0], limit)
                if limit == 1:
                    self.assertIn("request limit reached", text)
                else:
                    self.assertEqual(text, '{"action":"forward"}')

    @patch("nav.harness.llm_provider.time.sleep")
    @patch("nav.harness.llm_provider.time.time", side_effect=[100.0, 100.0, 105.0])
    def test_shared_pacing_waits_without_consuming_extra_requests(self, clock, sleep):
        with tempfile.TemporaryDirectory() as tmpdir:
            counter = str(Path(tmpdir) / "requests.sqlite3")
            with patch.dict(os.environ, {
                "OPENROUTER_MAX_REQUESTS": "2",
                "OPENROUTER_REQUEST_COUNTER_FILE": counter,
                "OPENROUTER_MIN_REQUEST_INTERVAL_SEC": "5",
            }):
                self.assertTrue(_reserve_openrouter_request())
                self.assertTrue(_reserve_openrouter_request())
                self.assertFalse(_reserve_openrouter_request())
            sleep.assert_called_once_with(5.0)
            with sqlite3.connect(counter) as conn:
                self.assertEqual(conn.execute("SELECT value FROM request_counter").fetchone()[0], 2)

    @patch("nav.harness.llm_provider.requests.post")
    def test_http_error_aborts_decision_instead_of_becoming_a_navigation_stop(self, post):
        from nav.harness.routing import execute_decision

        post.return_value = Mock(status_code=429, headers={})
        post.return_value.raise_for_status.side_effect = requests.HTTPError("429 Too Many Requests")
        result = {}
        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "test-key",
            "OPENROUTER_MAX_REQUEST_ATTEMPTS": "1",
        }):
            execute_decision("llm", {
                "prompt": "navigate", "images": [], "model_id": "qwen/test",
                "max_tokens": 500, "allowed_actions": ["forward", "stop"],
            }, result)
        self.assertTrue(result["error"])
        self.assertIn("API error:", result["reasoning"])

    @patch("nav.harness.llm_provider.requests.post")
    def test_optional_reasoning_and_json_mode(self, post: Mock) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": '{"action":"forward","reasoning":"clear"}'
                    },
                }
            ]
        }
        post.return_value = response

        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-key",
                "OPENROUTER_REASONING_ENABLED": "false",
                "OPENROUTER_JSON_MODE": "1",
            },
        ):
            text = call_openrouter("navigate", [], "deepseek/test", max_tokens=500)

        self.assertEqual(text, '{"action":"forward","reasoning":"clear"}')
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["reasoning"], {"enabled": False})
        self.assertEqual(payload["response_format"], {"type": "json_object"})

    @patch("nav.harness.llm_provider.requests.post")
    def test_empty_content_becomes_explicit_api_error(self, post: Mock) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": None},
                }
            ]
        }
        post.return_value = response

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            text = call_openrouter("navigate", [], "deepseek/test")

        self.assertIn('"error": true', text)
        self.assertIn("API error: unexpected OpenRouter response", text)

    @patch("nav.harness.llm_provider.requests.post")
    def test_cross_process_budget_blocks_requests_after_limit(self, post: Mock) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": '{"action":"stop"}'},
                }
            ]
        }
        post.return_value = response

        with tempfile.TemporaryDirectory() as tmpdir:
            counter = Path(tmpdir) / "requests.sqlite3"
            with patch.dict(
                os.environ,
                {
                    "OPENROUTER_API_KEY": "test-key",
                    "OPENROUTER_MAX_REQUESTS": "1",
                    "OPENROUTER_REQUEST_COUNTER_FILE": str(counter),
                },
            ):
                first = call_openrouter("navigate", [], "deepseek/test")
                second = call_openrouter("navigate", [], "deepseek/test")

        self.assertEqual(first, '{"action":"stop"}')
        self.assertIn("request limit reached", second)
        post.assert_called_once()


class GeminiProviderTest(unittest.TestCase):
    @patch("nav.harness.llm_provider.requests.post")
    def test_direct_gemini_multimodal_payload_and_response(self, post: Mock) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "hidden", "thought": True},
                            {
                                "text": '{"action":"forward","reasoning":"clear"}'
                            },
                        ]
                    }
                }
            ]
        }
        post.return_value = response
        image = np.zeros((2, 2, 3), dtype=np.uint8)

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            text = call_gemini(
                "navigate",
                [("ego", image)],
                "google/gemini-3.8-flash",
                max_tokens=123,
            )

        self.assertEqual(text, '{"action":"forward","reasoning":"clear"}')
        url = post.call_args.args[0]
        kwargs = post.call_args.kwargs
        self.assertTrue(url.endswith("/gemini-3.8-flash:generateContent"))
        self.assertEqual(kwargs["headers"]["x-goog-api-key"], "test-key")
        self.assertEqual(kwargs["json"]["generationConfig"]["maxOutputTokens"], 123)
        parts = kwargs["json"]["contents"][0]["parts"]
        self.assertEqual(parts[0], {"text": "navigate"})
        self.assertEqual(parts[1]["inline_data"]["mime_type"], "image/png")
        self.assertTrue(parts[1]["inline_data"]["data"])

    @patch("nav.harness.llm_provider.call_gemini")
    def test_provider_neutral_wrapper_parses_action(self, call: Mock) -> None:
        call.return_value = '{"action":"turn right","reasoning":"align"}'

        action, reasoning = llm_generate(
            "navigate",
            [],
            "gemini-3.8-flash",
            provider="gemini",
            allowed_actions=["forward", "turn right", "turn left", "stop"],
        )

        self.assertEqual(action, "turn right")
        self.assertEqual(reasoning, "align")

    @patch("nav.harness.llm_provider.time.sleep")
    @patch("nav.harness.llm_provider.requests.post")
    def test_transient_503_is_retried(self, post: Mock, sleep: Mock) -> None:
        unavailable = Mock(status_code=503, headers={})
        unavailable.raise_for_status.side_effect = AssertionError(
            "transient response should be retried before raise_for_status"
        )
        success = Mock(status_code=200, headers={})
        success.raise_for_status.return_value = None
        success.json.return_value = {
            "candidates": [
                {"content": {"parts": [{"text": '{"action":"stop"}'}]}}
            ]
        }
        post.side_effect = [unavailable, success]

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            text = call_gemini("stop", [], "gemini-3.8-flash")

        self.assertEqual(text, '{"action":"stop"}')
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(1)


class OpenAIProviderTest(unittest.TestCase):
    @patch("nav.harness.llm_provider.time.sleep")
    @patch("nav.harness.llm_provider.requests.post")
    def test_every_retry_reserves_a_slot_and_stops_at_limit(self, post, sleep):
        unavailable = Mock(status_code=503, headers={})
        success = Mock(status_code=200, headers={})
        success.json.return_value = {"output_text": '{"action":"forward"}'}
        for failure in (unavailable, requests.Timeout("timeout"), requests.ConnectionError("offline")):
            for limit in (1, 2):
                with self.subTest(failure=type(failure).__name__, limit=limit), tempfile.TemporaryDirectory() as tmpdir:
                    post.reset_mock()
                    post.side_effect = [failure, success]
                    counter = str(Path(tmpdir) / "requests.sqlite3")
                    with patch.dict(os.environ, {
                        "OPENAI_API_KEY": "test-key",
                        "OPENAI_MAX_REQUESTS": str(limit),
                        "OPENAI_REQUEST_COUNTER_FILE": counter,
                        "OPENAI_MIN_REQUEST_INTERVAL_SEC": "0",
                        "OPENAI_MAX_REQUEST_ATTEMPTS": "3",
                    }):
                        result = call_openai("navigate", [], "gpt-5-mini")
                        blocked = call_openai("navigate", [], "gpt-5-mini")
                    self.assertEqual(post.call_count, limit)
                    self.assertIn("request limit reached", blocked)
                    with sqlite3.connect(counter) as conn:
                        self.assertEqual(conn.execute("SELECT value FROM request_counter").fetchone()[0], limit)
                    if limit == 1:
                        self.assertIn("request limit reached", result)
                    else:
                        self.assertEqual(result, '{"action":"forward"}')

    @patch("nav.harness.llm_provider.requests.post")
    def test_budget_misconfiguration_fails_closed(self, post):
        for limit in ("0", "invalid", "2"):
            with self.subTest(limit=limit), patch.dict(os.environ, {
                "OPENAI_API_KEY": "test-key", "OPENAI_MAX_REQUESTS": limit,
                "OPENAI_REQUEST_COUNTER_FILE": "", "OPENAI_MIN_REQUEST_INTERVAL_SEC": "0",
            }):
                self.assertIn('"error": true', call_openai("navigate", [], "gpt-5.2"))
        post.assert_not_called()

    @patch("nav.harness.llm_provider.time.sleep")
    @patch("nav.harness.llm_provider.time.time", side_effect=[100.0, 100.0, 105.0])
    def test_openai_shared_pacing(self, clock, sleep):
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "OPENAI_MAX_REQUESTS": "2",
            "OPENAI_REQUEST_COUNTER_FILE": str(Path(tmpdir) / "requests.sqlite3"),
            "OPENAI_MIN_REQUEST_INTERVAL_SEC": "5",
        }):
            self.assertTrue(_reserve_openai_request())
            self.assertTrue(_reserve_openai_request())
            self.assertFalse(_reserve_openai_request())
        sleep.assert_called_once_with(5.0)

    def test_independent_model_counters(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "OPENAI_MAX_REQUESTS": "1", "OPENAI_MIN_REQUEST_INTERVAL_SEC": "0",
        }):
            for model in ("gpt-5-mini", "gpt-5.2"):
                with patch.dict(os.environ, {
                    "OPENAI_REQUEST_COUNTER_FILE": str(Path(tmpdir) / (model + ".sqlite3")),
                }):
                    self.assertTrue(_reserve_openai_request())
                    self.assertFalse(_reserve_openai_request())

    def test_concurrent_processes_cannot_overspend_or_reset_counter(self):
        code = (
            "from nav.harness.llm_provider import _reserve_openai_request; "
            "print(sum(_reserve_openai_request() for _ in range(10)))"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            env = {**os.environ, "OPENAI_MAX_REQUESTS": "7",
                   "OPENAI_REQUEST_COUNTER_FILE": str(Path(tmpdir) / "requests.sqlite3"),
                   "OPENAI_MIN_REQUEST_INTERVAL_SEC": "0"}
            workers = [subprocess.Popen([sys.executable, "-c", code], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(4)]
            total = 0
            for worker in workers:
                output, errors = worker.communicate(timeout=30)
                self.assertEqual(worker.returncode, 0, errors)
                total += int(output.strip())
            self.assertEqual(total, 7)
            restarted = subprocess.run([sys.executable, "-c", code], env=env,
                                       capture_output=True, text=True, check=True, timeout=30)
            self.assertEqual(restarted.stdout.strip(), "0")

    @patch("nav.harness.llm_provider.requests.post")
    def test_openai_multimodal_payload_and_response(self, post: Mock) -> None:
        response = Mock(status_code=200, headers={})
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"action":"forward","reasoning":"clear"}',
                        }
                    ],
                }
            ]
        }
        post.return_value = response
        image = np.zeros((2, 2, 3), dtype=np.uint8)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            text = call_openai(
                "navigate",
                [("ego", image)],
                "gpt-5.6-sol",
                max_tokens=123,
            )

        self.assertEqual(text, '{"action":"forward","reasoning":"clear"}')
        self.assertEqual(post.call_args.args[0], "https://api.openai.com/v1/responses")
        kwargs = post.call_args.kwargs
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(kwargs["json"]["model"], "gpt-5.6-sol")
        self.assertEqual(kwargs["json"]["max_output_tokens"], 123)
        self.assertFalse(kwargs["json"]["store"])
        self.assertFalse(kwargs["allow_redirects"])
        self.assertNotIn("temperature", kwargs["json"])
        self.assertNotIn("reasoning", kwargs["json"])
        parts = kwargs["json"]["input"][0]["content"]
        self.assertEqual(parts[0], {"type": "input_text", "text": "navigate"})
        self.assertEqual(parts[1]["type"], "input_image")
        self.assertTrue(parts[1]["image_url"].startswith("data:image/png;base64,"))

    @patch("nav.harness.llm_provider.call_openai")
    def test_provider_neutral_wrapper_parses_openai_action(self, call: Mock) -> None:
        call.return_value = '{"action":"turn left","reasoning":"avoid"}'

        action, reasoning = llm_generate(
            "navigate",
            [],
            "gpt-5.6-sol",
            provider="openai",
            allowed_actions=["forward", "turn right", "turn left", "stop"],
        )

        self.assertEqual(action, "turn left")
        self.assertEqual(reasoning, "avoid")


def _anthropic_message(text, stop_reason="end_turn"):
    message = Mock(stop_reason=stop_reason, stop_details=None)
    message.content = [Mock(type="thinking", thinking=""), Mock(type="text", text=text)]
    message.usage = Mock(input_tokens=1, cache_read_input_tokens=0, output_tokens=1)
    return message


def _anthropic_client(messages):
    """Build a mocked SDK client whose ``messages.stream`` yields ``messages`` in order."""
    import anthropic as anthropic_sdk

    client = Mock()
    outcomes = list(messages)

    def stream(**kwargs):
        outcome = outcomes.pop(0)
        cm = Mock()
        if isinstance(outcome, Exception):
            cm.__enter__ = Mock(side_effect=outcome)
        else:
            cm.__enter__ = Mock(return_value=Mock(get_final_message=Mock(return_value=outcome)))
        cm.__exit__ = Mock(return_value=False)
        return cm

    client.messages.stream.side_effect = stream
    client.sdk = anthropic_sdk
    return client


class AnthropicProviderTest(unittest.TestCase):
    @patch("nav.harness.llm_provider.anthropic.Anthropic")
    def test_multimodal_payload_and_response(self, ctor: Mock) -> None:
        client = _anthropic_client([_anthropic_message('{"action":"forward","reasoning":"clear"}')])
        ctor.return_value = client
        image = np.zeros((2, 2, 3), dtype=np.uint8)

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            text = call_anthropic("navigate", [("ego", image)], "claude-test-model", max_tokens=12345)

        self.assertEqual(text, '{"action":"forward","reasoning":"clear"}')
        self.assertEqual(ctor.call_args.kwargs["api_key"], "test-key")
        self.assertEqual(ctor.call_args.kwargs["max_retries"], 0)
        kwargs = client.messages.stream.call_args.kwargs
        self.assertEqual(kwargs["model"], "claude-test-model")
        self.assertEqual(kwargs["max_tokens"], 12345)
        self.assertEqual(kwargs["thinking"], {"type": "adaptive"})
        self.assertNotIn("temperature", kwargs)
        content = kwargs["messages"][0]["content"]
        self.assertEqual(kwargs["messages"][0]["role"], "user")
        self.assertEqual(content[0]["type"], "image")
        self.assertEqual(content[0]["source"]["media_type"], "image/png")
        self.assertFalse(content[0]["source"]["data"].startswith("data:"))
        self.assertEqual(content[-1], {"type": "text", "text": "navigate"})

    @patch("nav.harness.llm_provider.anthropic.Anthropic")
    def test_refusal_and_empty_reply_become_error_blobs(self, ctor: Mock) -> None:
        import json as _json

        for message in (_anthropic_message("", stop_reason="refusal"),
                        _anthropic_message("   ", stop_reason="max_tokens")):
            ctor.return_value = _anthropic_client([message])
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
                text = call_anthropic("navigate", [], "claude-test-model")
            blob = _json.loads(text)
            self.assertTrue(blob["error"])
            self.assertIn("unexpected Anthropic response", blob["reasoning"])

    @patch("nav.harness.llm_provider.time.sleep")
    @patch("nav.harness.llm_provider.anthropic.Anthropic")
    def test_rate_limit_is_retried_then_succeeds(self, ctor: Mock, sleep: Mock) -> None:
        import anthropic as anthropic_sdk

        limited = anthropic_sdk.RateLimitError(
            "429", response=Mock(status_code=429, headers={"retry-after": "3"}), body=None,
        )
        ctor.return_value = _anthropic_client([limited, _anthropic_message('{"action":"stop"}')])
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            text = call_anthropic("navigate", [], "claude-test-model")
        self.assertEqual(text, '{"action":"stop"}')
        self.assertEqual(ctor.return_value.messages.stream.call_count, 2)
        sleep.assert_called_once_with(3.0)

    @patch("nav.harness.llm_provider.call_anthropic")
    def test_provider_neutral_wrapper_parses_anthropic_action(self, call: Mock) -> None:
        call.return_value = '{"action":"turn left","reasoning":"avoid"}'
        action, reasoning = llm_generate(
            "navigate", [], "claude-test-model", provider="anthropic",
            allowed_actions=["forward", "turn left", "turn right", "stop"],
        )
        self.assertEqual(action, "turn left")
        self.assertEqual(reasoning, "avoid")


if __name__ == "__main__":
    unittest.main()
