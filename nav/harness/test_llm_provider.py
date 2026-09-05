"""Regression tests for direct Gemini provider routing."""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import requests

from nav.harness.llm_provider import (
    _reserve_openrouter_request,
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


if __name__ == "__main__":
    unittest.main()
