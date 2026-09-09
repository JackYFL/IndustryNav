"""Tests for the in-episode LLM decision retry in ``execute_decision``."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from nav.harness import routing


def _payload(retries: int) -> dict:
    return {
        "prompt": "p",
        "images": [],
        "model_id": "test/model",
        "max_tokens": 10,
        "allowed_actions": ["forward", "stop"],
        "llm_decision_retries": retries,
    }


OK = {"action": "forward", "reasoning": "clear aisle", "observation": "", "error": False}
BAD = {"action": "stop", "reasoning": "API error: Read timed out.", "observation": "", "error": True}


class ExecuteDecisionRetryTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(routing.time, "sleep", lambda _s: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_recovers_after_transient_failures(self):
        with patch.object(routing, "llm_generate_decision", side_effect=[BAD, BAD, OK]) as gen:
            result = {}
            routing.execute_decision("llm", _payload(retries=2), result)
        self.assertEqual(gen.call_count, 3)
        self.assertEqual(result["action"], "forward")
        self.assertFalse(result.get("error"))
        self.assertEqual(result["decision_attempts"], 3)
        self.assertTrue(result["finished"])

    def test_surfaces_error_when_retries_exhausted(self):
        with patch.object(routing, "llm_generate_decision", side_effect=[BAD, BAD, BAD]) as gen:
            result = {}
            routing.execute_decision("llm", _payload(retries=2), result)
        self.assertEqual(gen.call_count, 3)
        self.assertTrue(result["error"])
        self.assertEqual(result["action"], "stop")
        self.assertEqual(result["decision_attempts"], 3)

    def test_zero_retries_restores_abort_on_first_failure(self):
        with patch.object(routing, "llm_generate_decision", side_effect=[BAD, OK]) as gen:
            result = {}
            routing.execute_decision("llm", _payload(retries=0), result)
        self.assertEqual(gen.call_count, 1)
        self.assertTrue(result["error"])

    def test_missing_payload_key_defaults_to_no_retry(self):
        payload = _payload(retries=0)
        del payload["llm_decision_retries"]
        with patch.object(routing, "llm_generate_decision", side_effect=[BAD, OK]) as gen:
            result = {}
            routing.execute_decision("llm", payload, result)
        self.assertEqual(gen.call_count, 1)
        self.assertTrue(result["error"])

    def test_success_does_not_retry(self):
        with patch.object(routing, "llm_generate_decision", side_effect=[OK, BAD]) as gen:
            result = {}
            routing.execute_decision("llm", _payload(retries=2), result)
        self.assertEqual(gen.call_count, 1)
        self.assertEqual(result["decision_attempts"], 1)
        self.assertFalse(result.get("error"))

    def test_error_sentinel_in_reasoning_triggers_retry(self):
        sentinel = {"action": "stop", "reasoning": "api error: something", "observation": "", "error": False}
        with patch.object(routing, "llm_generate_decision", side_effect=[sentinel, OK]) as gen:
            result = {}
            routing.execute_decision("llm", _payload(retries=1), result)
        self.assertEqual(gen.call_count, 2)
        self.assertEqual(result["action"], "forward")


if __name__ == "__main__":
    unittest.main()
