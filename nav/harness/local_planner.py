"""
Local Planner Agent - Obstacle avoidance and immediate safety.
"""

import json
import re
import logging
from typing import List, Dict, Optional
import numpy as np

from nav.harness.types import LocalPlannerInput, LocalPlannerOutput
from nav.harness.prompt_assembly import get_local_prompt
from nav.harness.llm_provider import call_openrouter
from nav.config import LLM_SUBAGENT_MAX_TOKENS

logger = logging.getLogger(__name__)


class LocalPlannerAgent:
    """
    Local Planner: Analyzes egocentric view (and optionally BEV) to detect
    obstacles and recommend safe actions.
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        use_bev: bool = True,
        max_tokens: int = LLM_SUBAGENT_MAX_TOKENS,
    ):
        """
        Args:
            model: LLM model ID for OpenRouter
            use_bev: Whether to use BEV in addition to egocentric view
            max_tokens: Max tokens for LLM response
        """
        self.model = model
        self.use_bev = use_bev
        self.max_tokens = max_tokens
        self.prompt_template = get_local_prompt(use_bev)

        logger.info(f"LocalPlannerAgent initialized: model={model}, use_bev={use_bev}")

    def plan(self, input_data: LocalPlannerInput) -> LocalPlannerOutput:
        """
        Generate local obstacle avoidance recommendation.

        Args:
            input_data: LocalPlannerInput with egocentric (and optional BEV) view

        Returns:
            LocalPlannerOutput with safe/unsafe directions
        """
        # Prepare prompt
        prompt = self.prompt_template.format(
            heading=input_data.current_heading,
            target_bearing=input_data.target_bearing,
            target_distance=input_data.target_distance,
            allowed_actions=input_data.allowed_actions,
        )

        # Prepare images (egocentric is required, BEV is optional)
        images = [("egocentric", input_data.egocentric_view)]

        if self.use_bev and input_data.bev_view is not None:
            images.append(("bev", input_data.bev_view))

        # Call LLM
        try:
            response_text = call_openrouter(
                prompt=prompt,
                images=images,
                model=self.model,
                max_tokens=self.max_tokens,
            )

            # Parse output
            output = self._parse_output(response_text, input_data.allowed_actions)
            logger.info(
                f"[Local] Safe: {output.safe_directions}, "
                f"Unsafe: {output.unsafe_actions}, "
                f"Emergency: {output.emergency}"
            )
            return output

        except Exception as e:
            logger.error(f"[Local] Error during planning: {e}")
            return self._emergency_fallback(input_data.allowed_actions)

    def _parse_output(
        self, text: str, allowed_actions: List[str]
    ) -> LocalPlannerOutput:
        """Parse LLM JSON response into structured output."""
        import re
        import json

        # Try to extract JSON
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in response")

        json_str = match.group(0)

        # Try to parse
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}. Attempting to fix...")

            # Common fixes for LLM JSON errors
            # Fix 1: Missing commas in lists
            json_str = re.sub(r'"\s*\n\s*"', '", "', json_str)
            json_str = re.sub(r"}\s*\n\s*{", "}, {", json_str)

            # Fix 2: Trailing commas
            json_str = re.sub(r",\s*}", "}", json_str)
            json_str = re.sub(r",\s*]", "]", json_str)

            try:
                data = json.loads(json_str)
                logger.info("Successfully recovered from JSON error")
            except json.JSONDecodeError:
                raise ValueError(
                    f"Could not parse JSON even after fixes: {json_str[:200]}"
                )

        return LocalPlannerOutput(
            immediate_obstacles=data.get("immediate_obstacles", []),
            safe_directions=data.get("safe_directions", []),
            unsafe_actions=data.get("unsafe_actions", []),
            path_clearance=data.get("path_clearance", "unknown"),
            recommended_actions=data.get("recommended_actions", ["stop"]),
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", ""),
            emergency=bool(data.get("emergency", False)),
        )
