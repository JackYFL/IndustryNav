"""
Global Planner Agent - High-level navigation strategy.
"""

import json
import re
import logging
from typing import List, Tuple
import numpy as np

from nav.harness.types import GlobalPlannerInput, GlobalPlannerOutput
from nav.harness.prompt_assembly import get_global_prompt
from nav.harness.llm_provider import call_openrouter
from nav.config import LLM_SUBAGENT_MAX_TOKENS

logger = logging.getLogger(__name__)


class GlobalPlannerAgent:
    """
    Global Planner: Analyzes minimap (and optionally trajectory) to suggest
    high-level navigation direction toward target.
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        use_trajectory: bool = True,
        max_tokens: int = LLM_SUBAGENT_MAX_TOKENS,
    ):
        """
        Args:
            model: LLM model ID for OpenRouter
            use_trajectory: Whether to use trajectory history in planning
            max_tokens: Max tokens for LLM response
        """
        self.model = model
        self.use_trajectory = use_trajectory
        self.max_tokens = max_tokens
        self.prompt_template = get_global_prompt(use_trajectory)

        logger.info(
            f"GlobalPlannerAgent initialized: model={model}, use_trajectory={use_trajectory}"
        )

    def plan(self, input_data: GlobalPlannerInput) -> GlobalPlannerOutput:
        """
        Generate global navigation recommendation.

        Args:
            input_data: GlobalPlannerInput with minimap and optional trajectory

        Returns:
            GlobalPlannerOutput with strategic direction and recommendations
        """
        # Prepare prompt
        prompt = self.prompt_template.format(
            current_x=input_data.current_position[0],
            current_y=input_data.current_position[1],
            heading=input_data.agent_rotation,
            target_x=input_data.target_position[0],
            target_y=input_data.target_position[1],
            distance=input_data.distance_to_target,
            allowed_actions=input_data.allowed_actions,
        )

        # Prepare images
        images = [("minimap", input_data.minimap_annotated)]

        if self.use_trajectory and input_data.history_trajectory is not None:
            images.append(("trajectory", input_data.history_trajectory))

        # Call LLM
        try:
            response_text = call_openrouter(
                prompt=prompt,
                images=images,
                model=self.model,
                max_tokens=self.max_tokens,
            )

            # Parse output
            output = self._parse_output(response_text)
            logger.info(
                f"[Global] Direction: {output.strategic_direction}, "
                f"Actions: {output.recommended_actions}, "
                f"Confidence: {output.confidence:.2f}"
            )
            return output

        except Exception as e:
            logger.error(f"[Global] Error during planning: {e}")
            return self._fallback_output(input_data)

    def _parse_output(self, text: str) -> GlobalPlannerOutput:
        """Parse LLM JSON response into structured output."""
        # Extract JSON
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in response")

        data = json.loads(match.group(0))

        return GlobalPlannerOutput(
            strategic_direction=data.get("strategic_direction", "unknown"),
            recommended_actions=data.get("recommended_actions", ["forward"]),
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", ""),
        )

    def _fallback_output(self, input_data: GlobalPlannerInput) -> GlobalPlannerOutput:
        """Provide fallback output if LLM fails."""
        # Simple heuristic using the shared world-distance success threshold.
        if input_data.distance_to_target > input_data.reach_threshold_m:
            action = "forward"
            reason = "Fallback: Moving forward toward distant target"
        else:
            action = "stop"
            reason = "Fallback: Close to target, stopping"

        return GlobalPlannerOutput(
            strategic_direction="unknown",
            recommended_actions=[action],
            confidence=0.3,
            reasoning=reason,
        )
