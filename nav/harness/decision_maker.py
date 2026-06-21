"""Decision Maker — synthesize global + local planner outputs into a final action.

Currently a pure rule-based synthesizer: goal-check → emergency override →
cardinal-to-egocentric global preference → local-safe fallback → stop. An
earlier JSON-parsing LLM decision path was never wired into ``decide()`` and
was removed from this module in the harness refactor (git history retains
it; the reference prompt still lives in ``prompt_assembly.DECISION_MAKER_PROMPT``).
The ``model`` / ``max_tokens`` constructor args are kept as reserved API for
a future learned decision policy.
"""

import logging

from nav.config import LLM_SUBAGENT_MAX_TOKENS
from nav.harness.types import DecisionMakerInput, DecisionMakerOutput
from nav.utils import convert_cardinal_actions

logger = logging.getLogger(__name__)


class DecisionMakerAgent:
    """
    Decision Maker: Synthesizes global and local plans to make final action choice.
    """

    def __init__(self, model: str = "openai/gpt-4o", max_tokens: int = LLM_SUBAGENT_MAX_TOKENS):
        """
        Args:
            model: LLM model ID (use stronger model for critical decisions)
            max_tokens: Max tokens for LLM response
        """
        self.model = model
        self.max_tokens = max_tokens

        logger.info(f"DecisionMakerAgent initialized: model={model}")

    def decide(self, input_data: DecisionMakerInput) -> DecisionMakerOutput:
        """
        Make final action decision with GLOBAL PRIORITY in non-emergency.
        """
        # Quick pre-checks before calling LLM

        # 1. Goal reached?
        if input_data.distance_to_target <= input_data.reach_threshold:
            logger.info("[Decision] Goal reached!")
            return DecisionMakerOutput(
                action="stop",
                reasoning="Goal reached: within threshold distance",
                confidence=1.0,
                is_emergency_stop=False,
                is_goal_reached=True,
                is_stuck=False,
            )

        # 2. Emergency override? ONLY use safe directions
        if input_data.local_plan.emergency:
            safe_action = (
                input_data.local_plan.safe_directions[0]
                if input_data.local_plan.safe_directions
                else "stop"
            )
            logger.warning(
                f"[Decision] EMERGENCY! Ignoring global, choosing {safe_action}"
            )
            return DecisionMakerOutput(
                action=safe_action,
                reasoning=f"EMERGENCY: {input_data.local_plan.reasoning}. Ignoring global planner for safety.",
                confidence=input_data.local_plan.confidence,
                is_emergency_stop=True,
                is_goal_reached=False,
                is_stuck=False,
            )

        # 3. NON-EMERGENCY: Convert cardinal to egocentric, then prioritize Global
        # Convert global's cardinal recommendations to egocentric actions
        rot_x, rot_y, rot_z = input_data.agent_rotation
        agent_heading = float(rot_y)

        global_cardinal = input_data.global_plan.recommended_actions
        global_egocentric = convert_cardinal_actions(global_cardinal, agent_heading)

        logger.info(
            f"[Decision] Converted global cardinal {global_cardinal} → "
            f"egocentric {global_egocentric} (heading={agent_heading:.1f}°)"
        )

        # Try each converted egocentric action in order
        for ego_action in global_egocentric:
            if ego_action not in input_data.local_plan.unsafe_actions:
                logger.info(
                    f"[Decision] Non-emergency: Global (converted) suggests '{ego_action}', "
                    f"not in unsafe list. CHOOSING IT."
                )
                return DecisionMakerOutput(
                    action=ego_action,
                    reasoning=(
                        f"Non-emergency. Global suggests cardinal {global_cardinal}, "
                        f"converted to egocentric '{ego_action}' (heading {agent_heading:.1f}°). "
                        f"Local confirms it's not unsafe. Following global strategy."
                    ),
                    confidence=input_data.global_plan.confidence,
                    is_emergency_stop=False,
                    is_goal_reached=False,
                    is_stuck=False,
                )

        # 4. All global actions are unsafe - fallback to local safe
        if input_data.local_plan.safe_directions:
            fallback_action = input_data.local_plan.safe_directions[0]
            logger.warning(
                f"[Decision] All global actions unsafe. Fallback to local safe: {fallback_action}"
            )
            return DecisionMakerOutput(
                action=fallback_action,
                reasoning=(
                    f"All global recommendations (converted: {global_egocentric}) "
                    f"are unsafe. Temporarily using local safe action '{fallback_action}' "
                    f"to avoid obstacles."
                ),
                confidence=input_data.local_plan.confidence,
                is_emergency_stop=False,
                is_goal_reached=False,
                is_stuck=False,
            )

        # 5. No safe actions at all - stop
        logger.error("[Decision] No safe actions available. Stopping.")
        return DecisionMakerOutput(
            action="stop",
            reasoning="No safe actions available from either planner. Stopping for safety.",
            confidence=0.0,
            is_emergency_stop=True,
            is_goal_reached=False,
            is_stuck=False,
        )
