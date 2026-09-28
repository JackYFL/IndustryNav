"""DAgger behavior-policy mixing for PointGoal data collection."""

from __future__ import annotations

from nav.core.geometry import transformed_path_length_m


def decide_dagger(payload: dict) -> dict:
    """Query policy and expert, select behavior action, and return labels."""
    policy = payload["bc_controller"]
    expert = payload["astar_planner"]
    policy_action = policy.predict_action(
        ego_obs=payload["ego_obs"],
        depth_obs=payload["depth_obs"],
        curr_world_x=payload["curr_world_x"],
        curr_world_z=payload["curr_world_z"],
        curr_yaw_deg=payload["curr_yaw_deg"],
        target_world_x=payload["target_world_x"],
        target_world_z=payload["target_world_z"],
    )
    expert_action, expert_reasoning, expert_path = expert.decide(
        minimap_rgb=payload["minimap_rgb"],
        curr_xy=payload["curr_xy"],
        target_xy=payload["target_xy"],
        agent_theta=payload["agent_theta"],
        reach_m=payload["reach_m"],
        step=payload["step"],
        curr_world_xz=payload["curr_world_xz"],
        target_world_xz=payload.get("target_world_xz"),
        point_to_world=payload.get("point_to_world"),
    )
    beta = float(payload["dagger_beta"])
    use_expert = bool(payload["dagger_rng"].random() < beta)
    action = expert_action if use_expert else policy_action
    policy.observe_executed_action(action)
    expert.observe_executed_action(action)
    return {
        "action": action,
        "reasoning": (
            f"DAgger behavior={'expert' if use_expert else 'policy'}; "
            f"policy={policy_action}; expert={expert_action}."
        ),
        "dagger_step": int(payload["step"]),
        "dagger_beta": beta,
        "dagger_used_expert": use_expert,
        "dagger_policy_action": policy_action,
        "dagger_expert_action": expert_action,
        "dagger_expert_reasoning": expert_reasoning,
        "astar_path_length_m": transformed_path_length_m(
            expert_path, payload.get("point_to_world")
        ),
        "astar_path": [[int(point[0]), int(point[1])] for point in expert_path],
        "astar_step": int(payload["step"]),
        "astar_plan_revision": expert.plan_revision,
        "astar_plan_status": expert.last_plan_status,
        "astar_tracking": dict(expert.last_tracking_metrics),
    }
