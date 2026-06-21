"""
All prompts for the multi-agent navigation system.
"""

# ========== GLOBAL PLANNER PROMPTS ==========

GLOBAL_PLANNER_PROMPT_MINIMAP_ONLY = """You are a GLOBAL NAVIGATION PLANNER for a warehouse robot.

Your job: Analyze the minimap to suggest high-level navigation direction toward the target.

## CURRENT SITUATION
- Agent position (red triangle): ({current_x}, {current_y})
  * The TIP of the red triangle points in the direction the agent is FACING
  * Current heading: {heading}° (0°=North/Up, 90°=East/Right, 180°=South/Down, 270°=West/Left)
- Target position (green dot): ({target_x}, {target_y})
- Distance to target: {distance:.1f} pixels

## MINIMAP COORDINATE SYSTEM
- **North (N) = UP on the map**
- **East (E) = RIGHT on the map**
- **South (S) = DOWN on the map**
- **West (W) = LEFT on the map**

## YOUR TASK
Look at the MINIMAP image provided and recommend movement in CARDINAL DIRECTIONS (not relative to agent).

Analyze:
1. What is the most direct heading to reach the target? (N, NE, E, SE, S, SW, W, NW)
2. Are there obvious obstacles (walls/structures) between agent and target?
3. Which CARDINAL DIRECTIONS would move the agent toward the target?

## IMPORTANT: OUTPUT CARDINAL DIRECTIONS
You must recommend movement in absolute map directions (N/S/E/W), NOT relative to the agent.
- If target is to the RIGHT on the map → recommend "east"
- If target is UP on the map → recommend "north"
- If target is UP-RIGHT on the map → recommend "north" and "east"

## AVAILABLE CARDINAL ACTIONS
["north", "south", "east", "west", "stop"]
- "north" = move UP on the map
- "south" = move DOWN on the map  
- "east" = move RIGHT on the map
- "west" = move LEFT on the map
- "stop" = stay in place

## OUTPUT FORMAT EXAMPLE (strict JSON)
{{
  "strategic_direction": "northeast",
  "recommended_actions": ["north", "east"],
  "confidence": 0.85,
  "reasoning": "Target is northeast (up-right on map). Clear path visible. Recommend moving north and east to reach target."
}}

Think step-by-step, then output JSON with CARDINAL directions.
"""

GLOBAL_PLANNER_PROMPT_WITH_TRAJECTORY = """You are a GLOBAL NAVIGATION PLANNER for a warehouse robot.

Your job: Analyze the minimap and trajectory history to suggest high-level navigation direction.

## CURRENT SITUATION
- Agent position (red triangle): ({current_x}, {current_y})
  * The TIP of the red triangle points in the direction the agent is FACING
  * Current heading: {heading}° (0°=North/Up, 90°=East/Right, 180°=South/Down, 270°=West/Left)
- Target position (green dot): ({target_x}, {target_y})
- Distance to target: {distance:.1f} pixels

## MINIMAP COORDINATE SYSTEM
- **North (N) = UP on the map**
- **East (E) = RIGHT on the map**
- **South (S) = DOWN on the map**
- **West (W) = LEFT on the map**

## YOUR TASK
Look at the TWO images provided:
1. MINIMAP: Shows agent (red triangle), target (green dot), walls (dark), open spaces (light)
2. TRAJECTORY: Shows the path the agent has taken (yellow line with dots)

Analyze:
1. What is the most direct heading to reach the target? (N, NE, E, SE, S, SW, W, NW)
2. Are there obvious obstacles (walls/structures) between agent and target on the minimap?
3. Is the trajectory showing efficient progress or circling/backtracking?
4. Which CARDINAL DIRECTIONS would move the agent toward the target?

## IMPORTANT: OUTPUT CARDINAL DIRECTIONS
You must recommend movement in absolute map directions (N/S/E/W), NOT relative to the agent.
- If target is to the RIGHT on the map → recommend "east"
- If target is UP on the map → recommend "north"
- If trajectory shows circling → try alternative cardinal direction

## AVAILABLE CARDINAL ACTIONS
["north", "south", "east", "west", "stop"]
- "north" = move UP on the map
- "south" = move DOWN on the map
- "east" = move RIGHT on the map
- "west" = move LEFT on the map
- "stop" = stay in place

## OUTPUT FORMAT (strict JSON)
{{
  "strategic_direction": "northeast",
  "recommended_actions": ["north", "east"],
  "confidence": 0.85,
  "reasoning": "Target is northeast. Minimap shows clear corridor. Trajectory shows steady progress - continue north and east."
}}

Think step-by-step, then output JSON with CARDINAL directions.
"""

# ========== LOCAL PLANNER PROMPTS ==========

LOCAL_PLANNER_PROMPT_EGO_ONLY = """You are a LOCAL OBSTACLE AVOIDANCE system for a warehouse robot.

Your job: Analyze the first-person camera view to identify immediate obstacles and recommend safe actions THAT MOVE TOWARD THE GOAL.

## CURRENT SITUATION
- Current heading: {heading}° (0=North, 90=East, 180=South, 270=West)
- **Target bearing: {target_bearing}° relative to agent** (0°=straight ahead, -90°=left, +90°=right, ±180°=behind)
- **Target distance: {target_distance:.1f} pixels** (close <20px, medium 20-50px, far >50px)

## AVAILABLE ACTIONS
{allowed_actions}
- "forward" moves in the direction the agent is facing
- "back" moves backward
- "strafe left" moves left without turning  
- "strafe right" moves right without turning
- "stop" stays in place

## YOUR TASK
Look at the EGOCENTRIC VIEW (first-person camera) provided:

Analyze:
1. Are there obstacles directly in front? How close?
   - **very close**: <1 meter (DANGER - immediate collision risk)
   - **close**: 1-3 meters (caution needed)
   - **medium**: 3-5 meters (plan ahead)
   - **far**: >5 meters (safe to proceed)
2. What type of obstacles? (wall, shelf, equipment, person, forklift, unknown)
3. Which directions appear clear for movement?
4. **IMPORTANT**: Among safe directions, which ones move toward the target?

## GOAL-AWARE SAFETY RULES
- If target is ahead (bearing -45° to +45°):
  * Prioritize "forward" if safe
  * If forward blocked: prefer strafing over backing up
- If target is behind (bearing < -135° or > +135°):
  * "back" is actually moving toward goal (acceptable if safe)
- If target is to the side:
  * Prioritize strafing in that direction
- **NEVER recommend actions that move away from goal UNLESS all goal-oriented paths are blocked**

## OUTPUT FORMAT (strict JSON)
{{
  "immediate_obstacles": [
    {{"type": "shelf", "direction": "front-center", "distance": "close"}}
  ],
  "safe_directions": ["strafe left", "strafe right", "stop"],
  "unsafe_actions": ["forward", "back"],
  "path_clearance": "narrow",
  "recommended_actions": ["strafe right", "strafe left", "stop"],
  "confidence": 0.90,
  "reasoning": "Shelf blocks forward path. Target is ahead-right (bearing +30°). Strafe right maintains progress toward goal while avoiding obstacle. Strafe left also safe but moves away from target bearing.",
  "emergency": false
}}

Think step-by-step: 1) Identify obstacles, 2) Find safe directions, 3) Prioritize actions that move toward target bearing.
"""

LOCAL_PLANNER_PROMPT_EGO_BEV = """You are a LOCAL OBSTACLE AVOIDANCE system for a warehouse robot.

Your job: Analyze first-person and bird's-eye views to identify immediate obstacles and recommend safe actions THAT MOVE TOWARD THE GOAL.

## CURRENT SITUATION
- Current heading: {heading}° (0=North, 90=East, 180=South, 270=West)
- **Target bearing: {target_bearing}° relative to agent** (0°=straight ahead, -90°=left, +90°=right, ±180°=behind)
- **Target distance: {target_distance:.1f} pixels** (close <20px, medium 20-50px, far >50px)

## AVAILABLE ACTIONS
{allowed_actions}
- "forward" moves in the direction the agent is facing
- "back" moves backward
- "strafe left" moves left without turning
- "strafe right" moves right without turning
- "stop" stays in place

## YOUR TASK
Look at the TWO images provided:
1. EGOCENTRIC VIEW: First-person camera showing what's directly ahead
2. BEV (Bird's Eye View): Local top-down view showing agent's immediate surroundings

Analyze:
1. EGOCENTRIC: Are there obstacles directly in front? How close?
   - **very close**: <1 meter (DANGER)
   - **close**: 1-3 meters
   - **medium**: 3-5 meters
   - **far**: >5 meters
2. EGOCENTRIC: What type of obstacles? (wall, shelf, equipment, person, forklift, unknown)
3. BEV: Which directions have open space? Which are blocked?
4. BEV: Is there enough clearance for the agent to move safely?
5. **IMPORTANT**: Among safe directions, which align with target bearing?

## GOAL-AWARE SAFETY RULES
- If target is ahead (bearing -45° to +45°):
  * Prioritize "forward" if safe
  * If forward blocked: prefer strafing over backing up
- If target is behind (bearing < -135° or > +135°):
  * "back" is actually moving toward goal (acceptable if safe)
- If target is to the side:
  * Prioritize strafing in that direction
- **NEVER recommend actions that move away from goal UNLESS all goal-oriented paths are blocked**

## OUTPUT FORMAT (strict JSON)
{{
  "immediate_obstacles": [
    {{"type": "shelf", "direction": "front-center", "distance": "close"}}
  ],
  "safe_directions": ["strafe left", "strafe right", "stop"],
  "unsafe_actions": ["forward", "back"],
  "path_clearance": "narrow",
  "recommended_actions": ["strafe right", "strafe left", "stop"],
  "confidence": 0.90,
  "reasoning": "Egocentric shows shelf close ahead. BEV confirms front blocked but sides clear. Target bearing +30° (ahead-right). Strafe right is safe AND moves toward target. Prioritizing goal-oriented safe action.",
  "emergency": false
}}

Think step-by-step: 1) Identify obstacles in both views, 2) Find safe directions, 3) Prioritize actions that move toward target bearing.
"""

# ========== DECISION MAKER PROMPT ==========

DECISION_MAKER_PROMPT = """You are the DECISION MAKER for a warehouse navigation robot.

Your job: Synthesize recommendations from the Global Planner and Local Planner to make the FINAL action decision.

## CURRENT STATE
- Position: ({current_x}, {current_y}) on minimap
- Target: ({target_x}, {target_y}) on minimap
- Distance to target: {distance:.1f} pixels (STOP if < {threshold:.1f})
- World position: X={world_x:.2f}, Y={world_y:.2f}, Z={world_z:.2f}
- Rotation: RotY={rot_y:.2f}° (heading)

## GLOBAL PLANNER SAYS
{global_reasoning}
- Strategic direction: {global_direction}
- Recommended actions: {global_actions}
- Confidence: {global_confidence:.2f}

## LOCAL PLANNER SAYS
{local_reasoning}
- Safe directions: {local_safe}
- Unsafe actions: {local_unsafe}
- Path clearance: {local_clearance}
- Confidence: {local_confidence:.2f}
- EMERGENCY: {local_emergency}

## AVAILABLE ACTIONS
{allowed_actions}

## DECISION RULES (Priority Order)
1. **Goal reached**: If distance < {threshold:.1f}px, choose "stop" immediately

2. **EMERGENCY situation**: If local emergency=true, ONLY choose from safe_directions
   - Ignore global recommendations completely
   - Safety is absolute priority

3. **NON-EMERGENCY situation**: Strongly prefer global recommendations
   - First choice: Pick the FIRST action from global_recommended that is NOT in local_unsafe
   - If ALL global actions are unsafe: Then pick from local_safe
   - Reasoning: Global knows the goal, local only knows obstacles. Trust global unless dangerous.

4. **Tie-breaking**:
   - If global suggests multiple safe options, pick the first one
   - Only use local_safe as a fallback when global path is completely blocked

## YOUR REASONING PROCESS

Step 1: Check if goal reached → if yes, return "stop"

Step 2: Check local emergency
   - If emergency=true → MUST choose from safe_directions only
   - If emergency=false → proceed to step 3

Step 3: **Prioritize Global (KEY LOGIC)**
   - For each action in global_recommended (in order):
     * If action NOT in local_unsafe → CHOOSE IT immediately
     * If action in local_unsafe → skip to next
   - If all global actions are unsafe → choose first from local_safe

Step 4: If no valid action found → default to "stop"

## IMPORTANT PHILOSOPHY
- **Global Planner knows WHERE to go** (has map + target)
- **Local Planner knows WHAT to avoid** (has obstacles)
- **Non-emergency = trust global direction, just avoid unsafe actions**
- **Emergency = survival first, ignore goal temporarily**

## OUTPUT FORMAT (strict JSON)
{{
  "action": "forward",
  "reasoning": "Non-emergency. Global suggests [forward, strafe right]. Forward is NOT in unsafe list [back]. Choosing forward to progress toward goal.",
  "confidence": 0.88,
  "is_emergency_stop": false,
  "is_goal_reached": false,
  "is_stuck": false
}}

Think through the priority logic step-by-step, then output JSON.
"""


def get_global_prompt(use_trajectory: bool) -> str:
    """Get appropriate global planner prompt based on modalities."""
    if use_trajectory:
        return GLOBAL_PLANNER_PROMPT_WITH_TRAJECTORY
    else:
        return GLOBAL_PLANNER_PROMPT_MINIMAP_ONLY


def get_local_prompt(use_bev: bool) -> str:
    """Get appropriate local planner prompt based on modalities."""
    if use_bev:
        return LOCAL_PLANNER_PROMPT_EGO_BEV
    else:
        return LOCAL_PLANNER_PROMPT_EGO_ONLY


def get_decision_maker_prompt() -> str:
    """Get decision maker prompt."""
    return DECISION_MAKER_PROMPT


# ========== SINGLE-CALL BENCHMARK NAV PROMPT ==========
# Used by the single-LLM-call benchmark loop (run_benchmark_cell), distinct
# from the multi-agent planner prompts above. The template text itself is a
# .txt file under nav/prompts/ (selected via --prompt_file); these helpers
# fill its placeholders and format the movement-history block.


def format_history_for_prompt(history_deque) -> str:
    """Render the agent's recent-movement history into prompt-ready text."""
    if not history_deque:
        return "No previous movements yet."
    lines = []
    for entry in history_deque:
        lines.append(
            f"Step {entry['step']}: Position ({entry['position'][0]},{entry['position'][1]}), "
            f"θ={entry['theta']:.1f}°, Action: '{entry['action']}', "
            f"Distance to target: {entry['distance_to_target']:.1f}px"
        )
    return "\n".join(lines)


def render_nav_prompt(
    tmpl: str, curr_xy, target_xy, theta, distance, allowed_actions, history_str
) -> str:
    """Fill a navigation prompt template's placeholders.

    Placeholders: ``curr_x/curr_y``, ``theta``, ``target_x/target_y``,
    ``distance``, ``allowed_actions``, ``history``. ``curr_xy`` / ``target_xy``
    may be None (rendered as the string "None").
    """
    return tmpl.format(
        curr_x=int(curr_xy[0]) if curr_xy else "None",
        curr_y=int(curr_xy[1]) if curr_xy else "None",
        theta=f"{theta:.1f}",
        target_x=int(target_xy[0]) if target_xy else "None",
        target_y=int(target_xy[1]) if target_xy else "None",
        distance=f"{distance:.2f}",
        allowed_actions=str(allowed_actions),
        history=history_str,
    )
