# Action Speed and Turn Consistency Fix

This document explains the issue where the same forward / turn actions appeared to produce different movement and rotation speeds across scenes, and documents the fixes applied in both the Unity project and the Python-side data-collection / probe code.

## Cause

The root problem was that the same Python action values were not always converted into the same amount of Unity simulation motion. The action space itself was globally consistent, but the runtime interpretation of those actions could still vary because it depended on scene timing, frame pacing, and per-scene Unity state.

The main failure modes were:

1. Agent motion used frame-dependent delta time.
2. Different scenes could have different render cost, quality settings, or frame pacing.
3. Unity could run a different number of effective update / decision ticks per wall-clock interval.
4. Some scene setups could use inconsistent player / mapper transforms after teleport or scene load.
5. Moving environment objects driven by `SplineAnimate` could use scene-authored speeds that were hard to adjust consistently at runtime.

As a result, two scenes could receive the same continuous action from Python, for example:

```text
forward:    move=15, look=0
turn right: move=0,  look=9
turn left:  move=0,  look=-9
```

but still show different forward displacement or yaw change after the same number of Python `env.step()` calls.

The important distinction is:

- The Python action value was consistent.
- The Unity-side integration of that action was not guaranteed to be consistent until frame pacing and action delta time were fixed.

## Fix

The final fix makes action execution deterministic with respect to ML-Agents simulation steps:

1. Use a fixed action delta time in `PlayerController.MoveAgent(...)`.
2. Ensure the ML-Agents `DecisionRequester` requests decisions every simulation step.
3. Disable V-Sync and set a predictable target frame rate.
4. Avoid noisy / expensive quality levels during data collection.
5. Keep Python-side `sim_steps_per_decision` explicit and logged.
6. Add probe scripts that measure actual forward distance and turn angle across scenes.
7. Add a runtime spline speed multiplier so moving environment-object speed can be adjusted consistently from Python.

The most important Unity-side change is that agent movement no longer depends directly on the variable frame `Time.deltaTime` during AI control. It uses a fixed `agentStepDeltaTime`, currently:

```csharp
public float agentStepDeltaTime = 0.02f;
```

This makes a single ML-Agents action step correspond to a stable amount of motion across scenes.

## Background

During testing, the Python scripts sent the same global action values to every scene, but observed behavior did not always match:

- A forward action could move farther in one scene than another.
- A turn action could rotate by a different number of degrees in different scenes.
- The issue was easier to notice when comparing scene-by-scene data collection or benchmark runs.

At first glance this looked like an action-space bug, but the action-space values were already shared. The issue was further downstream: Unity was applying those actions through a runtime movement controller that was sensitive to timing and scene conditions.

## Modified Unity Files

These Unity files live under:

```text
/Users/liyifan/Documents/UnityProjects/IndustryNav/Assets/
```

### `UnityWarehouseSceneHDRP/PlayerController.cs`

This file contains the core movement fix.

The AI-control path now exposes a fixed step delta:

```csharp
public float agentStepDeltaTime = 0.02f;
```

`MoveAgent(...)` uses that fixed delta time:

```csharp
float dt = agentStepDeltaTime > 0f ? agentStepDeltaTime : Time.fixedDeltaTime;
if (dt <= 0f)
{
    dt = Time.deltaTime;
}
```

Then both movement and gravity are integrated with `dt`:

```csharp
velocity.y += gravity * dt;

Vector3 moveDirection = transform.right * moveInput.x + transform.forward * moveInput.y;

characterController.Move(moveDirection * moveSpeed * dt);
characterController.Move(velocity * dt);
```

The turn action is applied once per action step:

```csharp
transform.rotation *= Quaternion.Euler(0, lookInputX * lookSpeed, 0);
```

Why this matters:

- Forward displacement becomes tied to a fixed simulation step instead of variable render frame time.
- Gravity accumulation becomes stable across scenes.
- The AI-control path behaves differently from the manual-control path on purpose: manual keyboard movement can still use `Time.deltaTime`, while ML-Agents control uses the fixed action step.

The file also adds `EnsureInitialized()` so the `CharacterController` and input wrapper are initialized before teleport or action execution. This prevents scene-load / reset timing from leaving the controller in a partially initialized state.

### `UnityWarehouseSceneHDRP/Player/Scripts/WarehouseAgent.cs`

This file ensures the ML-Agents wrapper consistently drives the `PlayerController`.

The important runtime settings are:

```csharp
var dr = GetComponent<DecisionRequester>() ?? gameObject.AddComponent<DecisionRequester>();
dr.DecisionPeriod = 1;
dr.TakeActionsBetweenDecisions = true;
```

This ensures the agent receives decisions at every simulation step and continues applying actions between decisions.

`OnActionReceived(...)` passes the Python continuous action values into `PlayerController.MoveAgent(...)`:

```csharp
float moveForwardBack = actions.ContinuousActions[0];
float moveLeftRight = actions.ContinuousActions[1];
float lookLeftRight = actions.ContinuousActions[2];

playerController.MoveAgent(new Vector2(moveLeftRight, moveForwardBack), lookLeftRight);
```

The script also keeps the minimap mapper tied to the actual player controller transform:

```csharp
if (mapper != null && playerController != null)
    mapper.agentTf = playerController.transform;
```

Why this matters:

- The movement controller, vector observation, and minimap marker all refer to the same actual player transform.
- The decision frequency is explicit and not left to scene-specific prefab differences.
- Python can compare position / yaw changes across scenes using the same underlying action execution path.

### `GlobalSettings.cs`

This file makes frame pacing more predictable:

```csharp
[SerializeField] private int targetFPS = 30;

QualitySettings.vSyncCount = 0;
Application.targetFrameRate = targetFPS;
```

Why this matters:

- V-Sync can make runtime behavior depend on monitor refresh rate or platform display state.
- Disabling V-Sync lets Unity and the ML-Agents `EngineConfigurationChannel` control timing more predictably.
- A fixed target frame rate makes wall-clock behavior less scene-dependent.

This does not replace the fixed action delta time, but it reduces scene-to-scene timing noise and makes profiling easier.

### `UnityWarehouseSceneHDRP/SplineSpeedRuntimeScaler.cs`

This file adds runtime control for moving environment-object speed:

```csharp
private const string ParameterName = "spline_speed_multiplier";
```

It reads the value from ML-Agents environment parameters:

```csharp
float multiplier = Academy.Instance.EnvironmentParameters.GetWithDefault(ParameterName, 1f);
```

Then applies it to every `SplineAnimate`:

```csharp
animator.AnimationMethod = SplineAnimate.Method.Speed;
animator.MaxSpeed = baseSpeed * multiplier;
```

Why this matters:

- Environment object speed can be adjusted from Python without editing scene files or rebuilding per-scene settings.
- All spline-driven objects use `SplineAnimate.Method.Speed`, which is easier to reason about than duration-based animation when comparing scenes.
- The default multiplier is `1.0`, so existing scene-authored speeds are preserved unless Python explicitly changes them.

## Modified Python Files

### `nav/scripts/collect_data.py`

The data-collection script sends the spline speed multiplier to Unity:

```python
env_params.set_float_parameter(
    "spline_speed_multiplier",
    float(args.spline_speed_multiplier),
)
```

It also uses a less noisy quality level:

```python
engine.set_configuration_parameters(
    time_scale=1,
    quality_level=3,
    target_frame_rate=-1,
    width=args.screen_width,
    height=args.screen_height,
)
```

The script keeps `sim_steps_per_decision` explicit:

```python
SIM_STEPS_PER_DECISION = max(1, int(args.sim_steps_per_decision))
```

and applies each chosen action for that number of Unity steps:

```python
last_continuous_actions = continuous_actions
for step in range(max(1, SIM_STEPS_PER_DECISION)):
    if step == SIM_STEPS_PER_DECISION - 1:
        env.set_actions(BEHAVIOR_NAME, ActionTuple(continuous=last_continuous_actions))
    env.step()
```

The run summary records:

```python
"sim_steps_per_decision": int(SIM_STEPS_PER_DECISION),
"spline_speed_multiplier": float(args.spline_speed_multiplier),
```

Why this matters:

- Python-side action duration is no longer implicit.
- Every run records how many Unity simulation steps were used per decision.
- Moving-object speed settings are also recorded in `results.csv`.

### `nav/config.py`

The fixed `results.csv` schema includes:

```python
"sim_steps_per_decision",
"spline_speed_multiplier",
```

This ensures downstream readers can compare runs with the same action timing settings.

### `tools/probe_action_consistency.py`

This probe launches `scene_all`, selects scenes, sends deterministic actions, and reads vector observations after each action.

It measures:

- `turn_right_delta_deg`
- `turn_left_delta_deg`
- `forward_delta_x`
- `forward_delta_z`
- `forward_distance`
- `forward_y_delta`
- `final_yaw`

The relevant action sequence is:

```python
after_stop = step_action(env, BEHAVIOR_NAME, "stop", args.settle_steps)
after_turn_right = step_action(env, BEHAVIOR_NAME, "turn right", args.sim_steps)
after_turn_left = step_action(env, BEHAVIOR_NAME, "turn left", args.sim_steps)
before_forward = after_turn_left
after_forward = step_action(env, BEHAVIOR_NAME, "forward", args.sim_steps)
```

This is the direct test for whether the same action produces the same displacement / yaw change across scenes.

### `tools/probe_scene_timing.py`

This probe measures wall-clock Unity step timing per scene:

- mean step time
- median step time
- p90 step time
- min / max step time
- steps per second

It is useful for distinguishing two different questions:

1. Are actions physically consistent in simulation space?
2. Do scenes run at the same wall-clock speed?

After the fixed action delta-time change, the first should be consistent. The second can still differ because some scenes are heavier to render or simulate.

## Runtime Behavior

### Data collection

Recommended command shape:

```bash
python nav/scripts/collect_data.py \
  --sim_steps_per_decision 2 \
  --spline_speed_multiplier 1.0
```

If you want faster moving environment objects:

```bash
python nav/scripts/collect_data.py \
  --spline_speed_multiplier 2.0
```

This changes spline-driven environment objects, not the agent's own action speed.

### Action consistency probe

Run:

```bash
python tools/probe_action_consistency.py \
  --file_name auto \
  --scenes yifan1 yifan2 yifan3 \
  --sim_steps 2 \
  --output_csv outputs/action_consistency_probe.csv
```

Expected interpretation:

- `turn_right_delta_deg` should be close across scenes.
- `turn_left_delta_deg` should be close across scenes.
- `forward_distance` should be close across scenes, assuming the spawn point is not immediately blocked or on a slope / collision boundary.
- Large differences usually mean a Unity-side controller, collision, spawn, or scene setup issue remains.

### Scene timing probe

Run:

```bash
python tools/probe_scene_timing.py \
  --file_name auto \
  --scenes yifan1 yifan2 yifan3 \
  --steps 80 \
  --action forward \
  --quality_level 3 \
  --output_csv outputs/scene_step_timing.csv
```

Expected interpretation:

- Step time can still vary across scenes because some scenes have more geometry, lights, render cost, or active objects.
- This does not necessarily mean the action speed is inconsistent.
- If action deltas are consistent but wall-clock step times differ, the simulation is behaving correctly but scene performance is different.

## Why This Fix Works

The fix works because it separates simulation-space consistency from wall-clock performance.

| Problem | Fix |
| --- | --- |
| Movement depended on variable frame time | `MoveAgent(...)` uses fixed `agentStepDeltaTime` |
| Decision frequency could vary by prefab / scene setup | `DecisionRequester.DecisionPeriod = 1` |
| V-Sync / frame pacing could differ across machines | `QualitySettings.vSyncCount = 0`, target FPS configured |
| Heavy scenes could make wall-clock speed look different | Probe simulation deltas separately from step timing |
| Moving spline objects had scene-authored speeds | Runtime `spline_speed_multiplier` controls `SplineAnimate.MaxSpeed` |
| Logs did not fully identify action timing settings | `sim_steps_per_decision` and `spline_speed_multiplier` are recorded |

The agent's physical displacement and yaw change are now based on:

```text
action value × controller speed × fixed action dt × number of sim steps
```

instead of:

```text
action value × controller speed × variable rendered frame delta
```

That is the key reason the same action can now be compared across scenes.

## Limitations

Even after this fix, some differences can still be legitimate:

- Collisions can reduce forward distance if the spawn point is near an obstacle.
- Slopes, stairs, or uneven ground can affect `CharacterController` motion.
- Different spawn heights can produce different gravity settling behavior.
- Scene performance can still differ in wall-clock time.
- Environment objects can move at different authored base speeds if `spline_speed_multiplier=1.0`; the multiplier scales those base speeds but does not make every spline path identical.

The goal of this fix is not to make every scene have identical wall-clock runtime. The goal is to make the same agent action correspond to the same simulation-space command under the same local physical conditions.

## Debug Checklist

If forward or turn behavior still looks inconsistent:

1. Confirm the Unity build includes the fixed `PlayerController.MoveAgent(...)`.
2. Confirm `agentStepDeltaTime` is positive, usually `0.02`.
3. Confirm `WarehouseAgent` sets:

   ```csharp
   dr.DecisionPeriod = 1;
   dr.TakeActionsBetweenDecisions = true;
   ```

4. Confirm Python is using the expected `--sim_steps_per_decision`.
5. Compare `keyboard_actions.csv` rows:

   - `move`
   - `look`
   - `curr_world_x`
   - `curr_world_z`
   - `curr_direction_y`
   - `distance_world`

6. Run `tools/probe_action_consistency.py` on the suspicious scenes.
7. If action deltas are consistent but the run feels slower, run `tools/probe_scene_timing.py`.
8. If only one spawn point is inconsistent, test another spawn point away from walls, shelves, stairs, and dynamic objects.
9. If spline objects move too fast or too slowly, adjust `--spline_speed_multiplier` and confirm Unity logs:

   ```text
   [SplineSpeedRuntimeScaler] Applied spline_speed_multiplier=...
   ```

## Summary

The action speed / turn consistency issue was caused by Unity-side timing and scene-dependent execution, not by different Python action values.

The fix:

- Uses a fixed `agentStepDeltaTime` for AI-controlled movement.
- Forces ML-Agents decisions every simulation step.
- Disables V-Sync and sets predictable frame pacing.
- Keeps Python `sim_steps_per_decision` explicit and logged.
- Adds runtime spline speed control through `--spline_speed_multiplier`.
- Adds probe scripts to measure action deltas and scene timing separately.

This makes forward movement and yaw changes comparable across scenes while still allowing scene wall-clock performance to differ naturally.
