# Baseline Workflows and Extension

This document describes how to run the built-in A* navigation baseline and how to extend the benchmark with additional baselines.

## Overview

The benchmark routes all decision makers through one baseline interface. Built-in baselines currently include:

```text
llm
astar
bc
random
```

All baselines share the same Unity startup path, point format, action execution loop, output layout, and evaluation tools. A* is the classical, non-LLM navigation baseline. It plans on the minimap, follows waypoints with discrete actions, and writes the same benchmark artifacts as the other baselines.

Relevant code:

- `shs/run_Astar.sh`: convenience wrapper for scene/point runs.
- `nav/baselines/astar.py`: A* planner and action selection.
- `nav/scripts/run_benchmark_cell.py`: unified benchmark entry; `--baseline astar` constructs and runs the planner.
- `nav/config.py`: default A* tuning values in `ASTAR_DEFAULTS`.

A* does not need `OPENROUTER_API_KEY`.

## Built-in Baseline Interface

At runtime, `run_benchmark_cell.py` builds a per-step payload and calls:

```text
nav.harness.routing.execute_decision(baseline, payload, result_container)
```

Every baseline returns the same core fields through `result_container`:

```text
action
reasoning
finished
error              # optional
prompt             # optional, mainly LLM
```

`action` must be one of:

```text
forward
turn right
turn left
stop
```

This shared contract is what lets LLM, A*, BC, random, and future baselines reuse the same benchmark runner and output/evaluation code.

## Quick Commands

Run all points in one scene:

```bash
bash shs/run_Astar.sh scene1
```

Run one point:

```bash
bash shs/run_Astar.sh scene1 point1
```

Run all scenes:

```bash
bash shs/run_Astar.sh all
```

Enable debug visualizations:

```bash
ASTAR_DEBUG_VIZ=1 bash shs/run_Astar.sh scene1 point1
```

Dry-run the generated command without launching Unity:

```bash
DRY_RUN=1 bash shs/run_Astar.sh scene1 point1
```

## What the Wrapper Does

`shs/run_Astar.sh` handles the repetitive setup:

1. Validates `scene1`-`scene24` or `all`; `all` iterates the canonical 24-scene
   order and requires point data for every scene.
2. Maps scene code to `scene_id`.
3. Reads the selected point(s) from `input_points.json`.
4. Passes start world coordinates and target minimap pixels into `run_benchmark_cell`.
5. Chooses a free gRPC `base_port` for each sequential Unity process.
6. Enables Linux `xvfb-run` automatically when needed.
7. Writes outputs under `outputs/<scene_code>/<point_id>/<run_name>/`.

The wrapper calls:

```bash
python -m nav.scripts.run_benchmark_cell --baseline astar ...
```

## Inputs

A* uses the same point format as the other benchmarks:

```json
{
  "scene1": [
    {
      "point_id": "point1",
      "start": {"x": 31.0, "z": 49.63, "direction": 180.0},
      "target": {"x": 550.0, "y": 450.0}
    }
  ]
}
```

Coordinate interpretation:

- `start.x/start.z`: Unity world coordinates.
- `start.direction`: initial Unity yaw in degrees.
- `target.x/target.y`: visual minimap pixels.

The wrapper passes start coordinates through `--init_world_x` / `--init_world_z`, not through legacy pixel-spawn args.

## Outputs

Default output directory:

```text
outputs/<scene_code>/<point_id>/astar/
```

Typical contents:

```text
astar_actions.csv
astar_depth/
astar_fp/
astar_minimap_target/
astar_paths.jsonl
results.csv
unity_log.txt
```

The run summary in `results.csv` is world-coordinate only. It records the
initial, target, and final Unity X/Z positions plus `distance_world` in meters;
pixel coordinates remain available in `astar_actions.csv` for visualization.

Each new A* run also stores the exact per-decision route in `astar_paths.jsonl`.
The GIF exporter draws this route as a yellow line on the top view. For older
runs without a path log, it reconstructs the route by replaying A* over the
saved minimap and pose sequence. New records also include `tracking` diagnostics
with the controller name, path segment, cross-track error, heading error, and
combined steering angle.

When `ASTAR_DEBUG_VIZ=1`, debug images are written to:

```text
outputs/<scene_code>/<point_id>/astar/astar_debug/
```

or to `ASTAR_DEBUG_DIR` if explicitly provided.

Export one run or all available runs:

```bash
python -m nav.scripts.export_astar_preview --run-dir outputs/scene1/point1/astar --output outputs/astar_gifs/scene1/point1/astar.gif
python -m nav.scripts.export_astar_preview --batch-root outputs --output-dir outputs/astar_gifs
```

## Important Runtime Settings

The most useful environment variables are:

| Variable | Default | Meaning |
|---|---:|---|
| `MAX_STEPS` | unset | Explicit fixed budget; setting it disables dynamic budgeting unless overridden. |
| `ASTAR_DYNAMIC_STEP_BUDGET` | `1` | Derive and update the budget from world route length. |
| `ASTAR_STEP_BUDGET_MIN`, `ASTAR_STEP_BUDGET_MAX` | `50`, `160` | Lower and hard upper bounds for dynamic budgets. |
| `ASTAR_STEPS_PER_PATH_METER` | `1.25` | Decision steps allocated per meter of route. |
| `ASTAR_STEP_BUDGET_OVERHEAD` | `20` | Reserve for turns, recovery, and replanning. |
| `SIM_STEPS_PER_DECISION` | `2` | Unity simulation steps per A* action. |
| `REACH_M` | `2.0` | Success radius in Unity world meters. |
| `MODALITIES` | `ego,minimap,depth` | Saved sensor modalities. |
| `EGO_WIDTH`, `EGO_HEIGHT` | `640`, `480` | Egocentric RGB and depth sensor resolution (4:3 VGA). |
| `MINIMAP_WIDTH` | `862` | Optional minimap width; derived from height when only height is set. |
| `MINIMAP_HEIGHT` | unset | Optional minimap height; derived from width when only width is set. |
| `DYNAMIC_OBJECTS` | `moving` | `moving` runs environment motion; `static` freezes non-agent scene motion. |
| `MARKER_SOURCE` | `vector` | `vector` uses Unity vector observations; `red` uses legacy red-marker detection. |
| `HIDE_UNITY_RED_MARKER` | `1` | Removes the old Unity red marker when Python draws the vector marker. |
| `ASTAR_DEBUG_VIZ` | `0` | Saves walkable-grid/path debug frames when enabled. |
| `ASTAR_DEBUG_DIR` | unset | Explicit debug image directory. |
| `ASTAR_OBSTACLE_CLEARANCE_M` | `0.6` | Physical clearance around minimap obstacles in Unity world meters. |
| `ASTAR_DYNAMIC_REPLAN_LOOKAHEAD_M` | `8.0` | Distance ahead checked against each new minimap obstacle grid. |
| `ASTAR_DYNAMIC_REPLAN_CONFIRM_STEPS` | `2` | Consecutive blocked observations required before replanning. |
| `ASTAR_PROXY_STOP_DISTANCE_M` | `4.9` | World-distance threshold for switching from a reached proxy to direct terminal approach. |
| `RUN_NAME` | `astar` | Output subfolder name. |
| `DRY_RUN` | `0` | Prints the command without launching Unity. |

By default, the initial budget is computed from start-target world distance as
`ceil(20 + 1.25 * distance_m)`, clamped to `50..160`. Each new A* plan is
measured in world meters. The total budget may increase to the larger of the
longest planned-route allowance and the current step plus the new route's
remaining allowance. This also applies to dynamic-obstacle replans. The budget
never decreases and never exceeds 160. Setting `MAX_STEPS` selects a fixed
budget for reproducibility.

World coordinates are required for success, grid resolution, obstacle
clearance, waypoint spacing, waypoint arrival, off-path replanning, lookahead,
terminal approach, stuck recovery, and proxy approach. Pixel values remain
internal implementation details for raster masks and marker rendering.

## Tuning Examples

Use a fixed budget for a reproducibility run:

```bash
MAX_STEPS=120 bash shs/run_Astar.sh scene1 point4
```

Tune the dynamic budget bounds or scaling:

```bash
ASTAR_STEP_BUDGET_MAX=200 ASTAR_STEPS_PER_PATH_METER=1.4 \
  bash shs/run_Astar.sh scene1 point4
```

Make obstacle avoidance more conservative:

```bash
ASTAR_OBSTACLE_CLEARANCE_M=0.9 bash shs/run_Astar.sh scene1 point1
```

Save only minimap/debug outputs:

```bash
MODALITIES=minimap ASTAR_DEBUG_VIZ=1 bash shs/run_Astar.sh scene1 point1
```

Change the egocentric RGB and depth resolution together:

```bash
EGO_WIDTH=768 EGO_HEIGHT=432 bash shs/run_Astar.sh scene1 point1
```

Resize the minimap while preserving its aspect ratio:

```bash
MINIMAP_WIDTH=431 bash shs/run_Astar.sh scene1 point1
```

The saved minimap is `431 x 256`. A* converts its meter-based grid and clearance
settings to that runtime resolution using the calibrated minimap-to-world
projection, while CSV pixel output remains in the canonical `862 x 512` space.

Keep the raw Unity red marker visible:

```bash
HIDE_UNITY_RED_MARKER=0 bash shs/run_Astar.sh scene1 point1
```

Use a custom output folder name:

```bash
RUN_NAME=astar_clearance09 ASTAR_OBSTACLE_CLEARANCE_M=0.9 bash shs/run_Astar.sh scene1 point1
```

## Direct Python Invocation

For wrapper-free debugging, call the benchmark cell directly:

```bash
python -m nav.scripts.run_benchmark_cell \
  --baseline astar \
  --file_name auto \
  --scene_id 0 \
  --scene_name scene1 \
  --point_id point1 \
  --max_steps 70 \
  --sim_steps_per_decision 2 \
  --ego_width 512 \
  --ego_height 512 \
  --reach_m 2.0 \
  --modalities ego,minimap,depth \
  --marker_source vector \
  --hide_unity_red_marker \
  --frame_save_dir outputs/scene1/point1/astar \
  --model_id astar \
  --astar_obstacle_clearance_m 0.6 \
  --astar_proxy_stop_distance_m 4.9 \
  --init_world_x 31.0 \
  --init_world_z 49.63 \
  --init_curr_direction 180 \
  --target_x 550 \
  --target_y 450
```

Use the wrapper for normal runs. The direct command is mainly useful when isolating a single argument or debugger session.

## How A* Plans

At each decision step, the planner receives:

- current minimap RGB;
- current marker pixel and heading;
- target pixel;
- current Unity world position;
- target Unity world position and a minimap-pixel-to-world projection;
- A* tuning parameters.

The planner:

1. Converts the minimap into a walkable/blocked grid.
2. Clears marker artifacts around the current and target pixels.
3. Converts `ASTAR_OBSTACLE_CLEARANCE_M` through the calibrated projection and
   inflates obstacle regions with an axis-aware elliptical kernel.
4. Finds a path to the target or a reachable proxy near the target.
5. Smooths the grid path with obstacle-aware line-of-sight string pulling, then
   rounds validated corners into shorter heading changes. Diagonal grid moves
   cannot pass between two blocked cardinal neighbors.
6. Densely resamples each segment so deviation checks remain stable, then
   converts path samples to world coordinates for metric waypoint spacing,
   lookahead, path-deviation replanning, and heading control.
7. Checks the next 8 meters of the cached route against each new minimap grid.
   A blockage confirmed for two decisions triggers a new plan and updates the
   dynamic step budget from the changed route length. This check pauses while
   the agent is only rotating, which avoids repeated replans from the same view.
8. Projects the current pose onto the smoothed route and applies Stanley
   feedback using both heading error and signed cross-track error.
9. Converts the resulting steering angle into one of:

```text
forward
turn right
turn left
stop
```

A* intentionally uses the annotation action space in `ACTION_SPACE_ANNOTATION`, while learned/LLM agents use `ACTION_SPACE_AGENTS`.

## Debugging Checklist

If A* immediately stops:

- Check whether `distance_world` is already within `REACH_M` of the target.
- Check the target side-channel acknowledgement if world mapping is unavailable.
- Check `astar_actions.csv` for `stop_reason`.
- Enable `ASTAR_DEBUG_VIZ=1` and inspect the target/proxy debug frame.

If A* drives into shelves or corners:

- Increase `ASTAR_OBSTACLE_CLEARANCE_M`.
- Check whether the obstacle appears in the minimap mask.
- Verify `MARKER_SOURCE=vector` unless you explicitly need legacy red-marker detection.

If the target is unreachable:

- Inspect `astar_debug/` for the reachable proxy.
- Adjust `ASTAR_PROXY_STOP_DISTANCE_M` only to control when the final direct approach starts. Success still requires `REACH_M`.
- Confirm the target pixel in `input_points.json` is inside the intended navigable region.

If the current marker appears offset:

- Check the spawn mapping lines in `unity_log.txt`.
- Confirm `input_points.json` uses world coordinates for `start`.
- Compare `astar_minimap_target/` against `astar_debug/`.

If Linux launch fails:

- Confirm `xvfb-run` is installed.
- Keep `USE_XVFB=1` unless a real display is intentionally configured.

## Adding a New Baseline

A new baseline should be added through the shared routing path instead of creating a separate benchmark runner.

### 1. Add the Baseline Name

Add the new token to `BENCHMARK_BASELINES` in `nav/config.py`:

```python
BENCHMARK_BASELINES = ["random", "llm", "bc", "astar", "my_baseline"]
```

Also add its output prefix to `EVAL_RUN_PREFIXES` if the evaluator should discover its outputs:

```python
EVAL_RUN_PREFIXES = ["llm", "bc", "astar", "random", "my_baseline", ...]
```

### 2. Implement the Decision Logic

For a planner-style baseline, put the implementation under:

```text
nav/baselines/
```

For a learned controller, use the existing pattern in:

```text
nav/train/controller.py
```

The implementation should expose a small method that returns a benchmark action string. For example:

```python
action, reasoning = planner.decide(...)
```

or:

```python
action = controller.predict_action(...)
```

### 3. Route It in `execute_decision`

Add a branch in `nav/harness/routing.py`:

```python
elif baseline == "my_baseline":
    result_container["action"] = payload["my_controller"].predict_action(...)
    result_container["reasoning"] = "My baseline decision."
```

Keep failures contained. `execute_decision` already catches exceptions and turns them into a safe `stop`.

### 4. Build the Payload in `run_benchmark_cell.py`

In `nav/scripts/run_benchmark_cell.py`, initialize any controller/planner before the Unity loop, then build the per-step payload inside the decision loop.

The payload can use the same information already passed to existing baselines:

- egocentric RGB observation;
- depth observation;
- minimap RGB observation;
- current minimap pixel;
- target minimap pixel;
- current Unity world position;
- current Unity yaw;
- target world position;
- step count;
- reach threshold.

If the baseline needs new CLI flags, add them to `parse_args()`.

### 5. Choose the Action Space

`run_benchmark_cell.py` chooses the action space with `_action_space_for_baseline()`:

- learned/LLM agents use `ACTION_SPACE_AGENTS`;
- A* uses `ACTION_SPACE_ANNOTATION`.

If the new baseline needs annotation-style actions or agent-style actions, update that helper deliberately.

### 6. Add a Wrapper Only If Useful

Most baselines can run through:

```bash
BASELINE=my_baseline bash shs/run_headless_benchmark.sh scene1
```

Add a dedicated shell wrapper only when the baseline has many repeated tuning variables, like A* does.

### 7. Document Outputs

A baseline named `my_baseline` should write outputs under:

```text
outputs/<scene_code>/<point_id>/my_baseline/
```

and per-step actions should follow:

```text
my_baseline_actions.csv
```

This keeps downstream eval discovery and manual inspection predictable.
