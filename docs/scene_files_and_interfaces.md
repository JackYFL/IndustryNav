# Scene Files and Interfaces

This note documents the runtime interfaces that must stay in sync between Python and Unity. It is intended for internal maintenance when adding scenes or debugging spawn/target mapping.

The current benchmark uses one unified Unity runtime. Python selects the active warehouse scene through the ML-Agents environment parameter `scene_id`.

## Scene Codes

User-facing commands use anonymized scene codes:

```text
scene1 scene2 ... scene24
```

The unified client uses zero-based Unity build indices:

```text
scene1  -> scene_id 0
scene2  -> scene_id 1
...
scene24 -> scene_id 23
```

The full table is in [`scene_list.md`](scene_list.md). The canonical mapping is
`nav.config.SCENE_ID_MAP`; Python and both benchmark shell wrappers read it
directly.

`input_points.json` is separate task data and supplies four benchmark points for
every code from `scene1` through `scene24`.

## Benchmark Point Format

`input_points.json` stores the start/target tasks for each scene:

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

The coordinate spaces are intentionally mixed:

- `start.x` and `start.z` are Unity world coordinates on the X/Z plane.
- `start.direction` is Unity yaw in degrees.
- `target.x` and `target.y` are visual minimap pixels.

Do not pass `start.x/start.z` as minimap pixels. The benchmark path sends them as `--init_world_x` and `--init_world_z`.

## Runtime Startup Pipeline

The benchmark startup sequence is implemented in `nav/harness/env_setup.py`:

1. Launch `UnityEnvironment` with `no_graphics=False`. macOS uses `-batchmode` by default; Linux defaults to windowed launch because some Linux Unity builds can crash during Input System initialization under `-batchmode`. Override with `INDUSTRYNAV_UNITY_BATCHMODE=1` or `0`.
2. Attach side channels:
   - `EngineConfigurationChannel`
   - `EnvironmentParametersChannel`
   - `BoundsSideChannel`
   - `TargetSideChannel`
3. Set `scene_id` before the first useful `env.reset()`.
4. Set `use_ai_control=1` so the Unity `WarehouseAgent` accepts ML-Agents continuous actions even when the player is launched windowed.
5. Read a minimap observation and detect the rendered minimap bounds.
6. Set Unity's world/pixel mapper to the selected runtime minimap size via `minimap_px_width` / `minimap_px_height`.
7. Prime spawn and target through environment parameters.
8. Read Unity's side-channel acknowledgements for pixel/world mappings.

The behavior name expected by Python is:

```text
WarehouseAgent?team=0
```

This value lives in `nav.config.BEHAVIOR_NAME` and must match the Unity ML-Agents behavior name.

## Environment Parameters

Python sends these values through ML-Agents `EnvironmentParametersChannel`; Unity reads them in `WarehouseAgent.cs`.

| Parameter | Direction | Meaning |
|---|---|---|
| `scene_id` | Python -> Unity | Selects which scene inside `scene_all` to load. Must be set before the reset whose observations are used. |
| `use_ai_control` | Python -> Unity | Enables action-driven locomotion in `WarehouseAgent`. This is required for windowed Linux launches because `Application.isBatchMode` is false. |
| `dynamic_objects_enabled` | Python -> Unity | `1` runs environment motion; `0` freezes environment splines, animations, non-agent physics, NavMesh agents, particles, and timelines. The navigation agent is excluded. |
| `human_speed_mps` | Python -> Unity | Absolute speed for workers and Gley pedestrians. Default benchmark value: `1.2 m/s`. |
| `vehicle_speed_mps` | Python -> Unity | Absolute speed for forklifts and other vehicles. Default benchmark value: `2.5 m/s`. |
| `robot_speed_mps` | Python -> Unity | Absolute speed for robots, AGVs, and AMRs. Default benchmark value: `1.5 m/s`. |
| `light_intensity_multiplier` | Python -> Unity | Non-negative global multiplier applied to each Light's authored intensity. `global_light_intensity` is accepted as a legacy alias. |
| `light_fixed_exposure` | Python -> Unity | Switches enabled global HDRP volumes to Fixed exposure at this EV while runtime lighting is active. |
| `minimap_px_width`, `minimap_px_height` | Python -> Unity | Keeps Unity world/pixel mapping aligned with the selected sensor output size. |
| `spawn_x`, `spawn_y`, `spawn_z` | Python -> Unity | Preferred spawn path: direct Unity world coordinates. |
| `spawn_px`, `spawn_py`, `spawn_wy` | Python -> Unity | Legacy spawn path: Unity minimap pixel coordinates converted to world by Unity. |
| `spawn_rot` | Python -> Unity | Initial yaw in degrees. |
| `target_px`, `target_py` | Python -> Unity | Target location in Unity minimap pixel coordinates. |
| `goal_x`, `goal_y`, `goal_z`, `goal_rot` | Python -> Unity | Fallback world-space goal parameters used when no target pixel is supplied. |

The current benchmark uses world-coordinate spawn plus pixel-coordinate target:

```text
spawn_x/spawn_y/spawn_z/spawn_rot
target_px/target_py
```

## Camera Sensor Resolution

Egocentric resolution is passed as Unity process arguments before ML-Agents
creates the behavior specification. It is not sent through an environment
parameter and cannot be changed without restarting the Unity client.

| Layer | Width | Height | Default |
|---|---|---|---:|
| Python CLI | `--ego_width` | `--ego_height` | `512 x 512` |
| Shell wrappers | `EGO_WIDTH` | `EGO_HEIGHT` | `512 x 512` |
| Unity launch arguments | `--ego-width` | `--ego-height` | `512 x 512` |

The Python entry points validate that both dimensions are positive and forward
them to Unity. `WarehouseAgent.cs` applies the same dimensions to `AgentSensor`
(RGB) and `DepthSensor`, so their observations remain aligned:

```text
RGB observation:   (3, ego_height, ego_width)
depth observation: (1, ego_height, ego_width)
saved RGB/depth:   ego_width x ego_height
```

The minimap sensor uses the same startup mechanism, but its aspect ratio is
fixed to the canonical `862:512` map:

| Layer | Width | Height | Default |
|---|---|---|---:|
| Python CLI | `--minimap_width` (optional) | `--minimap_height` (optional) | `862 x 512` |
| Shell wrappers | `MINIMAP_WIDTH` (optional) | `MINIMAP_HEIGHT` (optional) | `862 x 512` |
| Unity launch arguments | `--minimap-width` | `--minimap-height` | `862 x 512` |

Passing only one dimension derives the other to the nearest integer. Width
`431` or height `256` therefore produces `431 x 256`. If both dimensions are
supplied, Python and Unity reject the configuration unless they match the
canonical aspect ratio.

Python runs coordinate mapping, success checks, and A* planning directly on the
selected sensor size. Existing `input_points.json` targets and pixel parameters
remain defined in the canonical `862 x 512` space and are scaled automatically
at runtime. Actions and result CSV files are converted back to canonical pixels
so evaluations remain comparable across resolutions.

Examples:

```bash
# General benchmark wrapper
EGO_WIDTH=768 EGO_HEIGHT=432 bash shs/run_headless_benchmark.sh scene1

# A* wrapper
EGO_WIDTH=768 EGO_HEIGHT=432 bash shs/run_Astar.sh scene1 point1

# Smaller minimap while preserving the benchmark coordinate space
MINIMAP_WIDTH=431 bash shs/run_Astar.sh scene1 point1

# Freeze environment motion without freezing the navigation agent
DYNAMIC_OBJECTS=static bash shs/run_Astar.sh scene1 point1

# Set distinct fixed speeds in meters/second
HUMAN_SPEED_MPS=1.0 VEHICLE_SPEED_MPS=3.0 ROBOT_SPEED_MPS=1.4 \
  bash shs/run_Astar.sh scene1 point1

# Sample reproducible category-specific speeds
HUMAN_SPEED_MIN_MPS=0.9 HUMAN_SPEED_MAX_MPS=1.4 \
VEHICLE_SPEED_MIN_MPS=2.0 VEHICLE_SPEED_MAX_MPS=3.5 \
ROBOT_SPEED_MIN_MPS=1.0 ROBOT_SPEED_MAX_MPS=2.0 MOTION_RANDOM_SEED=42 \
  bash shs/run_Astar.sh scene1 point1

# Direct benchmark, grid, or collector invocation
python -m nav.scripts.run_benchmark_cell --ego_width 768 --ego_height 432 ...
python -m nav.scripts.run_benchmark_grid --ego_width 768 --ego_height 432 ...
python -m nav.scripts.collect_data --ego_width 768 --ego_height 432 ...
python -m nav.scripts.run_benchmark_cell --minimap_width 431 ...
python -m nav.scripts.run_benchmark_cell --dynamic_objects static ...
python -m nav.scripts.run_benchmark_cell \
  --human_speed_mps 1.0 --vehicle_speed_mps 3.0 --robot_speed_mps 1.4 ...
```

`--screen_width` / `--screen_height` configure the Unity Player window, not the
camera observations. These features require a Unity client built after the
corresponding runtime resolution support was added; older clients ignore the
launch arguments.

## Side Channels

Custom side channels are defined in Python in `nav/harness/side_channels.py`. Their UUIDs and message opcodes live in `nav/config.py` and must match the C# side-channel registration.

### BoundsSideChannel

Unity sends warehouse world bounds once after reset:

```text
minX, maxX, minZ, maxZ
```

Python stores them as `bounds_sc.bounds`. This is useful for diagnostics and legacy/random spawn paths.

### TargetSideChannel

This channel carries the spawn/target mapping acknowledgements that the regular observation tensor does not expose.

For a world-coordinate spawn, Unity converts world -> minimap pixel and sends back:

```text
spawn_pixel = (px, py)
spawn_world = (x, z)
```

For a target pixel, Unity converts pixel -> world and sends back:

```text
target_pixel = (px, py)
target_world = (x, z)
```

Python stores these in:

```text
target_sc.last_spawn_pixel
target_sc.last_spawn_world
target_sc.last_target_pixel
target_sc.last_target_world
```

These acknowledgements are the source of truth when debugging mismatches such as "world spawn does not land on the expected minimap pixel."

## Coordinate Conversion

There are three coordinate systems in play:

| Space | Used by | Notes |
|---|---|---|
| Unity world X/Z | Unity physics, player spawn, world distance | `start.x/start.z` live here. |
| Visual minimap pixel | JSON targets and saved debug images | User-facing pixel coordinate system; it is not used for success thresholds. |
| Unity minimap pixel | Unity `MinimapWorldMapper` | Python converts visual pixels into this space after detecting the minimap margin. |

Python detects the visual minimap bounds with `find_exact_map_bounds()` and converts visual pixels through `visual_to_unity_coords()`. Unity then performs pixel/world conversion using the `MinimapWorldMapper` component in the scene.

## Unity-Side Responsibilities

The Unity-side agent/controller code is responsible for:

- reading `EnvironmentParameters`;
- selecting predefined vs random spawn/target logic;
- teleporting the player controller;
- converting spawn/target pixels through `MinimapWorldMapper`;
- sending spawn/target mapping acknowledgements through `TargetSideChannel`;
- sending warehouse bounds through `BoundsSideChannel`.

When adding or rebuilding a scene, verify that the scene contains:

- an ML-Agents agent with behavior name `WarehouseAgent?team=0`;
- the camera sensors used by Python (`ego`, `depth`, `minimap`);
- a valid `MinimapWorldMapper`;
- the target/goal marker object expected by `WarehouseAgent.cs`;
- the side-channel host/registration objects required by the bounds and target channels.

## Common Maintenance Tasks (for scene developers)

### Add a New Scene

1. Add the scene to the Unity build and assign it the next `scene_id`.
2. Extend `SCENE_ID_MAP` if the new ID is outside its current `0..23` range.
3. Add a new `sceneN` key to `input_points.json` when benchmark wrappers
   should run it.
4. Rebuild `scene_all`.
5. Run a cheap dry boot before launching full benchmarks:

```bash
BASELINE=random MAX_STEPS=2 bash shs/run_headless_benchmark.sh sceneN
```

### Debug Spawn/Target Mapping

Check `unity_log.txt` and the Python run log for lines like:

```text
Spawn world(31.00,49.63) -> unity(550,455) yaw=180.0
Target visual(550,450) -> unity(551,455) -> world(9.63,25.66)
```

If the reported pixel/world pair is wrong, inspect:

- whether `scene_id` selected the intended scene;
- whether `input_points.json` uses world coordinates for `start`;
- whether the minimap margin was detected correctly;
- whether `MinimapWorldMapper` in the Unity scene matches the rendered minimap;
- whether Python and C# side-channel UUIDs/opcodes still match.

### Validate a Runtime Build

After changing scene/runtime code, run:

```bash
BASELINE=random MAX_STEPS=2 bash shs/run_headless_benchmark.sh scene1
ASTAR_DEBUG_VIZ=1 bash shs/run_Astar.sh scene1 point1
```

This verifies both the ML-Agents launch path and the minimap/planning path.
