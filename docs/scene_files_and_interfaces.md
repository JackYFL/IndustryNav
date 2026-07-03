# Scene Files and Interfaces

This note documents the runtime interfaces that must stay in sync between Python and Unity. It is intended for internal maintenance when adding scenes or debugging spawn/target mapping.

The current benchmark uses one unified Unity runtime. Python selects the active warehouse scene through the ML-Agents environment parameter `scene_id`.

## Scene Codes

User-facing commands use anonymized scene codes:

```text
scene1 scene2 scene3 scene4 scene5 scene6 scene7 scene8 scene9 scene10 scene11 scene12
```

The mapping is:

```text
scene1  -> scene_id 1
scene2  -> scene_id 2
...
scene12 -> scene_id 12
```

This mapping appears in three places and should be updated together:

- `input_points.json`: top-level keys and per-scene benchmark points.
- `nav/config.py`: `SCENE_ID_MAP`, used by grid runs and shared Python utilities.
- `shs/run_headless_benchmark.sh` and `shs/run_Astar.sh`: shell wrapper validation and `scene_id` lookup.

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

1. Launch `UnityEnvironment` with `no_graphics=False` and `-batchmode`.
2. Attach side channels:
   - `EngineConfigurationChannel`
   - `EnvironmentParametersChannel`
   - `BoundsSideChannel`
   - `TargetSideChannel`
3. Set `scene_id` before the first useful `env.reset()`.
4. Read a minimap observation and detect the rendered minimap bounds.
5. Send the minimap resolution to Unity as `minimap_px_width` / `minimap_px_height`.
6. Prime spawn and target through environment parameters.
7. Read Unity's side-channel acknowledgements for pixel/world mappings.

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
| `minimap_px_width`, `minimap_px_height` | Python -> Unity | Tells Unity the minimap resolution detected from the current observation. |
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
| Visual minimap pixel | JSON targets, saved debug images, benchmark success distance | User-facing pixel coordinate system. |
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

## Common Maintenance Tasks

### Add a New Scene

1. Add the scene to the Unity build and assign it the next `scene_id`.
2. Rebuild `scene_all`.
3. Add a new `sceneN` key to `input_points.json`.
4. Add `sceneN -> scene_id` to `SCENE_ID_MAP`.
5. Add the same mapping to both shell wrappers.
6. Run a cheap dry boot before launching full benchmarks:

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
