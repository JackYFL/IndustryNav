# Red Dot Disappearance Fix

This document explains the fix for the disappearing red dot / red cone marker on the minimap during interactive data collection. The core change is to stop using a Unity-rendered red object as the only source of truth for the agent marker, and instead draw the red marker and heading arrow in Python from Unity vector observations.

## Cause

The root cause was that the old marker pipeline tied "where the agent is" to a physical red marker rendered inside the Unity scene. For Python to detect that marker correctly, all of the following had to be true:

1. The red cone / red dot had to stay synchronized with the agent transform.
2. The minimap camera had to see the red object.
3. The red object could not be occluded by roofs, walls, shelves, elevated platforms, or other taller scene geometry.
4. The rendered red pixels still had to match the HSV detection thresholds.

If any of those conditions failed, Python would either lose the red dot entirely or detect a position that no longer matched the actual agent.

The observed failure modes included:

- When the agent entered a high area, the minimap camera often saw roofs or upper-level structures, hiding the red cone.
- Some scenes had slightly different hierarchy / prefab setup, so the red marker could drift away from the agent transform.
- If the red cone had a collider, it could also interfere with the agent or the surrounding environment.
- Quality settings, ray tracing, realtime GI, anti-aliasing, and lighting changes could alter the red pixels enough for HSV detection to fail.

So the issue was not simply that the marker was "not red enough." The old localization logic depended on a rendered scene object that was vulnerable to rendering, occlusion, hierarchy, and physics issues.

## Fix

The final fix moves the marker from a Unity-rendered object to a Python-side overlay:

1. Unity still provides the agent's true `pos_x`, `pos_z`, and `rot_y` through ML-Agents vector observations.
2. Python uses the spawn/target side-channel calibration to project the agent world position into minimap pixels.
3. Python post-processes the minimap image and draws the red dot plus red heading arrow directly on top.
4. Saved minimap frames use the same overlay logic.
5. The old HSV red detection path is kept as a fallback with `--marker_source red`, mainly for debugging whether the Unity-side red cone is still visible.

The recommended default is:

```bash
--marker_source vector
```

This has several important benefits:

- The marker is drawn on top of the final minimap image, so it cannot be occluded by tall Unity scene objects.
- The marker position comes from the agent vector observation, so it does not depend on whether a red cone still follows the agent hierarchy.
- The marker color is drawn directly by Python/OpenCV, so it is not affected by Unity lighting, shadows, post-processing, or material changes.
- The red arrow makes the agent's current heading explicit, which is useful for deciding and debugging turns.

## Background

The earlier data-collection script used the red cone / red dot visible in the minimap image to estimate the agent's current position. This was intuitive, but it depended on a Unity marker being rendered correctly by the minimap camera.

In practice, several things made that fragile:

- When the agent moved to stairs, elevated platforms, or partially occluded regions, the red cone could be hidden by roofs, walls, shelves, or other taller objects.
- If the minimap camera only saw the roof or higher structures, the red cone could disappear from the minimap even though it still existed in the scene.
- Scene hierarchy / prefab inconsistencies could detach the marker from the agent transform, causing the detected image position to differ from the true agent position.
- HSV red detection was sensitive to lighting, render quality, post-processing, compression noise, and anti-aliasing.

The new design decouples "agent marker visualization" from "Unity scene object visibility." Unity provides the true world position and yaw; Python projects them into the minimap coordinate system and draws the marker in the displayed and saved minimap images.

## Solution Overview

The data-collection script now supports two marker sources:

```bash
--marker_source vector
--marker_source red
```

The default is:

```bash
--marker_source vector
```

In `vector` mode:

1. Python reads `pos_x`, `pos_z`, and `rot_y` from the ML-Agents vector observation.
2. Python uses the spawn/target side-channel pixel-to-world calibration to project world coordinates back into minimap pixels.
3. Python draws these overlays on the minimap:
   - red dot: projected agent position
   - red arrow: current agent heading
   - green dot: target position
4. Saved frames under `keyboard_minimap/` include the same Python-side red marker / arrow.

In `red` mode:

1. Python runs HSV red detection on the minimap image.
2. It tries to detect the red cone / red dot actually rendered by Unity.
3. This path is useful for debugging the old Unity marker, or for checking whether the red cone still appears in the minimap.

## Modified Files

### `nav/scripts/collect_data.py`

This is the main modified file.

#### 1. Added `--marker_source`

New argument:

```python
p.add_argument(
    "--marker_source",
    choices=("red", "vector"),
    default="vector",
    help="How to locate/draw the current agent marker on the minimap. "
    "'red' uses cone color detection; 'vector' projects Unity vector obs "
    "(pos_x,pos_z,rot_y) into the minimap and draws a Python-side arrow.",
)
```

Meaning:

- `vector`: project the marker from vector observations. This is the recommended default.
- `red`: keep the old red cone image-detection behavior.

#### 2. Added Python-side red dot / arrow drawing

These functions were added or extended:

```python
draw_curr_target_action(...)
annotate_minimap_for_save(...)
heading_endpoint_from_screen_direction(...)
vector_marker_from_obs(...)
signed_angle_to_target_deg(...)
```

`draw_curr_target_action(...)` now draws both the current red dot and an optional heading arrow:

```python
cv2.circle(bgr, (int(curr_xy[0]), int(curr_xy[1])), 4, (0, 0, 255), -1)
cv2.arrowedLine(..., (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.35)
```

Because this is a post-processing overlay, the marker is always drawn on top of the final minimap frame. It is not affected by roofs, walls, shelves, or other high objects in the Unity scene.

#### 3. Added world-to-minimap projection helpers

The following helpers were added to project Unity world coordinates into minimap pixels:

```python
visual_to_unity_coords(...)
world2pixel(...)
unity_to_visual_coords(...)
build_axis_aligned_projector(...)
world_to_unity_coords(...)
world_to_visual_coords(...)
vector_marker_from_obs(...)
```

The most important helper is `vector_marker_from_obs(...)`:

```python
def vector_marker_from_obs(margin, pos_x, pos_z, rot_y, img_shape=None, projector=None):
    center_xy = world_to_visual_coords(
        margin, pos_x, pos_z, img_shape=img_shape, projector=projector
    )
    yaw_rad = np.deg2rad(float(rot_y))
    ahead_world_x = float(pos_x) + np.sin(yaw_rad) * 1.5
    ahead_world_z = float(pos_z) + np.cos(yaw_rad) * 1.5
    heading_xy = world_to_visual_coords(
        margin,
        ahead_world_x,
        ahead_world_z,
        img_shape=img_shape,
        projector=projector,
    )
    return center_xy, heading_xy
```

It uses the Unity yaw convention:

- `rot_y = 0`: the agent faces `+Z`
- `rot_y = 90`: the agent faces `+X`

So the heading endpoint can be computed as:

```python
ahead_world_x = pos_x + sin(yaw) * distance
ahead_world_z = pos_z + cos(yaw) * distance
```

#### 4. Selection UI now shows the marker

The interactive selection windows now show the red marker so the user can visually confirm the point and direction before committing:

- Initial point selection shows the clicked red dot.
- Direction selection shows the red dot plus a red arrow preview.
- Target selection shows the initial red dot / red arrow.

Relevant functions:

```python
select_init_point(...)
select_init_direction(...)
```

Initial point selection now uses `Enter` or `Space` to confirm. This gives the user a chance to verify that the red dot is in the intended location.

#### 5. Saved minimap frames include the Python-side marker

Previously, saved minimap frames were mostly raw minimaps, or only included the target overlay. Now the script calls:

```python
minimap_save = annotate_minimap_for_save(
    minimap_obs_raw,
    curr_xy,
    state["target_xy"],
    curr_heading_xy=curr_heading_xy,
)
```

As a result, `keyboard_minimap/<step>.png` contains:

- red dot: current agent position
- red arrow: current agent heading
- green dot: target

This overlay is drawn in Python and does not depend on whether a Unity-side marker is visible.

#### 6. Added `heading_delta_deg`

The action log now records:

```python
"heading_delta_deg": heading_delta_deg,
"marker_source": args.marker_source,
```

`heading_delta_deg` is the signed angle between the agent's current heading and the direction to the target:

- Positive and negative values can be used to reason about which direction to turn.
- Values near `0` mean the agent is roughly facing the target.
- Values near `180` or `-180` mean the agent is facing away from the target.

This is useful for debugging, checking behavior-cloning data, and analyzing turn decisions.

#### 7. Added `--spline_speed_multiplier`

New argument:

```python
p.add_argument(
    "--spline_speed_multiplier",
    type=float,
    default=1.0,
    help="Runtime multiplier for Unity SplineAnimate environment-object speed. "
         "1.0 uses the speed serialized in the scene.",
)
```

The value is sent to Unity through an ML-Agents environment parameter:

```python
env_params.set_float_parameter(
    "spline_speed_multiplier",
    float(args.spline_speed_multiplier),
)
```

This is not directly part of the red dot disappearance issue, but it was included in the same collect-data runtime-control update. If the Unity side implements the matching reader, spline object speed can be adjusted without rebuilding scenes.

#### 8. Changed `quality_level` from 0 to 3

Before:

```python
quality_level=0
```

After:

```python
quality_level=3
```

In the current Unity project, quality index `0` corresponds to a Ray Tracing / Realtime GI configuration. That setting can introduce visible stochastic noise in CameraSensor frames. Using quality level `3` makes collected images more stable and also reduces false negatives in the old red-detection fallback mode.

### `nav/config.py`

`results.csv` uses a fixed schema. Extra fields are dropped by `extrasaction="ignore"`, so the new metadata fields must be added explicitly:

```python
RESULTS_CSV_FIELDS = [
    ...
    "marker_source",
    "spline_speed_multiplier",
]
```

This ensures each run records whether it used the `vector` or `red` marker source, along with the spline speed multiplier.

## Runtime Behavior

### Recommended command

Use the default vector marker:

```bash
python nav/scripts/collect_data.py \
  --marker_source vector \
  --spline_speed_multiplier 1.0
```

Or run as a module:

```bash
python -m nav.scripts.collect_data \
  --marker_source vector \
  --spline_speed_multiplier 1.0
```

### Fallback to old red detection

To check whether the Unity-side red cone is still visible, switch back to the old detection path:

```bash
python nav/scripts/collect_data.py --marker_source red
```

If `red` mode fails but `vector` mode works, the problem is likely Unity rendering / hierarchy / occlusion rather than the agent pose itself.

## Why This Fix Works

The fix works because the marker source changed:

| Old approach | New approach |
| --- | --- |
| Real red cone / red dot rendered in the Unity scene | Python post-processing overlay from vector observations |
| Can be hidden by roofs, shelves, or other tall objects | Drawn on top of the final minimap image |
| Depends on color detection | Depends on the agent's true position/yaw |
| Scene hierarchy differences can detach the marker | Uses the agent vector observation, not a marker prefab |
| Lighting and render noise can break detection | Marker color is fixed by OpenCV drawing |

Therefore, even if the Unity minimap camera sees only roofs, the red cone is occluded, or the red cone detaches from the agent hierarchy, the Python-side marker can still show the agent's true position and heading.

## Limitations

`vector` mode still relies on two assumptions:

1. The Unity vector observation must report the true agent transform through `pos_x`, `pos_z`, and `rot_y`.
2. The spawn/target side-channel calibration must correctly return the pixel/world mapping.

If the Unity vector observation is wrong, the Python overlay will also be wrong.

If the minimap projection has a systematic offset, check these values first:

- `target_sc.last_spawn_pixel`
- `target_sc.last_spawn_world`
- `target_sc.last_target_pixel`
- `target_sc.last_target_world`
- the minimap margin returned by `find_exact_map_bounds(...)`

## Debug Checklist

If the red marker is still wrong, check the following in order:

1. Confirm the run is using `--marker_source vector`.
2. Inspect the position and rotation logs at each step:

   ```text
   Position: X=..., Y=..., Z=...
   Rotation: RotX=..., RotY=..., RotZ=...
   ```

3. Open `keyboard_actions.csv` and inspect:

   - `curr_world_x`
   - `curr_world_z`
   - `curr_direction_y`
   - `curr_px`
   - `curr_py`
   - `heading_delta_deg`
   - `marker_source`

4. If `curr_world_x/z` is correct but `curr_px/py` is offset, debug the minimap projection calibration first.
5. If `curr_world_x/z` is already wrong, debug the Unity agent transform / vector observation first.
6. If only `--marker_source red` fails, the Unity red cone rendering or hierarchy still has a problem, but the recommended vector marker path is unaffected.

## Related Unity-side Notes

The Python-side marker is the stable solution used for data collection. The Unity scene may still keep a red cone or another marker for human inspection in Scene/Game view, but that object should no longer be treated as the only localization source.

If the Unity project still keeps a red cone:

- It can remain as a visual aid for humans.
- It should not be required for minimap localization.
- If it can physically collide with the agent or environment, set its collider to trigger or remove the collider.
- If it detaches from the agent transform, check the hierarchy parent, local transform, and any runtime reset / teleport / scene-load synchronization scripts.

## Summary

This update moves the red marker from a Unity-rendered object to a Python-side overlay:

- Default `--marker_source vector`
- Red dot / red arrow come from the agent vector observation
- Marker cannot be hidden by tall scene objects
- Marker does not disappear when the Unity red cone is occluded or detached
- Saved minimap frames include the marker / arrow
- Action logs record `heading_delta_deg` and `marker_source`
- `results.csv` records `marker_source` and `spline_speed_multiplier`

This makes the minimap marker more stable during collect-data runs and easier to debug across scenes.
