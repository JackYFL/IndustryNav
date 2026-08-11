# Supported Scenes

The unified Unity client bundles 24 warehouse scenes. Unity uses a zero-based
build index, so the runtime `scene_id` range is `0..23`. User-facing commands
use anonymized scene codes and resolve them through `nav.config.SCENE_ID_MAP`.

For runtime environment parameters and the Python/Unity side-channel handshake,
see [`scene_files_and_interfaces.md`](scene_files_and_interfaces.md).

| `scene_code` | `scene_id` | `scene_code` | `scene_id` |
|---|---:|---|---:|
| `scene1`  | 0  | `scene13` | 12 |
| `scene2`  | 1  | `scene14` | 13 |
| `scene3`  | 2  | `scene15` | 14 |
| `scene4`  | 3  | `scene16` | 15 |
| `scene5`  | 4  | `scene17` | 16 |
| `scene6`  | 5  | `scene18` | 17 |
| `scene7`  | 6  | `scene19` | 18 |
| `scene8`  | 7  | `scene20` | 19 |
| `scene9`  | 8  | `scene21` | 20 |
| `scene10` | 9  | `scene22` | 21 |
| `scene11` | 10 | `scene23` | 22 |
| `scene12` | 11 | `scene24` | 23 |

## Benchmark Points

The checked-in `input_points.json` defines four benchmark tasks (`point1`
through `point4`) for every scene from `scene1` through `scene24`. Existing
tasks from the former 12-scene build were migrated to the scene codes matching
their physical scenes in the new build order.

Tasks added for the newly included scenes are initial templates based on valid
spawn coordinates and shared-map targets. Validate their complete routes in the
24-scene Unity client before using them for official benchmark reporting.

`bash shs/run_Astar.sh all` and the grid runner's default scene selection iterate
the canonical `SCENE_CODES` order and validate that all 24 scenes have point
data.

## Where The Mapping Is Used

- `nav/config.py`: canonical `SCENE_ID_MAP`.
- `shs/run_headless_benchmark.sh` and `shs/run_Astar.sh`: import the canonical
  Python mapping instead of maintaining separate shell tables.
- `nav.scripts.run_benchmark_grid`: validates requested scene codes against the
  canonical mapping.
- `nav.scripts.run_benchmark_cell`: accepts `--scene_id <int>` directly
  (`0..23`).

## Overriding The Mapping

If a custom Unity client was built with a different internal ordering, override
the ID for one wrapper invocation:

```bash
SCENE_ID=7 bash shs/run_headless_benchmark.sh scene1 google/gemini-3-flash-preview
```

For a permanent ordering change, update the Unity build settings and
`SCENE_ID_MAP` together.

## Updating Benchmark Points

Use the interactive minimap editor to replace or append points for any scene:

```bash
python -m nav.scripts.edit_input_points
```

On the first run, the editor launches each of the 24 scenes once and stores its
minimap PNG plus pixel/world projection under `.cache/input_point_editor/`.
Subsequent scene changes read only this cache; dragging points and saving no
longer relaunch Unity. Use `--refresh-cache` after rebuilding the Unity client
or changing scene cameras, lighting, or minimap settings.

The scene table reads the current counts directly from `input_points.json`.
Choose a scene and pair count, open its cached minimap, then drag from the start
to the target. Releasing keeps the pair as a draft; set the start direction and
click `Confirm pair` to map it through the cached Unity projection. A single
click still supports the click-start/click-target workflow. `Undo` (or
`Cmd/Ctrl+Z`) restores the latest click, drag, direction, or confirmed pair.
Saving only changes the selected scene and writes the JSON atomically. All
pairs use thick colored lines with solid green `S` and red `T` endpoints. Each
start also has a yaw arrow, and the numeric yaw is shown in the point table. The
benchmark convention is `0` left, `90` up, `180` right, and `270` down on the
minimap. Press `Ctrl+S` (or `Cmd+S` on macOS) to save the active existing-point
or new-point draft.

Every confirmed `S/T` endpoint is draggable. Release an endpoint on its new
location and the editor validates it through the cached projection before
updating the draft.

Selecting a textual point row makes its line and endpoints more prominent and
loads its current yaw into `Initial yaw`. Change that value and click
`Apply yaw` to update only the starting direction. With a pending start or a
point-table row selected, arrow keys immediately set the four cardinal
directions and update the minimap arrow: `Left=0`, `Up=90`, `Right=180`, and
`Down=270`. Without either selection, they set the default yaw for the next
start. The row-level destructive action is
`Delete selected`. Existing-point changes remain reversible until
`Save existing changes` is clicked; `Undo` restores the previous delete, drag,
or yaw state.

In `Append` mode, click or drag a start and target, choose the yaw, then confirm
the pair. The append save button becomes available after the first confirmed
pair, so the append limit is only a cap rather than a required count.

The initial scene and pair count can also be supplied on the command line:

```bash
python -m nav.scripts.edit_input_points --scene 17 --pairs 4 --auto-load
```

### Point Editor Options

| Argument | Default | Purpose |
|---|---:|---|
| `--scene` | `scene1` | Initial scene code or number (`1` through `24`). |
| `--pairs` | `4` | Exact replacement count, or maximum append count. |
| `--mode` | `replace` | Use `replace` to rewrite a scene or `append` to add points. |
| `--direction` | `180` | Default initial Unity yaw for a new start. |
| `--auto-load` | disabled | Open the selected cached scene after cache preparation. |
| `--cache-dir` | `.cache/input_point_editor` | Minimap PNG and pixel/world projection cache. |
| `--refresh-cache` | disabled | Rebuild all 24 caches after client or scene changes. |
| `--file-name` | `auto` | Unified Unity client used only when caches must be generated. |
| `--input-file` | `input_points.json` | Point database to read and update. |

Examples:

```bash
# Append at most two pairs to scene8 and open it immediately
python -m nav.scripts.edit_input_points \
  --scene scene8 --pairs 2 --mode append --auto-load

# Rebuild all minimaps and projection metadata after rebuilding the client
python -m nav.scripts.edit_input_points --refresh-cache --auto-load
```

### Rendering the 24-Scene Point Overview

`nav.scripts.render_input_points_overview` reads the cached minimaps and
`input_points.json`, projects every world-space start back onto its scene, and
renders all scenes in row-major order. Each panel shows colored task lines,
green start markers, red target markers, `S1..S4` / `T1..T4` labels, and the
initial-heading arrows. It does not launch Unity.

Generate the default 4-column × 6-row, full-resolution overview:

```bash
python -m nav.scripts.render_input_points_overview
```

The default output is:

```text
outputs/input_points_overview_4x6.png
```

The output directory is ignored by Git. To regenerate the smaller image used
by the repository README:

```bash
python -m nav.scripts.render_input_points_overview \
  --panel-width 500 \
  --output docs/assets/industrynav_24_scene_points_overview.png
```

| Argument | Default | Purpose |
|---|---:|---|
| `--input-file` | `input_points.json` | Point database rendered into every panel. |
| `--cache-dir` | `.cache/input_point_editor` | Source minimap PNGs and projection metadata. |
| `--output` | `outputs/input_points_overview_4x6.png` | Destination PNG. |
| `--columns` | `4` | Grid column count; it must divide 24 evenly. |
| `--panel-width` | `862` | Width of each scene panel; minimum `320`. |

Regenerate the overview whenever `input_points.json` changes. Rebuild the
minimap cache first only when the Unity client, scene camera, lighting, or
minimap resolution changes.

When replacing the initial task templates with manually curated routes:

1. Keep point IDs sequential (`point1..pointN`); the official checked-in setup
   currently uses four points per scene.
2. Store spawn positions in world X/Z coordinates and targets in canonical
   minimap pixels.
3. Run a dry check before launching Unity:

```bash
DRY_RUN=1 bash shs/run_Astar.sh scene13 point1
```
