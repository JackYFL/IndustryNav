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

When replacing the initial task templates with manually curated routes:

1. Keep point IDs sequential (`point1..pointN`); the official checked-in setup
   currently uses four points per scene.
2. Store spawn positions in world X/Z coordinates and targets in canonical
   minimap pixels.
3. Run a dry check before launching Unity:

```bash
DRY_RUN=1 bash shs/run_Astar.sh scene13 point1
```
