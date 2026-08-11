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

`input_points.json` defines four benchmark tasks (`point1` through `point4`) for
every scene from `scene1` through `scene24`. Existing tasks from the former
12-scene build were migrated to the scene codes matching their physical scenes
in the new build order.

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

When replacing the initial task templates with manually curated routes:

1. Keep exactly four `point1..point4` entries under the scene code.
2. Store spawn positions in world X/Z coordinates and targets in canonical
   minimap pixels.
3. Run a dry check before launching Unity:

```bash
DRY_RUN=1 bash shs/run_Astar.sh scene13 point1
```
