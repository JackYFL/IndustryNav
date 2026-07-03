# Supported scenes

The unified Unity client `scene_all.app` bundles 12 warehouse scenes. The Python side selects which scene to load at runtime via the `scene_id` env parameter (1..12). User-facing tools (`shs/run_headless_benchmark.sh`, `input_points.json`) refer to scenes by anonymized **scene code**, and the wrapper maps code → id.

For runtime environment parameters and the Python/Unity side-channel handshake, see [`scene_files_and_interfaces.md`](scene_files_and_interfaces.md).

| `scene_code` | `scene_id` |
|---|---:|
| `scene1`  | 1  |
| `scene2`  | 2  |
| `scene3`  | 3  |
| `scene4`  | 4  |
| `scene5`  | 5  |
| `scene6`  | 6  |
| `scene7`  | 7  |
| `scene8`  | 8  |
| `scene9`  | 9  |
| `scene10` | 10 |
| `scene11` | 11 |
| `scene12` | 12 |

## Where these codes show up

- **`input_points.json`** — top-level keys must be one of the codes above. Each key holds 4 evaluation points (`point1..point4`) with spawn world coordinates + direction + target pixel.
- **`shs/run_headless_benchmark.sh`** — first positional arg is `<scene_code>`. The wrapper maps to `scene_id` via a hardcoded `case` statement matching the table above.
- **`nav.scripts.run_benchmark_cell`** — accepts `--scene_id <int>` directly (1..12).

## Overriding the code → id mapping

If your `scene_all.app` was built with a different internal ordering, override per invocation:

```bash
SCENE_ID=7 bash shs/run_headless_benchmark.sh scene1 google/gemini-3-flash-preview
```

If the override turns out to be the right value for your build, update the `case` statement in `shs/run_headless_benchmark.sh` so teammates don't need the env-var workaround.

## Adding a new scene

Add the new scene to the Unity runtime build so the new `scene_id` resolves to it, then:

1. Add a new key to `input_points.json` with at least 4 points (world-space spawn + pixel-space target).
2. Add a new branch to the `case` statement in `shs/run_headless_benchmark.sh`.
3. Append a row to the table above.
