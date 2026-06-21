# Supported scenes

The unified Unity client `scene_all.app` bundles 12 warehouse scenes. The Python side selects which scene to load at runtime via the `scene_id` env parameter (1..12). Most user-facing tools (`shs/run_headless_benchmark_macos.sh`, `input_points.json`) refer to scenes by their human-readable **scene name** instead, and the wrapper maps name → id.

| `scene_name` | `scene_id` | Author / theme hint |
|---|---:|---|
| `yifan1`  | 1  | Yifan-authored warehouse layout #1 |
| `yifan2`  | 2  | Yifan #2 |
| `yifan3`  | 3  | Yifan #3 |
| `yifan4`  | 4  | Yifan #4 |
| `yicheng` | 5  | Yicheng-authored warehouse |
| `lichi1`  | 6  | Lichi #1 |
| `lichi2`  | 7  | Lichi #2 |
| `xinyu1`  | 8  | Xinyu #1 |
| `xinyu2`  | 9  | Xinyu #2 |
| `anh1`    | 10 | Anh #1 |
| `anh2`    | 11 | Anh #2 |
| `anh3`    | 12 | Anh #3 |

## Where these names show up

- **`input_points.json`** — top-level keys must be one of the names above. Each key holds 4 evaluation points (`point1..point4`) with spawn pixel + direction + target pixel.
- **`shs/run_headless_benchmark_macos.sh`** — first positional arg is `<scene_name>`. The wrapper maps to `scene_id` via a hardcoded `case` statement matching the table above.
- **`nav.scripts.run_benchmark_cell`** — accepts `--scene_id <int>` directly (1..12).

## Overriding the name → id mapping

If your `scene_all.app` was built with a different internal ordering, override per invocation:

```bash
SCENE_ID=7 bash shs/run_headless_benchmark_macos.sh yifan1 google/gemini-3-flash-preview
```

If the override turns out to be the right value for your build, update the `case` statement in `shs/run_headless_benchmark_macos.sh` so teammates don't need the env-var workaround.

## Adding a new scene

Bake the new scene into the Unity project, rebuild `scene_all.app` so the new `scene_id` resolves to it, then:

1. Add a new key to `input_points.json` with at least 4 points (pixel-space spawn + target).
2. Add a new branch to the `case` statement in `shs/run_headless_benchmark_macos.sh`.
3. Append a row to the table above.
