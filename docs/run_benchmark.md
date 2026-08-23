# Running benchmarks (macOS / Ubuntu Linux, unified Unity client)

This is the canonical guide for running AI navigation benchmarks against the **unified** Unity client (`scene_all`, 24 scenes bundled, scene chosen at runtime with zero-based `scene_id` values `0..23`). Legacy per-scene clients and the older `run_sync.py` / `run_async.py` / `shs/run_agent.sh` flow are **not** the latest; treat them as reference only.

The current entry point is `nav/scripts/run_benchmark_cell.py` (run via `python -m nav.scripts.run_benchmark_cell`), driven by:

- `shs/run_headless_benchmark.sh` on macOS, or
- `shs/run_headless_benchmark.sh` on Ubuntu Linux.

The Python script itself is OS-agnostic; the wrappers only differ in client-path defaults and (on Linux) wrapping each invocation in `xvfb-run` to provide a virtual display surface. Linux Unity launches default to windowed mode rather than `-batchmode`; set `INDUSTRYNAV_UNITY_BATCHMODE=1` only if your local Linux player supports batchmode.

---

## Prerequisites

1. **Python env** — set up via `uv`. See [`docs/python_env_options.md`](python_env_options.md). The wrappers default to `<repo>/.venv/bin/python`; override with `PYTHON_BIN=<path>`.
2. **Unity client**:
   - **Preferred (any OS):** drop your build into the gitignored in-repo `unity_client/` folder — macOS `unity_client/scene_all.app`, Linux `unity_client/scene_all/scene_all.x86_64`. `--file_name auto` (the default) finds it there first (anchored to `REPO_ROOT`), so no env var or absolute path is needed.
   - macOS — set up via the [`unity_client_setup_macos`](../.claude/skills/unity_client_setup_macos/SKILL.md) skill (covers the Gatekeeper `xattr -dr com.apple.quarantine` step). Legacy override: `SCENE_ALL_APP=/path/to/scene_all.app`.
   - Ubuntu Linux — set up via the [`unity_client_setup_linux`](../.claude/skills/unity_client_setup_linux/SKILL.md) skill. Legacy override: `SCENE_ALL_BIN=/path/to/scene_all/scene_all.x86_64` (or add a fallback path to `config.SCENE_ALL_BUILDS["Linux"]`).
   - **Auto-discovery**: the Python entry points (`nav.scripts.run_benchmark_cell`, `nav.scripts.run_benchmark_grid`) accept `--file_name auto` (the default), which scans `config.SCENE_ALL_BUILDS` for the current OS and uses the first listed build path that exists locally. Both wrappers default the env var to `auto` as well. To onboard a new dev box, add its scene_all path to `config.SCENE_ALL_BUILDS[<OS>]` rather than re-hardcoding it in scripts.
3. **Xvfb (Linux only)** — `sudo apt install -y xvfb`. The wrapper requires it; pass `USE_XVFB=0` only if you have a real X display exported as `DISPLAY` and don't mind it going to sleep mid-run.
4. **OpenRouter key** — `source tmp/secrets.sh` (or `export OPENROUTER_API_KEY=…`) before invoking the wrapper for `agent` mode runs.

---

## Quickstart

macOS:
```bash
source tmp/secrets.sh
bash shs/run_headless_benchmark.sh scene1 google/gemini-3-flash-preview
```

Ubuntu Linux:
```bash
source tmp/secrets.sh
bash shs/run_headless_benchmark.sh scene1 google/gemini-3-flash-preview
```

This iterates every point under `input_points.json["scene1"]` and produces (per-run
sub-dirs/CSVs are prefixed by the **baseline token** — `llm_*` here; `astar_*`, `random_*`, … otherwise):

```
outputs/<scene_code>/<point_id>/<model_short_name>/
├── llm_fp/           # ego camera frames (PNG)
├── llm_minimap/      # raw minimap frames
├── llm_depth/        # depth frames
├── llm_minimap_target/   # minimap annotated with curr + target dots
├── llm_actions.csv   # per-step action + pose + distance log
├── agent_qa.txt      # LLM prompts + responses, per step
├── results.csv       # one row summary of this run
└── unity_log.txt     # Unity-side log
```

`results.csv` stores navigation coordinates and final distance only in Unity
world units: `init_world_x/z`, `target_world_x/z`, `final_world_x/z`, and
`distance_world` (meters). Minimap pixels remain in the per-step action CSV for
trajectory visualization and image-based diagnostics.

The wrapper reads spawn + target from `input_points.json`, picks a free TCP base port for each
point (so multiple scenes can run in parallel from different terminals), and dispatches one
`python -m nav.scripts.run_benchmark_cell` per point.

The full list of valid `<scene_code>` values (and their `scene_id` mapping) is documented in [`scene_list.md`](scene_list.md). For the Unity scene files and Python/Unity interface details, see [`scene_files_and_interfaces.md`](scene_files_and_interfaces.md).

### Baselines and sweeps

- **Non-LLM baselines** (no API key): set `BASELINE` ∈ `{astar, bc, random}`, e.g.
  `BASELINE=astar bash shs/run_headless_benchmark.sh scene1`. The output subdir is named by the
  baseline token instead of a model name. `llm` (default) is the only one that calls OpenRouter.
  The same routing interface can be extended for additional baselines. For A* commands, tuning, debug visualizations, and extension notes, see [`astar_workflow.md`](astar_workflow.md).
  For BC data collection, training, and checkpoint inference, see [`bc_workflow.md`](bc_workflow.md).
- **Grid sweeps** across `model × scene × point × seed × vision × history_size` go through the
  orchestrator: `python -m nav.scripts.run_benchmark_grid --models … --scenes … [--history_sizes 0 5 10]`.
  Aggregates land under `analysis/grid_runs/<timestamp>/`; non-default history sizes are routed under
  `outputs/_history_size/hs<k>/` so the stats loader isn't polluted. Omit `--history_sizes` for a normal
  single-history sweep.

### Shell wrapper variables

The shell wrappers use environment variables as lightweight "macros": set them before the command to override defaults without editing the script.

Example:

```bash
BASELINE=astar MAX_STEPS=100 bash shs/run_headless_benchmark.sh scene1
```

#### `shs/run_headless_benchmark.sh`

This is the general benchmark wrapper for `llm`, `astar`, `bc`, and `random`.

| Variable | Default | Meaning |
|---|---|---|
| `BASELINE` | `llm` | Decision backend: `llm`, `astar`, `bc`, or `random`. |
| `MODEL_ID` | `google/gemini-3-flash-preview` | OpenRouter model id for `BASELINE=llm`. The second positional arg overrides this too. |
| `SCENE_ALL_APP` | `auto` | macOS Unity runtime path override. |
| `SCENE_ALL_BIN` | `auto` | Linux Unity runtime path override. |
| `SCENE_ID` | derived from `scene_code` | Overrides the wrapper's `scene_code -> scene_id` mapping. Useful only if a local runtime build has a different scene order. |
| `MAX_STEPS` | `70` | Per-point decision-step budget. |
| `REACH_M` | `2.0` | Success radius in Unity world meters. |
| `EGO_WIDTH` | `512` | Width of the egocentric RGB and depth observations and saved frames. |
| `EGO_HEIGHT` | `512` | Height of the egocentric RGB and depth observations and saved frames. |
| `DYNAMIC_OBJECTS` | `moving` | `moving` runs environment motion; `static` freezes environment objects while leaving the navigation agent controllable. |
| `HUMAN_SPEED_MPS` | `1.2` | Absolute worker and pedestrian speed in meters/second. |
| `VEHICLE_SPEED_MPS` | `2.5` | Absolute forklift and vehicle speed in meters/second. |
| `ROBOT_SPEED_MPS` | `1.5` | Absolute robot, AGV, and AMR speed in meters/second. |
| `<CATEGORY>_SPEED_MIN_MPS`, `<CATEGORY>_SPEED_MAX_MPS` | unset | Optional deterministic range for one category; both values are required. |
| `MOTION_RANDOM_SEED` | `0` | Base seed for independent category-speed sampling. |
| `GLOBAL_LIGHT_INTENSITY` | unset | Optional fixed global multiplier for Unity scene lighting. `LIGHT_INTENSITY_MULTIPLIER` is an equivalent alias. |
| `LIGHT_INTENSITY_MIN`, `LIGHT_INTENSITY_MAX` | unset | Optional deterministic per-run multiplier range. Both values are required. |
| `LIGHT_RANDOM_SEED` | `0` | Base seed for range sampling. |
| `LIGHT_FIXED_EXPOSURE` | `9.0` | HDRP fixed exposure EV used when runtime lighting is enabled. |
| `PYTHON_BIN` | `<repo>/.venv/bin/python` | Python interpreter used by the wrapper. |
| `USE_XVFB` | `1` on Linux | Whether to wrap Unity with `xvfb-run` on Linux. |
| `XVFB_SCREEN` | `1724x1024x24` | Virtual display size/depth passed to `xvfb-run`. |
| `INDUSTRYNAV_UNITY_BATCHMODE` | `0` on Linux, `1` elsewhere | Whether Python passes `-batchmode` to Unity. Linux defaults to `0` to avoid Input System startup crashes seen in some builds. |
| `OPENROUTER_API_KEY` | unset | Required only for `BASELINE=llm`. |

Output naming:

- `BASELINE=llm` writes to `outputs/<scene_code>/<point_id>/<model_short_name>/`.
- Non-LLM baselines write to `outputs/<scene_code>/<point_id>/<baseline>/`.

#### `shs/run_Astar.sh`

This is a convenience wrapper around `run_benchmark_cell --baseline astar`. It supports all variables above that relate to runtime launch, plus A*-specific controls.

| Variable | Default | Meaning |
|---|---|---|
| `INPUT_FILE` | `<repo>/input_points.json` | Alternate point JSON. |
| `MAX_STEPS` | unset | Explicit fixed A* budget; setting it disables the wrapper's dynamic default. |
| `ASTAR_DYNAMIC_STEP_BUDGET` | `1` | Scale the budget from direct distance and A* route length. |
| `ASTAR_STEP_BUDGET_MIN`, `ASTAR_STEP_BUDGET_MAX` | `50`, `160` | Dynamic budget bounds. |
| `ASTAR_STEPS_PER_PATH_METER`, `ASTAR_STEP_BUDGET_OVERHEAD` | `1.25`, `20` | Dynamic budget slope and turn/recovery reserve. |
| `SIM_STEPS_PER_DECISION` | `2` | Unity simulation steps per A* action. |
| `REACH_M` | `2.0` | Success radius in Unity world meters. |
| `MODALITIES` | `ego,minimap,depth` | Saved sensor streams. |
| `EGO_WIDTH`, `EGO_HEIGHT` | `512`, `512` | Egocentric RGB and depth observation size. |
| `MINIMAP_WIDTH`, `MINIMAP_HEIGHT` | `862`, derived | Set either minimap dimension; the other is derived using `862:512`. |
| `DYNAMIC_OBJECTS` | `moving` | `moving` or `static`; forwarded to `--dynamic_objects`. |
| `HUMAN_SPEED_MPS` | `1.2` | Absolute worker and pedestrian speed in meters/second. |
| `VEHICLE_SPEED_MPS` | `2.5` | Absolute forklift and vehicle speed in meters/second. |
| `ROBOT_SPEED_MPS` | `1.5` | Absolute robot, AGV, and AMR speed in meters/second. |
| `<CATEGORY>_SPEED_MIN_MPS`, `<CATEGORY>_SPEED_MAX_MPS` | unset | Deterministic per-run range for one category. |
| `MOTION_RANDOM_SEED` | `0` | Category-speed sampling seed. |
| `LIGHT_INTENSITY_MULTIPLIER` | unset | Fixed global light multiplier. |
| `LIGHT_INTENSITY_MIN`, `LIGHT_INTENSITY_MAX` | unset | Deterministic per-run multiplier range. |
| `LIGHT_RANDOM_SEED`, `LIGHT_FIXED_EXPOSURE` | `0`, `9.0` | Sampling seed and fixed HDRP exposure EV. |
| `MARKER_SOURCE` | `vector` | `vector` projects Unity position/heading; `red` uses legacy red-marker detection. |
| `HIDE_UNITY_RED_MARKER` | `1` | Hide old Unity red marker when drawing Python-side vector marker. |
| `RUN_NAME` | `astar` | Output directory name under `outputs/<scene_code>/<point_id>/`. |
| `ASTAR_DEBUG_VIZ` | `0` | Save A* walkable-grid/path debug frames. |
| `ASTAR_DEBUG_DIR` | `<frame_save_dir>/astar_debug` | Explicit debug frame directory. |
| `ASTAR_OBSTACLE_CLEARANCE_M` | `0.6` | Physical obstacle clearance in Unity world meters. |
| `ASTAR_PROXY_STOP_DISTANCE_M` | `4.9` | World-distance threshold for switching from a reached proxy to direct terminal approach. |
| `DRY_RUN` | `0` | Print the generated command without launching Unity. |
| `BASE_PORT_START` | `5507` | Fallback base port if automatic free-port probing fails. |

Common A* examples:

World coordinates are required for all navigation distance thresholds. A*
converts metric grid and clearance settings to pixels internally; marker pixels
remain an internal image-processing detail.

```bash
# Freeze workers, spline vehicles, animations, and environment physics.
DYNAMIC_OBJECTS=static bash shs/run_Astar.sh scene1 point1

# Equivalent direct Python flag.
python -m nav.scripts.run_benchmark_cell --dynamic_objects static ...
```

```bash
ASTAR_DEBUG_VIZ=1 bash shs/run_Astar.sh scene1 point1
ASTAR_OBSTACLE_CLEARANCE_M=0.9 RUN_NAME=astar_clearance09 bash shs/run_Astar.sh scene1 point1
ASTAR_STEP_BUDGET_MAX=200 bash shs/run_Astar.sh all
MAX_STEPS=120 bash shs/run_Astar.sh all  # fixed-budget override
```

#### `shs/train_bc.sh`

This wrapper is intentionally small. It forwards most configuration to `python -m nav.scripts.train_bc`.

| Variable / arg | Default | Meaning |
|---|---|---|
| first positional arg | `resnet50` | BC preset: `cnn`, `resnet50`, or `dinov2`. |
| `PYTHON_BIN` | `<repo>/.venv/bin/python` | Python interpreter. |
| extra args | none | Forwarded directly to `nav.scripts.train_bc`, e.g. `--data_root`, `--epochs`, `--output_dir`. |

Example:

```bash
bash shs/train_bc.sh resnet50 --data_root collect_data --epochs 20
```

### Single-point run (for debugging)

Call the Python script directly. macOS:

```bash
SCENE_ALL_APP=/home/liyifa11/MyCodes/IndustryNav/scene_files/Linux/scene_all/scene_all.x86_64
.venv/bin/python -m nav.scripts.run_benchmark_cell \
  --baseline llm \
  --file_name "$SCENE_ALL_APP" \
  --scene_id 0 --base_port 5520 --max_steps 70 \
  --ego_width 512 --ego_height 512 \
  --frame_save_dir outputs/scene1/point1/gemini-3-flash-preview \
  --model_id google/gemini-3-flash-preview \
  --init_world_x 31.0 --init_world_z 49.63 --init_curr_direction 180 \
  --target_x 550 --target_y 450
```

Ubuntu Linux (note the `xvfb-run` prefix and the ELF path):

```bash
xvfb-run -a -s "-screen 0 1724x1024x24" .venv/bin/python -m nav.scripts.run_benchmark_cell \
  --baseline llm \
  --file_name /mnt/ss2/devops/sandbox/industrynav2/client/scene_all/scene_all.x86_64 \
  --scene_id 0 --base_port 5520 --max_steps 70 \
  --ego_width 512 --ego_height 512 \
  --frame_save_dir outputs/scene1/point1/gemini-3-flash-preview \
  --model_id google/gemini-3-flash-preview \
  --init_world_x 31.0 --init_world_z 49.63 --init_curr_direction 180 \
  --target_x 550 --target_y 450
```

### Camera sensor resolution

`--screen_width` and `--screen_height` control the Unity Player window (and are
normally matched by `XVFB_SCREEN` on Linux). They do not change ML-Agents camera
observations. Use `--ego_width` and `--ego_height`, or the wrapper variables
`EGO_WIDTH` and `EGO_HEIGHT`, to resize both the `AgentSensor` RGB observation
and the `DepthSensor` observation:

| Invocation | Resolution parameters |
|---|---|
| `run_benchmark_cell`, `run_benchmark_grid`, `collect_data` | `--ego_width <W> --ego_height <H>` |
| `run_headless_benchmark.sh`, `run_Astar.sh` | `EGO_WIDTH=<W> EGO_HEIGHT=<H>` |
| `run_benchmark_cell`, `run_benchmark_grid`, `collect_data` | `--minimap_width <W>` or `--minimap_height <H>` |
| `run_headless_benchmark.sh`, `run_Astar.sh` | `MINIMAP_WIDTH=<W>` or `MINIMAP_HEIGHT=<H>` |

```bash
EGO_WIDTH=768 EGO_HEIGHT=432 bash shs/run_headless_benchmark.sh scene1
MINIMAP_WIDTH=431 bash shs/run_headless_benchmark.sh scene1

python -m nav.scripts.run_benchmark_cell \
  --ego_width 768 --ego_height 432 --minimap_width 431 \
  ...
```

### Environment motion

The benchmark, grid runner, and data collector accept
`--dynamic_objects moving|static`; the default is `moving`. Static mode pauses
non-agent spline motion, animation, worker/traffic scripts, physics, NavMesh
agents, particles, and timelines. The controlled `WarehouseAgent` and its
cameras remain active.

```bash
DYNAMIC_OBJECTS=static bash shs/run_headless_benchmark.sh scene1
DYNAMIC_OBJECTS=static bash shs/run_Astar.sh scene1 point1
python -m nav.scripts.collect_data --dynamic_objects static ...
python -m nav.scripts.run_benchmark_grid --dynamic_objects static ...
```

Human, vehicle, and robot movers use separate absolute speeds in Unity world
meters/second. The defaults are `1.2`, `2.5`, and `1.5 m/s`. Use fixed values
or independent ranges:

```bash
HUMAN_SPEED_MPS=1.0 VEHICLE_SPEED_MPS=3.0 ROBOT_SPEED_MPS=1.4 \
  bash shs/run_headless_benchmark.sh scene1
HUMAN_SPEED_MIN_MPS=0.9 HUMAN_SPEED_MAX_MPS=1.4 \
VEHICLE_SPEED_MIN_MPS=2.0 VEHICLE_SPEED_MAX_MPS=3.5 \
ROBOT_SPEED_MIN_MPS=1.0 ROBOT_SPEED_MAX_MPS=2.0 MOTION_RANDOM_SEED=42 \
  bash shs/run_Astar.sh scene1 point1
python -m nav.scripts.collect_data \
  --human_speed_mps 1.0 --vehicle_speed_mps 3.0 --robot_speed_mps 1.4 ...
python -m nav.scripts.run_benchmark_grid \
  --human_speed_min_mps 0.9 --human_speed_max_mps 1.4 \
  --vehicle_speed_min_mps 2.0 --vehicle_speed_max_mps 3.5 \
  --robot_speed_min_mps 1.0 --robot_speed_max_mps 2.0 \
  --motion_random_seed 42 ...
```

Each configured range is sampled independently from the base seed plus category,
scene, point, and benchmark seed identifiers. Unity applies absolute speed to
classified Spline and NavMesh movement and synchronizes child Animator playback.
The Gley pedestrian system reads the human speed directly. The controlled
`WarehouseAgent` is excluded. Use `--dynamic_objects static` when all
environment motion must be frozen.

### Global lighting

Use `--global_light_intensity <value>` for a fixed multiplier, or provide
both `--light_intensity_min` and `--light_intensity_max` to sample one global
multiplier per run. Sampling is deterministic for the base seed plus scene,
point, and benchmark seed identifiers. Each Unity Light keeps its authored
relative intensity; repeated episode resets do not accumulate scaling.

```bash
GLOBAL_LIGHT_INTENSITY=0.8 bash shs/run_Astar.sh scene1 point1
LIGHT_INTENSITY_MIN=0.7 LIGHT_INTENSITY_MAX=1.3 LIGHT_RANDOM_SEED=42 \
  bash shs/run_headless_benchmark.sh scene1
python -m nav.scripts.collect_data \
  --light_intensity_min 0.7 --light_intensity_max 1.3 \
  --light_random_seed 42 ...
```

When enabled, HDRP exposure is fixed at `9.0` EV by default; override it with
`--light_fixed_exposure`. The actual sampled multiplier is stored in each
`results.csv`. Baked indirect lighting does not change at runtime.

Runtime scaling changes realtime lights and the direct component of mixed
lights. Indirect illumination already baked into lightmaps is unchanged.

The selected dimensions become part of the ML-Agents behavior specification at
startup and must be positive integers. Egocentric RGB/depth PNGs and depth NPY
maps are saved at that size. Either minimap dimension is derived from the
canonical `862:512` aspect ratio when omitted; mismatched explicit dimensions
are rejected.
Python scales canonical target coordinates and all A* image-space parameters
into the selected runtime size. A* converts path samples back to Unity world
coordinates for waypoint spacing, lookahead, replanning, heading, and success
checks. CSV files retain both metric and canonical-pixel distances.
These features require a Unity client rebuilt after the corresponding runtime
resolution support was added to `WarehouseAgent.cs`; older clients ignore the
new launch arguments.

### Cheap dry boot (no API key, 2 random steps)

Useful when you've changed the script or the Unity client and want to verify the side-channel handshake before spending tokens. macOS:

```bash
SCENE_ALL_APP=/home/liyifa11/MyCodes/IndustryNav/scene_files/Linux/scene_all/scene_all.x86_64
.venv/bin/python -m nav.scripts.run_benchmark_cell \
  --baseline random \
  --file_name "$SCENE_ALL_APP" \
  --scene_id 0 --max_steps 2 \
  --frame_save_dir outputs/_dryboot/scene1 \
  --init_world_x 31.0 --init_world_z 49.63 --init_curr_direction 180 \
  --target_x 550 --target_y 450
```

Ubuntu Linux:

```bash
xvfb-run -a -s "-screen 0 1724x1024x24" .venv/bin/python -m nav.scripts.run_benchmark_cell \
  --baseline random \
  --file_name /mnt/ss2/devops/sandbox/industrynav2/client/scene_all/scene_all.x86_64 \
  --scene_id 0 --max_steps 2 \
  --frame_save_dir outputs/_dryboot/scene1 \
  --init_world_x 31.0 --init_world_z 49.63 --init_curr_direction 180 \
  --target_x 550 --target_y 450
```

Look for these log lines — they confirm the Unity build is honoring the v2 side-channel protocol:

```
Spawn world(31.00, 49.63) yaw=180.0
Target visual(550, 450) -> unity(551, 455) -> world(9.63, 25.66)
```

(For the legacy pixel-spawn path you'd instead see `Spawn visual(...) -> unity(...) -> world(...)`.)

---

## Coordinate interpretation (important)

`input_points.json` mixes two coordinate spaces, and getting them wrong silently spawns the agent in invalid space:

- **`start.{x, z, direction}` are Unity world coordinates.** The fields are named `x`/`z` (Unity's horizontal axes) and `direction` is yaw in degrees. Pass these to `nav.scripts.run_benchmark_cell` as `--init_world_x` / `--init_world_z` / `--init_curr_direction`. The unified client spawns directly at that world point.
- **`target.{x, y}` are visual minimap pixels** (range 0..862 × 0..512). Pass as `--target_x` / `--target_y`. The script converts to Unity pixel via the letterbox-aware margin and the client maps to world internally.

The legacy `--init_curr_x` / `--init_curr_y` (visual minimap pixels) are still accepted for the spawn, but only used when the world args are absent. **Do not feed `start.x`/`start.z` into `--init_curr_x/y`** — their magnitudes (~30–50) are world units, not pixels, so treating them as pixels lands you at the top-left edge of the minimap and the agent walks into a wall. `nav.scripts.run_benchmark_grid` already routes them correctly.

---

## Models

Tested OpenRouter model ids:

| Model | OpenRouter id | Relative input cost |
|---|---|---|
| Gemini 3 Flash Preview | `google/gemini-3-flash-preview` | cheapest |
| Claude Sonnet 4.6 | `anthropic/claude-sonnet-4.6` | ~10× Gemini Flash |
| GPT-5.2 | `openai/gpt-5.2` | high |
| Qwen 3.5 Plus | `qwen/qwen3.5-plus-02-15` | low |

Pass via `bash shs/run_headless_benchmark.sh <scene_code> <model_id>`.

---

## Important: do not clobber historical results

- `experiment_results_v6.csv`, `eval_results.xlsx`, `history_size_metrics.xlsx` at the repo root are legit historical benchmark artifacts. **`nav.scripts.run_benchmark_cell` never writes to those by default** — it writes `results.csv` under `<frame_save_dir>` instead. Pass `--results_csv` only if you actually want to merge into an existing aggregate file (and even then, the script appends).
- The `outputs/` tree is the canonical location for new runs. Treat new runs as additive — pick a fresh `<model_short_name>` directory if you don't want to overwrite a previous run's frames.

---

## Where final aggregate metrics come from

`nav.scripts.run_benchmark_cell` does **not** compute final benchmark scores by itself. It only writes raw per-step actions + frames + a one-row session summary. Aggregate metrics (success rate, total steps, distance, etc. — the columns in `eval_results.xlsx`) are produced by the **post-hoc** `nav.scripts.eval_run`:

```bash
.venv/bin/python -m nav.scripts.eval_run --input-dir outputs/scene1/point1/gemini-3-flash-preview
```

Runtime and post-hoc evaluation use the final Unity world distance with a
default threshold of `2.0 m`. Set `--reach_m` for runtime and
`--success-dist-m` for post-hoc evaluation. Runs without `distance_world` are
not assigned success from their pixel distance.

Each saved depth step contains two files. `<step>.png` is a fixed-scale
grayscale view of the Unity sensor output; `<step>.npy` is a float32 depth map
in meters, decoded from sRGB and clipped by the Unity shader at 20 m. Warning
rate evaluation reads the `.npy` files. Do not compare old min-max-normalized
depth PNG brightness directly with newly generated runs.

Aggregation across runs into `eval_results.xlsx` is a manual / sheet-export step downstream of `nav.scripts.eval_run`.

---

## Headless caveats

Despite the name, this script does **not** run with `-nographics`. The unified Unity build needs an active renderer to populate the camera/depth sensors; passing `-nographics` or `no_graphics=True` (in mlagents) results in either a SIGKILL on launch or blank sensor frames. It always uses `no_graphics=False`; macOS adds `-batchmode`, while Linux defaults to a windowed player inside Xvfb because some Linux builds crash during Input System initialization in batch mode. Set `INDUSTRYNAV_UNITY_BATCHMODE=1` to override the Linux default. Python opens no display windows of its own.

- **macOS** — the Unity window is visible on the host's display while the run is in progress; that's expected.
- **Ubuntu Linux** — the wrapper invokes Python under `xvfb-run -a`, which provisions a fresh virtual display per invocation. No Unity window is visible to a remote SSH user. Don't rely on `DISPLAY` pointing at a real attached monitor — TVs sleep, login sessions get reclaimed, parallel runs collide on the same display. Use Xvfb. (Override with `USE_XVFB=0` only if you've thought about it.)

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Environment shut down with return code -9 (SIGKILL).` (macOS) | Gatekeeper quarantine or `-nographics` flag | `xattr -dr com.apple.quarantine $SCENE_ALL_APP`; remove `-nographics`. |
| `Environment shut down with return code -11 (SIGSEGV).` (Linux, at `UnityEnvironment(...)` init) | No display surface — running without `xvfb-run` and without a real `DISPLAY` | Wrap the invocation in `xvfb-run -a -s "-screen 0 1724x1024x24"`, or use `shs/run_headless_benchmark.sh`. Verified failure mode on this build. |
| `Minimap unavailable; cannot compute coordinate margin.` (right after the brain connects) | Camera sensors are returning blank frames. On Linux this happens when `-nographics` is in `additional_args` or `no_graphics=True` is passed to `UnityEnvironment(...)` — Unity launches but never populates sensor textures. | Remove `-nographics`; keep `no_graphics=False`. Verified on Linux (under `xvfb-run`, both flags reproduce this exact error). |
| `xvfb-run: error: Xvfb failed to start.` | Linux: Xvfb not installed or `/tmp` unwritable | `sudo apt install -y xvfb`; confirm `/tmp` is writable. |
| `dict contains fields not in fieldnames: 'X'` | Schema drift in `utils.append_results_csv` | Either drop `X` from the result dict in `nav.scripts.run_benchmark_cell` or extend the schema in `utils.append_results_csv`. |
| LLM connection failure | `OPENROUTER_API_KEY` missing or invalid | `source tmp/secrets.sh`; verify key. |
| Unity port already in use | Stale `worker_id` / `base_port` collision | The wrapper picks a fresh free port per point; if calling Python directly, pass a fresh `--base_port`. |
