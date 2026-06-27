"""Central configuration: constants, env-var-derived paths, and small helpers.

Per ``docs/cleanup.md``: every runtime hyperparameter, macro, or env variable
lives here. The only intentional exception is ``input_points.json`` at repo
root — it's scene-level start/target data that benefits from direct hand
editing.

Constants are grouped by domain. Add new constants in the matching section
or create a new one with a heading comment.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from nav.prompts import PromptName, path_of as _prompt_path_of

#: Absolute path to the repo root (this file is ``<repo>/nav/config.py``). Used
#: to anchor in-repo, machine-independent locations (the Unity client + project)
#: so callers can rely on auto-discovery instead of per-developer absolute paths.
REPO_ROOT: Path = Path(__file__).resolve().parents[1]

#: In-repo home of the compiled Unity client(s). Gitignored (large binaries);
#: each dev drops their OS build in here. macOS: ``unity_client/scene_all.app``;
#: Linux: ``unity_client/scene_all/scene_all.x86_64``.
UNITY_CLIENT_DIR: Path = REPO_ROOT / "unity_client"

#: In-repo home of the Unity *project* (editor-loadable sources). Gitignored.
#: Not consumed by the Python runtime today; reserved for editor-mode tooling.
UNITY_PROJECT_DIR: Path = REPO_ROOT / "unity_project"

# ---------------------------------------------------------------------------
# Action spaces
# ---------------------------------------------------------------------------
# Two distinct action-space magnitudes are in use. They are *not* aliases of
# each other; they map the same action names to different continuous control
# magnitudes. Pick one explicitly at the call site — never default.

#: Larger-magnitude action space (legacy default for early benchmark scripts).
#: Currently consumed only by the deprecated ``play_mlagents_*`` editor-mode
#: scripts; the surviving entry points use ``ACTION_SPACE_ANNOTATION``.
ACTION_SPACE_AGENTS: Dict[str, float] = {
    "forward": 25.5,
    "turn right": 45,
    "turn left": -45,
    "stop": 0,
}

#: Smaller-magnitude action space used by the surviving benchmark and
#: data-collection entry points. Finer granularity gave better navigation
#: results during experimentation.
ACTION_SPACE_ANNOTATION: Dict[str, float] = {
    "forward": 15,
    "turn right": 9,
    "turn left": -9,
    "stop": 0,
}

# ---------------------------------------------------------------------------
# Unity client / ml-agents channel
# ---------------------------------------------------------------------------

#: Behavior name registered by the unified ``scene_all`` client. Used when
#: calling ``env.set_actions(BEHAVIOR_NAME, ActionTuple(...))``.
BEHAVIOR_NAME: str = "WarehouseAgent?team=0"

#: Unity quality level used for CameraSensor capture. In the current Unity
#: project, quality index 0 maps to a Ray Tracing / Realtime GI setting that
#: introduces visible stochastic speckle noise in minimap/ego frames. Level 3
#: is the stable capture setting used by both benchmark and data collection.
UNITY_ENGINE_QUALITY_LEVEL: int = 3

# ---------------------------------------------------------------------------
# Red-dot detection (agent position marker on the minimap)
# ---------------------------------------------------------------------------
# HSV target + tolerances for locating the agent's red position marker.
# Hue/sat/val are on the human 0-360 / 0-1 scale; the detector converts to
# OpenCV's 0-179 / 0-255 internally. Tuned for the scene_all minimap's
# saturated-red agent triangle.

#: Target hue in degrees [0, 360) for the red agent marker.
RED_DOT_TARGET_HUE_DEG: float = 350.0
#: Target saturation [0, 1].
RED_DOT_TARGET_SAT: float = 1.0
#: Target value/brightness [0, 1].
RED_DOT_TARGET_VAL: float = 0.827
#: ± hue tolerance in degrees.
RED_DOT_HUE_TOL_DEG: float = 15.0
#: ± saturation tolerance as a fraction of the target.
RED_DOT_SAT_TOL_FRAC: float = 0.25
#: ± value tolerance as a fraction of the target.
RED_DOT_VAL_TOL_FRAC: float = 0.20
#: Ignore detected blobs smaller than this many pixels (noise rejection).
RED_DOT_MIN_BLOB_AREA: int = 5

#: Unity minimap dimensions in pixels (width, height). The pixel→world
#: transform lives on the Unity side; Python only needs these to size the
#: minimap canvas it draws overlays onto.
UNITY_MAP_SIZE: Tuple[float, float] = (862.0, 512.0)

#: Mapping of sensor modality name to the index used in ``env.get_steps``
#: observation tuples. Order is determined by the Unity client and must
#: match the agent's sensor registration order.
MODALITY_TO_IDX: Dict[str, int] = {"ego": 0, "depth": 1, "minimap": 2}

# ---------------------------------------------------------------------------
# Unity side-channel protocol
# ---------------------------------------------------------------------------
# UUIDs + message opcodes that must match the C# side-channel registration in
# the scene_all client. Changing these breaks the Python↔Unity handshake.

#: UUID for the minimap-bounds side channel.
SIDE_CHANNEL_BOUNDS_UUID: str = "591d9d3a-8a1f-4e9d-8a3b-5e6c7d8e9f0a"
#: UUID for the spawn/target mapping (ack) side channel.
SIDE_CHANNEL_TARGET_UUID: str = "11111111-2222-3333-4444-555555555555"
#: Opcode: the message carries a spawn pixel→world mapping ack.
SIDE_CHANNEL_MSG_SPAWN_MAPPING: int = 1001
#: Opcode: the message carries a target pixel→world mapping ack.
SIDE_CHANNEL_MSG_TARGET_MAPPING: int = 1002

# ---------------------------------------------------------------------------
# Minimap warmup + edge detection
# ---------------------------------------------------------------------------
# After env.reset() the Unity camera sensors take several sim steps to render
# non-blank frames; we poll with STOP actions until the minimap has enough
# Canny edges to be usable.

#: Max STOP-step attempts to wait for a non-blank minimap frame.
MINIMAP_WARMUP_MAX_ATTEMPTS: int = 40
#: Minimum Canny-edge pixel count for a minimap frame to count as rendered.
MINIMAP_WARMUP_MIN_EDGES: int = 50
#: Canny hysteresis thresholds (low, high) for minimap edge detection.
MINIMAP_CANNY_LO: int = 30
MINIMAP_CANNY_HI: int = 100

# ---------------------------------------------------------------------------
# Decision routing
# ---------------------------------------------------------------------------

#: Substrings in an LLM's reasoning that mark a hard provider failure (abort
#: the run). Kept narrow so legitimate chain-of-thought mentioning "failed"
#: (e.g. "forward failed to change position") doesn't trip a false abort.
LLM_ERROR_SENTINELS: Tuple[str, ...] = (
    "api connection failed",
    "api error",
    "rate limit",
)

# ---------------------------------------------------------------------------
# Baselines (classical / non-VLLM navigation methods)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AStarParams:
    """Tuning defaults for the A* minimap baseline (``nav.baselines.astar``).

    These are the algorithm's fallback defaults; the benchmark runner can
    override any of them per-run (the run scripts expose matching CLI
    flags). Debug-viz / IO knobs are NOT here — those are runtime concerns
    passed to the baseline separately.
    """

    grid_step: int = 8
    obstacle_threshold: int = 55
    min_free_ratio: float = 0.55
    obstacle_inflate_px: int = 4
    waypoint_distance_px: float = 28.0
    waypoint_reach_px: float = 24.0
    path_replan_distance_px: float = 56.0
    target_change_replan_px: float = 8.0
    turn_tolerance_deg: float = 25.0
    forward_priority_tolerance_deg: float = 35.0
    drive_turn_tolerance_deg: float = 65.0
    lookahead_px: float = 80.0
    front_cone_deg: float = 140.0
    hysteresis_reset_deg: float = 45.0
    hysteresis_lock_deg: float = 165.0
    stuck_world_epsilon: float = 0.05
    stuck_px_epsilon: float = 3.0
    stuck_steps: int = 4
    recovery_turn_steps: int = 3


#: Shared default A* tuning params instance (the baseline's kwarg defaults
#: reference these fields).
ASTAR_DEFAULTS = AStarParams()


@dataclass(frozen=True)
class NaVidParams:
    """Default hyperparameters for the NaVid baseline adapter (``nav.baselines.navid``)."""

    max_new_tokens: int = 256
    temperature: float = 0.2
    conv_mode: str = "vicuna_v1"
    instruction: str = "Navigate to the target location using the visual observations."


#: Shared default NaVid params instance.
NAVID_DEFAULTS = NaVidParams()

#: Mapping of scene name to the integer ``scene_id`` env-parameter the
#: unified client expects. Scene names match folder names under
#: ``outputs/<scene>/`` and keys in ``input_points.json``.
SCENE_ID_MAP: Dict[str, int] = {
    "yifan1": 1, "yifan2": 2, "yifan3": 3, "yifan4": 4,
    "yicheng": 5,
    "lichi1": 6, "lichi2": 7,
    "xinyu1": 8, "xinyu2": 9,
    "anh1": 10, "anh2": 11, "anh3": 12,
}

# ---------------------------------------------------------------------------
# LLM call defaults
# ---------------------------------------------------------------------------

#: Default ``max_tokens`` for top-level benchmark LLM calls (``--max_tokens``
#: CLI default in the run scripts). Sized generously so a single reasoning +
#: JSON action response fits without truncation.
LLM_DEFAULT_MAX_TOKENS: int = 20000

#: Larger ``max_tokens`` budget for the multi-step sub-agents (decision
#: maker, global planner, local planner) that emit chained reasoning rather
#: than a single action JSON.
LLM_SUBAGENT_MAX_TOKENS: int = 50000

#: Request timeout (seconds) for OpenRouter calls. Sized for slow first-token
#: behavior on larger models behind the proxy.
LLM_REQUEST_TIMEOUT_SEC: int = 120

#: Default number of past (step, position, action, …) entries fed back into the
#: LLM prompt as navigation history. The benchmark entry's ``--history_size``
#: and the grid runner's ``--history_sizes`` sweep both default to this; the
#: grid keeps default-history runs in the canonical ``outputs/`` tree and routes
#: other sizes under ``outputs/_history_size/hs<k>/`` so the stats loader (which
#: only reads the canonical tree) isn't polluted.
LLM_DEFAULT_HISTORY_SIZE: int = 5

# ---------------------------------------------------------------------------
# Evaluation thresholds (post-hoc metrics)
# ---------------------------------------------------------------------------

#: Pixel-space distance below which a run is counted as a success. Applied
#: to the final minimap distance between the agent and the target.
EVAL_SUCCESS_DIST_PX: int = 65

#: Manhattan pixel-distance threshold below which a ``forward`` step is
#: counted as a collision (agent commanded forward but barely moved).
EVAL_COLLISION_PX_THRESH: int = 34

#: Depth (in metres) below which a frame counts as "warning" — i.e. the
#: agent is within ``EVAL_WARNING_THRESHOLD_M`` of an obstacle inside the
#: forward ROI of the depth map.
EVAL_WARNING_THRESHOLD_M: float = 0.3

#: Default trapezoidal ROI in the depth image for warning detection.
#: Margins/pads are fractions of image height/width. Tuned for the
#: scene_all client's first-person depth output.
EVAL_ROI_PARAMS: Dict[str, float] = {
    "bottom_margin": 0.20,
    "top_margin": 0.55,
    "bottom_pad": 0.12,
    "top_pad": 0.36,
}

#: Default output target for ``nav.scripts.aggregate_eval``. Lives under
#: ``analysis/`` per the cleanup policy: aggregate stats are git-tracked.
EVAL_AGGREGATE_DEFAULT_OUT: str = "analysis/aggregate_eval.xlsx"

#: Default log directory for ``nav.scripts.eval_run`` in glob mode (single
#: -dir mode logs into the run's own output directory instead).
EVAL_DEFAULT_LOG_DIR: str = "logs/eval"

# ---------------------------------------------------------------------------
# Result schemas (CSV)
# ---------------------------------------------------------------------------
# The two CSVs ``utils.append_results_csv`` and ``run_benchmark_grid`` write
# into. Keeping the schemas here means any column add/remove is a one-line
# config change with no schema drift between writer and reader.

#: Per-cell result row written by ``nav.utils.append_results_csv``.
#:
#: Schema notes:
#:   - ``exp_name`` is the legacy column name for what is now also called
#:     ``scene_name``. Both are populated with the same value (e.g. ``"yifan1"``)
#:     to keep historical readers — including ``experiment_results_v6.csv``
#:     consumers — working. New code should reference ``scene_name``.
#:   - ``scene_name``, ``point_id``, ``seed_id``, ``vision_input`` were added so
#:     post-hoc statistical analysis (paired permutation, scene-clustered
#:     bootstrap) can locate each run's identity directly from the CSV.
RESULTS_CSV_FIELDS: List[str] = [
    "timestamp",
    "exp_name",
    "scene_name",
    "point_id",
    "seed_id",
    "exec_mode",
    "provider",
    "model",
    "vision_input",
    "max_steps",
    "reach_px",
    "target_x",
    "target_y",
    "init_direction",
    "final_x",
    "final_y",
    "distance_px",
    "stop_reason",
    "steps_taken",
    "frame_sleep",
    "modalities",
    "sim_steps_per_decision",
    "marker_source",
    "spline_speed_multiplier",
]

#: Per-frame actions log written by the benchmark runners (one row per
#: agent decision). 24 columns covering action, control signal, agent
#: state in pixel + world coordinates, full rotation triplet (so
#: pitch/roll near-zero invariant stays grep-able), target, and current
#: distance in both pixel and world space.
#:
#: Consumed by ``nav.eval.metrics`` and ``nav.eval.collision``. The
#: benchmark run scripts still inline this header today; PR 5/8 will
#: switch them to import from here so writer + readers share one
#: source of truth.
ACTIONS_CSV_FIELDS: List[str] = [
    "step", "action", "move", "strafe", "look",
    "init_px", "init_py", "init_world_x", "init_world_z", "init_direction",
    "curr_px", "curr_py", "curr_world_x", "curr_world_y", "curr_world_z",
    "curr_direction_x", "curr_direction_y", "curr_direction_z",
    "target_px", "target_py", "target_world_x", "target_world_z",
    "distance_px", "distance_world",
]

#: Per-cell summary row written by ``run_benchmark_grid.append_grid_row``.
GRID_CSV_FIELDS: List[str] = [
    "timestamp", "scene_name", "point_id", "model", "seed_id", "vision_input",
    "history_size", "ok", "returncode", "duration_sec", "frame_save_dir",
    "results_csv_present", "log_path",
]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
# Default prompt choices for the grid runner's vision/no-vision branch.
# Source of truth for the prompt files themselves is :mod:`nav.prompts`;
# these constants are kept as absolute path *strings* so CLI consumers
# (``--prompt_file``) and subprocess invocations keep working unchanged.

#: Prompt used when vision (egocentric RGB) is sent to the LLM.
DEFAULT_PROMPT_VISION: str = str(_prompt_path_of(PromptName.EGO_STATE_HISTORY))

#: Prompt used when vision is disabled and only textual state is sent.
DEFAULT_PROMPT_NOVISION: str = str(_prompt_path_of(PromptName.STATE_HISTORY_NO_VISION))

# ---------------------------------------------------------------------------
# Analysis output layout
# ---------------------------------------------------------------------------

#: Root directory for git-tracked aggregate statistical outputs.
ANALYSIS_ROOT: Path = Path("analysis")

#: Default subdirectory under ``analysis/`` where statistical-analysis scripts
#: drop their outputs. Each "version" of the project gets its own subdir.
DEFAULT_ANALYSIS_SUBDIR: str = "nav1"

#: Metric columns subject to the supersession rule in
#: :func:`nav.stats.merge.merge`. The merger applies the rule per metric,
#: so a missing distance_ratio in the primary source can still fall back
#: to the secondary source without losing the other metrics for that cell.
STATS_METRIC_FIELDS: Tuple[str, ...] = (
    "success",
    "distance_px",
    "distance_ratio",
    "collision_rate",
    "warning_rate",
    "efficiency_steps",
    "steps_taken",
)

#: Ordered ``(short_key, source_field, human_label)`` triples that drive the
#: full-analysis report layout. WR + distance are appended conditionally at
#: runtime based on the input source's coverage (see
#: :func:`nav.stats.full._active_metrics`); the base triplet (SR/DR/CR) is
#: always present.
STATS_REPORT_BASE_METRICS: Tuple[Tuple[str, str, str], ...] = (
    ("sr", "success", "SR"),
    ("dr", "distance_ratio", "DR"),
    ("cr", "collision_rate", "CR"),
)

#: Optional metric triples that get appended to the report only when the
#: source data carries finite values for them. WR requires raw depth NPYs
#: (xlsx-derived rows only); ``distance_px`` requires grid-tree-walked
#: rows (xlsx leaves it as NaN).
STATS_REPORT_OPTIONAL_METRICS: Dict[str, Tuple[str, str, str]] = {
    "warning_rate": ("wr", "warning_rate", "WR"),
    "distance_px": ("dist", "distance_px", "Mean dist (px)"),
}

# ---------------------------------------------------------------------------
# Unity ``scene_all`` client discovery
# ---------------------------------------------------------------------------

#: Known local paths to the unified Unity client across developer machines.
#: Each entry should point at the file or directory ml-agents will actually
#: launch — i.e. the ``.app`` bundle on macOS, the ``.x86_64`` ELF on Linux,
#: the ``.exe`` on Windows.
#:
#: ``resolve_scene_all_path("auto")`` scans entries for the current OS and
#: returns the first existing path. The **preferred** location is in-repo under
#: ``unity_client/`` (anchored to :data:`REPO_ROOT`, so it works from any CWD and
#: needs no per-developer absolute path); legacy absolute paths are kept as
#: fallbacks. Drop your OS build into ``unity_client/`` (see the
#: ``unity_client_setup_*`` skills) and ``--file_name auto`` finds it.
SCENE_ALL_BUILDS: Dict[str, List[str]] = {
    "Darwin": [
        str(UNITY_CLIENT_DIR / "scene_all.app"),
        str(REPO_ROOT / "scene_files" / "mac" / "scene_all.app"),
        "/Users/lichili/dev/IndustryNav2/scene_all.app",  # legacy fallback
    ],
    "Linux": [
        str(UNITY_CLIENT_DIR / "scene_all" / "scene_all.x86_64"),
        "/mnt/ss2/devops/sandbox/industrynav2/client/scene_all/scene_all.x86_64",
        "/home/liyifa11/MyCodes/IndustryNav/scene_files/Linux/scene_all/scene_all.x86_64",
    ],
    "Windows": [
        # The Windows scene_all.zip unzips to a scene_all/ folder whose launch exe
        # is IndustryNav.exe (the product name — unlike the macOS/Linux builds which
        # use the scene_all name). Not validated end-to-end yet (#19).
        str(UNITY_CLIENT_DIR / "scene_all" / "IndustryNav.exe"),
    ],
}


def resolve_scene_all_path(value: str) -> str:
    """Resolve a ``--file_name`` argument to an existing local Unity-client path.

    - If ``value`` is ``"auto"`` (case-insensitive) or empty, scan
      ``SCENE_ALL_BUILDS`` for the current OS and return the first listed
      path that exists locally.
    - Otherwise treat ``value`` as a literal path and return it unchanged.
      The caller is responsible for validating that it exists.

    Raises ``FileNotFoundError`` on ``"auto"`` when no candidate exists,
    including the list of paths tried so the user can either pass an
    explicit ``--file_name`` or extend ``SCENE_ALL_BUILDS`` for the
    current OS.
    """
    if value is None:
        value = ""
    if value.strip().lower() not in {"", "auto"}:
        return value
    os_name = platform.system()
    candidates = SCENE_ALL_BUILDS.get(os_name, [])
    for c in candidates:
        if Path(c).exists():
            return c
    tried = "\n  ".join(candidates) if candidates else "(no entries for this OS)"
    raise FileNotFoundError(
        f"--file_name=auto: no local scene_all build found on {os_name}.\n"
        f"Tried:\n  {tried}\n"
        f"Either pass --file_name <explicit-path>, or add your local build "
        f"to config.SCENE_ALL_BUILDS[\"{os_name}\"]."
    )


# ---------------------------------------------------------------------------
# Behavior cloning (nav.models / nav.train)
# ---------------------------------------------------------------------------

#: Discrete navigation action -> integer class label used by the BC datasets,
#: training loss, and the inference controller. The 4-class space is fixed by
#: the collected teleop data schema (``keyboard_actions.csv``).
BC_ACTION_TO_LABEL: Dict[str, int] = {
    "forward": 0,
    "stop": 1,
    "turn right": 2,
    "turn left": 3,
}

#: Inverse mapping (label -> action name), used by the inference controller.
BC_LABEL_TO_ACTION: Dict[int, str] = {v: k for k, v in BC_ACTION_TO_LABEL.items()}

#: Per-episode teleop data schema written by the data-collection script and
#: consumed by the BC datasets: ``<data_root>/<scene>/<point>/{csv,rgb,depth}``.
BC_EPISODE_CSV: str = "keyboard_actions.csv"
BC_RGB_SUBDIR: str = "keyboard_fp"
BC_DEPTH_SUBDIR: str = "keyboard_depth"

#: ImageNet RGB normalization stats. Applied to RGB frames in the BC datasets
#: and the inference controller so train/inference preprocessing match the
#: timm-pretrained backbones' expected input distribution.
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

#: Default visual backbone for the BC policy heads when a caller doesn't pass
#: one. The training presets (:data:`BC_BASE_PRESETS`) always set it explicitly;
#: this is the fallback for direct/ad-hoc construction.
NAV_DEFAULT_BACKBONE: str = "resnet50"


@dataclass(frozen=True)
class NavPolicyArch:
    """Fixed architecture hyperparameters shared by the BC policy heads.

    These are the dimensions/regularization that don't vary across the
    ``--base`` presets (which only swap backbone + policy_type + sequence
    shape). The policy ``__init__`` kwargs default to these fields so callers
    can still override per-instance, but the canonical values live here.
    """

    visual_dim: int = 256
    state_hidden: int = 128
    fusion_hidden: int = 256
    goal_hidden: int = 128
    action_embed: int = 16
    lstm_hidden: int = 512
    d_model: int = 256
    nhead: int = 4
    dropout: float = 0.1
    diffusion_steps: int = 100
    diffusion_beta_start: float = 1e-4
    diffusion_beta_end: float = 2e-2


#: Shared architecture-defaults instance referenced by the policy signatures.
NAV_POLICY_ARCH = NavPolicyArch()


@dataclass(frozen=True)
class BCTrainConfig:
    """Full BC training configuration (was ``navigation_bc/train.py:TrainConfig``).

    The ``--base`` flag on ``nav.scripts.train_bc`` resolves to one of the
    :data:`BC_BASE_PRESETS` instances below; individual CLI flags then override
    fields via :func:`dataclasses.replace`. Frozen so a preset can't be mutated
    in place by accident — overrides always produce a new instance.
    """

    data_root: str = "collect_data"
    output_dir: str = "outputs/nav_bc"
    img_size: int = 256
    batch_size: int = 16
    num_workers: int = 4
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42
    state_mode: str = "relative"  # relative | absolute | full (mlp policy only)
    use_depth: bool = True
    use_rgb: bool = False
    normalize_rgb: bool = True
    split_ratios: Tuple[float, float, float] = (0.9, 0.1, 0.0)
    rgb_backbone: str = "resnet50"
    depth_backbone: str = "resnet50"
    pretrained_rgb: bool = True
    pretrained_depth: bool = False
    half_width: bool = True
    backbone_lr_scale: float = 0.1
    policy_type: str = "transformer"  # mlp | lstm | transformer | diffusion
    seq_len: int = 8
    num_layers: int = 2
    goal_rep: str = "cartesian"  # cartesian | polar
    chunk_size: int = 1  # action chunk size; 1 = standard single-step BC


#: Preset bundles for ``--base``. Each reproduces one of the legacy
#: ``shs/train_{vanilla_cnn,resnet50,dinov2}.sh`` scripts (the ad-hoc encoded
#: ``output_dir`` names are normalized to ``outputs/nav_bc_<base>``; the model +
#: training behavior is unchanged).
BC_BASE_PRESETS: Dict[str, BCTrainConfig] = {
    "cnn": BCTrainConfig(
        output_dir="outputs/nav_bc_cnn",
        policy_type="mlp",
    ),
    "resnet50": BCTrainConfig(
        output_dir="outputs/nav_bc_resnet50",
        policy_type="transformer",
        rgb_backbone="resnet50",
        depth_backbone="resnet50",
        half_width=False,
        seq_len=28,
        num_layers=3,
        chunk_size=4,
        batch_size=4,
        num_workers=8,
    ),
    "dinov2": BCTrainConfig(
        output_dir="outputs/nav_bc_dinov2",
        policy_type="transformer",
        rgb_backbone="vit_small_patch14_dinov2.lvd142m",
        depth_backbone="vit_small_patch14_dinov2.lvd142m",
        backbone_lr_scale=0.05,
    ),
}


# ---------------------------------------------------------------------------
# Benchmark baselines (run_headless_benchmark.py --baseline / nav.harness.routing)
# ---------------------------------------------------------------------------

#: Decision baselines selectable via ``--baseline`` on the unified-client
#: benchmark entry (and the routing dispatcher). ``random`` is the no-API
#: smoke baseline; ``llm`` is the OpenRouter agent; ``bc`` the behavior-cloning
#: controller; ``astar``/``navid`` the classical / VLN baselines. (Interactive
#: ``human`` teleop is NOT a headless baseline — it lives in the data-collection
#: flow against the unified client.)
BENCHMARK_BASELINES: List[str] = ["random", "llm", "bc", "astar", "navid"]

#: Per-run output subdirectories/CSVs are prefixed by the baseline token, so a
#: completed run dir looks like ``<run>/<baseline>_fp/`` etc. The eval loaders
#: probe this set of candidate prefixes; legacy tokens (``agent``/``bc_agent``/
#: ``manual``) are kept so historical output trees still resolve.
EVAL_RUN_PREFIXES: List[str] = [
    "llm", "bc", "astar", "navid", "random",  # current (--baseline tokens)
    "agent", "bc_agent", "manual",            # legacy/historical
]

#: Candidate names for the depth-frame subdirectory across baselines.
EVAL_DEPTH_DIR_CANDIDATES: List[str] = [f"{p}_depth" for p in EVAL_RUN_PREFIXES]

#: Candidate names for the per-frame actions CSV across baselines.
EVAL_ACTIONS_CSV_CANDIDATES: List[str] = [f"{p}_actions.csv" for p in EVAL_RUN_PREFIXES]
