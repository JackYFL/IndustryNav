"""Shared runtime constants and small path helpers for IndustryNav."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from nav.prompts import PromptName, path_of as _prompt_path_of


# Paths

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
UNITY_CLIENT_DIR: Path = REPO_ROOT / "unity_client"


# Action spaces

ACTION_SPACE_AGENTS: Dict[str, float] = {
    "forward": 25.5,
    "turn right": 45,
    "turn left": -45,
    "stop": 0,
}

ACTION_SPACE_ANNOTATION: Dict[str, float] = {
    "forward": 15,
    "turn right": 9,
    "turn left": -9,
    "stop": 0,
}


# Unity runtime

BEHAVIOR_NAME: str = "WarehouseAgent?team=0"
UNITY_ENGINE_QUALITY_LEVEL: int = 3
UNITY_MAP_SIZE: Tuple[float, float] = (862.0, 512.0)
UNITY_DEPTH_MAX_DISTANCE_M: float = 20.0
MODALITY_TO_IDX: Dict[str, int] = {"ego": 0, "depth": 1, "minimap": 2}

UNITY_SCENE_COUNT: int = 24
SCENE_ID_MAP: Dict[str, int] = {
    f"scene{scene_id + 1}": scene_id for scene_id in range(UNITY_SCENE_COUNT)
}
SCENE_CODES: Tuple[str, ...] = tuple(SCENE_ID_MAP)

# UUIDs/opcodes must match the Unity C# side-channel registration.
SIDE_CHANNEL_BOUNDS_UUID: str = "591d9d3a-8a1f-4e9d-8a3b-5e6c7d8e9f0a"
SIDE_CHANNEL_TARGET_UUID: str = "11111111-2222-3333-4444-555555555555"
SIDE_CHANNEL_MSG_SPAWN_MAPPING: int = 1001
SIDE_CHANNEL_MSG_TARGET_MAPPING: int = 1002


# Minimap perception

RED_DOT_TARGET_HUE_DEG: float = 350.0
RED_DOT_TARGET_SAT: float = 1.0
RED_DOT_TARGET_VAL: float = 0.827
RED_DOT_HUE_TOL_DEG: float = 15.0
RED_DOT_SAT_TOL_FRAC: float = 0.25
RED_DOT_VAL_TOL_FRAC: float = 0.20
RED_DOT_MIN_BLOB_AREA: int = 5

MINIMAP_WARMUP_MAX_ATTEMPTS: int = 40
MINIMAP_WARMUP_MIN_EDGES: int = 50
MINIMAP_CANNY_LO: int = 30
MINIMAP_CANNY_HI: int = 100


# LLM runtime

LLM_ERROR_SENTINELS: Tuple[str, ...] = (
    "api connection failed",
    "api error",
    "rate limit",
)
LLM_DEFAULT_MAX_TOKENS: int = 20000
LLM_SUBAGENT_MAX_TOKENS: int = 50000
LLM_REQUEST_TIMEOUT_SEC: int = 120
LLM_DEFAULT_HISTORY_SIZE: int = 5


# Environment runtime defaults

MOTION_CATEGORIES: Tuple[str, ...] = ("human", "vehicle", "robot")


@dataclass(frozen=True)
class MotionParams:
    """Absolute-speed defaults for dynamic Unity objects."""

    human_speed_mps: float = 1.2
    vehicle_speed_mps: float = 2.5
    robot_speed_mps: float = 1.5
    random_seed: int = 0


@dataclass(frozen=True)
class LightingParams:
    """Defaults for deterministic runtime lighting overrides."""

    intensity_multiplier: float = 1.0
    random_seed: int = 0
    fixed_exposure: float = 9.0


MOTION_DEFAULTS = MotionParams()
LIGHTING_DEFAULTS = LightingParams()


# Baseline defaults

@dataclass(frozen=True)
class AStarParams:
    """A* minimap planner defaults."""

    grid_cell_m: float = 0.6
    obstacle_threshold: int = 55
    obstacle_bright_threshold: int = 190
    # Low-contrast structures (painted rails, partition tops) render at almost
    # exactly the floor gray, so no global band separates them. They are found
    # instead by local deviation from a median-filtered background.
    contrast_background_px: int = 21
    contrast_threshold: int = 35
    contrast_min_area_px: int = 300
    min_free_ratio: float = 0.55
    obstacle_clearance_m: float = 0.6
    path_smoothing: bool = True
    path_corner_smoothing_m: float = 1.0
    stanley_gain: float = 1.0
    stanley_softening_m: float = 1.5
    stanley_max_correction_deg: float = 45.0
    stanley_forward_tolerance_deg: float = 8.0
    turn_tolerance_deg: float = 25.0
    forward_priority_tolerance_deg: float = 20.0
    drive_turn_tolerance_deg: float = 30.0
    waypoint_distance_m: float = 2.0
    waypoint_reach_m: float = 1.2
    path_replan_distance_m: float = 2.4
    dynamic_replan_lookahead_m: float = 8.0
    dynamic_replan_confirm_steps: int = 2
    terminal_approach_m: float = 3.0
    lookahead_m: float = 3.0
    front_cone_deg: float = 140.0
    hysteresis_reset_deg: float = 45.0
    hysteresis_lock_deg: float = 165.0
    stuck_distance_m: float = 0.05
    stuck_steps: int = 4
    recovery_turn_steps: int = 3
    stuck_block_ahead_m: float = 2.1
    stuck_block_radius_m: float = 1.35
    stuck_block_ttl_steps: int = 24
    proxy_stop_distance_m: float = 4.9
    step_budget_min: int = 50
    step_budget_max: int = 160
    steps_per_path_meter: float = 1.25
    step_budget_overhead: int = 20


ASTAR_DEFAULTS = AStarParams()


# Evaluation

DEFAULT_REACH_DISTANCE_M: float = 2.0
EVAL_SUCCESS_DIST_M: float = DEFAULT_REACH_DISTANCE_M
EVAL_COLLISION_PX_THRESH: int = 34
EVAL_WARNING_THRESHOLD_M: float = 0.3
EVAL_ROI_PARAMS: Dict[str, float] = {
    "bottom_margin": 0.20,
    "top_margin": 0.55,
    "bottom_pad": 0.12,
    "top_pad": 0.36,
}
EVAL_AGGREGATE_DEFAULT_OUT: str = "analysis/aggregate_eval.xlsx"
EVAL_DEFAULT_LOG_DIR: str = "logs/eval"

MOTION_SPEED_CSV_FIELDS: List[str] = [
    "motion_random_seed",
    *[
        f"{category}_speed_{suffix}"
        for category in MOTION_CATEGORIES
        for suffix in ("mode", "mps", "min_mps", "max_mps")
    ],
]

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
    "step_budget_mode",
    "initial_step_budget",
    "astar_max_planned_path_m",
    "reach_m",
    "init_world_x",
    "init_world_z",
    "init_direction",
    "target_world_x",
    "target_world_z",
    "final_world_x",
    "final_world_z",
    "distance_world",
    "stop_reason",
    "steps_taken",
    "frame_sleep",
    "modalities",
    "sim_steps_per_decision",
    "marker_source",
    "dynamic_objects",
    *MOTION_SPEED_CSV_FIELDS,
    "global_light_intensity",
    "light_randomization_mode",
    "light_intensity_multiplier",
    "light_intensity_min",
    "light_intensity_max",
    "light_random_seed",
    "light_fixed_exposure",
]

ACTIONS_CSV_FIELDS: List[str] = [
    "step",
    "action",
    "move",
    "strafe",
    "look",
    "init_px",
    "init_py",
    "init_world_x",
    "init_world_z",
    "init_direction",
    "curr_px",
    "curr_py",
    "curr_world_x",
    "curr_world_y",
    "curr_world_z",
    "curr_direction_x",
    "curr_direction_y",
    "curr_direction_z",
    "target_px",
    "target_py",
    "target_world_x",
    "target_world_z",
    "marker_source",
    "distance_px",
    "distance_world",
]

GRID_CSV_FIELDS: List[str] = [
    "timestamp",
    "scene_name",
    "point_id",
    "model",
    "seed_id",
    "vision_input",
    "history_size",
    "dynamic_objects",
    *MOTION_SPEED_CSV_FIELDS,
    "global_light_intensity",
    "light_randomization_mode",
    "light_intensity_multiplier",
    "light_intensity_min",
    "light_intensity_max",
    "light_random_seed",
    "light_fixed_exposure",
    "ok",
    "returncode",
    "duration_sec",
    "frame_save_dir",
    "results_csv_present",
    "log_path",
]


# Prompts and analysis output

DEFAULT_PROMPT_VISION: str = str(_prompt_path_of(PromptName.EGO_STATE_HISTORY))
DEFAULT_PROMPT_NOVISION: str = str(_prompt_path_of(PromptName.STATE_HISTORY_NO_VISION))

ANALYSIS_ROOT: Path = Path("analysis")
DEFAULT_ANALYSIS_SUBDIR: str = "nav1"
STATS_METRIC_FIELDS: Tuple[str, ...] = (
    "success",
    "distance_world",
    "distance_ratio",
    "collision_rate",
    "warning_rate",
    "efficiency_steps",
    "steps_taken",
)
STATS_REPORT_BASE_METRICS: Tuple[Tuple[str, str, str], ...] = (
    ("sr", "success", "SR"),
    ("dr", "distance_ratio", "DR"),
    ("cr", "collision_rate", "CR"),
)
STATS_REPORT_OPTIONAL_METRICS: Dict[str, Tuple[str, str, str]] = {
    "warning_rate": ("wr", "warning_rate", "WR"),
    "distance_world": ("dist", "distance_world", "Mean dist (m)"),
}


# Unity client discovery

SCENE_ALL_BUILDS: Dict[str, List[str]] = {
    "Darwin": [
        str(REPO_ROOT / "unity_clients" / "scene_all_24scenes_absolute_speed_v10.app"),
        str(REPO_ROOT / "unity_clients" / "scene_all_24scenes_motion_speed_v9.app"),
        str(REPO_ROOT / "unity_clients" / "scene_all_24scenes_dynamic_flag_v8.app"),
        str(REPO_ROOT / "unity_clients" / "scene_all_resizable.app"),
        str(REPO_ROOT / "unity_clients" / "scene_all_rebuilt.app"),
        str(REPO_ROOT / "unity_clients" / "scene_all.app"),
        str(UNITY_CLIENT_DIR / "scene_all.app"),
        str(REPO_ROOT / "scene_files" / "mac" / "scene_all.app"),
    ],
    "Linux": [
        str(REPO_ROOT / "clients" / "scene_all_24scenes_latest" / "scene_all.x86_64"),
        str(UNITY_CLIENT_DIR / "scene_all" / "scene_all.x86_64"),
        "/mnt/ss2/devops/sandbox/industrynav2/client/scene_all/scene_all.x86_64",
    ],
    "Windows": [
        str(UNITY_CLIENT_DIR / "scene_all" / "IndustryNav.exe"),
    ],
}


def resolve_scene_all_path(value: str) -> str:
    """Return an explicit path or the first existing scene_all build for this OS."""
    if value is None:
        value = ""
    if value.strip().lower() not in {"", "auto"}:
        return value

    os_name = platform.system()
    candidates = SCENE_ALL_BUILDS.get(os_name, [])
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate

    tried = "\n  ".join(candidates) if candidates else "(no entries for this OS)"
    raise FileNotFoundError(
        f"--file_name=auto: no local scene_all build found on {os_name}.\n"
        f"Tried:\n  {tried}\n"
        f"Either pass --file_name <explicit-path>, or add your local build "
        f"to config.SCENE_ALL_BUILDS[\"{os_name}\"]."
    )


# Behavior cloning

BC_ACTION_TO_LABEL: Dict[str, int] = {
    "forward": 0,
    "stop": 1,
    "turn right": 2,
    "turn left": 3,
}

BC_LABEL_TO_ACTION: Dict[int, str] = {v: k for k, v in BC_ACTION_TO_LABEL.items()}

BC_EPISODE_CSV: str = "keyboard_actions.csv"
BC_RGB_SUBDIR: str = "keyboard_fp"
BC_DEPTH_SUBDIR: str = "keyboard_depth"

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
NAV_DEFAULT_BACKBONE: str = "resnet50"


@dataclass(frozen=True)
class NavPolicyArch:
    """Shared BC policy architecture defaults."""

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


NAV_POLICY_ARCH = NavPolicyArch()


@dataclass(frozen=True)
class BCTrainConfig:
    """Behavior-cloning training preset."""

    data_root: str = "collect_data"
    output_dir: str = "outputs/nav_bc"
    img_size: int = 256
    batch_size: int = 16
    num_workers: int = 4
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42
    state_mode: str = "relative"  # relative | absolute | full
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
    chunk_size: int = 1


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


# Benchmark/eval file discovery

BENCHMARK_BASELINES: List[str] = ["random", "llm", "bc", "astar"]
EVAL_RUN_PREFIXES: List[str] = [
    "llm",
    "bc",
    "astar",
    "random",
    "agent",
    "bc_agent",
    "manual",
]
EVAL_DEPTH_DIR_CANDIDATES: List[str] = [f"{p}_depth" for p in EVAL_RUN_PREFIXES]
EVAL_ACTIONS_CSV_CANDIDATES: List[str] = [f"{p}_actions.csv" for p in EVAL_RUN_PREFIXES]
