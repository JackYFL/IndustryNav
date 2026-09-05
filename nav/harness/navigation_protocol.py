"""Reproducible API navigation settings aligned to the archived Kiro runs.

Only observable task settings are aligned; provider/model internals and Unity
build equivalence cannot be recovered from the historical CLI logs.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from nav.config import ACTION_SPACE_AGENTS, LLM_DEFAULT_MAX_TOKENS, LLM_REQUEST_TIMEOUT_SEC
from nav.harness.coordinates import resolve_minimap_resolution


NAVIGATION_PROTOCOL_VERSION = "kiro-aligned-v1"
KIRO_EGO_SIZE = (320, 240)
KIRO_MINIMAP_SIZE = (431, 256)


def resolve_navigation_sensors(args, baseline="llm") -> None:
    """Apply Kiro sensor defaults only to LLM; preserve explicit overrides."""
    ego_default = KIRO_EGO_SIZE if baseline == "llm" else (512, 512)
    for key, default in zip(("ego_width", "ego_height"), ego_default):
        if getattr(args, key) is None:
            setattr(args, key, default)
        if getattr(args, key) <= 0:
            raise ValueError(f"--{key} must be positive.")
    if args.minimap_width is None and args.minimap_height is None and baseline == "llm":
        args.minimap_width, args.minimap_height = KIRO_MINIMAP_SIZE
    args.minimap_width, args.minimap_height = resolve_minimap_resolution(
        args.minimap_width, args.minimap_height,
    )


def navigation_run_config(args, prompt_template: str) -> dict:
    """Whitelist experiment settings; never serialize credentials or env dumps."""
    defaults = {
        "model_id": None, "llm_provider": "openrouter", "vision_input": True,
        "history_size": 5, "max_tokens": LLM_DEFAULT_MAX_TOKENS,
        "llm_min_request_interval_sec": 0.0,
        "scene_id": None, "scene_name": "", "point_id": "", "seed_id": "",
        "init_world_x": None, "init_world_z": None, "init_curr_direction": 180.0,
        "init_curr_x": None, "init_curr_y": None, "target_x": None, "target_y": None,
        "ego_width": 320, "ego_height": 240,
        "minimap_width": 431, "minimap_height": 256,
        "screen_width": 1724, "screen_height": 1024,
        "sim_steps_per_decision": 2, "reach_m": 2.0, "frame_sleep": 0.0,
        "max_steps": 70, "dynamic_step_budget": True,
        "step_budget_min": 50, "step_budget_max": 160,
        "steps_per_path_meter": 1.25, "step_budget_overhead": 20,
        "dynamic_objects": "moving", "marker_source": "vector",
        "hide_unity_red_marker": True, "file_name": "auto",
        "motion_random_seed": 0, "global_light_intensity": None,
        "light_intensity_multiplier": None, "light_intensity_min": None,
        "light_intensity_max": None, "light_random_seed": 0, "light_fixed_exposure": 9.0,
    }
    for category in ("human", "vehicle", "robot"):
        for suffix in ("mps", "min_mps", "max_mps"):
            defaults[f"{category}_speed_{suffix}"] = None
    values = {name: getattr(args, name, default) for name, default in defaults.items()}
    if values["dynamic_step_budget"] is None:
        values["dynamic_step_budget"] = True
    config = {
        "protocol": NAVIGATION_PROTOCOL_VERSION,
        "prompt_sha256": hashlib.sha256(prompt_template.encode("utf-8")).hexdigest(),
        "action_space": ACTION_SPACE_AGENTS,
        "request_timeout_sec": LLM_REQUEST_TIMEOUT_SEC,
        "input_modalities": ["ego"] if values["vision_input"] else [],
        "settings": values,
    }
    if values["llm_provider"] == "openrouter":
        config["openrouter_options"] = {
            "reasoning_enabled": os.getenv("OPENROUTER_REASONING_ENABLED", "").strip().lower(),
            "json_mode": os.getenv("OPENROUTER_JSON_MODE", "").strip().lower(),
            "min_request_interval_sec": os.getenv("OPENROUTER_MIN_REQUEST_INTERVAL_SEC", "0"),
            "max_request_attempts": os.getenv("OPENROUTER_MAX_REQUEST_ATTEMPTS", "1"),
        }
    return config


def check_navigation_run_config(folder: Path, expected: dict) -> None:
    """Refuse to reuse/overwrite outputs from a different or unknown protocol."""
    path = folder / "run_config.json"
    if path.exists():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Unreadable run configuration: {path}") from exc
        if saved == expected:
            return
    elif not folder.exists() or not any(p.name != "run.log" for p in folder.iterdir()):
        # The grid opens run.log before starting the cell process.
        return
    raise ValueError(
        f"Existing outputs use a different or unrecorded navigation configuration: {folder}. "
        "Choose a fresh --output_root (grid) or --frame_save_dir (cell); "
        "old results must not be mixed with this protocol."
    )


def prepare_navigation_run(folder: Path, expected: dict) -> None:
    check_navigation_run_config(folder, expected)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "run_config.json").write_text(
        json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
