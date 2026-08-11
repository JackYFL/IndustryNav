"""Category-specific absolute environment speeds shared by all entry points."""

from __future__ import annotations

import argparse
import hashlib
import math
import random
from dataclasses import dataclass


MOTION_CATEGORIES = ("human", "vehicle", "robot")
DEFAULT_SPEED_MPS = {
    "human": 1.2,
    "vehicle": 2.5,
    "robot": 1.5,
}


@dataclass(frozen=True)
class CategorySpeedConfig:
    """Resolved absolute speed for one category in one Unity run."""

    mode: str
    speed_mps: float | None
    minimum_mps: float | None
    maximum_mps: float | None


@dataclass(frozen=True)
class MotionSpeedConfig:
    """Resolved absolute speeds for all moving-object categories."""

    human: CategorySpeedConfig
    vehicle: CategorySpeedConfig
    robot: CategorySpeedConfig
    random_seed: int

    def for_category(self, category: str) -> CategorySpeedConfig:
        return getattr(self, category)


def add_motion_speed_args(parser: argparse.ArgumentParser) -> None:
    """Add fixed/range absolute-speed controls to an argument parser."""
    for category in MOTION_CATEGORIES:
        parser.add_argument(
            f"--{category}_speed_mps",
            f"--{category}-speed-mps",
            type=float,
            default=None,
            help=(
                f"Fixed absolute {category} speed in Unity world meters/second "
                f"(default: {DEFAULT_SPEED_MPS[category]})."
            ),
        )
        parser.add_argument(
            f"--{category}_speed_min_mps",
            f"--{category}-speed-min-mps",
            type=float,
            default=None,
            help=f"Lower bound for deterministic per-run {category} speed in m/s.",
        )
        parser.add_argument(
            f"--{category}_speed_max_mps",
            f"--{category}-speed-max-mps",
            type=float,
            default=None,
            help=f"Upper bound for deterministic per-run {category} speed in m/s.",
        )
    parser.add_argument(
        "--motion_random_seed",
        "--motion-random-seed",
        type=int,
        default=0,
        help="Base seed used when sampling category speed ranges.",
    )


def _validate_speed(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and greater than or equal to zero.")
    return value


def _stable_sample_seed(args, base_seed: int, category: str) -> int:
    identity = "|".join(
        str(getattr(args, name, ""))
        for name in ("scene_id", "scene_name", "point_id", "seed_id")
    )
    digest = hashlib.sha256(
        f"{base_seed}|{category}|{identity}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _resolve_category(args, category: str, base_seed: int) -> CategorySpeedConfig:
    fixed = getattr(args, f"{category}_speed_mps", None)
    minimum = getattr(args, f"{category}_speed_min_mps", None)
    maximum = getattr(args, f"{category}_speed_max_mps", None)
    fixed_flag = f"--{category}_speed_mps"
    min_flag = f"--{category}_speed_min_mps"
    max_flag = f"--{category}_speed_max_mps"

    if fixed is not None and (minimum is not None or maximum is not None):
        raise ValueError(
            f"{fixed_flag} cannot be combined with {min_flag}/{max_flag}."
        )
    if (minimum is None) != (maximum is None):
        raise ValueError(f"{min_flag} and {max_flag} must be provided together.")

    if fixed is not None:
        return CategorySpeedConfig(
            mode="fixed",
            speed_mps=_validate_speed(fixed_flag, fixed),
            minimum_mps=None,
            maximum_mps=None,
        )
    if minimum is None:
        return CategorySpeedConfig(
            mode="default",
            speed_mps=DEFAULT_SPEED_MPS[category],
            minimum_mps=None,
            maximum_mps=None,
        )

    minimum = _validate_speed(min_flag, minimum)
    maximum = _validate_speed(max_flag, maximum)
    if minimum > maximum:
        raise ValueError(f"{min_flag} must be less than or equal to {max_flag}.")
    speed_mps = random.Random(
        _stable_sample_seed(args, base_seed, category)
    ).uniform(minimum, maximum)
    return CategorySpeedConfig(
        mode="range",
        speed_mps=speed_mps,
        minimum_mps=minimum,
        maximum_mps=maximum,
    )


def resolve_motion_speed_config(args) -> MotionSpeedConfig:
    """Resolve category speeds without changing Python's global random state."""
    cached = getattr(args, "_motion_speed_config", None)
    if cached is not None:
        return cached

    base_seed = int(getattr(args, "motion_random_seed", 0))
    config = MotionSpeedConfig(
        human=_resolve_category(args, "human", base_seed),
        vehicle=_resolve_category(args, "vehicle", base_seed),
        robot=_resolve_category(args, "robot", base_seed),
        random_seed=base_seed,
    )
    args._motion_speed_config = config
    return config


def configure_unity_motion_speed(env_params, args, logger) -> MotionSpeedConfig:
    """Resolve and send category speeds before Unity's first reset."""
    config = resolve_motion_speed_config(args)
    labels: list[str] = []
    for category in MOTION_CATEGORIES:
        setting = config.for_category(category)
        env_params.set_float_parameter(
            f"{category}_speed_mps", float(setting.speed_mps)
        )
        labels.append(f"{category}={setting.mode}:{setting.speed_mps:.6f}m/s")
    logger.info(
        "Environment absolute speeds: "
        + ", ".join(labels)
        + f", seed={config.random_seed}"
    )
    return config


def motion_speed_result_fields(args) -> dict[str, object]:
    """Return normalized absolute-speed columns for a results row."""
    config = resolve_motion_speed_config(args)
    fields: dict[str, object] = {"motion_random_seed": config.random_seed}
    for category in MOTION_CATEGORIES:
        setting = config.for_category(category)
        fields.update(
            {
                f"{category}_speed_mode": setting.mode,
                f"{category}_speed_mps": setting.speed_mps,
                f"{category}_speed_min_mps": setting.minimum_mps,
                f"{category}_speed_max_mps": setting.maximum_mps,
            }
        )
    return fields
