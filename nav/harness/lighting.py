"""Deterministic runtime-lighting configuration shared by all entry points."""

from __future__ import annotations

import argparse
import hashlib
import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class LightingConfig:
    """Resolved lighting values for one Unity run."""

    enabled: bool
    mode: str
    multiplier: float
    minimum: float | None
    maximum: float | None
    random_seed: int
    fixed_exposure: float | None


def add_lighting_args(parser: argparse.ArgumentParser) -> None:
    """Add the common runtime-lighting arguments to an argument parser."""
    parser.add_argument(
        "--global_light_intensity",
        "--global-light-intensity",
        type=float,
        default=None,
        help="Fixed global multiplier for Unity scene lighting.",
    )
    parser.add_argument(
        "--light_intensity_multiplier",
        "--light-intensity-multiplier",
        type=float,
        default=None,
        help="Alias of --global_light_intensity.",
    )
    parser.add_argument(
        "--light_intensity_min",
        "--light-intensity-min",
        type=float,
        default=None,
        help="Lower bound for a deterministic per-run light multiplier.",
    )
    parser.add_argument(
        "--light_intensity_max",
        "--light-intensity-max",
        type=float,
        default=None,
        help="Upper bound for a deterministic per-run light multiplier.",
    )
    parser.add_argument(
        "--light_random_seed",
        "--light-random-seed",
        type=int,
        default=0,
        help="Base seed used when sampling a light multiplier range.",
    )
    parser.add_argument(
        "--light_fixed_exposure",
        "--light-fixed-exposure",
        type=float,
        default=9.0,
        help="HDRP fixed exposure EV used whenever runtime lighting is enabled.",
    )


def _validate_non_negative_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"{name} must be a finite value greater than or equal to zero."
        )
    return value


def _stable_sample_seed(args, base_seed: int) -> int:
    identity = "|".join(
        str(getattr(args, name, ""))
        for name in ("scene_id", "scene_name", "point_id", "seed_id")
    )
    digest = hashlib.sha256(f"{base_seed}|{identity}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def resolve_lighting_config(args) -> LightingConfig:
    """Validate and resolve one fixed multiplier without mutating global RNG state."""
    cached = getattr(args, "_lighting_config", None)
    if cached is not None:
        return cached

    fixed = getattr(args, "global_light_intensity", None)
    fixed_alias = getattr(args, "light_intensity_multiplier", None)
    minimum = getattr(args, "light_intensity_min", None)
    maximum = getattr(args, "light_intensity_max", None)
    base_seed = int(getattr(args, "light_random_seed", 0))

    if fixed is not None and fixed_alias is not None:
        raise ValueError(
            "--global_light_intensity and --light_intensity_multiplier are aliases; "
            "provide only one."
        )
    if fixed is None:
        fixed = fixed_alias
    if fixed is not None and (minimum is not None or maximum is not None):
        raise ValueError(
            "--global_light_intensity cannot be combined with "
            "--light_intensity_min/--light_intensity_max."
        )
    if (minimum is None) != (maximum is None):
        raise ValueError(
            "--light_intensity_min and --light_intensity_max must be provided together."
        )

    if fixed is None and minimum is None:
        config = LightingConfig(
            enabled=False,
            mode="disabled",
            multiplier=1.0,
            minimum=None,
            maximum=None,
            random_seed=base_seed,
            fixed_exposure=None,
        )
    else:
        exposure = float(getattr(args, "light_fixed_exposure", 9.0))
        if not math.isfinite(exposure):
            raise ValueError("--light_fixed_exposure must be finite.")

        if fixed is not None:
            multiplier = _validate_non_negative_finite(
                "--light_intensity_multiplier", fixed
            )
            config = LightingConfig(
                enabled=True,
                mode="fixed",
                multiplier=multiplier,
                minimum=None,
                maximum=None,
                random_seed=base_seed,
                fixed_exposure=exposure,
            )
        else:
            minimum = _validate_non_negative_finite("--light_intensity_min", minimum)
            maximum = _validate_non_negative_finite("--light_intensity_max", maximum)
            if minimum > maximum:
                raise ValueError(
                    "--light_intensity_min must be less than or equal to "
                    "--light_intensity_max."
                )
            sample_seed = _stable_sample_seed(args, base_seed)
            multiplier = random.Random(sample_seed).uniform(minimum, maximum)
            config = LightingConfig(
                enabled=True,
                mode="range",
                multiplier=multiplier,
                minimum=minimum,
                maximum=maximum,
                random_seed=base_seed,
                fixed_exposure=exposure,
            )

    args._lighting_config = config
    args.resolved_light_intensity_multiplier = config.multiplier
    return config


def configure_unity_lighting(env_params, args, logger) -> LightingConfig:
    """Resolve lighting and send it before Unity's first reset."""
    config = resolve_lighting_config(args)
    if not config.enabled:
        logger.info("Runtime lighting: disabled (using scene-authored lighting/exposure).")
        return config

    env_params.set_float_parameter(
        "light_intensity_multiplier", float(config.multiplier)
    )
    # Keep compatibility with clients built during the initial fixed-only rollout.
    env_params.set_float_parameter(
        "global_light_intensity", float(config.multiplier)
    )
    env_params.set_float_parameter(
        "light_fixed_exposure", float(config.fixed_exposure)
    )
    logger.info(
        "Runtime lighting: "
        f"mode={config.mode}, multiplier={config.multiplier:.6f}, "
        f"fixed_exposure={config.fixed_exposure:.3f}, seed={config.random_seed}"
    )
    return config


def lighting_result_fields(args) -> dict[str, object]:
    """Return the normalized lighting columns for a results row."""
    config = resolve_lighting_config(args)
    return {
        "light_randomization_mode": config.mode,
        "light_intensity_multiplier": config.multiplier,
        "light_intensity_min": config.minimum,
        "light_intensity_max": config.maximum,
        "light_random_seed": config.random_seed,
        "light_fixed_exposure": config.fixed_exposure,
        "global_light_intensity": config.multiplier,
    }
