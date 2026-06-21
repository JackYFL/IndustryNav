"""CLI entry for behavior-cloning training.

``--base {cnn,resnet50,dinov2}`` selects a preset bundle from
:data:`nav.config.BC_BASE_PRESETS` (reproducing the old per-base shell
scripts); any explicit flag then overrides the corresponding preset field.

    python -m nav.scripts.train_bc --base resnet50 --data_root collect_data
"""

from __future__ import annotations

import argparse
from dataclasses import fields, replace

from nav.config import BC_BASE_PRESETS, BCTrainConfig
from nav.train.loop import train
from nav.utils import logger_config


def parse_args() -> BCTrainConfig:
    p = argparse.ArgumentParser(description="Train a behavior-cloning navigation policy.")
    p.add_argument("--base", choices=sorted(BC_BASE_PRESETS), default="resnet50",
                   help="Preset bundle (backbone + policy_type + hyperparams).")

    # Overrides. Default None / "unset" so only explicitly-passed flags replace
    # the preset's value. Booleans use --flag/--no-flag (argparse tri-state).
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--img_size", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--state_mode", type=str, default=None, choices=["relative", "absolute", "full"])
    p.add_argument("--policy_type", type=str, default=None,
                   choices=["mlp", "lstm", "transformer", "diffusion"])
    p.add_argument("--seq_len", type=int, default=None)
    p.add_argument("--num_layers", type=int, default=None)
    p.add_argument("--chunk_size", type=int, default=None)
    p.add_argument("--goal_rep", type=str, default=None, choices=["cartesian", "polar"])
    p.add_argument("--rgb_backbone", type=str, default=None)
    p.add_argument("--depth_backbone", type=str, default=None)
    p.add_argument("--backbone_lr_scale", type=float, default=None)
    p.add_argument("--use_depth", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--use_rgb", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--normalize_rgb", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--pretrained_rgb", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--pretrained_depth", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--half_width", action=argparse.BooleanOptionalAction, default=None)

    args = p.parse_args()
    preset = BC_BASE_PRESETS[args.base]

    overridable = {f.name for f in fields(BCTrainConfig)}
    overrides = {
        k: v for k, v in vars(args).items()
        if k in overridable and v is not None
    }
    return replace(preset, **overrides)


def main() -> None:
    cfg = parse_args()
    logger = logger_config(cfg.output_dir)
    logger.info(f"BC training | policy={cfg.policy_type} | backbone={cfg.rgb_backbone} | "
                f"output_dir={cfg.output_dir}")
    train(cfg)


if __name__ == "__main__":
    main()
