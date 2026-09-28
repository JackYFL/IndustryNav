"""Warm-start the PPO PointGoal actor from exported A*/DAgger episodes."""

from __future__ import annotations

import argparse
from pathlib import Path

from nav.baselines.rl.imitation import run_offline_imitation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        action="append",
        required=True,
        help="Exported BC dataset root; repeat to combine A* and DAgger data.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    init_group = parser.add_mutually_exclusive_group()
    init_group.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Initialize from an existing PPO-compatible actor checkpoint.",
    )
    init_group.add_argument(
        "--init-bc-checkpoint",
        type=Path,
        default=None,
        help=(
            "Initialize the depth encoder from a BC/DAgger checkpoint before "
            "fitting the PPO-compatible recurrent actor."
        ),
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument(
        "--episode-limit",
        type=int,
        default=0,
        help="Optional per-split episode cap for smoke tests; zero uses all data.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bptt-len", type=int, default=16)
    parser.add_argument("--encoder-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--action-history-noise", type=float, default=0.05)
    parser.add_argument(
        "--route-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train only the image-free nonlinear GPS/Compass route residual.",
    )
    parser.add_argument(
        "--checkpoint-metric",
        choices=("accuracy", "macro_accuracy"),
        default="macro_accuracy",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=7301)

    parser.add_argument("--depth-backbone", default="resnet50")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--visual-dim", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--action-history-len", type=int, default=5)
    parser.add_argument("--visual-history-len", type=int, default=5)
    parser.add_argument(
        "--half-width",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--goal-distance-scale-m", type=float, default=50.0)
    parser.add_argument("--analytic-goal-prior-strength", type=float, default=3.0)
    parser.add_argument(
        "--absolute-scene-state",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--num-scenes", type=int, default=24)
    parser.add_argument("--world-coordinate-scale-m", type=float, default=50.0)
    parser.add_argument("--coordinate-fourier-bands", type=int, default=0)
    parser.add_argument("--route-actor-hidden", type=int, default=0)
    parser.add_argument("--base-policy-logit-scale", type=float, default=1.0)
    parser.add_argument("--route-actor-logit-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    run_offline_imitation(parse_args())


if __name__ == "__main__":
    main()
