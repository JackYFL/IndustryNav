"""CLI entry for safety-aware recurrent PointGoal PPO/DDP-PPO training."""

from __future__ import annotations

import argparse
from pathlib import Path

from nav.baselines.rl.trainer import PPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--navigation-map-dir", type=Path, default=None,
        help="Training-only calibrated occupancy maps for progress rewards; no teacher actions.",
    )
    parser.add_argument("--reference-policy-checkpoint", type=Path, default=None)
    parser.add_argument("--reference-kl-coef", type=float, default=0.0,
                        help="KL(current || frozen initialization) regularization; not expert labels.")
    parser.add_argument("--reference-kl-final-coef", type=float, default=0.0)
    parser.add_argument("--reference-kl-decay-updates", type=int, default=500)
    parser.add_argument("--resample-invalid-spawns", action="store_true",
                        help="Log and reject displaced sampled spawns; never alter benchmark evaluation.")
    parser.add_argument(
        "--extra-manifest",
        type=Path,
        action="append",
        default=[],
        help=(
            "Additional task manifest to mix into training. Repeat the flag "
            "to combine multiple distance curricula."
        ),
    )
    parser.add_argument("--unity", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split",
        default="train",
        help="Manifest split to train on; use 'all' when input_points is the external holdout.",
    )
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--parallel-env-steps", action="store_true",
                        help="Step independent Unity players concurrently; keep resets on the main thread.")
    parser.add_argument("--scene-limit", type=int, default=0)
    parser.add_argument("--episodes-per-scene", type=int, default=0)
    parser.add_argument("--base-port", type=int, default=43000)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu", "auto"),
        default="cuda",
        help="Training device; CPU is intended only for small local smoke tests.",
    )
    parser.add_argument("--total-updates", type=int, default=200)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--bptt-len", type=int, default=8)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--chunks-per-minibatch", type=int, default=8)
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0e-4,
        help="Base learning rate used by the critic.",
    )
    parser.add_argument("--encoder-lr-scale", type=float, default=0.1)
    parser.add_argument("--adaptation-lr-scale", type=float, default=1.0,
                        help="LR multiplier for newly added persistent-memory parameters; does not affect the preserved actor.")
    parser.add_argument(
        "--policy-lr-scale",
        type=float,
        default=2.5,
        help="LR multiplier for actor, recurrent, action, goal, and temporal parameters.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--clip-param", type=float, default=0.2)
    parser.add_argument(
        "--policy-loss-coef",
        type=float,
        default=1.0,
        help="Scale the PPO actor objective (default: 1).",
    )
    parser.add_argument(
        "--rollout-action-mode",
        choices=("sample", "argmax"),
        default="sample",
        help=(
            "Use stochastic PPO samples or deterministic argmax rollouts. "
            "Argmax is restricted to imitation-only policy-loss-coef=0 runs."
        ),
    )
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument(
        "--policy-temperature", type=float, default=1.0,
        help=(
            "Stochastic pure-PPO training distribution softmax(logits / T). "
            "Use the same T for sampling, replay likelihoods, entropy and reference KL; "
            "evaluation remains deterministic argmax. Default 1 preserves existing behavior."
        ),
    )
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--gradient-clip-mode", choices=("global", "actor_value"), default="global",
        help=("Global clipping preserves the original update. actor_value independently "
              "clips the detached-critic DAgger PPO actor and value parameters to "
              "max-grad-norm each; their combined norm may exceed max-grad-norm."),
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restore optimizer state when resuming (default: true).",
    )
    parser.add_argument(
        "--resume-optimizer-lr-mode", choices=("checkpoint", "config"), default="checkpoint",
        help=("With optimizer resume, retain saved rates (default) or keep Adam moments "
              "and apply the newly configured named-group learning rates. Config mode "
              "requires --resume --resume-optimizer and an existing latest.pt."),
    )
    parser.add_argument("--init-bc-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--init-dagger-actor-checkpoint", type=Path, default=None,
        help="Keep a complete DAgger Transformer actor and add a PPO critic.",
    )
    parser.add_argument(
        "--critic-warmup-updates", type=int, default=0,
        help="Fit only the new critic before PPO actor updates (Transformer only).",
    )
    parser.add_argument(
        "--freeze-dagger-transformer", action=argparse.BooleanOptionalAction,
        default=None,
        help="Fine-tune only the DAgger action head and new critic, preserving its Transformer.",
    )
    parser.add_argument(
        "--target-kl", type=float, default=0.0,
        help="Stop optimization for an update when sampled KL exceeds this value; 0 disables.",
    )
    parser.add_argument(
        "--init-ppo-checkpoint",
        type=Path,
        default=None,
        help=(
            "Initialize model weights from a PPO checkpoint while starting a "
            "new update/optimizer schedule."
        ),
    )
    parser.add_argument(
        "--teacher-bc-checkpoint",
        type=Path,
        default=None,
        help="Optional DAgger/BC policy used for an auxiliary imitation loss.",
    )
    parser.add_argument(
        "--online-astar-teacher",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the minimap A* planner only during training to label on-policy "
            "states. The minimap is not part of the learned policy input."
        ),
    )
    parser.add_argument(
        "--teacher-bc-coef",
        type=float,
        default=0.0,
        help="Initial cross-entropy coefficient for the BC teacher (default: 0).",
    )
    parser.add_argument(
        "--teacher-bc-decay-updates",
        type=int,
        default=200,
        help="Updates over which the teacher coefficient decays.",
    )
    parser.add_argument(
        "--teacher-bc-min-scale",
        type=float,
        default=0.1,
        help="Minimum fraction of the initial teacher coefficient.",
    )
    parser.add_argument(
        "--teacher-action-prob",
        type=float,
        default=0.0,
        help=(
            "Initial probability of executing the teacher action during a "
            "rollout. Expert-executed steps are excluded from PPO actor loss."
        ),
    )
    parser.add_argument("--teacher-action-decay-updates", type=int, default=100)
    parser.add_argument("--teacher-action-min-prob", type=float, default=0.0)
    parser.add_argument(
        "--teacher-class-balance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Inverse-frequency balance A* or BC teacher labels per rollout "
            "to prevent deterministic forward-action collapse."
        ),
    )
    parser.add_argument(
        "--teacher-replay-capacity",
        type=int,
        default=0,
        help="Recent A*-labeled recurrent states retained across updates; zero disables.",
    )
    parser.add_argument("--teacher-replay-steps", type=int, default=0)
    parser.add_argument("--teacher-replay-batch-size", type=int, default=64)
    parser.add_argument("--teacher-replay-coef", type=float, default=1.0)
    parser.add_argument(
        "--teacher-replay-class-weight-power",
        type=float,
        default=0.5,
        help="Tempered inverse-frequency exponent for replay labels.",
    )
    parser.add_argument("--astar-obstacle-clearance-m", type=float, default=0.6)
    parser.add_argument(
        "--astar-policy-forward-tolerance-deg",
        type=float,
        default=12.5,
        help=(
            "Heading-error deadband for the atomic online A* teacher. "
            "Larger values reduce corrective turn labels on off-policy states."
        ),
    )
    parser.add_argument(
        "--geodesic-progress-reward",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use progress along the training-only online A* path instead of "
            "straight-line goal progress. Requires --online-astar-teacher."
        ),
    )
    parser.add_argument("--depth-backbone", default="resnet50")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--visual-dim", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument(
        "--action-history-len",
        type=int,
        default=5,
        help="Number of previous actions explicitly embedded by the policy (minimum 5).",
    )
    parser.add_argument(
        "--visual-history-len",
        type=int,
        default=5,
        help="Number of previous visual encoder features used by the temporal policy (minimum 5).",
    )
    parser.add_argument(
        "--half-width",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--pretrained-depth",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--goal-distance-scale-m", type=float, default=50.0)
    parser.add_argument(
        "--absolute-scene-state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Append normalized current/target world coordinates and a scene "
            "one-hot vector to the relative PointGoal state."
        ),
    )
    parser.add_argument("--num-scenes", type=int, default=24)
    parser.add_argument("--world-coordinate-scale-m", type=float, default=50.0)
    parser.add_argument(
        "--coordinate-fourier-bands",
        type=int,
        default=0,
        help=(
            "Dyadic sine/cosine bands for each normalized current/target "
            "world coordinate; requires --absolute-scene-state."
        ),
    )
    parser.add_argument(
        "--route-actor-hidden",
        type=int,
        default=0,
        help=(
            "Optional hidden width for a nonlinear GPS/Compass route "
            "residual; requires --absolute-scene-state."
        ),
    )
    parser.add_argument("--base-policy-logit-scale", type=float, default=1.0)
    parser.add_argument("--route-actor-logit-scale", type=float, default=1.0)
    parser.add_argument("--analytic-goal-prior-strength", type=float, default=3.0)
    parser.add_argument("--reach-m", type=float, default=2.0)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument(
        "--dynamic-step-budget", action=argparse.BooleanOptionalAction,
        default=False,
        help="Train with distance-dependent episode limits, matching benchmark defaults.",
    )
    parser.add_argument("--step-budget-min", type=int, default=80)
    parser.add_argument("--step-budget-max", type=int, default=320)
    parser.add_argument("--steps-per-path-meter", type=float, default=2.2)
    parser.add_argument("--step-budget-overhead", type=int, default=60)
    parser.add_argument(
        "--training-safety-shield", action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the fixed benchmark contact/warning/turn-escape shield during PPO rollouts.",
    )
    parser.add_argument(
        "--task-sampling",
        choices=("random", "sequential"),
        default="random",
    )
    parser.add_argument(
        "--scene-block-episodes",
        type=int,
        default=4,
        help="Episodes sampled within one scene before choosing another scene.",
    )
    parser.add_argument(
        "--fresh-unity-per-episode", action="store_true",
        help=(
            "Restart the Unity player for every training episode, including "
            "consecutive tasks in one scene. Slower, but matches the canonical "
            "evaluator's fresh-world lifecycle. Default reuses same-scene players."
        ),
    )
    parser.add_argument(
        "--dynamic-objects",
        choices=("moving", "static"),
        default="moving",
    )
    parser.add_argument("--ego-width", type=int, default=320)
    parser.add_argument("--ego-height", type=int, default=240)
    parser.add_argument("--minimap-width", type=int, default=862)
    parser.add_argument("--minimap-height", type=int, default=512)
    parser.add_argument("--progress-scale", type=float, default=1.5)
    parser.add_argument("--step-penalty", type=float, default=0.01)
    parser.add_argument("--success-bonus", type=float, default=10.0)
    parser.add_argument("--timeout-penalty", type=float, default=2.0)
    parser.add_argument("--collision-penalty", type=float, default=1.0)
    parser.add_argument(
        "--collision-step-penalty",
        type=float,
        default=0.0,
        help=(
            "Additional penalty on every physically blocked forward step, "
            "separate from the contact-onset penalty."
        ),
    )
    parser.add_argument("--warning-penalty", type=float, default=0.02)
    parser.add_argument("--rotation-penalty", type=float, default=0.005)
    parser.add_argument("--turn-reversal-penalty", type=float, default=0.05)
    parser.add_argument("--stagnation-penalty", type=float, default=0.02)
    parser.add_argument("--stagnation-start-steps", type=int, default=12)
    parser.add_argument(
        "--stagnation-progress-epsilon-m",
        type=float,
        default=0.05,
        help="Required best-distance improvement to reset stagnation.",
    )
    parser.add_argument("--safe-forward-bonus", type=float, default=0.02)
    parser.add_argument(
        "--stuck-recovery-steps",
        type=int,
        default=48,
        help="End and resample an episode after this many no-progress steps; zero disables.",
    )
    parser.add_argument("--stuck-recovery-penalty", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    PPOTrainer(parse_args()).run()


if __name__ == "__main__":
    main()
