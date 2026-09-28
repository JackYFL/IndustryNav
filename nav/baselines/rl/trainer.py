"""PPO/DDP-PPO training logic for the recurrent PointGoal baseline."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributions import Categorical

from nav.baselines.rl.actions import (
    PPO_ACTIONS,
    append_action_history,
    initial_action_history,
)
from nav.baselines.rl.reward import compute_pointgoal_reward
from nav.baselines.rl.regularization import reference_kl, reference_coefficient, temperature_logits
from nav.baselines.rl.replay import TeacherReplayBuffer, soft_inverse_class_weights
from nav.envs.unity_pointgoal import UnityPointGoalEnv
from nav.models.policies import (
    PointGoalActorCritic,
    PointGoalPPOConfig,
    append_visual_history,
    initial_visual_history,
    load_compatible_pointgoal_state_dict,
    build_pointgoal_actor_critic,
    dagger_ppo_config,
)
from nav.train.advantages import generalized_advantage_estimate
from nav.train.base import BaseTrainer, TrainerState
from nav.train.initialization import load_bc_depth_encoder
from nav.train.optimizers import clip_policy_value_gradients, gradient_l2_norm, restore_optimizer_state


def read_tasks(args: argparse.Namespace, rank: int, world_size: int) -> list[list[dict]]:
    records = []
    manifest_paths = [args.manifest, *getattr(args, "extra_manifest", [])]
    for manifest_path in manifest_paths:
        with manifest_path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                if args.split == "all" or record.get("split") == args.split:
                    records.append(record)
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for task in records:
        by_scene[str(task["scene_name"])].append(task)
    scene_names = sorted(
        by_scene, key=lambda name: int(name.removeprefix("scene"))
    )
    if args.scene_limit > 0:
        scene_names = scene_names[: args.scene_limit]
    scene_names = scene_names[rank::world_size]
    if not scene_names:
        raise ValueError(f"Rank {rank} received no scenes")
    env_shards: list[list[dict]] = [[] for _ in range(args.num_envs)]
    for scene_index, scene_name in enumerate(scene_names):
        tasks = by_scene[scene_name]
        if args.episodes_per_scene > 0:
            tasks = tasks[: args.episodes_per_scene]
        env_shards[scene_index % args.num_envs].extend(tasks)
    env_shards = [shard for shard in env_shards if shard]
    if len(env_shards) != args.num_envs:
        raise ValueError(
            f"Requested {args.num_envs} envs but only {len(env_shards)} received tasks"
        )
    return env_shards


def resolve_training_device(
    requested: str,
    *,
    world_size: int,
    local_rank: int,
) -> torch.device:
    """Resolve the PPO device while keeping multi-rank training CUDA-only."""

    device_name = requested.lower()
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "CUDA is unavailable; pass --device cpu only for local smoke tests"
            )
        return torch.device("cuda", local_rank)
    if device_name == "cpu":
        if world_size > 1:
            raise SystemExit("Distributed PPO currently requires CUDA")
        return torch.device("cpu")
    raise SystemExit(f"Unsupported training device: {requested}")


def setup_distributed(device_name: str) -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = resolve_training_device(
        device_name,
        world_size=world_size,
        local_rank=local_rank,
    )
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    elif device.type == "cuda":
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, device


def normalized_advantages(
    advantages: torch.Tensor, world_size: int
) -> torch.Tensor:
    flat = advantages.double().reshape(-1)
    stats = torch.stack((flat.sum(), (flat * flat).sum(), flat.new_tensor(flat.numel())))
    if world_size > 1:
        # NCCL only accepts CUDA collectives. Rollout storage intentionally
        # stays on CPU, so move only these three scalars for global moments.
        stats = stats.to(torch.device("cuda", torch.cuda.current_device()))
        dist.all_reduce(stats)
        stats = stats.cpu()
    mean = stats[0] / stats[2]
    variance = (stats[1] / stats[2] - mean * mean).clamp_min(1e-8)
    return (advantages - mean.float()) / variance.sqrt().float()


def serializable_args(args: argparse.Namespace) -> dict:
    def convert(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    return {
        key: convert(value)
        for key, value in vars(args).items()
    }


def is_value_parameter(model, name: str) -> bool:
    return name.startswith(getattr(model, "value_parameter_prefixes", ("critic.",)))


def optimizer_parameter_groups(
    model: PointGoalActorCritic,
    *,
    lr: float,
    encoder_lr_scale: float,
    policy_lr_scale: float,
    adaptation_lr_scale: float = 1.0,
) -> list[dict]:
    """Build disjoint LR groups for perception, policy/temporal, and value."""
    groups: dict[str, list[torch.nn.Parameter]] = {
        "depth_encoder": [],
        "policy_temporal": [],
        "value": [],
    }
    adaptation = []
    for name, parameter in model.named_parameters():
        if name.startswith("depth_encoder."):
            groups["depth_encoder"].append(parameter)
        elif is_value_parameter(model, name):
            groups["value"].append(parameter)
        elif name.startswith(getattr(model, "adaptation_parameter_prefixes", ())):
            adaptation.append(parameter)
        else:
            groups["policy_temporal"].append(parameter)
    if any(not parameters for parameters in groups.values()):
        empty = [name for name, parameters in groups.items() if not parameters]
        raise RuntimeError(f"Optimizer parameter groups are empty: {empty}")
    result = [
        {
            "params": groups["depth_encoder"],
            "lr": lr * encoder_lr_scale,
            "group_name": "depth_encoder",
        },
        {
            "params": groups["policy_temporal"],
            "lr": lr * policy_lr_scale,
            "group_name": "policy_temporal",
        },
        {
            "params": groups["value"],
            "lr": lr,
            "group_name": "value",
        },
    ]
    if adaptation:
        result.append({"params": adaptation, "lr": lr * adaptation_lr_scale,
                       "group_name": "policy_adaptation"})
    return result


def teacher_auxiliary_scale(
    update: int,
    decay_updates: int,
    minimum_scale: float,
) -> float:
    """Linearly decay imitation guidance while retaining a small anchor."""
    if decay_updates <= 0:
        return 1.0
    progress = min(max((int(update) - 1) / int(decay_updates), 0.0), 1.0)
    return max(float(minimum_scale), 1.0 - progress)


def scheduled_teacher_probability(
    update: int,
    initial_probability: float,
    decay_updates: int,
    minimum_probability: float,
) -> float:
    """Linearly decay DAgger action mixing to an absolute probability floor."""
    initial_probability = float(initial_probability)
    minimum_probability = float(minimum_probability)
    if initial_probability <= 0.0:
        return 0.0
    return initial_probability * teacher_auxiliary_scale(
        update,
        decay_updates,
        minimum_probability / initial_probability,
    )


def balanced_teacher_class_weights(
    counts: torch.Tensor,
    *,
    max_weight: float = 4.0,
) -> torch.Tensor:
    """Return inverse-frequency teacher weights with unit sample mean.

    A* rollouts contain many more forward labels than turn labels. Without
    balancing, deterministic argmax can collapse to forward even while the
    aggregate teacher agreement looks high.
    """
    counts = counts.float()
    total = counts.sum()
    if total.item() <= 0:
        return torch.ones_like(counts)
    weights = total / (counts.numel() * counts.clamp_min(1.0))
    weights = weights.clamp(max=float(max_weight))
    sample_mean = (weights * counts).sum() / total
    return weights / sample_mean.clamp_min(1.0e-8)


def select_rollout_action(
    logits: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Choose rollout actions for PPO sampling or deterministic DAgger."""
    if mode == "sample":
        return Categorical(logits=logits).sample()
    if mode == "argmax":
        return logits.argmax(dim=-1)
    raise ValueError(f"Unsupported rollout action mode: {mode}")


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_config: PointGoalPPOConfig,
    args: argparse.Namespace,
    update: int,
    global_steps: int,
) -> None:
    raw_model = model.module if hasattr(model, "module") else model
    payload = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": asdict(model_config),
        "train_config": serializable_args(args),
        "update": int(update),
        "global_steps": int(global_steps),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_teacher_replay(path: Path, replay: TeacherReplayBuffer) -> None:
    """Atomically persist one rank's bounded teacher replay state."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(replay.state_dict(), temporary)
    os.replace(temporary, path)


def run_training(
    args: argparse.Namespace,
    state: TrainerState | None = None,
) -> None:
    gradient_clip_mode = getattr(args, "gradient_clip_mode", "global")
    if gradient_clip_mode not in {"global", "actor_value"}:
        raise SystemExit("Unknown gradient-clip-mode")
    if gradient_clip_mode == "actor_value" and (
        not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0
    ):
        raise SystemExit("Separate gradient clipping requires a finite positive max-grad-norm")
    resume_lr_mode = getattr(args, "resume_optimizer_lr_mode", "checkpoint")
    if resume_lr_mode == "config" and (
        not args.resume or not getattr(args, "resume_optimizer", True)
        or not (Path(args.output_dir) / "latest.pt").is_file()
    ):
        raise SystemExit("Configured-rate optimizer resume requires --resume --resume-optimizer and latest.pt")
    reference_initial = getattr(args, "reference_kl_coef", 0.0)
    reference_final = getattr(args, "reference_kl_final_coef", 0.0)
    reference_decay = getattr(args, "reference_kl_decay_updates", 500)
    reference_path = getattr(args, "reference_policy_checkpoint", None)
    policy_temperature = float(getattr(args, "policy_temperature", 1.0))
    if not math.isfinite(policy_temperature) or policy_temperature <= 0:
        raise SystemExit("policy-temperature must be finite and positive")
    reference_coefficient(1, reference_initial, reference_final, reference_decay)
    if reference_initial and (reference_path is None or not reference_path.is_file()):
        raise SystemExit("Reference KL requires a saved frozen policy checkpoint")
    dagger_checkpoint = getattr(args, "init_dagger_actor_checkpoint", None)
    if dagger_checkpoint is not None and (
        args.init_bc_checkpoint is not None or args.init_ppo_checkpoint is not None
    ):
        raise SystemExit("DAgger actor initialization is exclusive with other initializers")
    if getattr(args, "critic_warmup_updates", 0) < 0 or getattr(args, "target_kl", 0) < 0:
        raise SystemExit("critic-warmup-updates and target-kl must be nonnegative")
    if args.num_envs <= 0 or args.total_updates <= 0 or args.rollout_steps <= 0:
        raise SystemExit("num-envs, total-updates, and rollout-steps must be positive")
    if args.rollout_steps % args.bptt_len:
        raise SystemExit("rollout-steps must be divisible by bptt-len")
    if args.chunks_per_minibatch <= 0:
        raise SystemExit("chunks-per-minibatch must be positive")
    if args.action_history_len < 5:
        raise SystemExit("action-history-len must be at least 5")
    if args.visual_history_len < 5:
        raise SystemExit("visual-history-len must be at least 5")
    if args.lr <= 0.0 or args.encoder_lr_scale <= 0.0 or args.policy_lr_scale <= 0.0:
        raise SystemExit("learning rate and LR scales must be positive")
    if not math.isfinite(getattr(args, "adaptation_lr_scale", 1.0)) or getattr(args, "adaptation_lr_scale", 1.0) <= 0:
        raise SystemExit("adaptation-lr-scale must be finite and positive")
    if args.policy_loss_coef < 0.0:
        raise SystemExit("--policy-loss-coef must be nonnegative")
    if args.rollout_action_mode == "argmax" and args.policy_loss_coef > 0.0:
        raise SystemExit(
            "--rollout-action-mode argmax requires --policy-loss-coef 0; "
            "PPO actor updates require stochastic on-policy samples"
        )
    if args.scene_block_episodes <= 0:
        raise SystemExit("scene-block-episodes must be positive")
    if args.stuck_recovery_steps < 0:
        raise SystemExit("stuck-recovery-steps must be nonnegative")
    if args.step_budget_min <= 0 or args.step_budget_max < args.step_budget_min:
        raise SystemExit("invalid dynamic step-budget bounds")
    if args.steps_per_path_meter <= 0.0 or args.step_budget_overhead < 0:
        raise SystemExit("invalid dynamic step-budget coefficients")
    if args.teacher_bc_coef < 0.0 or args.teacher_bc_decay_updates < 0:
        raise SystemExit("teacher BC coefficient and decay must be nonnegative")
    if not 0.0 <= args.teacher_bc_min_scale <= 1.0:
        raise SystemExit("teacher-bc-min-scale must be in [0, 1]")
    if args.online_astar_teacher and args.teacher_bc_checkpoint is not None:
        raise SystemExit(
            "--online-astar-teacher and --teacher-bc-checkpoint are mutually exclusive"
        )
    teacher_enabled = bool(
        args.online_astar_teacher or args.teacher_bc_checkpoint is not None
    )
    if policy_temperature != 1.0 and (
        args.policy_loss_coef <= 0 or teacher_enabled
        or args.teacher_action_prob > 0 or args.teacher_replay_capacity > 0
    ):
        raise SystemExit("Non-default policy-temperature is supported only for pure stochastic PPO")
    if args.teacher_bc_coef > 0.0 and not teacher_enabled:
        raise SystemExit(
            "--teacher-bc-coef requires --teacher-bc-checkpoint or "
            "--online-astar-teacher"
        )
    if not 0.0 <= args.teacher_action_prob <= 1.0:
        raise SystemExit("--teacher-action-prob must be in [0, 1]")
    if not 0.0 <= args.teacher_action_min_prob <= 1.0:
        raise SystemExit("--teacher-action-min-prob must be in [0, 1]")
    if args.teacher_action_min_prob > args.teacher_action_prob:
        raise SystemExit(
            "--teacher-action-min-prob cannot exceed --teacher-action-prob"
        )
    if args.teacher_action_decay_updates < 0:
        raise SystemExit("--teacher-action-decay-updates must be nonnegative")
    if args.teacher_action_prob > 0.0 and not teacher_enabled:
        raise SystemExit("--teacher-action-prob requires an enabled teacher")
    if args.teacher_class_balance and not teacher_enabled:
        raise SystemExit("--teacher-class-balance requires an enabled teacher")
    if args.teacher_replay_capacity < 0 or args.teacher_replay_steps < 0:
        raise SystemExit("teacher replay capacity/steps must be nonnegative")
    if args.teacher_replay_capacity == 0 and args.teacher_replay_steps > 0:
        raise SystemExit("teacher replay steps require a positive capacity")
    if args.teacher_replay_capacity > 0 and args.teacher_replay_steps <= 0:
        raise SystemExit("teacher replay capacity requires positive replay steps")
    if args.teacher_replay_capacity > 0 and not teacher_enabled:
        raise SystemExit("teacher replay requires an enabled teacher")
    if args.teacher_replay_batch_size <= 0 or args.teacher_replay_coef < 0.0:
        raise SystemExit("teacher replay batch size must be positive and coef nonnegative")
    if not 0.0 <= args.teacher_replay_class_weight_power <= 1.0:
        raise SystemExit("teacher replay class-weight power must be in [0, 1]")
    if args.astar_obstacle_clearance_m < 0.0:
        raise SystemExit("--astar-obstacle-clearance-m must be nonnegative")
    if not 0.0 <= args.astar_policy_forward_tolerance_deg <= 90.0:
        raise SystemExit(
            "--astar-policy-forward-tolerance-deg must be in [0, 90]"
        )
    if args.geodesic_progress_reward and not args.online_astar_teacher:
        raise SystemExit(
            "--geodesic-progress-reward requires --online-astar-teacher"
        )
    if args.init_bc_checkpoint is not None and args.init_ppo_checkpoint is not None:
        raise SystemExit(
            "--init-bc-checkpoint and --init-ppo-checkpoint are mutually exclusive"
        )
    if (
        args.init_ppo_checkpoint is not None
        and not args.init_ppo_checkpoint.is_file()
    ):
        raise SystemExit(
            f"Initial PPO checkpoint does not exist: {args.init_ppo_checkpoint}"
        )
    if (
        args.teacher_bc_checkpoint is not None
        and not args.teacher_bc_checkpoint.is_file()
    ):
        raise SystemExit(
            f"Teacher BC checkpoint does not exist: {args.teacher_bc_checkpoint}"
        )
    nonnegative_reward_args = (
        "progress_scale",
        "step_penalty",
        "success_bonus",
        "timeout_penalty",
        "collision_penalty",
        "collision_step_penalty",
        "warning_penalty",
        "rotation_penalty",
        "turn_reversal_penalty",
        "stagnation_penalty",
        "safe_forward_bonus",
        "stagnation_progress_epsilon_m",
        "stuck_recovery_penalty",
    )
    if any(getattr(args, name) < 0.0 for name in nonnegative_reward_args):
        raise SystemExit("reward scales and penalties must be nonnegative")
    if args.stagnation_start_steps <= 0:
        raise SystemExit("stagnation-start-steps must be positive")
    if args.num_scenes <= 0 or args.world_coordinate_scale_m <= 0.0:
        raise SystemExit("num-scenes and world-coordinate-scale-m must be positive")
    if args.coordinate_fourier_bands < 0:
        raise SystemExit("--coordinate-fourier-bands must be nonnegative")
    if args.coordinate_fourier_bands and not args.absolute_scene_state:
        raise SystemExit(
            "--coordinate-fourier-bands requires --absolute-scene-state"
        )
    if args.route_actor_hidden < 0:
        raise SystemExit("--route-actor-hidden must be nonnegative")
    if args.route_actor_hidden and not args.absolute_scene_state:
        raise SystemExit("--route-actor-hidden requires --absolute-scene-state")
    if args.base_policy_logit_scale < 0.0 or args.route_actor_logit_scale < 0.0:
        raise SystemExit("policy logit scales must be nonnegative")
    manifest_paths = [args.manifest, *getattr(args, "extra_manifest", [])]
    missing_manifests = [path for path in manifest_paths if not path.is_file()]
    if missing_manifests or not args.unity.exists():
        raise SystemExit(
            "manifest(s) and Unity executable must exist; missing manifests: "
            + ", ".join(map(str, missing_manifests))
        )

    rank, world_size, local_rank, device = setup_distributed(args.device)
    seed = args.seed + rank * 100003
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    unity_device_ids = [
        value.strip()
        for value in os.environ.get("INDUSTRYNAV_UNITY_DEVICE_IDS", "").split(",")
        if value.strip()
    ]
    if unity_device_ids:
        os.environ["INDUSTRYNAV_UNITY_DEVICE_INDEX"] = unity_device_ids[
            local_rank % len(unity_device_ids)
        ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s | rank={rank} | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(args.output_dir / f"train_rank{rank}.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("pointgoal_ppo")

    model_config = PointGoalPPOConfig(
        depth_backbone=args.depth_backbone,
        img_size=args.img_size,
        visual_dim=args.visual_dim,
        hidden_size=args.hidden_size,
        action_history_len=args.action_history_len,
        visual_history_len=args.visual_history_len,
        half_width=args.half_width,
        pretrained_depth=args.pretrained_depth,
        goal_distance_scale_m=args.goal_distance_scale_m,
        analytic_goal_prior_strength=args.analytic_goal_prior_strength,
        absolute_scene_state=args.absolute_scene_state,
        num_scenes=args.num_scenes,
        world_coordinate_scale_m=args.world_coordinate_scale_m,
        coordinate_fourier_bands=args.coordinate_fourier_bands,
        route_actor_hidden=args.route_actor_hidden,
        base_policy_logit_scale=args.base_policy_logit_scale,
        route_actor_logit_scale=args.route_actor_logit_scale,
    )
    actor_payload = None
    source_path = (
        args.output_dir / "latest.pt"
        if args.resume and (args.output_dir / "latest.pt").is_file()
        else args.init_ppo_checkpoint
    )
    if dagger_checkpoint is not None and source_path is None:
        actor_payload = torch.load(dagger_checkpoint, map_location="cpu", weights_only=False)
        bc_config = actor_payload.get("config")
        if not bc_config:
            bc_config = json.loads((dagger_checkpoint.parent / "config.json").read_text())
        model_config = dagger_ppo_config(bc_config)
    elif source_path is not None:
        source_payload = torch.load(source_path, map_location="cpu", weights_only=False)
        if source_payload.get("model_config", {}).get("policy_architecture") in {
            "dagger_transformer", "dagger_transformer_memory",
        }:
            model_config = PointGoalPPOConfig(**source_payload["model_config"])
        del source_payload
    transformer_actor = model_config.uses_dagger_transformer
    if gradient_clip_mode == "actor_value" and not transformer_actor:
        raise SystemExit("Separate gradient clipping requires a detached-critic DAgger PPO actor")
    memory_actor = model_config.policy_architecture == "dagger_transformer_memory"
    freeze_transformer = getattr(args, "freeze_dagger_transformer", None)
    if freeze_transformer is not None:
        if not transformer_actor:
            raise SystemExit("--freeze-dagger-transformer requires a DAgger Transformer actor")
        model_config = replace(model_config, freeze_dagger_transformer=freeze_transformer)
    if transformer_actor:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        for name in ("img_size", "action_history_len", "visual_history_len", "goal_distance_scale_m", "half_width"):
            setattr(args, name, getattr(model_config, name))
        if args.absolute_scene_state or args.teacher_bc_coef or args.online_astar_teacher or args.teacher_bc_checkpoint:
            raise SystemExit("DAgger Transformer PPO uses relative goals and pure PPO without teachers")
    elif getattr(args, "critic_warmup_updates", 0):
        raise SystemExit("Critic-only warmup requires the detached Transformer critic")
    if memory_actor and args.bptt_len < 2:
        raise SystemExit("Persistent-memory PPO requires bptt-len >=2")
    if reference_initial and (not transformer_actor or (not memory_actor and args.bptt_len != 1)):
        raise SystemExit("Reference KL currently requires the exact cached Transformer window and bptt-len=1")
    if rank == 0:
        run_config_path = args.output_dir / "run_config.json"
        run_config_payload = {
            "train_config": serializable_args(args),
            "model_config": asdict(model_config),
            "world_size": int(world_size),
        }
        temporary_config = run_config_path.with_suffix(".json.tmp")
        temporary_config.write_text(
            json.dumps(run_config_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_config, run_config_path)
    model = build_pointgoal_actor_critic(model_config).to(device)
    if actor_payload is not None:
        model.load_dagger_actor(actor_payload["model"])
        logger.info("Preserved the full DAgger Transformer actor from %s", dagger_checkpoint)
        del actor_payload
    latest_path = args.output_dir / "latest.pt"
    start_update = 1
    global_steps = 0
    history_migration = None
    if args.resume and latest_path.is_file():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        history_migration = load_compatible_pointgoal_state_dict(
            model, checkpoint["model"]
        )
        start_update = int(checkpoint.get("update", 0)) + 1
        global_steps = int(checkpoint.get("global_steps", 0))
        logger.info("Resumed model at update %d", start_update - 1)
        if history_migration is not None:
            action_resize = history_migration.get("action_history")
            if action_resize is not None:
                logger.info(
                    "Expanded checkpoint action history from %d to %d slots",
                    *action_resize,
                )
            visual_resize = history_migration.get("visual_history")
            if visual_resize is not None:
                logger.info(
                    "Initialized checkpoint visual history from %d to %d slots",
                    *visual_resize,
                )
    elif args.init_ppo_checkpoint is not None:
        initial_checkpoint = torch.load(
            args.init_ppo_checkpoint,
            map_location=device,
            weights_only=False,
        )
        history_migration = load_compatible_pointgoal_state_dict(
            model,
            initial_checkpoint["model"],
        )
        logger.info(
            "Initialized PPO model weights from %s with a fresh optimizer/schedule",
            args.init_ppo_checkpoint,
        )
    elif args.init_bc_checkpoint is not None:
        loaded = load_bc_depth_encoder(model, args.init_bc_checkpoint)
        logger.info("Loaded %d BC depth-encoder tensors", loaded)
    if state is not None:
        state.iteration = start_update - 1
        state.global_step = global_steps

    reference_model = None
    if reference_initial:
        reference_payload = torch.load(reference_path, map_location="cpu", weights_only=False)
        reference_config = PointGoalPPOConfig(**reference_payload["model_config"])
        if reference_config.policy_architecture != "dagger_transformer":
            raise SystemExit("The frozen reference must be the memory-free Transformer actor")
        if reference_config.bc_actor_config != model_config.bc_actor_config:
            raise SystemExit("Reference policy must use identical vision/goal/history encoding")
        reference_model = build_pointgoal_actor_critic(reference_config).to(device)
        reference_model.load_state_dict(reference_payload["model"], strict=True)
        reference_model.requires_grad_(False).eval()
        for name, tensor in model.depth_encoder.state_dict().items():
            if not torch.equal(tensor, reference_model.depth_encoder.state_dict()[name]):
                raise SystemExit("Cached reference logits require identical frozen depth encoders")
        del reference_payload
        logger.info("Frozen reference KL enabled from %s; no action mixing or labels", reference_path)

    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(
            model,
            lr=args.lr,
            encoder_lr_scale=args.encoder_lr_scale,
            policy_lr_scale=args.policy_lr_scale,
            adaptation_lr_scale=getattr(args, "adaptation_lr_scale", 1.0),
        ),
        eps=1e-5,
        weight_decay=args.weight_decay,
    )
    if (
        args.resume
        and latest_path.is_file()
        and getattr(args, "resume_optimizer", True)
    ):
        if history_migration is not None:
            raise SystemExit(
                "Cannot restore optimizer across a temporal-input resize; "
                "resume with --no-resume-optimizer"
            )
        checkpoint_optimizer = checkpoint["optimizer"]
        if len(checkpoint_optimizer["param_groups"]) != len(optimizer.param_groups):
            raise SystemExit(
                "Cannot restore optimizer across an LR-group change; "
                "resume with --no-resume-optimizer"
            )
        restore_optimizer_state(optimizer, checkpoint_optimizer, use_config_lrs=resume_lr_mode == "config")
    elif args.resume and latest_path.is_file():
        logger.info("Reset optimizer state for resumed reward/curriculum tuning")
    logger.info("Effective optimizer LRs after initialization/resume: %s", ", ".join(
        f"{group['group_name']}={group['lr']:.3g}" for group in optimizer.param_groups
    ))
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            # Recurrent PPO calls the wrapped model once per BPTT timestep
            # before a single backward pass. Re-broadcasting mutable ResNet
            # BatchNorm buffers between those forwards invalidates autograd's
            # saved tensor versions. Parameters/gradients still synchronize.
            broadcast_buffers=False,
        )

    task_shards = read_tasks(args, rank, world_size)
    reward_kwargs = {
        "progress_scale": args.progress_scale,
        "step_penalty": args.step_penalty,
        "success_bonus": args.success_bonus,
        "timeout_penalty": args.timeout_penalty,
        "collision_penalty": args.collision_penalty,
        "collision_step_penalty": args.collision_step_penalty,
        "warning_penalty": args.warning_penalty,
        "rotation_penalty": args.rotation_penalty,
        "turn_reversal_penalty": args.turn_reversal_penalty,
        "stagnation_penalty": args.stagnation_penalty,
        "stagnation_start_steps": args.stagnation_start_steps,
        "safe_forward_bonus": args.safe_forward_bonus,
        "stuck_recovery_penalty": args.stuck_recovery_penalty,
    }
    envs: list[UnityPointGoalEnv] = []
    observations: list[dict] = []
    step_executor = None
    try:
        for env_index, tasks in enumerate(task_shards):
            env = UnityPointGoalEnv(
                tasks,
                unity_path=str(args.unity),
                output_dir=args.output_dir / "unity" / f"rank{rank}",
                worker_id=rank * args.num_envs + env_index,
                base_port=args.base_port,
                reach_m=args.reach_m,
                max_steps=args.max_episode_steps,
                dynamic_step_budget=args.dynamic_step_budget,
                step_budget_min=args.step_budget_min,
                step_budget_max=args.step_budget_max,
                steps_per_path_meter=args.steps_per_path_meter,
                step_budget_overhead=args.step_budget_overhead,
                ego_width=args.ego_width,
                ego_height=args.ego_height,
                minimap_width=args.minimap_width,
                minimap_height=args.minimap_height,
                dynamic_objects=args.dynamic_objects,
                goal_distance_scale_m=args.goal_distance_scale_m,
                absolute_scene_state=args.absolute_scene_state,
                num_scenes=args.num_scenes,
                world_coordinate_scale_m=args.world_coordinate_scale_m,
                coordinate_fourier_bands=args.coordinate_fourier_bands,
                online_astar_teacher=args.online_astar_teacher,
                astar_obstacle_clearance_m=args.astar_obstacle_clearance_m,
                astar_policy_forward_tolerance_deg=(
                    args.astar_policy_forward_tolerance_deg
                ),
                geodesic_progress_reward=args.geodesic_progress_reward,
                navigation_map_dir=getattr(args, "navigation_map_dir", None),
                resample_invalid_spawns=getattr(args, "resample_invalid_spawns", False),
                stagnation_progress_epsilon_m=(
                    args.stagnation_progress_epsilon_m
                ),
                task_sampling=args.task_sampling,
                task_sampling_seed=seed + env_index * 1009,
                scene_block_episodes=args.scene_block_episodes,
                reuse_same_scene=not getattr(args, "fresh_unity_per_episode", False),
                preserve_depth_precision=transformer_actor,
                training_safety_shield=args.training_safety_shield,
                auto_reset=not getattr(args, "parallel_env_steps", False),
                stuck_recovery_steps=args.stuck_recovery_steps,
                action_names=PPO_ACTIONS,
                reward_fn=compute_pointgoal_reward,
                reward_kwargs=reward_kwargs,
                logger=logger,
            )
            observations.append(env.reset())
            envs.append(env)

        num_envs = len(envs)
        if getattr(args, "parallel_env_steps", False):
            step_executor = ThreadPoolExecutor(max_workers=num_envs, thread_name_prefix="unity-step")
        teacher_controllers = []
        if args.teacher_bc_checkpoint is not None:
            from nav.baselines.bc.agent import BCNavController

            teacher_controllers = [
                BCNavController(
                    str(args.teacher_bc_checkpoint),
                    device=str(device),
                )
                for _ in range(num_envs)
            ]
            logger.info(
                "Enabled BC teacher auxiliary loss | checkpoint=%s | coef=%g",
                args.teacher_bc_checkpoint,
                args.teacher_bc_coef,
            )
        elif args.online_astar_teacher:
            logger.info(
                "Enabled online A* teacher | coef=%g | action_prob=%g",
                args.teacher_bc_coef,
                args.teacher_action_prob,
            )
        teacher_enabled = bool(teacher_controllers or args.online_astar_teacher)
        teacher_action_rng = random.Random(seed + 780291)
        teacher_replay = (
            TeacherReplayBuffer(
                args.teacher_replay_capacity,
                img_size=model_config.img_size,
                seed=seed + 910247,
            )
            if args.teacher_replay_capacity > 0
            else None
        )
        replay_path = args.output_dir / f"teacher_replay_rank{rank}.pt"
        if args.resume and teacher_replay is not None and replay_path.is_file():
            teacher_replay.load_state_dict(
                torch.load(replay_path, map_location="cpu", weights_only=False)
            )
            logger.info(
                "Restored %d cross-update teacher replay states",
                len(teacher_replay),
            )
        hidden = (model.module if hasattr(model, "module") else model).initial_hidden(
            num_envs, device
        )
        action_history = initial_action_history(
            num_envs,
            args.action_history_len,
            device=device,
        )
        visual_history = initial_visual_history(
            num_envs,
            args.visual_history_len,
            model_config.visual_dim,
            device=device,
        )
        masks = torch.zeros(num_envs, 1, device=device)
        metrics_path = args.output_dir / "metrics.jsonl"

        for update in range(start_update, args.total_updates + 1):
            reference_weight = reference_coefficient(update, reference_initial, reference_final, reference_decay)
            if state is not None:
                state.iteration = update
            teacher_action_probability = scheduled_teacher_probability(
                update,
                args.teacher_action_prob,
                args.teacher_action_decay_updates,
                args.teacher_action_min_prob,
            )
            storage: dict[str, list[torch.Tensor]] = defaultdict(list)
            completed_episodes: list[dict] = []
            model.eval()
            for _ in range(args.rollout_steps):
                depth = torch.from_numpy(
                    np.stack([obs["depth"] for obs in observations])
                ).to(device)
                goal = torch.from_numpy(
                    np.stack([obs["goal"] for obs in observations])
                ).to(device)
                storage["depth"].append(depth.cpu())
                storage["goal"].append(goal.cpu())
                storage["action_history"].append(action_history.cpu())
                assert visual_history is not None
                storage["visual_history"].append(visual_history.cpu())
                storage["hidden"].append(hidden.detach().cpu())
                storage["masks"].append(masks.cpu())
                teacher_actions_tensor = None
                if teacher_enabled:
                    teacher_actions = []
                    for env_index, env in enumerate(envs):
                        if args.online_astar_teacher:
                            teacher_action = env.predict_astar_teacher_action()
                        else:
                            teacher = teacher_controllers[env_index]
                            assert env.pose is not None and env.target_world is not None
                            teacher_action = teacher.predict_action(
                                ego_obs=None,
                                depth_obs=env.depth_obs,
                                curr_world_x=env.pose[0],
                                curr_world_z=env.pose[1],
                                curr_yaw_deg=env.pose[2],
                                target_world_x=env.target_world[0],
                                target_world_z=env.target_world[1],
                            )
                        if teacher_action not in PPO_ACTIONS:
                            raise RuntimeError(
                                "Teacher returned unsupported action "
                                f"{teacher_action!r}"
                            )
                        teacher_actions.append(PPO_ACTIONS.index(teacher_action))
                    teacher_actions_tensor = torch.tensor(
                        teacher_actions, dtype=torch.long
                    )
                    storage["teacher_actions"].append(teacher_actions_tensor)
                with torch.no_grad():
                    logits, value, hidden, current_visual = model.module(
                        depth,
                        goal,
                        action_history,
                        visual_history,
                        hidden,
                        masks,
                    ) if hasattr(model, "module") else model(
                        depth,
                        goal,
                        action_history,
                        visual_history,
                        hidden,
                        masks,
                    )
                    # This is the actual behavior policy, not an off-policy
                    # sampling trick: replay uses the identical transform.
                    logits = temperature_logits(logits, policy_temperature)
                    distribution = Categorical(logits=logits)
                    if reference_model is not None:
                        raw_policy = model.module if hasattr(model, "module") else model
                        ref_logits, _, _, _ = reference_model(
                            current_visual, goal, action_history, visual_history,
                            raw_policy.reference_window_hidden(storage["hidden"][-1].to(device)), masks,
                        )
                        storage["reference_logits"].append(ref_logits.cpu())
                    storage["policy_argmax"].append(logits.argmax(dim=-1).cpu())
                    action = select_rollout_action(logits, args.rollout_action_mode)
                    on_policy = torch.ones(
                        num_envs, dtype=torch.float32, device=device
                    )
                    if (
                        teacher_actions_tensor is not None
                        and teacher_action_probability > 0.0
                    ):
                        use_teacher = torch.tensor(
                            [
                                teacher_action_rng.random()
                                < teacher_action_probability
                                for _ in range(num_envs)
                            ],
                            dtype=torch.bool,
                            device=device,
                        )
                        action = torch.where(
                            use_teacher,
                            teacher_actions_tensor.to(device),
                            action,
                        )
                        on_policy = (~use_teacher).float()
                    log_prob = distribution.log_prob(action)
                if transformer_actor:
                    # Frozen encoder features are exact for PPO replay and
                    # avoid retaining 256x256 float depth windows on the CPU.
                    storage["depth"][-1] = current_visual.detach().cpu()
                next_observations = []
                rewards = []
                dones = []
                executed_actions = []
                shield_interventions = []
                map_valid = []
                selected_actions = action.cpu().tolist()
                if step_executor is not None:
                    futures = [step_executor.submit(env.step, selected_action)
                               for env, selected_action in zip(envs, selected_actions)]
                    step_results = [future.result() for future in futures]
                else:
                    step_results = [env.step(selected_action)
                                    for env, selected_action in zip(envs, selected_actions)]
                for env, (obs, reward, done, info) in zip(envs, step_results):
                    if step_executor is not None and done:
                        # Unity process construction/scene switches can need
                        # main-thread signal handlers. Only steps run in threads.
                        obs = env.reset()
                    next_observations.append(obs)
                    rewards.append(reward)
                    dones.append(done)
                    executed_actions.append(info["executed_action_index"])
                    shield_interventions.append(info["shield_intervened"])
                    map_valid.append(bool(info.get("map_progress_valid", False)))
                    if done:
                        completed_episodes.append(info)
                if teacher_controllers:
                    for teacher, selected_action, done in zip(
                        teacher_controllers,
                        executed_actions,
                        dones,
                    ):
                        teacher.observe_executed_action(
                            PPO_ACTIONS[selected_action]
                        )
                        if done:
                            teacher.reset()
                # The shield is a fixed environment action mapping. PPO must
                # keep the sampled proposal and its likelihood; only history
                # and physical-action statistics use the executed action.
                executed_action = torch.tensor(executed_actions, dtype=torch.long, device=device)
                storage["actions"].append(action.cpu())
                storage["executed_actions"].append(executed_action.cpu())
                storage["shield_interventions"].append(torch.tensor(shield_interventions, dtype=torch.float32))
                storage["map_valid"].append(torch.tensor(map_valid, dtype=torch.float32))
                storage["on_policy"].append(on_policy.cpu())
                storage["old_log_probs"].append(log_prob.cpu())
                storage["values"].append(value.cpu())
                storage["rewards"].append(torch.tensor(rewards, dtype=torch.float32))
                nonterminal = torch.tensor(
                    [not done for done in dones], dtype=torch.float32
                )
                storage["nonterminal"].append(nonterminal)
                observations = next_observations
                action_history = append_action_history(
                    action_history,
                    executed_action,
                    episode_done=~nonterminal.bool(),
                )
                visual_history = append_visual_history(
                    visual_history,
                    current_visual,
                    episode_done=~nonterminal.bool(),
                )
                masks = nonterminal.to(device).unsqueeze(-1)

            with torch.no_grad():
                depth = torch.from_numpy(
                    np.stack([obs["depth"] for obs in observations])
                ).to(device)
                goal = torch.from_numpy(
                    np.stack([obs["goal"] for obs in observations])
                ).to(device)
                _, next_value, _, _ = model(
                    depth,
                    goal,
                    action_history,
                    visual_history,
                    hidden,
                    masks,
                )
            tensors = {key: torch.stack(value) for key, value in storage.items()}
            if teacher_replay is not None:
                teacher_replay.add_rollout(tensors)
            policy_argmax_counts = torch.bincount(
                tensors["policy_argmax"].reshape(-1),
                minlength=len(PPO_ACTIONS),
            ).to(device=device, dtype=torch.float32)
            executed_action_counts = torch.bincount(
                tensors["executed_actions"].reshape(-1),
                minlength=len(PPO_ACTIONS),
            ).to(device=device, dtype=torch.float32)
            teacher_action_counts = (
                torch.bincount(
                    tensors["teacher_actions"].reshape(-1),
                    minlength=len(PPO_ACTIONS),
                ).to(device=device, dtype=torch.float32)
                if teacher_enabled
                else torch.zeros(len(PPO_ACTIONS), device=device)
            )
            action_counts = torch.stack(
                (
                    policy_argmax_counts,
                    executed_action_counts,
                    teacher_action_counts,
                )
            )
            if world_size > 1:
                dist.all_reduce(action_counts)
            teacher_class_weights = (
                balanced_teacher_class_weights(action_counts[2])
                if teacher_enabled and args.teacher_class_balance
                else None
            )
            advantages, returns = generalized_advantage_estimate(
                tensors["rewards"],
                tensors["values"],
                next_value.cpu(),
                tensors["nonterminal"],
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
            )
            advantages = normalized_advantages(advantages, world_size)
            teacher_coefficient = (
                args.teacher_bc_coef
                * teacher_auxiliary_scale(
                    update,
                    args.teacher_bc_decay_updates,
                    args.teacher_bc_min_scale,
                )
            )

            chunks = [
                (env_index, start)
                for env_index in range(num_envs)
                for start in range(0, args.rollout_steps, args.bptt_len)
            ]
            epoch_losses = []
            maximum_checked_kl = 0.0
            model.train()
            warmup = update <= getattr(args, "critic_warmup_updates", 0)
            kl_stopped = False
            for epoch in range(args.ppo_epochs):
                random.Random(seed + update * 101 + epoch).shuffle(chunks)
                for offset in range(0, len(chunks), args.chunks_per_minibatch):
                    batch_chunks = chunks[offset : offset + args.chunks_per_minibatch]
                    batch_hidden = torch.stack(
                        [tensors["hidden"][start, env] for env, start in batch_chunks]
                    ).to(device)
                    new_log_probs = []
                    new_values = []
                    entropies = []
                    old_log_probs = []
                    old_values = []
                    batch_returns = []
                    batch_advantages = []
                    batch_on_policy = []
                    teacher_losses = []
                    teacher_agreements = []
                    reference_losses = []
                    for relative_step in range(args.bptt_len):
                        indices = [
                            (start + relative_step, env) for env, start in batch_chunks
                        ]
                        step_depth = torch.stack(
                            [tensors["depth"][step, env] for step, env in indices]
                        ).to(device)
                        step_goal = torch.stack(
                            [tensors["goal"][step, env] for step, env in indices]
                        ).to(device)
                        step_history = torch.stack(
                            [tensors["action_history"][step, env] for step, env in indices]
                        ).to(device)
                        step_visual_history = torch.stack(
                            [tensors["visual_history"][step, env] for step, env in indices]
                        ).to(device)
                        step_masks = torch.stack(
                            [tensors["masks"][step, env] for step, env in indices]
                        ).to(device)
                        logits, value, batch_hidden, _ = model(
                            step_depth,
                            step_goal,
                            step_history,
                            step_visual_history,
                            batch_hidden,
                            step_masks,
                        )
                        selected = torch.stack(
                            [tensors["actions"][step, env] for step, env in indices]
                        ).to(device)
                        logits = temperature_logits(logits, policy_temperature)
                        distribution = Categorical(logits=logits)
                        if reference_model is not None:
                            ref_logits = torch.stack([
                                tensors["reference_logits"][step, env] for step, env in indices
                            ]).to(device)
                            reference_losses.append(reference_kl(
                                logits, temperature_logits(ref_logits, policy_temperature)
                            ))
                        new_log_probs.append(distribution.log_prob(selected))
                        new_values.append(value)
                        entropies.append(distribution.entropy())
                        old_log_probs.append(torch.stack([
                            tensors["old_log_probs"][step, env]
                            for step, env in indices
                        ]).to(device))
                        old_values.append(torch.stack([
                            tensors["values"][step, env] for step, env in indices
                        ]).to(device))
                        batch_returns.append(torch.stack([
                            returns[step, env] for step, env in indices
                        ]).to(device))
                        batch_advantages.append(torch.stack([
                            advantages[step, env] for step, env in indices
                        ]).to(device))
                        batch_on_policy.append(torch.stack([
                            tensors["on_policy"][step, env]
                            for step, env in indices
                        ]).to(device))
                        if teacher_enabled:
                            teacher_selected = torch.stack([
                                tensors["teacher_actions"][step, env]
                                for step, env in indices
                            ]).to(device)
                            teacher_losses.append(F.cross_entropy(
                                logits,
                                teacher_selected,
                                weight=teacher_class_weights,
                                reduction="none",
                            ))
                            teacher_agreements.append(
                                (logits.argmax(dim=-1) == teacher_selected).float()
                            )

                    new_log_probs_t = torch.stack(new_log_probs)
                    new_values_t = torch.stack(new_values)
                    entropy = torch.stack(entropies).mean()
                    reference_loss = (
                        torch.stack(reference_losses).mean() if reference_losses else logits.new_zeros(())
                    )
                    old_log_probs_t = torch.stack(old_log_probs)
                    old_values_t = torch.stack(old_values)
                    returns_t = torch.stack(batch_returns)
                    advantages_t = torch.stack(batch_advantages)
                    on_policy_t = torch.stack(batch_on_policy)
                    teacher_loss = (
                        torch.stack(teacher_losses).mean()
                        if teacher_losses
                        else logits.new_zeros(())
                    )
                    teacher_agreement = (
                        torch.stack(teacher_agreements).mean()
                        if teacher_agreements
                        else logits.new_zeros(())
                    )
                    ratio = (new_log_probs_t - old_log_probs_t).exp()
                    sampled_kl = (
                        ((ratio - 1) - (new_log_probs_t - old_log_probs_t)) * on_policy_t
                    ).sum() / on_policy_t.sum().clamp_min(1.0)
                    checked_kl = sampled_kl.detach().clone()
                    if world_size > 1:
                        dist.all_reduce(checked_kl, op=dist.ReduceOp.MAX)
                    maximum_checked_kl = max(maximum_checked_kl, checked_kl.item())
                    if (
                        not warmup and getattr(args, "target_kl", 0) > 0
                        and checked_kl.item() > args.target_kl and epoch_losses
                    ):
                        # Complete DDP's pending backward without an update.
                        optimizer.zero_grad(set_to_none=True)
                        (new_log_probs_t.sum() * 0 + new_values_t.sum() * 0).backward()
                        kl_stopped = True
                        break
                    surrogate = torch.minimum(
                        ratio * advantages_t,
                        ratio.clamp(1.0 - args.clip_param, 1.0 + args.clip_param)
                        * advantages_t,
                    )
                    policy_loss = -(
                        surrogate * on_policy_t
                    ).sum() / on_policy_t.sum().clamp_min(1.0)
                    clipped_value = old_values_t + (
                        new_values_t - old_values_t
                    ).clamp(-args.clip_param, args.clip_param)
                    value_loss = 0.5 * torch.maximum(
                        (new_values_t - returns_t).pow(2),
                        (clipped_value - returns_t).pow(2),
                    ).mean()
                    loss = (
                        (0.0 if warmup else args.policy_loss_coef) * policy_loss
                        + args.value_loss_coef * value_loss
                        - (0.0 if warmup else args.entropy_coef) * entropy
                        + teacher_coefficient * teacher_loss
                        + (0.0 if warmup else reference_weight) * reference_loss
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if warmup:
                        raw_model = model.module if hasattr(model, "module") else model
                        for name, parameter in raw_model.named_parameters():
                            if not is_value_parameter(raw_model, name):
                                parameter.grad = None
                    raw_model = model.module if hasattr(model, "module") else model
                    policy_grad_norm = gradient_l2_norm(
                        p for name, p in raw_model.named_parameters()
                        if not name.startswith("depth_encoder.") and not is_value_parameter(raw_model, name)
                    )
                    value_grad_norm = gradient_l2_norm(
                        p for name, p in raw_model.named_parameters() if is_value_parameter(raw_model, name)
                    )
                    total_grad_norm = clip_policy_value_gradients(
                        model, args.max_grad_norm, mode=gradient_clip_mode,
                    )
                    policy_grad_after_clip = gradient_l2_norm(
                        p for name, p in raw_model.named_parameters()
                        if not name.startswith("depth_encoder.") and not is_value_parameter(raw_model, name)
                    )
                    value_grad_after_clip = gradient_l2_norm(
                        p for name, p in raw_model.named_parameters() if is_value_parameter(raw_model, name)
                    )
                    optimizer.step()
                    approx_kl = (
                        ((old_log_probs_t - new_log_probs_t) * on_policy_t).sum()
                        / on_policy_t.sum().clamp_min(1.0)
                    ).detach()
                    epoch_losses.append(
                        torch.tensor(
                            [
                                loss.item(), policy_loss.item(), value_loss.item(),
                                entropy.item(), approx_kl.item(),
                                teacher_loss.item(), teacher_agreement.item(),
                                reference_loss.item(),
                                sampled_kl.detach().item(),
                                (((ratio.detach() - 1).abs() > args.clip_param).float() * on_policy_t).sum().item()
                                / on_policy_t.sum().clamp_min(1.0).item(),
                                total_grad_norm.item(), policy_grad_norm.item(), value_grad_norm.item(),
                                policy_grad_after_clip.item(), value_grad_after_clip.item(),
                            ],
                            device=device,
                        )
                    )
                if kl_stopped:
                    break

            replay_loss_stats = torch.zeros(3, device=device)
            replay_class_weights = None
            if teacher_replay is not None:
                replay_counts = teacher_replay.action_counts(
                    len(PPO_ACTIONS)
                ).to(device)
                if world_size > 1:
                    dist.all_reduce(replay_counts)
                replay_class_weights = soft_inverse_class_weights(
                    replay_counts,
                    power=args.teacher_replay_class_weight_power,
                )
                for _ in range(args.teacher_replay_steps):
                    replay_batch = teacher_replay.sample(
                        args.teacher_replay_batch_size,
                        device=device,
                    )
                    replay_logits, replay_values, _, _ = model(
                        replay_batch["depth"],
                        replay_batch["goal"],
                        replay_batch["action_history"],
                        replay_batch["visual_history"],
                        replay_batch["hidden"],
                        replay_batch["masks"],
                    )
                    replay_targets = replay_batch["teacher_actions"].long()
                    replay_loss = F.cross_entropy(
                        replay_logits,
                        replay_targets,
                        weight=replay_class_weights,
                    )
                    # Replay supervises only the actor, but DDP still expects
                    # every parameter in this standalone forward/backward to
                    # participate in reduction. Connect the critic with an
                    # exact zero term so it receives zero (not missing) grads.
                    replay_loss = replay_loss + 0.0 * replay_values.sum()
                    optimizer.zero_grad(set_to_none=True)
                    (args.teacher_replay_coef * replay_loss).backward()
                    clip_policy_value_gradients(
                        model, args.max_grad_norm, mode=gradient_clip_mode,
                    )
                    optimizer.step()
                    replay_loss_stats += torch.tensor(
                        [
                            replay_loss.item(),
                            (
                                replay_logits.argmax(dim=-1)
                                == replay_targets
                            ).float().mean().item(),
                            1.0,
                        ],
                        device=device,
                    )
                if world_size > 1:
                    dist.all_reduce(replay_loss_stats)

            global_steps += args.rollout_steps * num_envs * world_size
            if state is not None:
                state.global_step = global_steps
            loss_stats = torch.stack(epoch_losses).mean(0)
            rollout_stats = torch.tensor(
                [
                    tensors["rewards"].mean().item(),
                    len(completed_episodes),
                    sum(int(ep["success"]) for ep in completed_episodes),
                    sum(float(ep["episode_return"]) for ep in completed_episodes),
                    sum(int(ep["episode_collisions"]) for ep in completed_episodes),
                    sum(int(ep["episode_warnings"]) for ep in completed_episodes),
                    sum(int(ep["episode_forward_steps"]) for ep in completed_episodes),
                    sum(int(ep["episode_rotation_steps"]) for ep in completed_episodes),
                    sum(int(ep["episode_turn_reversals"]) for ep in completed_episodes),
                    sum(int(ep["episode_stagnant_steps"]) for ep in completed_episodes),
                    sum(int(ep["episode_collision_events"]) for ep in completed_episodes),
                    sum(int(ep["episode_stuck_recovery"]) for ep in completed_episodes),
                    sum(int(ep["episode_steps"]) for ep in completed_episodes),
                    sum(
                        float(ep["episode_initial_distance_m"])
                        for ep in completed_episodes
                    ),
                    tensors["shield_interventions"].mean().item(),
                    tensors["map_valid"].mean().item(),
                ],
                device=device,
            )
            if world_size > 1:
                dist.all_reduce(loss_stats)
                loss_stats /= world_size
                dist.all_reduce(rollout_stats)
            episode_count = int(rollout_stats[1].item())
            record = {
                "update": update,
                "global_steps": global_steps,
                "policy_temperature": policy_temperature,
                "mean_step_reward": float(rollout_stats[0].item() / world_size),
                "critic_warmup": warmup,
                "shield_intervention_fraction": float(rollout_stats[14].item() / world_size),
                "map_progress_coverage": (
                    float(rollout_stats[15].item() / world_size)
                    if getattr(args, "navigation_map_dir", None) else None
                ),
                "kl_early_stopped": kl_stopped,
                "episodes": episode_count,
                "success_rate": (
                    float(rollout_stats[2].item() / episode_count)
                    if episode_count else None
                ),
                "mean_episode_return": (
                    float(rollout_stats[3].item() / episode_count)
                    if episode_count else None
                ),
                "collisions_per_episode": (
                    float(rollout_stats[4].item() / episode_count)
                    if episode_count else None
                ),
                "warnings_per_episode": (
                    float(rollout_stats[5].item() / episode_count)
                    if episode_count else None
                ),
                "forward_steps_per_episode": (
                    float(rollout_stats[6].item() / episode_count)
                    if episode_count else None
                ),
                "rotation_steps_per_episode": (
                    float(rollout_stats[7].item() / episode_count)
                    if episode_count else None
                ),
                "turn_reversals_per_episode": (
                    float(rollout_stats[8].item() / episode_count)
                    if episode_count else None
                ),
                "stagnant_steps_per_episode": (
                    float(rollout_stats[9].item() / episode_count)
                    if episode_count else None
                ),
                "collision_events_per_episode": (
                    float(rollout_stats[10].item() / episode_count)
                    if episode_count else None
                ),
                "stuck_recovery_rate": (
                    float(rollout_stats[11].item() / episode_count)
                    if episode_count else None
                ),
                "mean_episode_steps": (
                    float(rollout_stats[12].item() / episode_count)
                    if episode_count else None
                ),
                "mean_initial_distance_m": (
                    float(rollout_stats[13].item() / episode_count)
                    if episode_count else None
                ),
                "loss": float(loss_stats[0].item()),
                "policy_loss": float(loss_stats[1].item()),
                "value_loss": float(loss_stats[2].item()),
                "entropy": float(loss_stats[3].item()),
                "approx_kl": float(loss_stats[4].item()),
                "sampled_kl_k3": float(loss_stats[8].item()),
                "maximum_checked_kl": maximum_checked_kl,
                "ppo_clip_fraction": float(loss_stats[9].item()),
                "gradient_norm_total": float(loss_stats[10].item()),
                "gradient_norm_policy": float(loss_stats[11].item()),
                "gradient_norm_value": float(loss_stats[12].item()),
                "gradient_clip_mode": gradient_clip_mode,
                "gradient_norm_policy_after_clip": float(loss_stats[13].item()),
                "gradient_norm_value_after_clip": float(loss_stats[14].item()),
                "effective_optimizer_lrs": {group["group_name"]: group["lr"] for group in optimizer.param_groups},
                "teacher_loss": float(loss_stats[5].item()),
                "teacher_agreement": float(loss_stats[6].item()),
                "reference_kl": float(loss_stats[7].item()),
                "reference_kl_coefficient": float(reference_weight),
                "teacher_coefficient": float(teacher_coefficient),
                "teacher_action_probability": float(teacher_action_probability),
                "teacher_action_fraction": float(
                    1.0 - tensors["on_policy"].mean().item()
                ),
                "teacher_source": (
                    "astar"
                    if args.online_astar_teacher
                    else "bc"
                    if teacher_controllers
                    else None
                ),
                "rollout_action_mode": args.rollout_action_mode,
                "policy_argmax_action_rates": (
                    action_counts[0] / action_counts[0].sum().clamp_min(1.0)
                ).tolist(),
                "executed_action_rates": (
                    action_counts[1] / action_counts[1].sum().clamp_min(1.0)
                ).tolist(),
                "teacher_action_rates": (
                    action_counts[2] / action_counts[2].sum().clamp_min(1.0)
                ).tolist() if teacher_enabled else None,
                "teacher_class_weights": (
                    teacher_class_weights.tolist()
                    if teacher_class_weights is not None
                    else None
                ),
                "teacher_replay_size_per_rank": (
                    len(teacher_replay) if teacher_replay is not None else 0
                ),
                "teacher_replay_loss": (
                    float(replay_loss_stats[0].item() / replay_loss_stats[2].item())
                    if replay_loss_stats[2].item() > 0
                    else None
                ),
                "teacher_replay_agreement": (
                    float(replay_loss_stats[1].item() / replay_loss_stats[2].item())
                    if replay_loss_stats[2].item() > 0
                    else None
                ),
                "teacher_replay_class_weights": (
                    replay_class_weights.tolist()
                    if replay_class_weights is not None
                    else None
                ),
            }
            if rank == 0:
                with metrics_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                logger.info("PPO %s", json.dumps(record, sort_keys=True))
                save_checkpoint(
                    latest_path,
                    model,
                    optimizer,
                    model_config,
                    args,
                    update,
                    global_steps,
                )
                if update % args.checkpoint_interval == 0:
                    save_checkpoint(
                        args.output_dir / f"update_{update:06d}.pt",
                        model,
                        optimizer,
                        model_config,
                        args,
                        update,
                        global_steps,
                    )
            if (
                teacher_replay is not None
                and update % args.checkpoint_interval == 0
            ):
                save_teacher_replay(replay_path, teacher_replay)
            if world_size > 1:
                dist.barrier(device_ids=[local_rank])
    except BaseException:
        # NCCL teardown can wait for other ranks; expose the actual cause
        # immediately rather than only a later peer collective timeout.
        logger.exception("PPO rank %d failed before environment/distributed cleanup", rank)
        raise
    finally:
        if step_executor is not None:
            step_executor.shutdown(wait=True)
        for env in envs:
            env.close()
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


class PPOTrainer(BaseTrainer[argparse.Namespace, None]):
    """PointGoal PPO trainer using the shared trainer lifecycle."""

    def fit(self) -> None:
        run_training(self.config, self.state)
