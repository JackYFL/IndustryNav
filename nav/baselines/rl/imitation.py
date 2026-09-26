"""Offline imitation warm-up for the recurrent PointGoal actor-critic."""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nav.baselines.rl.actions import (
    PPO_ACTIONS,
    append_action_history,
    initial_action_history,
)
from nav.data.pointgoal_imitation import (
    PointGoalImitationEpisodeDataset,
    collate_pointgoal_imitation_episodes,
)
from nav.models.policies import (
    PointGoalActorCritic,
    PointGoalPPOConfig,
    append_visual_history,
    initial_visual_history,
    load_compatible_pointgoal_state_dict,
)
from nav.train.initialization import load_bc_depth_encoder


def imitation_class_weights(
    class_counts: dict[int, int],
    *,
    power: float,
    num_actions: int = len(PPO_ACTIONS),
) -> torch.Tensor:
    """Return softly balanced weights with unit mean over training samples."""
    if not 0.0 <= power <= 1.0:
        raise ValueError("class-weight power must be in [0, 1]")
    counts = torch.tensor(
        [class_counts.get(index, 0) for index in range(num_actions)],
        dtype=torch.float32,
    )
    present = counts > 0
    if not bool(present.all()):
        raise ValueError("Every PPO action needs at least one imitation sample")
    inverse = counts.sum() / (counts * num_actions)
    weights = inverse.pow(float(power))
    return weights / ((weights * counts).sum() / counts.sum())


def _freeze_batch_norm_statistics(module: torch.nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            child.eval()


def _encode_valid_depth_frames(
    model: PointGoalActorCritic,
    depth: torch.Tensor,
    valid: torch.Tensor,
    *,
    encoder_batch_size: int,
) -> torch.Tensor:
    """Encode only real padded-sequence entries in memory-bounded batches."""
    if encoder_batch_size <= 0:
        raise ValueError("encoder_batch_size must be positive")
    batch_size, time_steps = depth.shape[:2]
    flat_depth = depth.flatten(0, 1)
    flat_valid = valid.flatten()
    valid_indices = flat_valid.nonzero(as_tuple=False).flatten()
    flat_features = torch.zeros(
        batch_size * time_steps,
        model.config.visual_dim,
        device=depth.device,
    )
    with torch.no_grad():
        for start in range(0, valid_indices.numel(), encoder_batch_size):
            indices = valid_indices[start : start + encoder_batch_size]
            flat_features[indices] = model.encode_depth(flat_depth[indices])
    return flat_features.reshape(batch_size, time_steps, -1)


def _empty_metrics() -> dict[str, Any]:
    return {
        "loss_sum": 0.0,
        "samples": 0,
        "correct": 0,
        "class_correct": [0] * len(PPO_ACTIONS),
        "class_total": [0] * len(PPO_ACTIONS),
        "predicted": [0] * len(PPO_ACTIONS),
    }


def _finalize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    samples = max(int(metrics["samples"]), 1)
    class_accuracy = [
        correct / total if total else 0.0
        for correct, total in zip(
            metrics["class_correct"], metrics["class_total"]
        )
    ]
    return {
        "loss": float(metrics["loss_sum"]) / samples,
        "accuracy": int(metrics["correct"]) / samples,
        "macro_accuracy": float(sum(class_accuracy)) / len(class_accuracy),
        "class_accuracy": class_accuracy,
        "class_total": metrics["class_total"],
        "predicted_action_rates": [
            count / samples for count in metrics["predicted"]
        ],
        "samples": int(metrics["samples"]),
    }


def run_imitation_epoch(
    model: PointGoalActorCritic,
    loader: DataLoader,
    *,
    device: torch.device,
    class_weights: torch.Tensor,
    bptt_len: int,
    encoder_batch_size: int,
    optimizer: torch.optim.Optimizer | None = None,
    max_grad_norm: float = 1.0,
    action_history_noise: float = 0.0,
) -> dict[str, Any]:
    """Replay complete episodes with teacher-forced recurrent histories."""
    training = optimizer is not None
    if bptt_len <= 0:
        raise ValueError("bptt_len must be positive")
    if not 0.0 <= action_history_noise <= 1.0:
        raise ValueError("action_history_noise must be in [0, 1]")
    model.train(training)
    # Offline warm-up intentionally preserves the online-trained visual
    # representation; small per-timestep batches would corrupt BatchNorm.
    model.depth_encoder.eval()
    metrics = _empty_metrics()
    weights = class_weights.to(device)

    for batch in loader:
        depth = batch["depth"].to(device, non_blocking=True)
        goal = batch["goal"].to(device, non_blocking=True)
        target = batch["action"].to(device, non_blocking=True)
        behavior = batch["behavior_action"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        batch_size, time_steps = valid.shape
        features = _encode_valid_depth_frames(
            model,
            depth,
            valid,
            encoder_batch_size=encoder_batch_size,
        )
        hidden = model.initial_hidden(batch_size, device)
        action_history = initial_action_history(
            batch_size,
            model.config.action_history_len,
            device=device,
        )
        visual_history = initial_visual_history(
            batch_size,
            model.config.visual_history_len,
            model.config.visual_dim,
            device=device,
        )

        for chunk_start in range(0, time_steps, bptt_len):
            chunk_end = min(time_steps, chunk_start + bptt_len)
            hidden = hidden.detach()
            if visual_history is not None:
                visual_history = visual_history.detach()
            loss_sum = torch.zeros((), device=device)
            valid_count = 0
            if training:
                optimizer.zero_grad(set_to_none=True)
            grad_context = torch.enable_grad() if training else torch.no_grad()
            with grad_context:
                for step in range(chunk_start, chunk_end):
                    valid_step = valid[:, step]
                    if not bool(valid_step.any()):
                        break
                    masks = (
                        valid_step.float().unsqueeze(-1)
                        if step > 0
                        else torch.zeros(batch_size, 1, device=device)
                    )
                    logits, _, hidden, current_visual = model.forward_from_visual(
                        features[:, step],
                        goal[:, step],
                        action_history,
                        visual_history,
                        hidden,
                        masks,
                    )
                    losses = F.cross_entropy(
                        logits,
                        target[:, step],
                        weight=weights,
                        reduction="none",
                    )
                    loss_sum = loss_sum + losses[valid_step].sum()
                    count = int(valid_step.sum().item())
                    valid_count += count
                    prediction = logits.argmax(dim=-1)
                    valid_prediction = prediction[valid_step]
                    valid_target = target[:, step][valid_step]
                    metrics["correct"] += int(
                        (valid_prediction == valid_target).sum().item()
                    )
                    metrics["samples"] += count
                    for action_id in range(len(PPO_ACTIONS)):
                        action_mask = valid_target == action_id
                        metrics["class_total"][action_id] += int(
                            action_mask.sum().item()
                        )
                        metrics["class_correct"][action_id] += int(
                            (
                                (valid_prediction == action_id)
                                & action_mask
                            ).sum().item()
                        )
                        metrics["predicted"][action_id] += int(
                            (valid_prediction == action_id).sum().item()
                        )

                    history_action = behavior[:, step]
                    if training and action_history_noise > 0.0:
                        noise_mask = valid_step & (
                            torch.rand(batch_size, device=device)
                            < action_history_noise
                        )
                        random_action = torch.randint(
                            len(PPO_ACTIONS),
                            (batch_size,),
                            device=device,
                        )
                        history_action = torch.where(
                            noise_mask, random_action, history_action
                        )
                    next_valid = (
                        valid[:, step + 1]
                        if step + 1 < time_steps
                        else torch.zeros_like(valid_step)
                    )
                    action_history = append_action_history(
                        action_history,
                        history_action,
                        episode_done=~next_valid,
                    )
                    visual_history = append_visual_history(
                        visual_history,
                        current_visual,
                        episode_done=~next_valid,
                    )
            if valid_count:
                loss = loss_sum / valid_count
                metrics["loss_sum"] += float(loss_sum.detach().item())
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [
                            parameter
                            for parameter in model.parameters()
                            if parameter.requires_grad
                        ],
                        max_grad_norm,
                    )
                    optimizer.step()
    return _finalize_metrics(metrics)


def run_route_epoch(
    model: PointGoalActorCritic,
    loader: DataLoader,
    *,
    device: torch.device,
    class_weights: torch.Tensor,
    optimizer: torch.optim.Optimizer | None = None,
    max_grad_norm: float = 1.0,
) -> dict[str, Any]:
    """Train the nonlinear GPS/Compass residual without loading image files."""
    if not model.config.route_actor_hidden:
        raise ValueError("route-only training requires an enabled route actor")
    training = optimizer is not None
    model.train(training)
    metrics = _empty_metrics()
    weights = class_weights.to(device)
    for batch in loader:
        valid = batch["valid"]
        goal = batch["goal"][valid].to(device, non_blocking=True)
        target = batch["action"][valid].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad() if training else torch.no_grad():
            logits = model.route_actor(goal)
            loss = F.cross_entropy(logits, target, weight=weights)
        prediction = logits.argmax(dim=-1)
        count = int(target.numel())
        metrics["loss_sum"] += float(loss.detach().item()) * count
        metrics["samples"] += count
        metrics["correct"] += int((prediction == target).sum().item())
        for action_id in range(len(PPO_ACTIONS)):
            action_mask = target == action_id
            metrics["class_total"][action_id] += int(action_mask.sum().item())
            metrics["class_correct"][action_id] += int(
                ((prediction == action_id) & action_mask).sum().item()
            )
            metrics["predicted"][action_id] += int(
                (prediction == action_id).sum().item()
            )
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.route_actor.parameters() if p.requires_grad],
                max_grad_norm,
            )
            optimizer.step()
    return _finalize_metrics(metrics)


def _serializable_args(args) -> dict:
    return {
        key: (
            [str(item) for item in value]
            if isinstance(value, list)
            else str(value)
            if isinstance(value, Path)
            else value
        )
        for key, value in vars(args).items()
    }


def _save_checkpoint(
    path: Path,
    *,
    model: PointGoalActorCritic,
    optimizer: torch.optim.Optimizer,
    args,
    epoch: int,
    best_metric: float,
    train_metrics: dict,
    val_metrics: dict,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": asdict(model.config),
        "train_config": _serializable_args(args),
        "pretrain_kind": "pointgoal_actor_imitation",
        "epoch": int(epoch),
        "update": 0,
        "global_steps": int(epoch * train_metrics["samples"]),
        "best_metric": float(best_metric),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def run_offline_imitation(args) -> None:
    """Train a PPO-compatible actor checkpoint from non-benchmark episodes."""
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise SystemExit("epochs/batch-size must be positive and workers nonnegative")
    if args.action_history_len < 5 or args.visual_history_len < 5:
        raise SystemExit("action and visual history lengths must both be at least 5")
    if args.coordinate_fourier_bands and not args.absolute_scene_state:
        raise SystemExit("coordinate Fourier features require absolute scene state")
    if args.route_actor_hidden < 0:
        raise SystemExit("route-actor-hidden must be nonnegative")
    if args.route_only and args.route_actor_hidden <= 0:
        raise SystemExit("--route-only requires --route-actor-hidden")
    if args.base_policy_logit_scale < 0.0 or args.route_actor_logit_scale < 0.0:
        raise SystemExit("policy logit scales must be nonnegative")
    if not 0.0 <= args.action_history_noise <= 1.0:
        raise SystemExit("action-history-noise must be in [0, 1]")
    for root in args.data_root:
        if not root.is_dir():
            raise SystemExit(f"Imitation data root does not exist: {root}")
    init_bc_checkpoint = getattr(args, "init_bc_checkpoint", None)
    if args.init_checkpoint is not None and init_bc_checkpoint is not None:
        raise SystemExit(
            "--init-checkpoint and --init-bc-checkpoint are mutually exclusive"
        )
    if args.init_checkpoint is not None and not args.init_checkpoint.is_file():
        raise SystemExit(f"Initial checkpoint does not exist: {args.init_checkpoint}")
    if init_bc_checkpoint is not None and not init_bc_checkpoint.is_file():
        raise SystemExit(
            f"Initial BC checkpoint does not exist: {init_bc_checkpoint}"
        )
    if args.route_only and init_bc_checkpoint is not None:
        raise SystemExit("--init-bc-checkpoint is not used with --route-only")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; pass --device cpu for a smoke test")
    device = torch.device(device_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(args.output_dir / "train.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("pointgoal_imitation")

    model_config = PointGoalPPOConfig(
        depth_backbone=args.depth_backbone,
        img_size=args.img_size,
        visual_dim=args.visual_dim,
        hidden_size=args.hidden_size,
        action_history_len=args.action_history_len,
        visual_history_len=args.visual_history_len,
        half_width=args.half_width,
        pretrained_depth=False,
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
    model = PointGoalActorCritic(model_config).to(device)
    if args.route_only:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.route_actor.parameters():
            parameter.requires_grad_(True)
    else:
        for parameter in model.depth_encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in model.critic.parameters():
            parameter.requires_grad_(False)

    latest_path = args.output_dir / "latest.pt"
    start_epoch = 1
    best_metric = float("-inf")
    resume_payload = None
    if args.resume and latest_path.is_file():
        resume_payload = torch.load(latest_path, map_location=device, weights_only=False)
        if PointGoalPPOConfig(**resume_payload["model_config"]) != model_config:
            raise SystemExit("Resume checkpoint model configuration does not match")
        model.load_state_dict(resume_payload["model"])
        start_epoch = int(resume_payload.get("epoch", 0)) + 1
        best_metric = float(resume_payload.get("best_metric", best_metric))
        logger.info("Resumed offline imitation at epoch %d", start_epoch - 1)
    elif args.init_checkpoint is not None:
        source = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        migration = load_compatible_pointgoal_state_dict(model, source["model"])
        logger.info("Initialized from %s | migration=%s", args.init_checkpoint, migration)
    elif init_bc_checkpoint is not None:
        loaded = load_bc_depth_encoder(model, init_bc_checkpoint)
        logger.info(
            "Loaded %d BC/DAgger depth-encoder tensors from %s",
            loaded,
            init_bc_checkpoint,
        )

    train_dataset = PointGoalImitationEpisodeDataset(
        args.data_root,
        split=args.train_split,
        goal_distance_scale_m=model_config.goal_distance_scale_m,
        absolute_scene_state=model_config.absolute_scene_state,
        num_scenes=model_config.num_scenes,
        world_coordinate_scale_m=model_config.world_coordinate_scale_m,
        coordinate_fourier_bands=model_config.coordinate_fourier_bands,
        episode_limit=args.episode_limit,
        load_depth=not args.route_only,
    )
    val_dataset = PointGoalImitationEpisodeDataset(
        args.data_root,
        split=args.val_split,
        goal_distance_scale_m=model_config.goal_distance_scale_m,
        absolute_scene_state=model_config.absolute_scene_state,
        num_scenes=model_config.num_scenes,
        world_coordinate_scale_m=model_config.world_coordinate_scale_m,
        coordinate_fourier_bands=model_config.coordinate_fourier_bands,
        episode_limit=args.episode_limit,
        load_depth=not args.route_only,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate_pointgoal_imitation_episodes,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    weights = imitation_class_weights(
        train_dataset.class_counts,
        power=args.class_weight_power,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=1.0e-5,
    )
    if resume_payload is not None and args.resume_optimizer:
        optimizer.load_state_dict(resume_payload["optimizer"])
    logger.info(
        "Loaded %d train / %d val episodes | train actions=%s | weights=%s",
        len(train_dataset),
        len(val_dataset),
        dict(train_dataset.class_counts),
        [round(float(value), 4) for value in weights],
    )

    metrics_path = args.output_dir / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs + 1):
        if args.route_only:
            train_metrics = run_route_epoch(
                model,
                train_loader,
                device=device,
                class_weights=weights,
                optimizer=optimizer,
                max_grad_norm=args.max_grad_norm,
            )
            val_metrics = run_route_epoch(
                model,
                val_loader,
                device=device,
                class_weights=weights,
            )
        else:
            train_metrics = run_imitation_epoch(
                model,
                train_loader,
                device=device,
                class_weights=weights,
                bptt_len=args.bptt_len,
                encoder_batch_size=args.encoder_batch_size,
                optimizer=optimizer,
                max_grad_norm=args.max_grad_norm,
                action_history_noise=args.action_history_noise,
            )
            val_metrics = run_imitation_epoch(
                model,
                val_loader,
                device=device,
                class_weights=weights,
                bptt_len=args.bptt_len,
                encoder_batch_size=args.encoder_batch_size,
            )
        selected_metric = float(val_metrics[args.checkpoint_metric])
        improved = selected_metric > best_metric
        best_metric = max(best_metric, selected_metric)
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "checkpoint_metric": args.checkpoint_metric,
            "best_metric": best_metric,
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        _save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            args=args,
            epoch=epoch,
            best_metric=best_metric,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )
        if improved:
            _save_checkpoint(
                args.output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                args=args,
                epoch=epoch,
                best_metric=best_metric,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
            )
        if epoch % args.checkpoint_interval == 0 or epoch == args.epochs:
            _save_checkpoint(
                args.output_dir / f"epoch_{epoch:03d}.pt",
                model=model,
                optimizer=optimizer,
                args=args,
                epoch=epoch,
                best_metric=best_metric,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
            )
        logger.info(
            "epoch=%d train_acc=%.4f train_macro=%.4f val_acc=%.4f "
            "val_macro=%.4f val_actions=%s",
            epoch,
            train_metrics["accuracy"],
            train_metrics["macro_accuracy"],
            val_metrics["accuracy"],
            val_metrics["macro_accuracy"],
            [round(rate, 3) for rate in val_metrics["predicted_action_rates"]],
        )


__all__ = [
    "imitation_class_weights",
    "run_imitation_epoch",
    "run_route_epoch",
    "run_offline_imitation",
]
