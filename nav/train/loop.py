"""Model-agnostic BC training loop.

Driven by a :class:`nav.config.BCTrainConfig`. Picks the dataset variant
(single-frame vs. sequence) and the policy (:func:`nav.models.build_policy`)
from ``cfg.policy_type``, then runs class-weighted cross-entropy (or the
diffusion MSE objective) with a lower learning rate on the visual backbone.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from nav.config import BC_ACTION_TO_LABEL, BC_NAV_ACTION_TO_LABEL, BCTrainConfig
from nav.models import build_policy
from nav.train.dataset import NavEpisodeDataset, NavEpisodeSequenceDataset
from nav.train.pointgoal import POINTGOAL_ENCODING_LEGACY

_SEQUENCE_POLICIES = {"lstm", "transformer", "diffusion"}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_class_weights(
    class_counts: Dict[int, int], power: float = 1.0
) -> torch.Tensor:
    if not 0.0 <= power <= 1.0:
        raise ValueError("class_weight_power must be in [0, 1]")
    counts = np.array(
        [class_counts[i] for i in range(len(class_counts))], dtype=np.float32
    )
    present = counts > 0
    if not np.any(present):
        raise ValueError("at least one action class must have training samples")

    # Keep absent output classes at exactly zero.  Clamping an absent class to
    # one sample gives it an enormous inverse-frequency weight; when label
    # smoothing is enabled that class then receives supervision from every
    # example and can dominate the learned policy (the v3 PointGoal stop
    # collapse).  Relative weights among classes that are actually present are
    # unchanged.
    weights = np.zeros_like(counts)
    inverse = counts[present].sum() / (counts[present] * present.sum())
    weights[present] = np.power(inverse, power)
    return torch.tensor(weights, dtype=torch.float32)


def mask_untrained_stop_logits(
    logits: torch.Tensor,
    include_stop_targets: bool,
    navigation_only_actions: bool = False,
) -> torch.Tensor:
    """Make ``stop`` impossible when termination is distance-controlled.

    PointGoal rollouts already terminate before policy inference once the
    agent is within ``reach_m``.  Runs trained without terminal stop targets
    therefore use only the three navigation actions.  Keeping the fourth
    output preserves checkpoint compatibility while this mask enforces the
    intended contract during training, validation, and deployment.
    """
    if include_stop_targets or navigation_only_actions:
        return logits
    masked = logits.clone()
    masked[..., BC_ACTION_TO_LABEL["stop"]] = torch.finfo(masked.dtype).min
    return masked


def _build_datasets(cfg: BCTrainConfig):
    if cfg.policy_type in _SEQUENCE_POLICIES:
        common = dict(
            data_root=cfg.data_root, seq_len=cfg.seq_len, img_size=cfg.img_size,
            split_ratios=cfg.split_ratios, seed=cfg.seed, goal_rep=cfg.goal_rep,
            goal_encoding=cfg.goal_encoding,
            goal_distance_scale_m=cfg.goal_distance_scale_m,
            horizontal_flip_prob=cfg.horizontal_flip_prob,
            previous_action_noise_prob=cfg.previous_action_noise_prob,
            normalize_rgb=cfg.normalize_rgb, use_depth=cfg.use_depth, use_rgb=cfg.use_rgb,
            chunk_size=cfg.chunk_size,
            action_target_offset=cfg.sequence_action_offset,
            include_stop_targets=cfg.include_stop_targets,
            navigation_only_actions=cfg.navigation_only_actions,
        )
        train_ds = NavEpisodeSequenceDataset(split="train", **common)
        val_ds = NavEpisodeSequenceDataset(split="val", **common)
        input_dim = train_ds[0]["goal"].shape[-1]
    else:
        common = dict(
            data_root=cfg.data_root, img_size=cfg.img_size, split_ratios=cfg.split_ratios,
            seed=cfg.seed, state_mode=cfg.state_mode, normalize_rgb=cfg.normalize_rgb,
            use_depth=cfg.use_depth,
        )
        train_ds = NavEpisodeDataset(split="train", **common)
        val_ds = NavEpisodeDataset(split="val", **common)
        input_dim = train_ds[0]["state"].shape[0]
    return train_ds, val_ds, input_dim


def _forward_logits(model, batch, device, policy_type: str):
    """Move a batch to ``device`` and return the model's per-step logits."""
    if policy_type in _SEQUENCE_POLICIES:
        rgb = batch["rgb"].to(device) if batch["rgb"].numel() > 0 else batch["rgb"]
        depth = batch["depth"].to(device) if batch["depth"].numel() > 0 else batch["depth"]
        goal = batch["goal"].to(device)
        prev_action = batch["prev_action"].to(device)
        if policy_type == "diffusion":
            return model.sample_logits(rgb, depth, goal, prev_action)
        return model(rgb, depth, goal, prev_action)
    rgb = batch["rgb"].to(device)
    depth = batch["depth"].to(device) if batch["depth"].numel() > 0 else batch["depth"]
    return model(rgb, depth, batch["state"].to(device))


def _chunk_ce_loss(criterion, logits: torch.Tensor, action: torch.Tensor, chunk_size: int) -> torch.Tensor:
    if chunk_size > 1:
        B, C, A = logits.shape
        return criterion(logits.reshape(B * C, A), action.reshape(B * C))
    return criterion(logits, action)


def turn_direction_loss(
    logits: torch.Tensor,
    action: torch.Tensor,
    action_to_label: Dict[str, int] = BC_ACTION_TO_LABEL,
) -> torch.Tensor:
    """Balance left-vs-right learning without duplicating forward samples."""
    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])
        action = action.reshape(-1)
    right = action_to_label["turn right"]
    left = action_to_label["turn left"]
    mask = (action == right) | (action == left)
    if not torch.any(mask):
        return logits.sum() * 0.0
    turn_logits = logits[mask][:, [right, left]]
    turn_target = (action[mask] == left).long()
    return F.cross_entropy(turn_logits, turn_target)


def evaluate(
    model,
    loader,
    device,
    policy_type: str,
    chunk_size: int = 1,
    include_stop_targets: bool = True,
    navigation_only_actions: bool = False,
) -> Dict[str, float]:
    action_to_label = (
        BC_NAV_ACTION_TO_LABEL
        if navigation_only_actions
        else BC_ACTION_TO_LABEL
    )
    model.eval()
    correct = total = 0
    per_class = {i: [0, 0] for i in range(len(action_to_label))}
    predicted = {i: 0 for i in range(len(action_to_label))}
    with torch.no_grad():
        for batch in loader:
            action = batch["action"].to(device)
            logits = _forward_logits(model, batch, device, policy_type)
            logits = mask_untrained_stop_logits(
                logits, include_stop_targets, navigation_only_actions
            )
            if chunk_size > 1:  # evaluate only the first step of the chunk
                step_logits, step_action = logits[:, 0, :], action[:, 0]
            else:
                step_logits, step_action = logits, action
            pred = step_logits.argmax(dim=1)
            correct += (pred == step_action).sum().item()
            total += step_action.numel()
            for cls in range(len(action_to_label)):
                mask = step_action == cls
                per_class[cls][1] += mask.sum().item()
                per_class[cls][0] += (pred[mask] == cls).sum().item()
                predicted[cls] += (pred == cls).sum().item()

    metrics = {"acc": correct / max(total, 1)}
    metrics.update({f"acc_class_{k}": v[0] / max(v[1], 1) for k, v in per_class.items()})
    metrics.update({f"support_class_{k}": v[1] for k, v in per_class.items()})
    metrics.update({f"predicted_class_{k}": value for k, value in predicted.items()})
    present = [metrics[f"acc_class_{k}"] for k, value in per_class.items() if value[1]]
    navigation_classes = tuple(
        action_to_label[name]
        for name in ("forward", "turn right", "turn left")
    )
    navigation = [
        metrics[f"acc_class_{k}"] for k in navigation_classes if per_class[k][1]
    ]
    metrics["macro_acc"] = float(np.mean(present)) if present else 0.0
    metrics["macro_nav_acc"] = float(np.mean(navigation)) if navigation else 0.0
    return metrics


def _make_optimizer(model, cfg: BCTrainConfig) -> torch.optim.Optimizer:
    # Differential LR: the visual backbone trains slower to preserve features.
    backbone_ids = {id(p) for n, p in model.named_parameters() if "rgb_encoder" in n or "depth_encoder" in n}
    backbone_group = [p for p in model.parameters() if id(p) in backbone_ids]
    other_group = [p for p in model.parameters() if id(p) not in backbone_ids]
    param_groups = []
    if backbone_group:
        param_groups.append({"params": backbone_group, "lr": cfg.lr * cfg.backbone_lr_scale})
    if other_group:
        param_groups.append({"params": other_group, "lr": cfg.lr})
    return torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)


def initialize_from_checkpoint(model, cfg: BCTrainConfig) -> None:
    """Load compatible weights for DAgger fine-tuning, without optimizer state."""
    if not cfg.init_checkpoint:
        return
    checkpoint = torch.load(cfg.init_checkpoint, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Invalid initialization checkpoint: {cfg.init_checkpoint}")
    source_cfg = checkpoint.get("config", {})
    source_encoding = source_cfg.get(
        "goal_encoding", POINTGOAL_ENCODING_LEGACY
    )
    if source_encoding != cfg.goal_encoding:
        raise ValueError(
            "Initialization checkpoint goal encoding mismatch: "
            f"{source_encoding!r} != {cfg.goal_encoding!r}"
        )
    source_navigation_only = bool(
        source_cfg.get("navigation_only_actions", False)
    )
    if source_navigation_only != cfg.navigation_only_actions:
        raise ValueError(
            "Initialization checkpoint action-head mismatch: "
            f"navigation_only_actions={source_navigation_only} != "
            f"{cfg.navigation_only_actions}"
        )
    model.load_state_dict(checkpoint["model"], strict=True)


def _save_ckpt(path, model, cfg_dict, epoch, metrics, best_score) -> None:
    torch.save(
        {"model": model.state_dict(), "config": cfg_dict, "epoch": epoch,
         "val_metrics": metrics, "best_score": best_score,
         "selection_metric": cfg_dict.get("checkpoint_metric", "acc"),
         # Retain the legacy field for older checkpoint readers.
         "best_acc": best_score},
        path,
    )


def train(cfg: BCTrainConfig) -> None:
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    cfg_dict = asdict(cfg)
    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=2)

    train_ds, val_ds, input_dim = _build_datasets(cfg)

    loader_kwargs = {"batch_size": cfg.batch_size, "num_workers": cfg.num_workers, "pin_memory": True}
    if cfg.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_policy(cfg, input_dim).to(device)
    initialize_from_checkpoint(model, cfg)

    if not 0.0 <= cfg.label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    if cfg.turn_aux_loss_weight < 0.0:
        raise ValueError("turn_aux_loss_weight must be nonnegative")
    if cfg.checkpoint_metric not in {"acc", "macro_acc", "macro_nav_acc"}:
        raise ValueError(f"Unsupported checkpoint_metric: {cfg.checkpoint_metric!r}")
    if cfg.policy_type == "diffusion" and cfg.turn_aux_loss_weight:
        raise ValueError("turn_aux_loss_weight is only supported by logit policies")

    criterion = torch.nn.CrossEntropyLoss(
        weight=get_class_weights(
            train_ds.class_counts, cfg.class_weight_power
        ).to(device),
        label_smoothing=cfg.label_smoothing,
    )
    optimizer = _make_optimizer(model, cfg)

    best_score = float("-inf")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        for batch in pbar:
            action = batch["action"].to(device)
            optimizer.zero_grad()
            if cfg.policy_type == "diffusion":
                rgb = batch["rgb"].to(device) if batch["rgb"].numel() > 0 else batch["rgb"]
                depth = batch["depth"].to(device) if batch["depth"].numel() > 0 else batch["depth"]
                goal = batch["goal"].to(device)
                prev_action = batch["prev_action"].to(device)
                loss = model.diffusion_loss(rgb, depth, goal, prev_action, action)
            else:
                logits = _forward_logits(model, batch, device, cfg.policy_type)
                logits = mask_untrained_stop_logits(
                    logits,
                    cfg.include_stop_targets,
                    cfg.navigation_only_actions,
                )
                loss = _chunk_ce_loss(criterion, logits, action, cfg.chunk_size)
                if cfg.turn_aux_loss_weight:
                    loss = loss + cfg.turn_aux_loss_weight * turn_direction_loss(
                        logits,
                        action,
                        (
                            BC_NAV_ACTION_TO_LABEL
                            if cfg.navigation_only_actions
                            else BC_ACTION_TO_LABEL
                        ),
                    )
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        metrics = evaluate(
            model,
            val_loader,
            device,
            cfg.policy_type,
            cfg.chunk_size,
            cfg.include_stop_targets,
            cfg.navigation_only_actions,
        )
        score = metrics[cfg.checkpoint_metric]
        print(
            f"Validation acc: {metrics['acc']:.4f} | "
            f"macro: {metrics['macro_acc']:.4f} | "
            f"macro-nav: {metrics['macro_nav_acc']:.4f}"
        )

        if score > best_score:
            best_score = score
            _save_ckpt(os.path.join(cfg.output_dir, "best.pt"), model, cfg_dict, epoch, metrics, best_score)
        _save_ckpt(os.path.join(cfg.output_dir, "last.pt"), model, cfg_dict, epoch, metrics, best_score)
        with open(os.path.join(cfg.output_dir, "metrics.json"), "a") as f:
            f.write(json.dumps({"epoch": epoch, **metrics}) + "\n")

    print(f"Best validation {cfg.checkpoint_metric}: {best_score:.4f}")
