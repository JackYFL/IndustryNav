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
from torch.utils.data import DataLoader
from tqdm import tqdm

from nav.config import BCTrainConfig
from nav.models import build_policy
from nav.train.dataset import NavEpisodeDataset, NavEpisodeSequenceDataset

_SEQUENCE_POLICIES = {"lstm", "transformer", "diffusion"}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_class_weights(class_counts: Dict[int, int]) -> torch.Tensor:
    counts = np.maximum(np.array([class_counts[i] for i in range(4)], dtype=np.float32), 1.0)
    return torch.tensor(counts.sum() / (counts * len(counts)), dtype=torch.float32)


def _build_datasets(cfg: BCTrainConfig):
    if cfg.policy_type in _SEQUENCE_POLICIES:
        common = dict(
            data_root=cfg.data_root, seq_len=cfg.seq_len, img_size=cfg.img_size,
            split_ratios=cfg.split_ratios, seed=cfg.seed, goal_rep=cfg.goal_rep,
            normalize_rgb=cfg.normalize_rgb, use_depth=cfg.use_depth, use_rgb=cfg.use_rgb,
            chunk_size=cfg.chunk_size,
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


def evaluate(model, loader, device, policy_type: str, chunk_size: int = 1) -> Dict[str, float]:
    model.eval()
    correct = total = 0
    per_class = {i: [0, 0] for i in range(4)}
    with torch.no_grad():
        for batch in loader:
            action = batch["action"].to(device)
            logits = _forward_logits(model, batch, device, policy_type)
            if chunk_size > 1:  # evaluate only the first step of the chunk
                step_logits, step_action = logits[:, 0, :], action[:, 0]
            else:
                step_logits, step_action = logits, action
            pred = step_logits.argmax(dim=1)
            correct += (pred == step_action).sum().item()
            total += step_action.numel()
            for cls in range(4):
                mask = step_action == cls
                per_class[cls][1] += mask.sum().item()
                per_class[cls][0] += (pred[mask] == cls).sum().item()

    metrics = {"acc": correct / max(total, 1)}
    metrics.update({f"acc_class_{k}": v[0] / max(v[1], 1) for k, v in per_class.items()})
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


def _save_ckpt(path, model, cfg_dict, epoch, metrics, best_acc) -> None:
    torch.save(
        {"model": model.state_dict(), "config": cfg_dict, "epoch": epoch,
         "val_metrics": metrics, "best_acc": best_acc},
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

    criterion = torch.nn.CrossEntropyLoss(weight=get_class_weights(train_ds.class_counts).to(device))
    optimizer = _make_optimizer(model, cfg)

    best_acc = 0.0
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
                loss = _chunk_ce_loss(criterion, logits, action, cfg.chunk_size)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        metrics = evaluate(model, val_loader, device, cfg.policy_type, cfg.chunk_size)
        val_acc = metrics["acc"]
        print(f"Validation acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            _save_ckpt(os.path.join(cfg.output_dir, "best.pt"), model, cfg_dict, epoch, metrics, best_acc)
        _save_ckpt(os.path.join(cfg.output_dir, "last.pt"), model, cfg_dict, epoch, metrics, best_acc)
        with open(os.path.join(cfg.output_dir, "metrics.json"), "a") as f:
            f.write(json.dumps({"epoch": epoch, **metrics}) + "\n")

    print(f"Best val acc: {best_acc:.4f}")
