"""Train a scene-conditioned local-waypoint planner from A* trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from nav.config import SCENE_ID_MAP
from nav.models.policies import SceneWaypointConfig, SceneWaypointPlanner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-manifest", type=Path, action="append", required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lookahead-m", type=float, default=5.0)
    parser.add_argument("--coordinate-scale-m", type=float, default=50.0)
    parser.add_argument("--coordinate-fourier-bands", type=int, default=4)
    parser.add_argument("--scene-embed-dim", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--angular-loss-coef", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _episode_csv(record: dict, manifest_path: Path) -> Path:
    episode_dir = Path(str(record["episode_dir"]))
    if episode_dir.is_absolute():
        root = episode_dir
    elif record.get("source_dataset"):
        root = Path(str(record["source_dataset"])) / episode_dir
    else:
        root = manifest_path.parent / episode_dir
    return root / "keyboard_actions.csv"


def _unique_positions(rows: list[dict]) -> tuple[np.ndarray, tuple[float, float]]:
    positions = []
    for row in rows:
        try:
            position = (
                float(row["curr_world_x"]),
                float(row["curr_world_z"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        if not positions or math.hypot(
            position[0] - positions[-1][0],
            position[1] - positions[-1][1],
        ) >= 0.05:
            positions.append(position)
    if not positions:
        return np.empty((0, 2), dtype=np.float32), (0.0, 0.0)
    last = rows[-1]
    target = (
        float(last.get("target_world_x", positions[-1][0])),
        float(last.get("target_world_z", positions[-1][1])),
    )
    return np.asarray(positions, dtype=np.float32), target


def episode_waypoint_samples(
    rows: list[dict],
    *,
    scene_id: int,
    lookahead_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build unique-position route samples with an arc-length lookahead."""
    positions, target = _unique_positions(rows)
    if len(positions) < 2:
        empty2 = np.empty((0, 2), dtype=np.float32)
        return (
            np.empty((0,), dtype=np.int64),
            empty2,
            empty2.copy(),
            empty2.copy(),
        )
    segments = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    cumulative = np.concatenate(
        (np.zeros(1, dtype=np.float32), np.cumsum(segments, dtype=np.float32))
    )
    targets = np.repeat(np.asarray(target, dtype=np.float32)[None], len(positions), 0)
    labels = np.empty_like(positions)
    for index in range(len(positions)):
        future_index = int(
            np.searchsorted(
                cumulative,
                cumulative[index] + float(lookahead_m),
                side="left",
            )
        )
        future_index = min(future_index, len(positions) - 1)
        future = positions[future_index]
        if future_index == index:
            future = targets[index]
        delta = future - positions[index]
        norm = float(np.linalg.norm(delta))
        if norm > lookahead_m:
            delta = delta * (lookahead_m / norm)
        labels[index] = delta / float(lookahead_m)
    return (
        np.full(len(positions), int(scene_id), dtype=np.int64),
        positions,
        targets,
        labels,
    )


def load_samples(
    manifests: list[Path], lookahead_m: float
) -> tuple[TensorDataset, TensorDataset, dict]:
    splits: dict[str, list[tuple[np.ndarray, ...]]] = {"train": [], "val": []}
    episodes = {"train": 0, "val": 0, "missing": 0}
    for manifest_path in manifests:
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            scene_name = str(record["scene_name"])
            if scene_name not in SCENE_ID_MAP:
                continue
            path = _episode_csv(record, manifest_path)
            if not path.is_file():
                episodes["missing"] += 1
                continue
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            sample = episode_waypoint_samples(
                rows,
                scene_id=SCENE_ID_MAP[scene_name],
                lookahead_m=lookahead_m,
            )
            if not len(sample[0]):
                continue
            split = "val" if str(record.get("split", "train")) == "val" else "train"
            splits[split].append(sample)
            episodes[split] += 1

    def dataset(parts: list[tuple[np.ndarray, ...]]) -> TensorDataset:
        if not parts:
            raise RuntimeError("Waypoint dataset split is empty")
        arrays = [np.concatenate([part[i] for part in parts]) for i in range(4)]
        return TensorDataset(
            torch.from_numpy(arrays[0]).long(),
            torch.from_numpy(arrays[1]).float(),
            torch.from_numpy(arrays[2]).float(),
            torch.from_numpy(arrays[3]).float(),
        )

    train = dataset(splits["train"])
    val = dataset(splits["val"])
    stats = {
        **episodes,
        "train_samples": len(train),
        "val_samples": len(val),
    }
    return train, val, stats


def waypoint_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    angular_loss_coef: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    regression = F.smooth_l1_loss(prediction, target)
    cosine = F.cosine_similarity(prediction, target, dim=-1, eps=1.0e-6)
    angular = (1.0 - cosine).mean()
    return regression + angular_loss_coef * angular, regression, angular


def main() -> None:
    args = parse_args()
    if args.lookahead_m <= 0.0 or args.epochs <= 0 or args.batch_size <= 0:
        raise SystemExit("lookahead, epochs, and batch size must be positive")
    if args.lr <= 0.0 or args.weight_decay < 0.0 or args.angular_loss_coef < 0.0:
        raise SystemExit("optimizer and loss scales are invalid")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    train_dataset, val_dataset, dataset_stats = load_samples(
        args.dataset_manifest, args.lookahead_m
    )
    config = SceneWaypointConfig(
        coordinate_scale_m=args.coordinate_scale_m,
        coordinate_fourier_bands=args.coordinate_fourier_bands,
        scene_embed_dim=args.scene_embed_dim,
        hidden_size=args.hidden_size,
        lookahead_m=args.lookahead_m,
    )
    model = SceneWaypointPlanner(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sums = np.zeros(4, dtype=np.float64)
        for scene, current, target, label in train_loader:
            scene, current, target, label = (
                scene.to(device), current.to(device), target.to(device), label.to(device)
            )
            prediction = model(scene, current, target)
            loss, regression, angular = waypoint_loss(
                prediction, label, angular_loss_coef=args.angular_loss_coef
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_sums += [loss.item(), regression.item(), angular.item(), 1.0]
        model.eval()
        val_sums = np.zeros(5, dtype=np.float64)
        with torch.no_grad():
            for scene, current, target, label in val_loader:
                scene, current, target, label = (
                    scene.to(device), current.to(device), target.to(device), label.to(device)
                )
                prediction = model(scene, current, target)
                loss, regression, angular = waypoint_loss(
                    prediction, label, angular_loss_coef=args.angular_loss_coef
                )
                error_m = (
                    torch.linalg.vector_norm(prediction - label, dim=-1).mean()
                    * args.lookahead_m
                )
                val_sums += [
                    loss.item(), regression.item(), angular.item(), error_m.item(), 1.0
                ]
        report = {
            "epoch": epoch,
            "train_loss": train_sums[0] / train_sums[3],
            "train_regression": train_sums[1] / train_sums[3],
            "train_angular": train_sums[2] / train_sums[3],
            "val_loss": val_sums[0] / val_sums[4],
            "val_regression": val_sums[1] / val_sums[4],
            "val_angular": val_sums[2] / val_sums[4],
            "val_waypoint_error_m": val_sums[3] / val_sums[4],
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(report, sort_keys=True) + "\n")
        print(json.dumps(report), flush=True)
        payload = {
            "model": model.state_dict(),
            "model_config": config.to_dict(),
            "train_config": {
                key: ([str(x) for x in value] if isinstance(value, list) else str(value) if isinstance(value, Path) else value)
                for key, value in vars(args).items()
            },
            "dataset_stats": dataset_stats,
            "epoch": epoch,
        }
        torch.save(payload, args.output_dir / "latest.pt")
        if report["val_loss"] < best_val:
            best_val = report["val_loss"]
            torch.save(payload, args.output_dir / "best.pt")
    (args.output_dir / "dataset_stats.json").write_text(
        json.dumps(dataset_stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
