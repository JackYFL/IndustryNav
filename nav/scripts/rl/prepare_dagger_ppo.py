"""Convert the complete DAgger actor and audit non-benchmark PPO task splits."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from nav.baselines.bc.agent import BCNavController
from nav.baselines.rl.agent import PPOPointGoalController
from nav.models.policies import DaggerTransformerActorCritic, dagger_ppo_config


def sha256(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--input-points", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20260912)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "initial_actor.pt"
    if checkpoint_path.exists():
        raise SystemExit("Initialization already exists; use a new directory")
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    bc_config = source.get("config") or json.loads((args.checkpoint.parent / "config.json").read_text())
    config = dagger_ppo_config(bc_config)
    model = DaggerTransformerActorCritic(config)
    model.load_dagger_actor(source["model"])
    source_hash = sha256(args.checkpoint)
    torch.save({
        "model": model.state_dict(), "model_config": asdict(config),
        "update": 0, "global_steps": 0,
        "initialization": {"checkpoint": str(args.checkpoint), "sha256": source_hash},
    }, checkpoint_path)
    del model, source
    original = BCNavController(str(args.checkpoint), device=args.device)
    converted = PPOPointGoalController(str(checkpoint_path), device=args.device)
    bc_logits, ppo_logits = [], []
    original.model.register_forward_hook(lambda _m, _a, out: bc_logits.append(out.detach()))
    converted.model.register_forward_hook(lambda _m, _a, out: ppo_logits.append(out[0].detach()))
    rng = np.random.default_rng(20260912)
    errors = []
    for step in range(20):
        if step == 15:
            original.reset()
            converted.reset()
        state = dict(
            depth_obs=rng.random((240, 320, 1), dtype=np.float32),
            curr_world_x=step * .4, curr_world_z=step * -.7,
            curr_yaw_deg=step * 22.5, target_world_x=12., target_world_z=17.,
        )
        expected = original.predict_action(ego_obs=None, **state)
        actual = converted.predict_action(**state)
        if expected != actual:
            raise RuntimeError(f"Actor changed its action at probe {step}")
        torch.testing.assert_close(bc_logits[-1], ppo_logits[-1], atol=1e-4, rtol=1e-4)
        errors.append(float((bc_logits[-1] - ppo_logits[-1]).abs().max()))
        executed = ("forward", "turn right", "turn left")[step % 3]
        original.observe_executed_action(executed)
        converted.observe_executed_action(executed)

    benchmark = json.loads(args.input_points.read_text())
    records, rejected, seen = defaultdict(list), [], set()
    for manifest in args.manifest:
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            task = json.loads(line)
            scene = task["scene_name"]
            key = (scene, *(round(float(task[k]), 3) for k in ("init_world_x", "init_world_z", "target_x", "target_y")))
            if key in seen:
                continue
            seen.add(key)
            start_clearance = min(math.hypot(task["init_world_x"] - p["start"]["x"], task["init_world_z"] - p["start"]["z"]) for p in benchmark[scene])
            target_clearance = min(math.hypot(task["target_x"] - p["target"]["x"], task["target_y"] - p["target"]["y"]) for p in benchmark[scene])
            if start_clearance < 3 or target_clearance < 20:
                rejected.append({"scene": scene, "episode": task["episode_id"], "start_clearance_m": start_clearance, "target_clearance_px": target_clearance})
                continue
            task["source_manifest"] = str(manifest)
            records[scene].append(task)
    if set(records) != {f"scene{i}" for i in range(1, 25)}:
        raise RuntimeError("Training task audit must cover all 24 scenes")
    train, validation = [], []
    counts = {}
    for scene in sorted(records, key=lambda s: int(s[5:])):
        ordered = sorted(records[scene], key=lambda t: hashlib.sha256(json.dumps(t, sort_keys=True).encode()).hexdigest())
        if len(ordered) < 6:
            raise RuntimeError(f"Too few independent tasks for {scene}")
        for index, task in enumerate(ordered):
            task["split"] = "val" if index < 2 else "train"
            task["episode_id"] = f"ppo_resampled_{index + 1:03d}"
            (validation if index < 2 else train).append(task)
        counts[scene] = {"train": len(ordered) - 2, "val": 2}
    for name, rows in (("train", train), ("val", validation)):
        (args.output_dir / f"{name}_manifest.jsonl").write_text("".join(json.dumps(t, sort_keys=True) + "\n" for t in rows))
    report = {
        "source_checkpoint": str(args.checkpoint), "source_sha256": source_hash,
        "initial_actor_sha256": sha256(checkpoint_path),
        "actor_probes": len(errors), "maximum_logit_error": max(errors),
        "benchmark_sha256": sha256(args.input_points),
        "source_manifests": {str(p): sha256(p) for p in args.manifest},
        "train_episodes": len(train), "validation_episodes": len(validation),
        "scenes": counts, "rejected_endpoints": rejected,
        "protocol": "All 24 scenes; independently sampled endpoints; input_points used for endpoint exclusion and benchmark model selection, never gradient updates.",
    }
    (args.output_dir / "initialization_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
