"""Build one uniformly averaged PPO policy from trusted, hash-pinned snapshots.

This is an initialization/evaluation artifact, not an optimizer-resume artifact.
There is no per-task selection, teacher supervision, or inference-time ensemble.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from nav.models.policies import PointGoalPPOConfig, build_pointgoal_actor_critic
from nav.train.checkpoint_averaging import average_state_dicts


def build_average(checkpoints: list[Path], expected_hashes: list[str], output_dir: Path) -> dict:
    if len(checkpoints) < 2 or len(checkpoints) != len(expected_hashes):
        raise ValueError("Supply at least two checkpoints, each with its expected SHA256")
    if output_dir.exists():
        raise FileExistsError("Use a new output directory; existing artifacts are never overwritten")
    sources, payloads = [], []
    for path, expected in zip(checkpoints, expected_hashes):
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected:
            raise ValueError(f"Checkpoint hash mismatch: {path}")
        # Only use user-owned, trusted experiment artifacts. Load the same
        # bytes just hashed, so an actively changing path cannot race the audit.
        import io
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
        sources.append({"checkpoint": str(path.resolve()), "sha256": actual,
                        "update": payload.get("update"), "global_steps": payload.get("global_steps"),
                        "train_config": payload.get("train_config"), "weight": 1 / len(checkpoints)})
        payloads.append(payload)
    if len({source["sha256"] for source in sources}) != len(sources):
        raise ValueError("Duplicate checkpoint bytes would silently overweight one parent")
    config = payloads[0]["model_config"]
    if config.get("policy_architecture") != "dagger_transformer":
        raise ValueError("This experiment supports the memory-free DAgger-initialized PPO actor only")
    if any(payload["model_config"] != config for payload in payloads[1:]):
        raise ValueError("Model/observation/history configurations differ")
    state = average_state_dicts([payload["model"] for payload in payloads],
                               fixed_prefixes=("depth_encoder.",))
    torch.manual_seed(20260913)
    model = build_pointgoal_actor_critic(PointGoalPPOConfig(**config)).eval()
    model.load_state_dict(state, strict=True)
    # Exercise the exact model forward and reset-aware temporal state. These
    # synthetic feature probes are a functional check, not a navigation score.
    batch = 3
    hidden = model.initial_hidden(batch, torch.device("cpu"))
    history = torch.full((batch, model.config.action_history_len), 3, dtype=torch.long)
    for step in range(27):
        masks = torch.ones(batch, 1)
        if step in (0, 16):
            masks.zero_()
        angle = torch.randn(batch)
        goal = torch.stack((torch.rand(batch), angle.sin(), angle.cos()), dim=-1)
        visual = torch.randn(batch, model.config.visual_dim)
        with torch.no_grad():
            logits, value, hidden, _ = model(visual, goal, history, None, hidden, masks)
        if any(not torch.isfinite(tensor).all() for tensor in (logits, value, hidden)):
            raise ValueError("Averaged model produced nonfinite temporal outputs")
        history = torch.roll(history, -1, dims=1)
        history[:, -1] = step % 3  # Recorded executed action, including overrides.
    metadata = {
        "method": "uniform_parameter_average", "parents": sources,
        "inference_models": 1, "teacher_or_action_mixing": False,
        "optimizer_resume_supported": False, "temporal_probe_steps": 27,
        "frozen_depth_encoder_identical": True,
        "changed_tensors_from_first": sum(not torch.equal(state[k], payloads[0]["model"][k]) for k in state),
        "evaluation_required": True,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    target = output_dir / "averaged_actor.pt"
    torch.save({"model": state, "model_config": config, "update": 0, "global_steps": 0,
                "initialization": metadata}, target)
    metadata = dict(metadata, checkpoint_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (output_dir / "averaging_audit.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--expected-sha256", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(build_average(args.checkpoints, args.expected_sha256, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
