"""Add audited, initially zero-output persistent memory to a complete PPO actor."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from nav.train.initialization import add_persistent_pointgoal_memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--memory-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--freeze-base-actor", action="store_true",
                        help="Train only persistent-memory residuals and critics, keeping all original actor weights fixed.")
    parser.add_argument("--motion-features", action="store_true",
                        help="Add relative-goal progress/forward-displacement cues to memory; no privileged state.")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("Initializers are immutable; choose a new output directory")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    payload, audit = add_persistent_pointgoal_memory(checkpoint, args.memory_size,
                                                   freeze_base_actor=args.freeze_base_actor,
                                                   motion_features=args.motion_features)
    audit.update(source_checkpoint=str(args.checkpoint),
                 source_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), seed=args.seed)
    args.output_dir.mkdir(parents=True)
    target = args.output_dir / "initial_memory_actor.pt"
    temporary = target.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)
    audit["checkpoint_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    (args.output_dir / "initialization_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
