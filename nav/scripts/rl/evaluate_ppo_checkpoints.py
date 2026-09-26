"""Evaluate immutable PPO snapshots with a fixed 96-point shield protocol."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def write_json(path, payload):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--reference-monitor", type=Path,
        help="Reuse a completed identical initialization baseline from an earlier monitor.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--input-points", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--unity", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base-port", type=int, default=51000)
    parser.add_argument("--milestones", type=int, nargs="+", default=[10, 20, 40, 80, 120, 160, 200])
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    status_path = args.output_root / "monitor.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"evaluations": [], "skipped_updates": []}
    if args.reference_monitor and not status["evaluations"]:
        reference = json.loads(args.reference_monitor.read_text())
        initial = next(item for item in reference["evaluations"] if item["update"] == 0)
        initial_hash = hashlib.sha256(args.initial_checkpoint.read_bytes()).hexdigest()
        if initial["checkpoint_sha256"] != initial_hash:
            raise SystemExit("Reference initialization checkpoint hash does not match")
        reference_root = args.reference_monitor.resolve().parent.parent
        for kind, expected_count in (("benchmark96", 96), ("validation48", 48)):
            summary_path = (reference_root / initial[kind]["summary"]).resolve()
            report = json.loads(summary_path.read_text())
            episodes = report["episodes"]
            keys = {(e["scene_name"], e["episode_id"]) for e in episodes}
            if len(episodes) != expected_count or len(keys) != expected_count or any(e.get("returncode", 1) != 0 for e in episodes):
                raise SystemExit("Reference evaluation is incomplete")
            expected = dict(
                reach_m=2.0, eval_seed=0, safety_shield=True,
                shield_collision_streak=8, shield_recovery_turns=6,
                shield_recovery_forward_steps=4, shield_max_clearance_turns=8,
                shield_warning_streak=8, shield_warning_turns=2,
                shield_turn_streak=12, shield_turn_escape_steps=4,
                shield_terminal_homing_distance_m=0.0,
            )
            if any(report.get(k) != v for k, v in expected.items()):
                raise SystemExit("Reference evaluation protocol does not match")
            if kind == "benchmark96":
                old_input = reference_root / report["input_points"]
                if old_input.read_bytes() != args.input_points.read_bytes():
                    raise SystemExit("Reference benchmark points do not match")
            else:
                old_command = json.loads((summary_path.parent / "command.json").read_text())
                old_manifest = reference_root / old_command[old_command.index("--manifest") + 1]
                def canonical_tasks(path):
                    return sorted(json.dumps(dict(json.loads(line), split="test"), sort_keys=True) for line in path.read_text().splitlines() if line.strip())
                if canonical_tasks(old_manifest) != canonical_tasks(args.validation_manifest):
                    raise SystemExit("Reference validation tasks do not match")
            initial[kind]["summary"] = str(summary_path)
        initial["checkpoint"] = str(args.initial_checkpoint.resolve())
        status["evaluations"].append(initial)
        status["initialization_reference"] = str(args.reference_monitor.resolve())
        write_json(status_path, status)
    # The shared evaluator calls all non-training records 'test'. These are
    # independently reserved validation pairs, not the canonical benchmark.
    validation = args.output_root / "validation_tasks.jsonl"
    rows = [dict(json.loads(line), split="test") for line in args.validation_manifest.read_text().splitlines() if line.strip()]
    validation.write_text("".join(json.dumps(row) + "\n" for row in rows))
    common = [
        sys.executable, "-m", "nav.scripts.evaluation.evaluate_pointgoal_policy",
        "--baseline", "ppo", "--unity", str(args.unity), "--device", "cuda",
        "--scenes", "24", "--eval-seed", "0", "--workers", str(args.workers),
        "--base-port", str(args.base_port), "--persistent-policy", "--resume",
        "--safety-shield", "--shield-collision-streak", "8", "--shield-recovery-turns", "6",
        "--shield-recovery-forward-steps", "4", "--shield-max-clearance-turns", "8",
        "--shield-warning-streak", "8", "--shield-warning-turns", "2",
        "--shield-turn-streak", "12", "--shield-turn-escape-steps", "4",
    ]
    while True:
        evaluated = {item["update"] for item in status["evaluations"]}
        candidates = [u for u in args.milestones if u not in evaluated and u not in status["skipped_updates"] and (args.checkpoint_dir / f"update_{u:06d}.pt").is_file()]
        if 0 not in evaluated:
            update, checkpoint = 0, args.initial_checkpoint
        elif candidates:
            # Avoid evaluating obsolete snapshots when training outpaces eval.
            update = max(candidates)
            status["skipped_updates"] = sorted(set(status["skipped_updates"] + [u for u in candidates if u < update]))
            checkpoint = args.checkpoint_dir / f"update_{update:06d}.pt"
        elif max(args.milestones) in evaluated:
            status["state"] = "completed"
            write_json(status_path, status)
            break
        else:
            status["state"] = "waiting_for_checkpoint"
            write_json(status_path, status)
            time.sleep(args.poll_seconds)
            continue
        if update in status["skipped_updates"]:
            # A newer completed checkpoint supersedes this one.
            time.sleep(args.poll_seconds)
            continue
        entry = {"update": update, "checkpoint": str(checkpoint)}
        for kind, count, source_flags in (
            ("benchmark96", 96, ["--input-points", str(args.input_points), "--episodes-per-scene", "4"]),
            ("validation48", 48, ["--manifest", str(validation), "--episodes-per-scene", "2"]),
        ):
            directory = args.output_root / f"update_{update:06d}" / kind
            directory.mkdir(parents=True, exist_ok=True)
            summary = directory / "summary.json"
            status.update(state="evaluating", current_update=update, current_split=kind)
            write_json(status_path, status)
            command = common + source_flags + ["--checkpoint", str(checkpoint), "--output-root", str(directory)]
            write_json(directory / "command.json", command)
            with (directory / "evaluation.log").open("a") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode or not summary.is_file():
                status.update(state="evaluation_failed", failed_update=update, failed_split=kind)
                write_json(status_path, status)
                raise SystemExit(f"Evaluation failed: {directory}")
            report = json.loads(summary.read_text())
            episodes = report["episodes"]
            keys = {(e["scene_name"], e["episode_id"]) for e in episodes}
            if len(episodes) != count or len(keys) != count or any(e.get("returncode", 1) != 0 for e in episodes):
                raise SystemExit(f"Incomplete evaluation: {directory}")
            successes = sum(bool(e["success"]) for e in episodes)
            entry[kind] = {"successes": successes, "episodes": count, "success_rate": successes / count, "summary": str(summary)}
            print(json.dumps({"update": update, "split": kind, **entry[kind]}), flush=True)
        entry["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        status["evaluations"].append(entry)
        best = max(status["evaluations"], key=lambda item: (item["benchmark96"]["successes"], item["validation48"]["successes"]))
        status["best"] = best
        status["target_reached"] = best["benchmark96"]["successes"] >= 58 and best["update"] > 0
        write_json(status_path, status)
        if status["target_reached"]:
            write_json(args.output_root / "target_reached.json", best)
            break


if __name__ == "__main__":
    main()
