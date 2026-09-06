"""Smoke-test a reconstructed bundle without ANY new model request.

Default: storage/rollback checks only. --unity additionally restores the pose,
replays exactly one previously saved decision, and checkpoints the observed
post-action pose. All mutations are confined to a new smoke copy.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from mlagents_envs.base_env import ActionTuple

from nav.config import BEHAVIOR_NAME
from nav.harness.checkpoint import CheckpointStore, RunLock, atomic_json, pose_from_steps, validate_restored_pose
from nav.harness.env_setup import setup_and_prime
from nav.scripts.reconstruct_gemini_runs import PROTOCOL, PROMPT_SHA, digest, read_rows, snapshot


def bundle_store(folder, config):
    if config.get("protocol") == "legacy-qwen-reconstructed-v1":
        from nav.scripts.reconstruct_qwen_runs import checkpoint_store
        return checkpoint_store(folder, config)
    return CheckpointStore(folder, config, folder / "llm_actions.csv", folder / "agent_qa.txt")


def check_bundle(folder):
    identity = json.loads((folder / "source_manifest.json").read_text())["sha256"]
    actual_files = snapshot(folder)
    if any(actual_files.get(name) != expected for name, expected in identity.items()):
        raise ValueError("Reconstructed source logs/images were modified.")
    config = json.loads((folder / "run_config.json").read_text())
    if config.get("protocol") not in {PROTOCOL, "legacy-qwen-reconstructed-v1"} or config.get("execution_policy") != {
        "cached_action_smoke_only": True, "api_resume_ready": False, "new_model_calls_allowed": False,
    }:
        raise ValueError("Not a cached-replay-only reconstructed bundle.")
    if digest((folder / "legacy_prompt.txt").read_bytes()) != PROMPT_SHA:
        raise ValueError("Reconstructed prompt was modified.")
    state = bundle_store(folder, config).load()
    if state["phase"] != "decision_ready" or state["step_count"] >= state["step_budget"]:
        raise ValueError("Smoke requires a pending saved decision within the original budget.")
    if config["protocol"] == PROTOCOL and (
        config["settings"]["llm_provider"] != "unverified_legacy" or state["request_budget"] is not None
    ):
        raise ValueError("Unexpected provider/budget binding for a no-model replay.")
    if config["protocol"] == "legacy-qwen-reconstructed-v1" and (
        config["settings"]["llm_provider"] != "openrouter" or not state.get("request_budget")
    ):
        raise ValueError("Reconstructed Qwen must retain its original provider and counter.")
    return config, state


def launch_args(config, state, output, base_port):
    settings = config["settings"]
    if not Path(settings["file_name"]).exists():
        raise ValueError("The recorded Unity client is missing; no alternative build is silently substituted.")
    return argparse.Namespace(**{**settings, "frame_save_dir": str(output), "base_port": base_port, "worker_id": 0,
                                 "init_world_x": state["pose"]["x"], "init_world_z": state["pose"]["z"],
                                 "init_curr_direction": state["pose"]["yaw"], "_resume_world_y": state["pose"]["y"]})


def save_rgb(steps, path, settings):
    rgb = np.asarray(steps.obs[0][0])
    if rgb.ndim != 3:
        raise ValueError("Missing RGB observation.")
    if rgb.shape[0] == 3:
        rgb = rgb.transpose(1, 2, 0)
    if rgb.shape != (settings["ego_height"], settings["ego_width"], 3):
        raise ValueError(f"Restored RGB resolution mismatch: {rgb.shape}")
    if not np.isfinite(rgb).all() or np.ptp(rgb) <= 0:
        raise ValueError("Restored RGB is non-finite or blank.")
    Image.fromarray((rgb.clip(0, 1) * 255).astype(np.uint8)).save(path)


def smoke(bundle, output, *, unity=False, setup=setup_and_prime, base_port=None, backend_name="unity"):
    bundle, output = bundle.resolve(), output.resolve()
    if output.exists() or bundle == output or bundle in output.parents or output in bundle.parents:
        raise ValueError("Use a new smoke directory separate from the reconstruction bundle.")
    config, state = check_bundle(bundle)
    before = snapshot(bundle)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(bundle, output)
    store = bundle_store(output, config)
    action_bytes = (output / "llm_actions.csv").read_bytes()
    qa_bytes = (output / "agent_qa.txt").read_bytes()
    row = read_rows(output / "llm_actions.csv")[-1]
    action_tail = action_bytes[state["files"]["actions"]["bytes"]:]
    qa_tail = qa_bytes[state["files"]["qa"]["bytes"]:]
    lock = RunLock(output)
    env = None
    report = {"status": "started", "backend": backend_name if unity else "storage_only",
              "model_calls": 0, "api_resume_ready": False, "source_bundle": str(bundle),
              "original_step_budget": state["step_budget"], "committed_steps_before": state["step_count"],
              "pending_action": state["pending_decision"]["action"], "scene_state_restored": False}
    try:
        archive = store.restore_logs(state)
        if len(read_rows(output / "llm_actions.csv")) != state["step_count"]:
            raise ValueError("Rollback did not preserve the committed action boundary.")
        if not (output / "llm_fp" / f"{state['step_count']}.png").exists():
            raise ValueError("Rollback lost the pending action's RGB.")
        if (state["step_budget"] != config["settings"].get("recorded_step_budget", 70)
                or state["initial_step_budget"] != config["settings"].get("recorded_initial_step_budget", 70)):
            raise ValueError("Original budget changed.")
        report.update(archive=str(archive), rollback_verified=True)
        if unity:
            if base_port is None:
                from nav.scripts.run_benchmark_grid import free_tcp_port
                base_port = free_tcp_port()
            args = launch_args(config, state, output, base_port)
            logger = logging.getLogger("reconstructed_gemini_smoke")
            primed = setup(args, logger)
            env = primed.env
            steps = env.get_steps(BEHAVIOR_NAME)[0]
            actual = pose_from_steps(steps)
            validate_restored_pose(state["pose"], actual, state["target_world"], primed.target_world)
            if abs(actual["y"] - state["pose"]["y"]) > .1:
                raise ValueError("Restored height differs by more than 0.1m.")
            save_rgb(steps, output / "smoke_before.png", config["settings"])
            report.update(pose_restored=True, restored_pose=actual, actual_target=list(primed.target_world),
                          position_error_m=math.dist([actual['x'], actual['z']], [state['pose']['x'], state['pose']['z']]))
            # Append the ORIGINAL saved logs, never manufacture a new model reply.
            with (output / "llm_actions.csv").open("ab") as actions, (output / "agent_qa.txt").open("ab") as qa:
                actions.write(action_tail)
                qa.write(qa_tail)
                actions.flush()
                qa.flush()
                signal = np.array([[float(row[k]) for k in ("move", "strafe", "look")]], dtype=np.float32)
                env.set_actions(BEHAVIOR_NAME, ActionTuple(continuous=signal))
                for _ in range(config["settings"]["sim_steps_per_decision"]):
                    env.step()
                post_steps = env.get_steps(BEHAVIOR_NAME)[0]
                post_pose = pose_from_steps(post_steps)
                save_rgb(post_steps, output / "smoke_after.png", config["settings"])
                pending = state["pending_decision"]
                history = state["history"] + [{**pending["history_entry"], "step": state["step_count"] + 1,
                                               "action": pending["action"], "observation": ""}]
                state = store.save({**state, "phase": "ready", "pose": post_pose,
                                    "step_count": state["step_count"] + 1, "pending_decision": None,
                                    "history": history[-config["settings"]["history_size"]:],
                                    "resume_count": state["resume_count"] + 1,
                                    "stop_reason": "cached_action_smoke_limit"}, streams=(actions, qa))
            checked = store.load()
            report.update(cached_decisions_replayed=1, post_action_pose=post_pose,
                          committed_steps_after=checked["step_count"], checkpoint_roundtrip=True,
                          final_distance_m=math.dist([post_pose['x'], post_pose['z']], state['target_world']),
                          simulated_action_ticks=config["settings"]["sim_steps_per_decision"])
        report.update(status="passed", bundle_unchanged=snapshot(bundle) == before)
        if not report["bundle_unchanged"]:
            raise ValueError("Source bundle unexpectedly changed.")
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            lock.close()
            atomic_json(output / "smoke_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--unity", action="store_true", help="Launch Unity and replay one cached action; still zero model calls.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(smoke(args.bundle, args.output_dir, unity=args.unity), indent=2))


if __name__ == "__main__":
    main()
