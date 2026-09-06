"""Reconstruct historical Gemini episodes into separate, cached-replay-only bundles.

No model calls, source edits or invented request allowance. Unknown transport
settings remain unknown: these bundles are NOT current-protocol API runs.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

from nav.harness.checkpoint import CheckpointStore, atomic_json, config_digest, digest


PROTOCOL = "legacy-gemini-reconstructed-v1"
PROMPT_REF = "0ae3c79:nav/prompts/nav_ego_state_history.txt"
PROMPT_SHA = "b4895fdd540fe7732c7603295a7bbd01d6f7cf3ee2eab5fcb5dea7db15dfbe3a"
ACTION_SPACE = {"forward": 15, "turn right": 22.5, "turn left": -22.5, "stop": 0}
NORMAL_STOPS = {"max_steps", "reached_vicinity"}
POSE_COLUMNS = dict(zip(("x", "y", "z", "rx", "yaw", "rz"),
                        ("curr_world_x", "curr_world_y", "curr_world_z",
                         "curr_direction_x", "curr_direction_y", "curr_direction_z")))
PATTERNS = (
    (r"^- World position X/Z: \(([-\d.]+), ([-\d.]+)\) m$", ("curr_world_x", "curr_world_z")),
    (r"^- World target X/Z: \(([-\d.]+), ([-\d.]+)\) m$", ("target_world_x", "target_world_z")),
    (r"^- World distance to target: ([-\d.]+) m$", ("distance_m",)),
    (r"^- Minimap position/target: \(([-\d.]+), ([-\d.]+)\) / \(([-\d.]+), ([-\d.]+)\) px$",
     ("curr_x", "curr_y", "target_x", "target_y")),
    (r"^- Heading: ([-\d.]+)°$", ("theta",)),
    (r"^- Environment motion: (\w+)$", ("dynamic_objects",)),
    (r"^- Allowed actions: (.*)$", ("allowed_actions",)),
    (r"^- `stop` is valid when world distance is at most ([-\d.]+) m\.$", ("reach_m",)),
)


def legacy_template():
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(["git", "show", PROMPT_REF], cwd=repo, check=True, capture_output=True)
    if digest(result.stdout) != PROMPT_SHA:
        raise ValueError("Historical prompt does not match the audited template hash.")
    return result.stdout.decode()


def read_rows(path):
    return list(csv.DictReader(io.StringIO(path.read_bytes().decode(), newline=""))) if path.exists() else []


def snapshot(folder):
    """Content identity, including images; refuse links outside the source tree."""
    files = {}
    for path in sorted(folder.rglob("*")):
        if path.is_symlink():
            raise ValueError("Source contains a symlink; refusing an ambiguous copy.")
        if path.is_file():
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
            files[str(path.relative_to(folder))] = hasher.hexdigest()
    return files


def parse_prompt(prompt, template):
    values = {}
    for pattern, names in PATTERNS:
        match = re.search(pattern, prompt, re.M)
        if not match:
            raise ValueError("Historical prompt lacks a required state field.")
        values.update(zip(names, match.groups()))
    values["history"] = prompt.split("## MOVEMENT HISTORY\n", 1)[1].split("\n## ACTIONS", 1)[0].strip()
    if template.format(**values).strip() != prompt:
        raise ValueError("Saved prompt differs from the audited historical template.")
    if ast.literal_eval(values["allowed_actions"]) != list(ACTION_SPACE):
        raise ValueError("Unexpected historical action space.")
    return values


def parse_history(text):
    if text == "No previous movements yet.":
        return []
    entries = []
    for line in text.splitlines():
        match = re.fullmatch(
            r"Step (\d+): World X/Z \(([-\d.]+), ([-\d.]+)\) m, θ=([-\d.]+)°, "
            r"Action: '(forward|turn right|turn left|stop)', Distance to target: ([-\d.]+) m", line)
        if not match:
            raise ValueError("Unsupported historical history entry.")
        step, x, z, yaw, action, distance = match.groups()
        entries.append({"step": int(step), "world_position": [float(x), float(z)],
                        "theta": float(yaw), "action": action,
                        "distance_to_target_m": float(distance), "observation": ""})
    return entries


def recorded_settings(folder, rows, values, *, model_id="google/gemini-3.8-flash", dynamic=False):
    import shlex

    launch_log = (folder / "run.log").read_text()
    commands = [line[len("# command: "):] for line in launch_log.splitlines() if line.startswith("# command: ")]
    if len(commands) != 1:
        raise ValueError("Need one unambiguous recorded launch command.")
    tokens = shlex.split(commands[0])
    raw = {token[2:]: tokens[i + 1] for i, token in enumerate(tokens[:-1]) if token.startswith("--")}
    types = {"model_id": str, "file_name": str, "scene_id": int, "scene_name": str,
             "point_id": str, "seed_id": str, "history_size": int, "max_steps": int,
             "reach_m": float, "ego_width": int, "ego_height": int, "minimap_width": int,
             "minimap_height": int, "init_world_x": float, "init_world_z": float,
             "init_curr_direction": float, "target_x": int, "target_y": int, "dynamic_objects": str}
    if set(types) - set(raw):
        raise ValueError("Launch command is missing required reconstruction settings.")
    settings = {key: convert(raw[key]) for key, convert in types.items()}
    if (settings["model_id"] != model_id or settings["max_steps"] != 70
            or settings["history_size"] != 5 or raw.get("vision_input") != "true"
            or settings["scene_name"] != folder.parents[2].name
            or settings["point_id"] != folder.parents[1].name
            or settings["seed_id"] != folder.name.removeprefix("seed")):
        raise ValueError("Not an audited historical model/task launch.")
    if (settings["ego_width"], settings["ego_height"], settings["minimap_width"], settings["minimap_height"]) != (512, 512, 862, 512):
        raise ValueError("Unexpected sensor settings.")
    if math.dist([settings["init_world_x"], settings["init_world_z"]],
                 [float(rows[0][k]) for k in ("init_world_x", "init_world_z")]) > .01:
        raise ValueError("Launch spawn and action log disagree.")
    for key in ("target_x", "target_y", "reach_m"):
        if float(settings[key]) != float(values[key]):
            raise ValueError("Launch target/reach threshold and prompt disagree.")
    if settings["dynamic_objects"] != values["dynamic_objects"]:
        raise ValueError("Launch motion setting and prompt disagree.")
    runtime = "\n".join(p.read_text(errors="replace") for p in folder.glob("*.log") if p.name != "unity_log.txt")
    engines = set(re.findall(r"Engine config: quality_level=(\d+), screen=(\d+)x(\d+)", runtime))
    motions = set(re.findall(r"human_speed=([\d.]+)m/s vehicle_speed=([\d.]+)m/s robot_speed=([\d.]+)m/s lighting=(\w+)", runtime))
    if len(engines) != 1 or len(motions) != 1:
        raise ValueError("Missing or conflicting runtime engine/motion records.")
    quality, width, height = next(iter(engines))
    human, vehicle, robot, lighting = next(iter(motions))
    if lighting != "disabled":
        raise ValueError("Only recorded authored lighting is supported.")
    summaries = read_rows(folder / "results.csv")
    if not dynamic and summaries and (summaries[-1].get("step_budget_mode") not in {None, "", "fixed"}
                      or int(summaries[-1]["max_steps"]) != 70):
        raise ValueError("Result budget disagrees with historical fixed-budget protocol.")
    settings.update(vision_input=True, llm_provider="unverified_legacy", max_tokens=None,
                    dynamic_step_budget=False, sim_steps_per_decision=2,
                    quality_level=int(quality), screen_width=int(width), screen_height=int(height),
                    human_speed_mps=float(human), vehicle_speed_mps=float(vehicle), robot_speed_mps=float(robot),
                    marker_source="vector", hide_unity_red_marker=True)
    if dynamic:
        if ("--dynamic_step_budget" not in tokens or "--no-dynamic_step_budget" in tokens
                or raw.get("llm_provider") != "openrouter" or not summaries):
            raise ValueError("Dynamic recovery needs the original provider, budget flag and result summary.")
        for key, convert in {"step_budget_min": int, "step_budget_max": int,
                             "steps_per_path_meter": float, "step_budget_overhead": int,
                             "max_tokens": int, "llm_min_request_interval_sec": float}.items():
            settings[key] = convert(raw[key])
        summary = summaries[-1]
        expected = max(settings["step_budget_min"], min(settings["step_budget_max"], math.ceil(
            settings["step_budget_overhead"] + settings["steps_per_path_meter"] * math.dist(
                [float(rows[0][k]) for k in ("init_world_x", "init_world_z")],
                [float(rows[0][k]) for k in ("target_world_x", "target_world_z")]))))
        if (summary.get("step_budget_mode") != "dynamic" or int(summary["max_steps"]) != expected
                or int(summary["initial_step_budget"]) != expected or int(summary["sim_steps_per_decision"]) != 2):
            raise ValueError("Recorded dynamic budget or simulation step count is inconsistent.")
        settings.update(llm_provider="openrouter", dynamic_step_budget=True,
                        recorded_step_budget=expected, recorded_initial_step_budget=expected)
    return settings


def reconstruct(folder, template, *, settings_reader=recorded_settings, protocol=PROTOCOL,
                require_pending_reasoning=False):
    action_bytes = (folder / "llm_actions.csv").read_bytes()
    reader = csv.DictReader(io.StringIO(action_bytes.decode(), newline=""))
    rows, line_boundaries = [], []
    for row in reader:
        rows.append(row)
        line_boundaries.append(reader.line_num)
    if not rows or [int(r["step"]) for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("Actions must be nonempty and contiguous from step 1.")
    qa = (folder / "agent_qa.txt").read_bytes().decode()
    matches = list(re.finditer(r"^=== Step (\d+) ===\n(.*?)(?=^=== Step \d+ ===\n|\Z)", qa, re.M | re.S))
    if [int(m[1]) for m in matches] != list(range(1, len(rows) + 1)):
        raise ValueError("Q&A and action step numbers disagree.")
    decisions, warnings = [], []
    for row, match in zip(rows, matches):
        answer = re.fullmatch(r"\[Q\]\n(.*?)\n\[A\]\nAction: (.*?)\nReasoning: (.*?)\n={20,}\s*", match[2], re.S)
        if not answer or answer[2] != row["action"]:
            raise ValueError("Incomplete or mismatched historical answer.")
        if require_pending_reasoning and row is rows[-1] and not answer[3].strip():
            raise ValueError("Pending answer has empty reasoning; cannot distinguish a saved decision from an error fallback.")
        prompt = answer[1].strip()
        values = parse_prompt(prompt, template)
        pose = {k: float(row[v]) for k, v in POSE_COLUMNS.items()}
        if not all(math.isfinite(v) for v in pose.values()):
            raise ValueError("Non-finite recorded pose.")
        if any(abs((pose[k] + 180) % 360 - 180) > .1 for k in ("rx", "rz")):
            raise ValueError("Cannot restore a non-upright pose.")
        actual = [float(row[k]) for k in ("move", "strafe", "look")]
        expected = [ACTION_SPACE[row["action"]] if row["action"] == "forward" else 0., 0.,
                    ACTION_SPACE[row["action"]] if row["action"].startswith("turn ") else 0.]
        if actual != expected:
            raise ValueError("Recorded action signal differs from the audited action space.")
        world = [float(values[k]) for k in ("curr_world_x", "curr_world_z")]
        position_error = math.dist(world, [pose["x"], pose["z"]])
        yaw_error = abs((pose["yaw"] - float(values["theta"]) + 180) % 360 - 180)
        if position_error > .009 or yaw_error > .06:
            warning = {"step": int(row["step"]), "position_error_m": position_error, "yaw_error_deg": yaw_error}
            if row is rows[-1]:
                raise ValueError(f"Pending action pose conflict: XZ={position_error:.4f}m yaw={yaw_error:.4f}deg; no automatic rollback chosen.")
            warnings.append(warning)
        decisions.append({"action": answer[2], "reasoning": answer[3], "observation": "", "prompt": prompt,
                          "history_entry": {"position": [int(values[k]) for k in ("curr_x", "curr_y")],
                                            "world_position": world, "theta": float(values["theta"]),
                                            "distance_to_target_m": float(values["distance_m"])}})
    settings = settings_reader(folder, rows, values)
    budget = settings.get("recorded_step_budget", 70)
    initial_budget = settings.get("recorded_initial_step_budget", budget)
    history = parse_history(values["history"])
    expected_steps = list(range(max(1, len(rows) - settings["history_size"]), len(rows)))
    if [h["step"] for h in history] != expected_steps:
        raise ValueError("Last request does not contain the expected history window.")
    for entry in history:
        if entry["action"] != rows[entry["step"] - 1]["action"]:
            raise ValueError("Last request history and earlier actions disagree.")
    pending_image = folder / "llm_fp" / f"{len(rows) - 1}.png"
    if not pending_image.is_file():
        raise ValueError("Pending action's original RGB image is missing.")
    config = {"protocol": protocol, "prompt_sha256": PROMPT_SHA, "action_space": ACTION_SPACE,
              "input_modalities": ["ego"], "request_timeout_sec": None, "settings": settings,
              "execution_policy": {"cached_action_smoke_only": True, "api_resume_ready": False,
                                   "new_model_calls_allowed": False},
              "reconstruction": {"source_run": str(folder.resolve()), "historical_prompt_source": PROMPT_REF,
                  "unknown": ["original_provider_routing", "effective_inference_options", "request_timeout",
                              "request_attempts_and_cumulative_call_count", "original_unity_binary_hash"],
                  "inferred": {"sim_steps_per_decision": "2: historical CLI default; observed 45-degree turns support it",
                               "dynamic_step_budget": "false: historical fixed-budget runner and --max_steps 70",
                               "hide_unity_red_marker": "historical default; not a restored scene snapshot"},
                  "history_source": "exact history text of the final saved request; no fabricated observations",
                  "historical_pose_discrepancies": warnings}}
    boundary_lines = line_boundaries[-2] if len(rows) > 1 else 1
    action_prefix = "".join(action_bytes.decode().splitlines(keepends=True)[:boundary_lines]).encode()
    qa_prefix = qa[:matches[-1].start()].encode()
    state = {"version": 1, "config_sha256": config_digest(config), "phase": "decision_ready",
             "step_count": len(rows) - 1, "step_budget": budget, "initial_step_budget": initial_budget,
             "step_budget_mode": "dynamic" if settings["dynamic_step_budget"] else "fixed",
             "pose": pose, "init_world": [float(rows[0][k]) for k in ("init_world_x", "init_world_z")],
             "target_world": [float(rows[0][k]) for k in ("target_world_x", "target_world_z")],
             "history": history, "pending_decision": decisions[-1], "minimap_projector": None,
             "detected_init_xy": [int(rows[0][k]) for k in ("init_px", "init_py")], "resume_count": 0,
             "recovery_source": "reconstructed_legacy_pre_action_logs", "stop_reason": None,
             "scene_state_restored": False, "request_budget": None,
             "historical_api_calls": None, "modalities": ["depth", "ego", "minimap"],
             "files": {name: {"bytes": len(data), "sha256": digest(data)}
                       for name, data in (("actions", action_prefix), ("qa", qa_prefix))}}
    CheckpointStore(folder, config, folder / "llm_actions.csv", folder / "agent_qa.txt").validate(state)
    return config, state


def checkpoint_store(folder, config):
    return CheckpointStore(folder, config, folder / "llm_actions.csv", folder / "agent_qa.txt")


def create_bundle(source, destination, template, *, reconstructor=reconstruct, store_factory=checkpoint_store):
    source, destination = source.resolve(), destination.resolve()
    if destination == source or source in destination.parents or destination in source.parents or destination.exists():
        raise ValueError("Destination must be new and separate from the source.")
    identity = snapshot(source)
    config, state = reconstructor(source, template)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    if snapshot(destination) != identity or snapshot(source) != identity:
        raise ValueError("Source changed during reconstruction or copy integrity failed.")
    (destination / "legacy_prompt.txt").write_text(template)
    atomic_json(destination / "run_config.json", config)
    atomic_json(destination / "checkpoint.json", state)
    atomic_json(destination / "source_manifest.json", {"source_run": str(source), "sha256": identity})
    store = store_factory(destination, config)
    loaded = store.load()
    report = {"status": "offline_validated", "source_run": str(source), "bundle": str(destination),
              "committed_steps": loaded["step_count"], "pending_step": loaded["step_count"] + 1,
              "total_budget": loaded["step_budget"], "prompt_records_verified": loaded["step_count"] + 1,
              "history_entries": len(loaded["history"]), "source_unchanged": True,
              "model_calls": 0, "api_resume_ready": False,
              "historical_pose_discrepancies": config["reconstruction"]["historical_pose_discrepancies"]}
    atomic_json(destination / "reconstruction_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    source_root = args.source_root.resolve()
    if root.exists() or root == source_root or source_root in root.parents or root in source_root.parents:
        raise SystemExit("Choose a new output root outside the source tree.")
    template = legacy_template()
    root.mkdir(parents=True)
    report = {"protocol": PROTOCOL, "model_calls": 0, "api_resume_ready": False, "runs": []}
    for source in sorted(source_root.glob("scene*/point*/gemini-3.8-flash/seed0")):
        summaries = read_rows(source / "results.csv")
        if summaries and summaries[-1].get("stop_reason") in NORMAL_STOPS:
            continue
        if not read_rows(source / "llm_actions.csv"):
            continue
        try:
            item = create_bundle(source, root / source.relative_to(source_root), template)
        except (ValueError, OSError, KeyError) as exc:
            item = {"source_run": str(source), "status": "blocked", "error": str(exc)}
        report["runs"].append(item)
        atomic_json(root / "manifest.json", report)
        print(f"{source.parents[2].name}/{source.parents[1].name}: {item['status']}", flush=True)
    print(f"Manifest: {root / 'manifest.json'}")


if __name__ == "__main__":
    main()
