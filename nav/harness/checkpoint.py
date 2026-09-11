"""Transport-neutral, crash-safe navigation checkpoints (not Unity snapshots).

An action is committed only after its post-action pose is observed. A durable
pending decision can be replayed from its pre-action pose without a second model
call. Dynamic scene objects and provider-internal conversation state are not
restored; this limitation is recorded in every continuation event.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile


CHECKPOINT_NAME = "checkpoint.json"
CHECKPOINT_VERSION = 1
NORMAL_STOPS = {"max_steps", "reached_vicinity"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def config_digest(config):
    return digest(json.dumps(config, sort_keys=True, allow_nan=False).encode())


def atomic_json(path, data):
    path = Path(path)
    encoded = json.dumps(data, indent=2, allow_nan=False).encode() + b"\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class RunLock:
    """An OS-held lock is released even on process death; never delete its file."""

    def __init__(self, folder):
        self.stream = (Path(folder) / ".run.lock").open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                self.stream.seek(0)
                self.stream.write(b"0")
                self.stream.flush()
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            raise ValueError(f"Another process is using this run: {folder}") from exc

    def close(self):
        self.stream.close()


def pose_from_steps(decision_steps):
    values = decision_steps.obs[3][0]
    if len(values) < 6:
        raise ValueError("Checkpoint requires a six-value world pose observation.")
    pose = {name: float(value) for name, value in zip(("x", "y", "z", "rx", "yaw", "rz"), values)}
    if not all(math.isfinite(value) for value in pose.values()):
        raise ValueError("Checkpoint world pose is not finite.")
    return pose


def validate_restored_pose(expected, actual, target, actual_target):
    position_error = math.hypot(expected["x"] - actual["x"], expected["z"] - actual["z"])
    yaw_error = abs((expected["yaw"] - actual["yaw"] + 180) % 360 - 180)
    if position_error > 0.1 or yaw_error > 1:
        raise ValueError(f"Unity did not restore the checkpoint pose: position error={position_error:.3f}m, yaw error={yaw_error:.2f}deg")
    if math.hypot(target[0] - actual_target[0], target[1] - actual_target[1]) > 0.1:
        raise ValueError("Unity target changed while restoring the checkpoint.")


class CheckpointStore:
    def __init__(self, folder, config, actions_path, qa_path):
        self.folder = Path(folder).resolve()
        self.path = self.folder / CHECKPOINT_NAME
        self.config = config
        self.files = {"actions": Path(actions_path).resolve(), "qa": Path(qa_path).resolve()}
        # Rollback is intentionally restricted to per-run logs, never arbitrary
        # custom output paths or paths supplied inside an untrusted checkpoint.
        for path in self.files.values():
            if path.parent != self.folder:
                raise ValueError("Checkpointed runs require actions/Q&A logs directly inside frame_save_dir.")
        if len(set(self.files.values())) != 2:
            raise ValueError("Action and Q&A logs must be different files.")

    def budget_binding(self):
        provider = self.config["settings"]["llm_provider"].upper()
        counter = os.getenv(f"{provider}_REQUEST_COUNTER_FILE", "").strip()
        limit = os.getenv(f"{provider}_MAX_REQUESTS", "").strip()
        if not limit:
            return None
        if not counter:
            raise ValueError("A bounded run requires its persistent API request counter.")
        path = Path(counter).resolve()
        count = 0
        if path.exists():
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
                row = conn.execute("SELECT value FROM request_counter WHERE id=1").fetchone()
                count = int(row[0])
        return {"counter_file": str(path), "limit": int(limit), "requests_reserved": count}

    def save(self, state, streams=()):
        for stream in streams:
            stream.flush()
            os.fsync(stream.fileno())
        files = {}
        for name, path in self.files.items():
            content = path.read_bytes() if path.exists() else b""
            files[name] = {"bytes": len(content), "sha256": digest(content)}
        value = {**state, "version": CHECKPOINT_VERSION,
                 "config_sha256": config_digest(self.config), "files": files,
                 "request_budget": self.budget_binding()}
        self.validate(value)
        atomic_json(self.path, value)
        return value

    def validate(self, state):
        if state.get("version") != CHECKPOINT_VERSION:
            raise ValueError("Unsupported checkpoint version.")
        if state.get("config_sha256") != config_digest(self.config):
            raise ValueError("Checkpoint belongs to a different navigation configuration.")
        step = state.get("step_count")
        if type(step) is not int or step < 0:
            raise ValueError("Invalid checkpoint step count.")
        for name in ("step_budget", "initial_step_budget"):
            if type(state.get(name)) is not int or state[name] < 0:
                raise ValueError(f"Invalid checkpoint {name}.")
        if state["step_budget"] > 0 and step > state["step_budget"]:
            raise ValueError("Checkpoint exceeds its original step budget.")
        if state.get("phase") not in {"ready", "decision_ready", "completed"}:
            raise ValueError("Invalid checkpoint phase.")
        pose = state.get("pose", {})
        if set(pose) != {"x", "y", "z", "rx", "yaw", "rz"} or not all(
            isinstance(v, (int, float)) and math.isfinite(v) for v in pose.values()
        ):
            raise ValueError("Invalid checkpoint world pose.")
        if any(abs((pose[key] + 180) % 360 - 180) > 0.1 for key in ("rx", "rz")):
            raise ValueError("This Unity interface can only restore upright agents (yaw rotation).")
        for name in ("init_world", "target_world"):
            values = state.get(name, [])
            if len(values) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                raise ValueError(f"Invalid checkpoint {name}.")
        history = state.get("history")
        if not isinstance(history, list) or any(not isinstance(row, dict) for row in history):
            raise ValueError("Invalid checkpoint history.")
        if state["phase"] == "decision_ready":
            pending = state.get("pending_decision", {})
            if pending.get("error") or pending.get("action") not in self.config["action_space"]:
                raise ValueError("Invalid pending checkpoint decision.")
            if any(not isinstance(pending.get(k), str) for k in ("reasoning", "observation", "prompt")) or not isinstance(pending.get("history_entry"), dict):
                raise ValueError("Pending checkpoint decision lacks its prompt/visual history.")
        for name in self.files:
            record = state.get("files", {}).get(name, {})
            if type(record.get("bytes")) is not int or record["bytes"] < 0 or not isinstance(record.get("sha256"), str):
                raise ValueError("Invalid checkpoint log boundary.")

    def prefix(self, state, name):
        path = self.files[name]
        content = path.read_bytes() if path.exists() else b""
        record = state["files"][name]
        prefix = content[:record["bytes"]]
        if len(prefix) != record["bytes"] or digest(prefix) != record["sha256"]:
            raise ValueError(f"Checkpoint log prefix was modified or truncated: {path}")
        return prefix

    def load(self):
        with self.path.open(encoding="utf-8") as stream:
            state = json.load(stream)
        # The worker persists a reply before returning to the simulation loop.
        # This covers a Unity failure while the main thread is still idling.
        reply_path = self.folder / "decision_reply.json"
        if state.get("phase") == "ready" and reply_path.exists():
            with reply_path.open(encoding="utf-8") as stream:
                reply = json.load(stream)
            if (reply.get("config_sha256") == state.get("config_sha256")
                    and reply.get("step_count") == state.get("step_count")):
                state.update(phase="decision_ready", pending_decision=reply["decision"],
                             request_budget=reply.get("request_budget") or state.get("request_budget"))
        self.validate(state)
        binding = state.get("request_budget")
        if binding:
            actual = self.budget_binding()
            if (not actual or not Path(binding["counter_file"]).exists()
                    or actual["counter_file"] != binding["counter_file"]
                    or actual["limit"] > binding["limit"]
                    or actual["requests_reserved"] < binding["requests_reserved"]):
                raise ValueError("Resume must retain the original API counter and cannot reset/increase its allowance.")
        for name in self.files:
            self.prefix(state, name)
        rows = list(csv.DictReader(io.StringIO(self.prefix(state, "actions").decode())))
        if [int(row["step"]) for row in rows] != list(range(1, state["step_count"] + 1)):
            raise ValueError("Checkpoint step count and committed action rows disagree.")
        return state

    def save_reply(self, step_count, decision):
        if decision.get("error"):
            return
        atomic_json(self.folder / "decision_reply.json", {
            "config_sha256": config_digest(self.config), "step_count": step_count,
            "decision": {key: decision.get(key, {} if key == "history_entry" else "")
                         for key in ("action", "reasoning", "observation", "prompt", "history_entry")},
            "request_budget": self.budget_binding(),
        })

    def restore_logs(self, state):
        """Archive uncommitted tails before rollback; retain all completed frames."""
        prefixes = {name: self.prefix(state, name) for name in self.files}
        archive = Path(tempfile.mkdtemp(prefix="resume-", dir=self.folder))
        shutil.copy2(self.path, archive / CHECKPOINT_NAME)
        # Copy logs before truncating; uncommitted bytes remain recoverable.
        for name, path in self.files.items():
            if path.exists():
                shutil.copy2(path, archive / path.name)
            with path.open("wb") as stream:
                stream.write(prefixes[name])
                stream.flush()
                os.fsync(stream.fileno())
        for name in ("results.csv", "unity_log.txt"):
            path = self.folder / name
            if path.exists():
                shutil.move(str(path), str(archive / name))
        # Frame N accompanies action N+1. Preserve it only for a cached decision.
        first_discard = state["step_count"] + (state["phase"] == "decision_ready")
        for name in ("llm_fp", "llm_depth", "llm_minimap", "llm_minimap_target"):
            folder = self.folder / name
            if not folder.is_dir() or folder.is_symlink():
                continue
            for path in folder.iterdir():
                if path.is_file() and path.stem.isdigit() and int(path.stem) >= first_discard:
                    destination = archive / name / path.name
                    destination.parent.mkdir(exist_ok=True)
                    shutil.move(str(path), str(destination))
        return archive

    def recover_legacy(self):
        """Recover matching old API logs at the last action's PRE-action pose.

        The last logged action may or may not have executed. Treat it as pending
        and restore its pre-action pose, so it is neither lost nor double-applied.
        This is deliberately not a migration of unrecorded Kiro/API protocols.
        """
        content = self.files["actions"].read_bytes()
        rows = list(csv.DictReader(io.StringIO(content.decode())))
        if not rows or [int(row["step"]) for row in rows] != list(range(1, len(rows) + 1)):
            raise ValueError("Legacy recovery needs contiguous action rows beginning at step 1.")
        qa = self.files["qa"].read_text(encoding="utf-8")
        blocks = {int(m.group(1)): (m.start(), m.group(2)) for m in re.finditer(
            r"^=== Step (\d+) ===\n(.*?)(?=^=== Step \d+ ===\n|\Z)", qa, re.MULTILINE | re.DOTALL
        )}
        decisions = {}
        for row in rows:
            step = int(row["step"])
            if step not in blocks:
                raise ValueError("Legacy recovery needs the saved response for every logged action.")
            body = blocks[step][1]
            match = re.search(r"\[Q\]\n(.*?)\n\[A\]\nObservation: (.*?)\nAction: (.*?)\nReasoning: (.*?)\n={20,}", body, re.DOTALL)
            if not match or match.group(3) != row["action"]:
                raise ValueError("Legacy Q&A and action log disagree; cannot safely resume.")
            prompt, observation, action, reasoning = match.groups()
            world = re.search(r"World position X/Z: \(([-\d.]+), ([-\d.]+)\)", prompt)
            heading = re.search(r"Heading: ([-\d.]+)", prompt)
            distance = re.search(r"World distance to target: ([-\d.]+)", prompt)
            # World-coordinate prompts no longer print the minimap pixel line;
            # it is optional here because history rendering prefers the world
            # position and only falls back to pixels when that is missing.
            pixel = re.search(r"Minimap position/target: \(([-\d.]+), ([-\d.]+)\)", prompt)
            if not all((world, heading, distance)):
                raise ValueError("Legacy Q&A is missing request-time state; recovery is unsupported.")
            history_entry = {"world_position": list(map(float, world.groups())),
                             "theta": float(heading.group(1)),
                             "distance_to_target_m": float(distance.group(1))}
            if pixel:
                history_entry["position"] = list(map(float, pixel.groups()))
            decisions[step] = {"action": action, "observation": observation, "reasoning": reasoning,
                               "prompt": prompt.strip(), "error": False,
                               "history_entry": history_entry}
        settings = self.config["settings"]
        first, last = rows[0], rows[-1]
        initial = [float(first[k]) for k in ("init_world_x", "init_world_z")]
        target = [float(first[k]) for k in ("target_world_x", "target_world_z")]
        budget = int(settings["max_steps"])
        if settings["dynamic_step_budget"]:
            budget = max(settings["step_budget_min"], min(settings["step_budget_max"],
                         math.ceil(settings["step_budget_overhead"] + settings["steps_per_path_meter"] * math.dist(initial, target))))
        initial_budget = budget
        summary_path = self.folder / "results.csv"
        if summary_path.exists():
            with summary_path.open(newline="") as stream:
                summaries = list(csv.DictReader(stream))
            if summaries and summaries[-1].get("initial_step_budget") and summaries[-1].get("max_steps"):
                initial_budget = int(summaries[-1]["initial_step_budget"])
                budget = int(summaries[-1]["max_steps"])
        history_size = settings["history_size"]
        history = [{**decisions[s]["history_entry"], "step": s,
                    "action": decisions[s]["action"], "observation": decisions[s]["observation"]}
                   for s in range(1, len(rows))]
        # Preserve raw prefixes exactly, including CRLF and quoted CSV fields.
        stream = io.StringIO(content.decode(), newline="")
        reader = csv.DictReader(stream)
        boundary = stream.tell()
        for _ in rows[:-1]:
            next(reader)
            boundary = stream.tell()
        if len(rows) == 1:
            stream.seek(0)
            stream.readline()
            boundary = stream.tell()
        action_prefix = content.decode()[:boundary].encode()
        qa_prefix = qa[:blocks[len(rows)][0]].encode()
        state = {
            "version": CHECKPOINT_VERSION, "config_sha256": config_digest(self.config),
            "phase": "decision_ready", "step_count": len(rows) - 1,
            "step_budget": budget, "initial_step_budget": initial_budget,
            "step_budget_mode": "dynamic" if settings["dynamic_step_budget"] else "fixed",
            "pose": {key: float(last[column]) for key, column in zip(
                ("x", "y", "z", "rx", "yaw", "rz"),
                ("curr_world_x", "curr_world_y", "curr_world_z", "curr_direction_x", "curr_direction_y", "curr_direction_z"))},
            "init_world": initial, "target_world": target,
            "history": history[-history_size:] if history_size else [],
            "pending_decision": decisions[len(rows)], "minimap_projector": None,
            "detected_init_xy": None, "resume_count": 0,
            "recovery_source": "legacy_pre_action_logs", "stop_reason": None,
            "request_budget": self.budget_binding(),
            "files": {name: {"bytes": len(data), "sha256": digest(data)}
                      for name, data in (("actions", action_prefix), ("qa", qa_prefix))},
        }
        self.validate(state)
        atomic_json(self.path, state)
        return self.load()
