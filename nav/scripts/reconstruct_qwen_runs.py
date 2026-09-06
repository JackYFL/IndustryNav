"""Recover old Qwen episodes without altering outputs or spending API allowance.

Dynamic per-task budgets and the original shared request counter are retained.
Only cached replay is enabled; this does not authorize or start API continuation.
"""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import sqlite3

from nav.harness.checkpoint import CheckpointStore, atomic_json, config_digest, digest
from nav.scripts.reconstruct_gemini_runs import (
    NORMAL_STOPS, create_bundle, legacy_template, read_rows, recorded_settings, reconstruct,
)


PROTOCOL = "legacy-qwen-reconstructed-v1"
MODEL = "qwen/qwen3.8-flash"
DEFAULT_JOB = Path("analysis/queued_runs/qwen38_after_deepseek_20260904")


def read_counter(binding):
    path = Path(binding["counter_file"])
    if not path.is_file():
        raise ValueError("Original Qwen request counter is missing; no new allowance is created.")
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as db:
        row = db.execute("SELECT value FROM request_counter WHERE id=1").fetchone()
    if not row or type(row[0]) is not int or row[0] < binding["requests_reserved"]:
        raise ValueError("Original Qwen request counter was reset or corrupted.")
    return {**binding, "requests_reserved": row[0]}


def job_evidence(job):
    path = job.resolve() / "status.json"
    raw = path.read_bytes()
    status = json.loads(raw)
    expected = {"model": MODEL, "stage": "finished_with_errors", "request_limit": 7151,
                "seed": 0, "history_size": 5, "max_tokens": 500,
                "reasoning_enabled": False, "json_mode": True, "request_interval_sec": 5,
                "request_attempts": 3, "tasks_with_results": 96}
    if any(status.get(key) != value for key, value in expected.items()):
        raise ValueError("Qwen job metadata differs from the audited historical run.")
    counter = Path(status["counter_file"]).resolve()
    if counter != job.resolve() / "requests.sqlite3":
        raise ValueError("Qwen job points to a different request counter.")
    binding = read_counter({"counter_file": str(counter), "limit": status["request_limit"],
                            "requests_reserved": status["requests_reserved"]})
    return {"status_file": str(path), "status_sha256": digest(raw), "recorded_provider": "openrouter",
            "budget_binding": binding,
            "counter_scope": "entire Qwen job, including smoke tests and retries; not per episode",
            "transport": {key: status[key] for key in
                          ("max_tokens", "reasoning_enabled", "json_mode", "request_interval_sec", "request_attempts")},
            "invalid_tasks": [(v["scene"], v["point"]) for v in status["invalid_tasks"]]}


def qwen_settings(folder, rows, values):
    settings = recorded_settings(folder, rows, values, model_id=MODEL, dynamic=True)
    if (settings["max_tokens"] != 500 or settings["llm_min_request_interval_sec"] != 0
            or [settings[k] for k in ("step_budget_min", "step_budget_max", "steps_per_path_meter", "step_budget_overhead")]
            != [50, 160, 1.25, 20]):
        raise ValueError("Qwen launch differs from the recorded job protocol.")
    return settings


def reconstruct_qwen(folder, template, evidence):
    if (folder.parents[2].name, folder.parents[1].name) not in evidence["invalid_tasks"]:
        raise ValueError("Task is not a recorded interrupted Qwen task.")
    config, state = reconstruct(folder, template, settings_reader=qwen_settings, protocol=PROTOCOL,
                                require_pending_reasoning=True)
    config["reconstruction"]["unknown"] = ["request_timeout", "per_episode_request_count", "original_unity_binary_hash"]
    config["reconstruction"]["inferred"] = {
        "hide_unity_red_marker": "historical default; not a restored scene snapshot",
    }
    config["reconstruction"]["budget_source"] = "original result summary, cross-checked against launch formula and initial world distance"
    config["reconstruction"]["request_budget_evidence"] = evidence
    config["reconstruction"]["recorded_provider"] = "openrouter"
    config["recorded_openrouter_options"] = evidence["transport"]
    state["config_sha256"] = config_digest(config)
    state["request_budget"] = read_counter(evidence["budget_binding"])
    state["historical_shared_requests_reserved"] = state["request_budget"]["requests_reserved"]
    return config, state


class QwenReconstructionStore(CheckpointStore):
    """Read-only accounting for cached replay; never consult keys or reserve calls."""

    def budget_binding(self):
        if self.config.get("protocol") != PROTOCOL or self.config["execution_policy"]["new_model_calls_allowed"]:
            raise ValueError("This store is only for no-model Qwen reconstruction checks.")
        return read_counter(self.config["reconstruction"]["request_budget_evidence"]["budget_binding"])


def checkpoint_store(folder, config):
    return QwenReconstructionStore(folder, config, folder / "llm_actions.csv", folder / "agent_qa.txt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--job-dir", type=Path, default=DEFAULT_JOB)
    args = parser.parse_args()
    root, source_root = args.output_root.resolve(), args.source_root.resolve()
    if root.exists() or root == source_root or source_root in root.parents or root in source_root.parents:
        raise SystemExit("Choose a new output root outside the source tree.")
    evidence, template = job_evidence(args.job_dir), legacy_template()
    root.mkdir(parents=True)
    report = {"protocol": PROTOCOL, "model_calls": 0, "api_resume_ready": False,
              "original_request_budget": evidence["budget_binding"], "runs": []}
    for source in sorted(source_root.glob("scene*/point*/qwen3.8-flash/seed0")):
        item = {"source_run": str(source), "task": f"{source.parents[2].name}/{source.parents[1].name}"}
        try:
            summaries = read_rows(source / "results.csv")
            if summaries and summaries[-1].get("stop_reason") in NORMAL_STOPS:
                item["status"] = "skipped_complete"
            elif not read_rows(source / "llm_actions.csv"):
                item.update(status="no_actions", reason="No recorded navigation action; no checkpoint fabricated.")
            else:
                item.update(create_bundle(source, root / source.relative_to(source_root), template,
                            reconstructor=partial(reconstruct_qwen, evidence=evidence), store_factory=checkpoint_store))
        except (ValueError, OSError, KeyError, sqlite3.Error) as exc:
            item.update(status="blocked", error=str(exc))
        report["runs"].append(item)
        atomic_json(root / "manifest.json", report)
        if item["status"] != "skipped_complete":
            print(f"{item['task']}: {item['status']}", flush=True)
    report["request_counter_after"] = read_counter(evidence["budget_binding"])
    atomic_json(root / "manifest.json", report)
    print(f"Manifest: {root / 'manifest.json'}")


if __name__ == "__main__":
    main()
