"""Resume one recorded LLM run using its original configuration and API budget.

Usage: python -m nav.scripts.resume_benchmark outputs/.../seed0 [--dry-run]
Credentials must already be loaded in the environment. This command never
creates, resets or increases a request counter.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from nav.harness.checkpoint import CHECKPOINT_NAME
from nav.harness.navigation_protocol import navigation_run_config
from nav.scripts.run_benchmark_grid import free_tcp_port


def resume_command(folder, config, base_port=5507):
    allowed = navigation_run_config(argparse.Namespace(llm_provider="gemini"), "")["settings"]
    if set(config["settings"]) - set(allowed):
        raise ValueError("Run configuration contains unsupported settings; refusing to print/execute them.")
    if config["settings"]["llm_provider"] not in {"openai", "openrouter", "gemini"}:
        raise ValueError("This run needs its original CLI/Kiro adapter; it is not an API-provider run.")
    command = [sys.executable, "-m", "nav.scripts.run_benchmark_cell",
               "--baseline", "llm", "--resume", "--frame_save_dir", str(folder),
               "--worker_id", "0", "--base_port", str(base_port)]
    boolean_flags = {"dynamic_step_budget", "hide_unity_red_marker"}
    for name, value in config["settings"].items():
        if value is None:
            continue
        if name in boolean_flags:
            command.append(f"--{name}" if value else f"--no-{name}")
        elif name == "vision_input":
            command.extend(("--vision_input", "true" if value else "false"))
        else:
            command.extend((f"--{name}", str(value)))
    return command


def resume_environment(config, checkpoint, current=None):
    env = dict(os.environ if current is None else current)
    provider = config["settings"]["llm_provider"]
    prefix = provider.upper()
    if provider == "openai":
        options = config.get("openai_options", {})
        env["OPENAI_MAX_REQUEST_ATTEMPTS"] = str(options.get("max_request_attempts", 4))
        env["OPENAI_MIN_REQUEST_INTERVAL_SEC"] = str(options.get("shared_min_request_interval_sec", "0"))
    elif provider == "openrouter":
        mapping = {"reasoning_enabled": "REASONING_ENABLED", "json_mode": "JSON_MODE",
                   "min_request_interval_sec": "MIN_REQUEST_INTERVAL_SEC", "max_request_attempts": "MAX_REQUEST_ATTEMPTS"}
        for key, suffix in mapping.items():
            env[f"OPENROUTER_{suffix}"] = str(config.get("openrouter_options", {}).get(key, ""))
    binding = checkpoint.get("request_budget")
    if binding:
        counter = Path(binding["counter_file"])
        if not counter.is_file():
            raise ValueError("Original API request counter is missing; refusing to create fresh allowance.")
        env[f"{prefix}_REQUEST_COUNTER_FILE"] = str(counter)
        prior_limit = env.get(f"{prefix}_MAX_REQUESTS", "").strip()
        env[f"{prefix}_MAX_REQUESTS"] = str(min(int(prior_limit), binding["limit"]) if prior_limit else binding["limit"])
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Inspect the resume command; no Unity, model call or file mutation.")
    args = parser.parse_args()
    folder = args.run_dir.resolve()
    with (folder / "run_config.json").open() as stream:
        config = json.load(stream)
    checkpoint_path = folder / CHECKPOINT_NAME
    checkpoint = {}
    if checkpoint_path.exists():
        with checkpoint_path.open() as stream:
            checkpoint = json.load(stream)
    provider = config["settings"]["llm_provider"].upper()
    if not checkpoint and provider in {"OPENAI", "OPENROUTER"}:
        if not os.getenv(f"{provider}_MAX_REQUESTS") or not os.getenv(f"{provider}_REQUEST_COUNTER_FILE"):
            raise SystemExit("Legacy logs do not record the API budget. Export the original provider request limit and counter file before resuming.")
    env = resume_environment(config, checkpoint)
    command = resume_command(folder, config, base_port=5507 if args.dry_run else free_tcp_port())
    if checkpoint.get("modalities"):
        command += ["--modalities", ",".join(checkpoint["modalities"])]
    print(shlex.join(command))  # Only whitelisted run settings, never credentials.
    if args.dry_run:
        print("Dry run: original budget/counter retained; moving objects restart. A free port is selected only on execution.")
        return
    if not env.get(f"{config['settings']['llm_provider'].upper()}_API_KEY"):
        raise SystemExit("Provider API key is missing; load it securely before resuming.")
    raise SystemExit(subprocess.call(command, cwd=Path(__file__).resolve().parents[2], env=env))


if __name__ == "__main__":
    main()
