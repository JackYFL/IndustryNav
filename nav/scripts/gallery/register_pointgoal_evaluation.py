"""Add an audited 96-point report without pretending it contains GIF recordings."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

from nav.scripts.gallery.export_llm_gallery import write_gallery_page
from nav.scripts.rl.audit_pointgoal_ppo_result import validate_report


def register_evaluation(summary: Path, gallery: Path, *, run_id: str, name: str,
                        recording_note: str, model: str = "") -> dict:
    if not re.fullmatch(r"[a-z0-9_-]+", run_id):
        raise ValueError("Use a safe lowercase evaluation ID")
    data = summary.read_bytes()
    report = json.loads(data)
    episodes = validate_report(report)
    entry = {
        "id": run_id, "name": name, "model": model,
        "successes": report["successes"], "episodes": len(episodes),
        "success_rate": report["success_rate"], "recording_note": recording_note,
        "report": f"reports/{run_id}.json",
        "summary_sha256": hashlib.sha256(data).hexdigest(),
        "source_summary": str(summary.resolve()),
    }
    registry_path = gallery / "evaluation_runs.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else []
    existing = [row for row in registry if row["id"] == run_id]
    if existing and (len(existing) != 1 or existing[0] != entry):
        raise ValueError("Evaluation ID already records a different run; use a new ID")
    report_path = gallery / entry["report"]
    if report_path.exists() and report_path.read_bytes() != data:
        raise ValueError("Existing source report differs; refusing to overwrite")
    manifest = json.loads((gallery / "manifest.json").read_text())
    if model:
        replay_rows = [row for row in manifest if row["model"] == model]
        keys = {(row["scene"], row["point"]) for row in replay_rows}
        if len(replay_rows) != len(episodes) or keys != set(episodes):
            raise ValueError("Replay model must have exactly the same 96 tasks in the gallery")
        for row in replay_rows:
            episode = episodes[row["scene"], row["point"]]
            if bool(row["success"]) != episode["success"] or row["steps"] != episode["steps_taken"]:
                raise ValueError("Replay gallery outcomes do not match this report")
    if not existing:
        backup = Path(tempfile.mkdtemp(prefix="before-evaluation-", dir=gallery))
        for filename in ("manifest.json", "index.html", "evaluation_runs.json"):
            if (gallery / filename).exists():
                shutil.copy2(gallery / filename, backup / filename)
        registry.append(entry)
        report_path.parent.mkdir(exist_ok=True)
        if not report_path.exists():
            with report_path.open("xb") as stream:
                stream.write(data)
        registry_path.write_text(json.dumps(registry, indent=2) + "\n")
    write_gallery_page(gallery, manifest)
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--gallery-dir", type=Path, required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--recording-note", required=True)
    parser.add_argument("--model", default="")
    args = parser.parse_args()
    print(json.dumps(register_evaluation(args.summary, args.gallery_dir, run_id=args.id,
                                        name=args.name, recording_note=args.recording_note,
                                        model=args.model), indent=2))


if __name__ == "__main__":
    main()
