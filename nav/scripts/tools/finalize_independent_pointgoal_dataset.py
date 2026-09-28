"""Freeze physically audited training data without filtering policy outcomes."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir",type=Path,required=True)
    parser.add_argument("--physics-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--reuse-validation-dir",type=Path,
                        help="Previously finalized dataset with byte-identical reserved validation.")
    parser.add_argument("--validation-physics-dir",type=Path,
                        help="Original validation evidence, verified against the prior physical audit hashes.")
    parser.add_argument("--exclude-key",action="append",default=[],help="scene/task with separately retained invalid-placement evidence")
    args=parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("Finalized datasets are immutable; choose a new directory")
    source=[json.loads(line) for split in ("train","val") for line in (args.source_dir/f"{split}_manifest.jsonl").read_text().splitlines() if line.strip()]
    checks=[json.loads(line) for path in args.physics_dir.glob("scene*.jsonl") for line in path.read_text().splitlines() if line.strip()]
    key=lambda row:row["scene_name"]+"/"+row["episode_id"]
    reused_files = []
    reused_keys = set()
    if bool(args.reuse_validation_dir) != bool(args.validation_physics_dir):
        raise SystemExit("Both validation reuse paths are required")
    if args.reuse_validation_dir:
        for name in ("val_manifest.jsonl", "val_quick_manifest.jsonl"):
            if (args.source_dir/name).read_bytes() != (args.reuse_validation_dir/name).read_bytes():
                raise SystemExit("Reused validation must be byte-identical to the prior finalized cohort")
        previous = json.loads((args.reuse_validation_dir/"physical_audit.json").read_text())
        hashes = {Path(p).name: h for p, h in previous["physics_sha256"].items()}
        reused_files = sorted(args.validation_physics_dir.glob("scene*.jsonl"))
        for path in reused_files:
            if hashlib.sha256(path.read_bytes()).hexdigest() != hashes.get(path.name):
                raise SystemExit(f"Reused validation evidence hash mismatch: {path.name}")
        reused_checks = [json.loads(line) for path in reused_files for line in path.read_text().splitlines()
                         if line.strip() and json.loads(line)["split"] == "val"]
        reused_keys = {key(check) for check in reused_checks}
        checks.extend(reused_checks)
    evidence={key(row):row for row in checks}
    if len(evidence)!=len(checks) or set(evidence)!={key(row) for row in source}:
        raise SystemExit("Physical checks must cover every source task exactly once")
    rejected=[]
    retained=[]
    exclusions=set(args.exclude_key)
    if not exclusions<=set(evidence):
        raise SystemExit("Explicit exclusion is not a known task")
    for row in source:
        check=evidence[key(row)]
        invalid=check.get("status")=="invalid_spawn" or key(row) in exclusions
        if not invalid:
            errors = (check["spawn_error_m"],check["target_error_m"])
            # The earliest physics-probe schema omitted status on success.
            # Accept that schema only for hash-verified, previously finalized
            # validation evidence; all numeric/finite checks still apply.
            checked_status = check.get("status") == "checked" or (
                "status" not in check and key(row) in reused_keys
            )
            if (not checked_status or not check["finite_depth"]
                    or not all(math.isfinite(e) and 0 <= e <= .5 for e in errors)):
                raise SystemExit(f"Unexpected unclassified physical failure: {key(row)}")
        if invalid:
            if row["split"]!="train":
                raise SystemExit(f"Validation placement invalid; must repair before selecting models: {key(row)}")
            rejected.append(dict(key=key(row),reason="invalid_physical_spawn",evidence=check))
        else:
            retained.append(row)
    args.output_dir.mkdir(parents=True)
    for split in ("train","val"):
        (args.output_dir/f"{split}_manifest.jsonl").write_text("".join(json.dumps(row,sort_keys=True)+"\n" for row in retained if row["split"]==split))
    # Preserve the exact validation cohort, including ordering.
    (args.output_dir/"val_quick_manifest.jsonl").write_bytes((args.source_dir/"val_quick_manifest.jsonl").read_bytes())
    payload=dict(source=str(args.source_dir),physics=str(args.physics_dir),
        physical_checked=len(checks),rejected=rejected,
        training_pairs=sum(row["split"]=="train" for row in retained),
        validation_pairs=sum(row["split"]=="val" for row in retained),
        scene_counts=dict(Counter(row["scene_name"] for row in retained if row["split"]=="train")),
        physics_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(args.physics_dir.glob("scene*.jsonl"))},
        reused_validation_source=str(args.reuse_validation_dir) if args.reuse_validation_dir else None,
        reused_validation_physics_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in reused_files},
        note="No filtering by navigation success, action label, collision, or map-reward outcome; only invalid spawn placement")
    (args.output_dir/"physical_audit.json").write_text(json.dumps(payload,indent=2)+"\n")
    (args.output_dir/"sampling_audit.json").write_bytes((args.source_dir/"sampling_audit.json").read_bytes())
    print(json.dumps({k:v for k,v in payload.items() if k!="physics_sha256"}),flush=True)


if __name__=="__main__":
    main()
