"""Verify the portable assets for the best DAgger -> PPO training chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs/pointgoal_best_20260913/pipeline.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(1 for _ in stream)


def episode_count(path: Path) -> int:
    if not path.is_dir():
        return -1
    return sum(
        1
        for scene in path.iterdir()
        if scene.is_dir()
        for episode in scene.iterdir()
        if episode.is_dir()
    )


def default_unity_path() -> Path:
    if platform.system() == "Darwin":
        return REPO_ROOT / "unity_clients/scene_all.app"
    return REPO_ROOT / "clients/scene_all_24scenes_exact_pillar_markers_v14/scene_all.x86_64"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--unity", type=Path, default=default_unity_path())
    parser.add_argument(
        "--require-reference-outputs",
        action="store_true",
        help="Require the downloaded 52.08%% DAgger and Mixed400 PPO checkpoints.",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    failures: list[str] = []
    warnings: list[str] = []

    def local(relative: str) -> Path:
        return REPO_ROOT / relative

    def check_hash(relative: str, expected: str, *, required: bool = True) -> None:
        path = local(relative)
        if not path.is_file():
            message = f"missing file: {relative}"
            (failures if required else warnings).append(message)
            return
        actual = sha256(path)
        if actual != expected:
            failures.append(f"SHA256 mismatch: {relative}: {actual} != {expected}")

    dagger = config["dagger"]
    for snapshot in config.get("source_snapshots", []):
        snapshot_root = local(snapshot["path"])
        actual_files = (
            sum(1 for path in snapshot_root.rglob("*") if path.is_file())
            if snapshot_root.is_dir()
            else -1
        )
        if actual_files != int(snapshot["files"]):
            failures.append(
                "source snapshot file count mismatch: "
                f"{snapshot['path']}: {actual_files} != {snapshot['files']}"
            )

    check_hash(dagger["initial_checkpoint"], dagger["initial_checkpoint_sha256"])
    check_hash(dagger["aggregate_manifest"], dagger["aggregate_manifest_sha256"])
    check_hash(dagger["aggregate_summary"], dagger["aggregate_summary_sha256"])
    check_hash(
        dagger["reference_output"],
        dagger["reference_output_sha256"],
        required=args.require_reference_outputs,
    )
    for root_key, count_key in (
        ("base_data_root", "base_episodes"),
        ("recovery_data_root", "recovery_episodes"),
    ):
        root = local(dagger[root_key])
        actual = episode_count(root)
        expected = int(dagger[count_key])
        if actual != expected:
            failures.append(f"episode count mismatch: {dagger[root_key]}: {actual} != {expected}")

    aggregate = local(dagger["data_root"])
    if not aggregate.is_dir():
        failures.append(f"missing DAgger aggregate: {dagger['data_root']}")
    else:
        links = [path for path in aggregate.rglob("*") if path.is_symlink()]
        if len(links) != int(dagger["aggregate_symlinks"]):
            failures.append(
                f"aggregate symlink count mismatch: {len(links)} != {dagger['aggregate_symlinks']}"
            )
        broken = [path for path in links if not path.exists()]
        if broken:
            failures.append(f"broken DAgger aggregate symlinks: {len(broken)}")

    conversion = config["conversion"]
    for prefix in ("reference_actor", "train_manifest", "validation_manifest"):
        check_hash(conversion[prefix], conversion[f"{prefix}_sha256"])

    stage1 = config["ppo_stage1"]
    check_hash(
        stage1["reference_output"],
        stage1["reference_output_sha256"],
    )

    stage2 = config["ppo_stage2"]
    check_hash(stage2["manifest"], stage2["manifest_sha256"])
    check_hash(stage2["resume_anchor"], stage2["resume_anchor_sha256"])
    manifest = local(stage2["manifest"])
    if manifest.is_file() and line_count(manifest) != int(stage2["manifest_rows"]):
        failures.append(
            f"manifest row mismatch: {line_count(manifest)} != {stage2['manifest_rows']}"
        )
    check_hash(
        stage2["reference_output"],
        stage2["reference_output_sha256"],
        required=args.require_reference_outputs,
    )
    map_root = local(stage2["navigation_map_dir"])
    valid_maps = [
        scene
        for scene in map_root.glob("scene*")
        if all((scene / name).is_file() for name in ("metadata.json", "geometry.npz", "walkable.png"))
    ]
    if len(valid_maps) != 24:
        failures.append(f"complete navigation maps: {len(valid_maps)} != 24")

    unity = args.unity if args.unity.is_absolute() else REPO_ROOT / args.unity
    unity_executable = (
        unity / "Contents/MacOS/IndustryNav" if unity.suffix == ".app" else unity
    )
    if not unity_executable.is_file():
        failures.append(f"missing Unity executable: {unity}")

    report = {
        "ok": not failures,
        "config": str(args.config),
        "unity": str(unity),
        "failures": failures,
        "warnings": warnings,
    }
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
