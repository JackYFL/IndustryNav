#!/usr/bin/env bash
# Behavior-cloning training wrapper. Replaces the per-base scripts
# (train_vanilla_cnn.sh / train_resnet50.sh / train_dinov2.sh) with a single
# entry that selects the preset bundle via --base.
#
# Usage:
#   bash shs/train_bc.sh <base> [extra args forwarded to nav.scripts.train_bc]
#   bash shs/train_bc.sh resnet50 --data_root collect_data --epochs 20
#
# <base> is one of: cnn | resnet50 | dinov2  (default: resnet50)
#
# Env:
#   PYTHON_BIN   python interpreter (default: <repo>/.venv/bin/python)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"

BASE="${1:-resnet50}"
shift || true

exec "$PYTHON_BIN" -m nav.scripts.train_bc --base "$BASE" "$@"
