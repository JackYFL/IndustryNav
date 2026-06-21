# Python environment options

This project supports two interchangeable Python environment managers. Pick **one** for your local setup. The entry points (`nav.scripts.run_benchmark_cell`, `nav.scripts.run_benchmark_grid`, `nav.scripts.eval_run`, `nav.scripts.compile_stats`, the `shs/run_headless_benchmark.sh` wrapper, etc.) work identically once the environment is active.

> **Coding agents (Claude Code etc.): prefer `uv` unless the user explicitly asks for conda.** Use `uv run <cmd>` or activate `.venv` before invoking Python scripts.

> **Why two options?** Some collaborators are still on conda (consistent with their existing ML-Agents setup). New contributors and the project maintainer use `uv` for a faster, isolated, system-Python-free workflow.

---

## Option A — uv (recommended for new setups)

`uv` manages its own CPython interpreter, so it never touches your system Python or any conda installation. The lockfile (`uv.lock`) plus `pyproject.toml` are the source of truth.

### One-time setup

`mlagents` / `mlagents-envs` are installed by uv directly from a local source
checkout (see [ml-agents](#ml-agents-required-cloned-from-source) below), so the
clone must exist **before** `uv sync`:

```bash
# Install uv (macOS/Linux). Skip if already installed.
curl -LsSf https://astral.sh/uv/install.sh | sh

# From the repo root: install a managed CPython 3.10
uv python install 3.10

# Clone the ml-agents source into external/ (REQUIRED — see below for why)
mkdir -p external
git clone https://github.com/Unity-Technologies/ml-agents.git external/ml-agents

# Create .venv and install EVERYTHING (base deps + mlagents from external/) in one shot
uv sync
```

### Activating for future runs

```bash
source .venv/bin/activate     # macOS/Linux
# .\.venv\Scripts\Activate.ps1   # Windows PowerShell
```

Or skip activation and prefix any command with `uv run`:

```bash
uv run python -m nav.scripts.eval_run --input-dir outputs/yifan2/point1/glm-4.6v
uv run bash shs/run_headless_benchmark.sh yifan1 anthropic/claude-sonnet-4.6
```

### ml-agents (REQUIRED — cloned from source)

`mlagents` and `mlagents-envs` are **mandatory**, even if you have no intention of training a model. They are the only way the Python scripts can interface with the compiled Unity clients (scenes, player controllers, sensors). The PyPI builds are stale and unmaintained, while the GitHub repo at <https://github.com/Unity-Technologies/ml-agents> keeps receiving updates every few weeks — so we always install **from a local source checkout** under `external/ml-agents`, never from PyPI.

`pyproject.toml` wires this up via `[tool.uv.sources]`:

```toml
[tool.uv.sources]
mlagents      = { path = "external/ml-agents/ml-agents" }
mlagents-envs = { path = "external/ml-agents/ml-agents-envs" }
```

So once `external/ml-agents` exists, plain `uv sync` builds and installs both packages from that checkout automatically — there is **no separate `pip install` step**. Stay on the default `develop` branch (most up-to-date). If `develop` is ever broken on your platform, fall back to the minimum supported tag and re-sync:

```bash
git -C external/ml-agents checkout release_23_tag
uv sync --reinstall-package mlagents --reinstall-package mlagents-envs
```

> **`requires-python` note.** `mlagents-envs` declares `python_requires=">=3.10.1,<=3.10.12"`. uv [ignores the *upper* bound](https://docs.astral.sh/uv/concepts/resolution/#requires-python) during resolution, so the uv-managed 3.10.18 interpreter is fine — but the project's `requires-python` lower bound must be **`>=3.10.1`** (not `>=3.10`) to match mlagents' floor, otherwise `uv lock`/`uv sync` fails as "unsatisfiable" (uv reports it via a `sys_platform == 'darwin'` split, but it is *not* platform-specific). This is already set in `pyproject.toml`; don't lower it back to `>=3.10`.

> **For coding agents:** a fresh `uv sync` (with `external/ml-agents` already cloned) installs everything. If the mlagents build fails, retry with `release_23_tag` per above before reporting back.

### Updating dependencies

- Edit `pyproject.toml`, then run `uv sync` to update `.venv` and `uv.lock`.
- To add a package: `uv add <pkg>`.
- To regenerate the lockfile from scratch: `uv lock --upgrade`.

### Notes / gotchas

- `pyproject.toml` pins `opencv-python==4.10.0.84` (not 4.12 as in the legacy `requirements.txt`) because 4.12 requires `numpy>=2`, which is incompatible with the `numpy==1.23.5` that mlagents needs. Keep this in mind if you compare against conda envs created from `requirements.txt`.
- `[tool.uv].python-preference = "only-managed"` — uv will refuse to fall back to a system or conda Python.

---

## Option B — Conda (legacy, still supported)

```bash
conda update -n base -c defaults conda
conda create -n mlagents -c conda-forge python=3.10.12 -y
conda activate mlagents
pip install -r requirements.txt

# ml-agents from source (same as the uv path). The conda env is python 3.10.12,
# which is within mlagents-envs's declared range, so no --ignore-requires-python
# is needed here (the uv path uses 3.10.17 and does need it).
mkdir -p external && cd external
git clone git@github.com:Unity-Technologies/ml-agents.git
cd ml-agents
python -m pip install ./ml-agents-envs
python -m pip install ./ml-agents
```

For future sessions, just `conda activate mlagents`.

---

## Picking between them

| | uv | conda |
|---|---|---|
| Interpreter source | uv-managed CPython | conda-forge |
| Lockfile | `uv.lock` (committed) | none (relies on `requirements.txt`) |
| Speed of install | seconds | minutes |
| Recommended for | new setups, this project's maintainer, CI | collaborators with existing mlagents conda envs |

---

## Secrets (OpenRouter API key)

The `llm` baseline calls models through OpenRouter and needs `OPENROUTER_API_KEY`
in the environment. The repo convention is a gitignored `tmp/secrets.sh` that
`export`s it:

```bash
source tmp/secrets.sh          # exports OPENROUTER_API_KEY (and any other keys)
```

`tmp/secrets.sh` is in `.gitignore` (never commit real keys). Copy
`tmp/secrets.sh.example` to `tmp/secrets.sh` and fill in your key. The classical
baselines (`astar`, `random`) run fully offline and need no key.
