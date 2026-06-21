# Refactor execution plan

Companion to [`cleanup.md`](cleanup.md). That file states the *goals* and *file-level decisions*. This file states the *execution sequence* — PR by PR, with explicit "what changes inside each move" notes, not just folder mappings.

The rule from `cleanup.md` applies throughout: **no 1:1 file pasting**. Each move is an opportunity to delete dead code, collapse duplication, fix naming, and tighten interfaces. If a move is purely cosmetic, it is wrong.

---

## ⏯️ PROGRESS TRACKER — RESUME HERE (PRs 0–11 ✅ COMPLETE)

> This section is the authoritative status. The PR-by-PR plan below it is the original design; where they disagree, **this tracker wins** (it records what actually happened, including plan deviations).

> **Refactor complete (PRs 0–11).** All scattered root-level scripts now live under the `nav/` package (`config`, `utils`, `harness/`, `eval/`, `stats/`, `baselines/`, `models/`, `train/`, `scripts/`); one cross-OS shell; superseded scripts in `deprecated/`; 173 unit tests. **Fate of these planning docs:** `docs/cleanup.md` + this `docs/refactor_plan.md` are kept in place as the historical record **until PR #17 merges**, then move them to `docs/history/` (or `deprecated/`). `docs/cleanup_clarification.md` was deleted in PR 11 (always temporary).

### Branch / PR / remote state

- Work branch: **`nav2-refactor`** (all commits live here; local == `origin/nav2-refactor`).
- **`main` already contains PRs 0–3** — a collaborator merged PR #15 (the "PRs 0-3" PR) into main as merge commit `5267495`. A revert (PR #16) was opened then **closed unmerged** (merging it would trigger the revert-a-merge trap). Net: PRs 0–3 are on main; the refactor now lands **incrementally**, not as one big final merge.
- Open PR **[#17](https://github.com/JackYFL/IndustryNav/pull/17)** `nav2-refactor → main` carries **PR 4 + 5a + 5b + 6** (its diff is just the incremental work since PR 0–3 are already in main).
- Commit map: PR0 `aa771cb`+`c8c76ea`; stash-play_mlagents `34e6b72`; PR1 `43be3e4`; PR2 `d34309c`; PR3 `85eac3f`; PR4 `c22f2fb`+docs`ce23823`; PR5a `96b37bd`; (merge `b8b1848`); PR5b `a93c370`; PR6 `9ea06b0`.
- Issue **[#14](https://github.com/JackYFL/IndustryNav/issues/14)**: Unity project relocation — separate ticket, not in this branch.

### Status by PR

| PR | Scope | Status |
|----|-------|--------|
| 0  | pre-flight hygiene (deprecated/, tests/, tmp/secrets.sh.example, .gitignore, delete artifacts) | ✅ done |
| 1  | `nav/config.py` + `nav/utils.py` foundation | ✅ done |
| 2  | prompts → `nav/prompts/`, llm_providers → `nav/harness/llm_provider.py` | ✅ done |
| 3  | eval pipeline → `nav/eval/` + `nav/scripts/{eval_run,aggregate_eval}.py` | ✅ done |
| 4  | stats → `nav/stats/` + `nav/scripts/compile_stats.py` (CLI subcommands) | ✅ done |
| 5a | `agents/` + `red_detector.py` → `nav/harness/` (+ perception/) | ✅ done |
| 5b | extract side_channels/coordinates/observations/routing/prompt helpers from `run_headless_benchmark.py` → `nav/harness/` | ✅ done |
| 6  | baselines → `nav/baselines/{astar,navid}.py` | ✅ done |
| 7  | BC models → `nav/models/`, train loop → `nav/train/`, `nav/scripts/train_bc.py`, consolidate `shs/train_*.sh` → `train_bc.sh --base` | ✅ done |
| 8  | entry-point + shell consolidation (8a–8e, see below) | ✅ done |
| 9  | `third_party/` → `external/` rename + ml-agents reinstall + skill/doc updates | ✅ done |
| 10 | tests backfill (grid runner + config invariants) | ✅ done |
| 11 | docs sweep (README*, run_benchmark.md, AGENTS.md, python_env_options.md; delete cleanup_clarification.md) | ✅ done |

### Plan deviations / discoveries (IMPORTANT — these override the body below)

1. **`agents/` planner trio was orphaned** — only `deprecated/play_mlagents_multiagent.py` used it (live scripts call `llm_openrouter` directly). Per user decision: **kept**, moved to `nav/harness/` (global/local/decision planners + types + `prompt_assembly.py`) for future multi-agent revival. `decision_maker.decide()` is pure-heuristic; its dead LLM scaffolding was removed.
2. **stats `_partial` is NOT a dup of `_full`** — it's a genuinely different method (unpaired/episode/no-Spearman) for the label-less archive. Result: `nav/stats/full.py` + `partial.py` coexist, sharing deduped primitives (`bootstrap.py`, `permutation.py`, `spearman.py`, `load.py`, `merge.py`). PR 4 verified **byte-identical** outputs vs the old scripts at seed 0.
3. **PR 5 was split** into 5a (move/rename/fix) + 5b (extract from the live 867-line entry). 5b was **conservative**: it pulled out cohesive components but **left the async while-loop in `run_headless_benchmark.py`** as glue. Turning that loop into a reusable state-machine class is **deferred to PR 8** (natural when the entry moves; risky to do without an easy Unity iterate cycle).
4. **A* is now wired into the unified entry (pulled forward from PR 8); NaVid is paused.** `run_headless_benchmark.py` `exec_mode` now ∈ {random, agent, bc_agent, **astar**}. The A* half of the planned `--baseline` work was pulled forward: `nav/harness/routing.py:execute_decision` gained an `astar` branch that drives `nav.baselines.astar.AStarBaseline`, the entry constructs the planner + builds its per-step payload (minimap/curr_xy/theta/reach_px/step/world coords), and `--astar_debug_viz`/`--astar_debug_dir` are exposed (tuning params still default from `ASTAR_DEFAULTS`). **Verified end-to-end against `scene_all.app`** (scene 1, no API key): plans/replans, follows waypoints, actuates Unity, writes `results.csv` + `astar_actions.csv`. Added `tests/test_harness_runtime.py::test_execute_decision_astar_uses_planner` (suite now 150).
   - **NaVid stays paused** — still only in `run_sync.py --exec_mode navid` (needs torch + the external NaVid repo). **Future TODO after the refactor stabilizes:** absorb NaVid into the unified entry the same way A* was (an `navid` branch in `execute_decision` + planner construction in the entry), then stash `run_sync.py`. Not blocking any current PR.
   - PR 8 still owns the rest: the generalized `--baseline {llm,astar,navid,bc,human}` flag naming, folding the baseline shells, and superseding/stashing `run_sync.py`. A* being live in the entry does **not** close PR 8.
5. **Macro rule (now in global + project memory):** EVERY module-level constant in `nav/` goes in `nav/config.py`, never in the consumer module — incl. test fixtures. Single domain prefix only (`EVAL_*`, `LLM_*`, `STATS_*`, `ASTAR_*`…), no double prefix.
6. **PR 7: `nav/models/` splits by policy head, NOT by backbone.** The plan's `{cnn,resnet,dino}.py` assumed three model classes per backbone — wrong. The code has 4 *policy heads* (`NavPolicy`/mlp, `NavPolicyRNN`/lstm, `NavPolicyTransformer`, `NavPolicyDiffusion`) and the cnn/resnet/dino distinction is only a `timm` backbone string. So (user-approved): `nav/models/encoder.py` (`TimmEncoder` + `build_encoder_pair`, the deduped construction) + `nav/models/policy.py` (the 4 heads, flat attribute layout preserved so **existing checkpoints still `strict=True`-load** — verified state_dict keys match the old classes) + `__init__` exporting `build_policy()`. The 3 empty `{cnn,resnet,dino}.py` stubs were deleted; `llm.py` stub kept. `--base {cnn,resnet50,dinov2}` is a **preset bundle** → `nav.config.BC_BASE_PRESETS` (frozen `BCTrainConfig` instances reproducing the 3 old `train_*.sh`; CLI flags override via `dataclasses.replace`). Train loop → `nav/train/{dataset,controller,loop}.py` (the two datasets share a `_NavEpisodeBase`); fixed arch hyperparams → `nav.config.NavPolicyArch`/`NAV_POLICY_ARCH`. Fixed a latent bug: `NavPolicy`'s `resnet32` default backbone (not a real timm model; effective default was always `resnet50` via the train config) → `NAV_DEFAULT_BACKBONE="resnet50"`. `navigation_bc/` + `shs/train_{vanilla_cnn,resnet50,dinov2}.sh` deleted; `shs/train_bc.sh --base` is the single wrapper. `run_headless_benchmark.py` + `run_async.py` BC import now `from nav.train.controller import BCNavController`.

### Verification baseline (how to prove "still runnable")

- Unit tests: `.venv/bin/python -m pytest tests/ -q` → **173 passing** (149 at PR 6; +1 A* routing; +5 PR 7 BC; +1 PR 8a NaVid; +10 PR 10 grid runner; +7 PR 10 config invariants). (Use `.venv/bin/python` directly; `uv run` re-resolves deps and fails on the mlagents pin.)
- Unity client exists locally: **`/Users/lichili/dev/IndustryNav2/scene_all.app`** (`nav.config.SCENE_ALL_BUILDS`, `--file_name auto`).
- **Dry-boot** (no API key): `.venv/bin/python run_headless_benchmark.py --exec_mode random --file_name /Users/lichili/dev/IndustryNav2/scene_all.app --scene_id 1 --max_steps 2 --frame_save_dir outputs/_dryboot/x --init_curr_x 31 --init_curr_y 50 --init_curr_direction 180 --target_x 550 --target_y 450` → expect Spawn/Target world-coord handshake lines.
- **Real LLM run** (cheap): same but `--exec_mode agent --model_id google/gemini-3-flash-preview --base_port <fresh>` after `source tmp/secrets.sh`. Both verified working post-5b.

### PR 8 sub-sequencing (user chose sub-PRs; 8a first)

- **8a — `--baseline` routing ✅ done.** Renamed the entry's `--exec_mode {random,agent,bc_agent,astar}` → `--baseline {random,llm,bc,astar,navid}` (config.`BENCHMARK_BASELINES`); `agent`→`llm`, `bc_agent`→`bc`. Wired **NaVid** into `routing.py` + the entry (mirrors the A* wiring; `--navid_model_path/_repo/_instruction/...` flags). Per-run output dirs are now prefixed by the baseline token (`llm_fp/`, `bc_actions.csv`, …); the eval discovery prefix lists moved to config (`EVAL_DEPTH_DIR_CANDIDATES`/`EVAL_ACTIONS_CSV_CANDIDATES`) and were extended with `llm_*`/`bc_*` while keeping legacy `agent_*`/`bc_agent_*`/`manual_*` so historical trees still resolve. Updated callers: `run_benchmark_grid.py` (`--baseline llm`), `shs/run_headless_benchmark_{macos,linux}.sh`, and the benchmark skill docs + `docs/run_benchmark.md` command examples. results.csv keeps its `exec_mode` column name (value is now the baseline token). **`human` is dropped** from the headless entry — it was legacy interactive `manual` teleop against the now-deprecated per-scene client ([[unified-client-only]]); teleop lives in the collect_data flow. Verified: 156 tests green + A* end-to-end via `--baseline astar` + NaVid friendly error path.
- **8b — entry move + env-setup extraction ✅ done.** Moved `run_headless_benchmark.py` → `nav/scripts/run_benchmark_cell.py` (invoke `python -m nav.scripts.run_benchmark_cell`; callers + wrappers + skill docs updated; wrappers `export PYTHONPATH=$REPO_ROOT`). **Deviation from the original "loop→state-machine class" plan:** the async loop is a 2-state poll (decision-in-flight vs idle) with a *single* caller — wrapping it in a single-use class adds indirection without payoff (no second consumer), so the loop stays in the cell script. The real dedup win — env construction + scene-select + minimap-margin + spawn/target pixel→world handshake (the part collect_data *also* does) — was extracted to **`nav/harness/env_setup.py`** (`setup_and_prime(args, logger) -> PrimedEnv`, raising `EnvSetupError` instead of `sys.exit`). Cell script 670→578 lines (the loop is the irreducible remainder; the <300 target assumed the loop got classed out, which it didn't, by design). Verified: 156 tests + A* end-to-end with byte-identical spawn/target handshake + trajectory vs. pre-extraction.
- **8c — grid runner ✅ done.** Moved `run_benchmark_grid.py` → `nav/scripts/run_benchmark_grid.py` (`python -m`; `REPO_ROOT=parents[2]`; dispatches `python -m nav.scripts.run_benchmark_cell`). Dropped the duplicated `SCENE_ID_MAP` + `GRID_CSV_FIELDS` (now imported from config). Added **`history_size`** as the 6th sweep axis: `--history_sizes` (default `[LLM_DEFAULT_HISTORY_SIZE=5]`), `Cell.history_size`, threaded into the cmd + CSV rows + label. **Default-history cells stay in the canonical `outputs/` tree; non-default sizes route under `outputs/_history_size/hs<k>/`** so `nav.stats.load.discover_grid_runs` (which skips `_`-prefixed roots) isn't polluted. Aggregates moved from `outputs/grid_runs.csv` → `analysis/grid_runs/<timestamp>/{runs,failures}.csv`. Absorbed + deleted `shs/run_ablate_history_size.sh` (legacy per-scene-client ablation). Updated AGENTS.md + run_benchmark.md refs. Verified: 156 tests + `--dry_run` history sweep (hs0/hs5/hs10 → correct paths/labels).
- **8d — collect_data ✅ done.** Moved `run_headless_collect_data_assign_points_v2.py` → `nav/scripts/collect_data.py` (1124→958 lines). Removed its duplicated copies: local `BoundsSideChannel`/`TargetSideChannel` → `nav.harness.side_channels`; obs monkeypatch → `nav.harness.observations.patch_observation_decoding` (called at `main()` start); `MODALITY_TO_IDX`+`get_obs_safe` → harness/config; `find_exact_map_bounds`+`visual_to_unity_coords` → `nav.harness.coordinates`; local `BEHAVIOR_NAME` literal → `config`. Deleted dead `pixel2world`/`world2pixel`. **Deviation:** did NOT adopt `env_setup.setup_and_prime` — the teleop priming is *interactive* (cv2 point-selection, conditional re-priming) and materially differs from the benchmark's CLI-driven path. Kept the local `get_minimap_rgb_for_init` (intentionally lenient: no edge-gate, depth/ego fallback — differs from the harness warmup poll). Verified: import + `--help` + 156 tests; a full teleop run needs a human at a keyboard + display.
- **8e — shells + stash ✅ done.** Merged `run_headless_benchmark_{macos,linux}.sh` → one `shs/run_headless_benchmark.sh` (OS-detect via `uname`; Linux adds `xvfb-run`; `BASELINE={llm,astar,navid,bc,random}` env selects the decision baseline; output subdir = model short-name for llm else the baseline token). Stashed to `deprecated/`: the two per-OS wrappers, the legacy baseline shells (`run_agent.sh`, `run_bc_agent.sh`, `run_human_eval.sh`, `run_astar_all_scenes.sh` — all routed through `run_sync`/`run_async`/per-scene clients), `run_agent.ps1`, and the superseded entry scripts `run_sync.py` / `run_async.py` / `run_headless.py`. **Deviation:** the `.ps1` was *stashed*, not refreshed — there's no validated Windows `scene_all` build (`config.SCENE_ALL_BUILDS["Windows"]` is empty) and the unified shell can't be exercised there, so shipping a speculative refresh would be untested; add a Windows wrapper when a build exists. Updated skill + AGENTS.md + run_benchmark.md shell refs. Verified: `bash -n` + a full `BASELINE=random` macOS run over all 4 `yifan1` points (handshake + per-point results) + 156 tests.

### Expanded PR 8 scope (consolidated from later discoveries)

- `run_headless_benchmark.py` → `nav/scripts/run_benchmark_cell.py`; finish the loop→state-machine extraction (the 5b deferral).
- `run_benchmark_grid.py` → `nav/scripts/run_benchmark_grid.py`; add `history_size` as a 6th sweep axis (absorb `shs/run_ablate_history_size.sh`); write grid aggregates to `analysis/grid_runs/` not `outputs/`.
- `run_headless_collect_data_assign_points_v2.py` → `nav/scripts/collect_data.py`.
- **Add `--baseline {llm,astar,navid,bc,human}` routing** in `nav/harness/routing.py` + entry; this absorbs `run_sync.py`'s astar/navid, `shs/run_bc_agent.sh`, `shs/run_human_eval.sh`, `shs/run_astar_all_scenes.sh`, `shs/run_agent.sh`. **A* is already live** in the entry as `--exec_mode astar` (deviation #4) — PR 8 generalizes the flag naming and folds NaVid in (NaVid wiring is the paused future TODO from deviation #4).
- Merge `shs/run_headless_benchmark_{macos,linux}.sh` → `shs/run_headless_benchmark.sh` (OS branch only differs by `xvfb-run`). Refresh `shs/run_agent.ps1` for Windows parity.
- **Stash to `deprecated/`:** `run_sync.py`, `run_async.py`, `run_headless.py` (now superseded).
- Then **re-run the A* end-to-end test** the user requested (via `--baseline astar` against `scene_all.app`).

### Untracked, intentionally (not a bug)

- `nav/models/llm.py` — empty stub, reserved for a future LLM-policy head (PR 7 filled `encoder.py`/`policy.py`; the `cnn`/`dino`/`resnet` stubs were deleted, see deviation #6).

---

## 0. Pre-existing scaffolding to account for

The user has already created (these are not PRs in the sequence below; they're the starting state):

- `nav/` with empty/near-empty subfolders: `baselines/`, `eval/`, `harness/`, `models/`, `prompts/`, `scripts/`, `stats/`, `train/`.
- `nav/models/{llm,cnn,resnet,dino}.py` and `nav/baselines/{astar,navid}.py` exist as stubs — need to verify whether they're empty placeholders or already partially implemented before any PR touches them. **Action during PR 0: read these and either confirm empty-stub status or fold their content into the corresponding PR.**
- `unity_project/` exists as an empty directory at repo root. Issue [#14](https://github.com/JackYFL/IndustryNav/issues/14) tracks the actual Unity-project relocation; not in this plan's sequence.
- `tmp/secrets.sh` exists (101 bytes). Already in line with the post-refactor convention.
- `.gitignore` already ignores `third_party/` and `__pycache__/`. The `third_party/` line needs to flip to `external/` in PR 9.

---

## Target `nav/` layout (post-refactor)

```
nav/
├── __init__.py
├── config.py                   # ALL hyperparameters / env vars / macros
├── utils.py                    # generic stateless utilities (file I/O, logging, json/csv, encoding)
├── prompts/                    # *.txt prompt templates (loaded by harness)
├── harness/                    # runtime agent harness (replaces agents/ + red_detector.py)
│   ├── __init__.py
│   ├── decision_loop.py        # async state machine that drives one cell of a benchmark
│   ├── state_machine.py        # the explicit-state graph (was tangled inside run_async.py)
│   ├── routing.py              # dispatches inputs to LLM / baseline / BC controller
│   ├── local_planner.py        # BEV + egocentric local planner (from agents/local_planner.py)
│   ├── global_planner.py       # from agents/global_planner.py
│   ├── decision_maker.py       # from agents/decision_maker.py (after dedup vs decision_loop)
│   ├── types.py                # shared dataclasses (from agents/types.py, extended)
│   ├── llm_provider.py         # API connector (from llm_providers.py)
│   └── perception/
│       ├── __init__.py
│       └── red_detector.py     # from red_detector.py
├── eval/                       # post-experiment evaluation utilities
│   ├── __init__.py
│   ├── warning.py              # from warning_detect_mp.py
│   ├── collision.py            # from collision_analyzer.py
│   ├── metrics.py              # from eval_metrics.py
│   └── aggregate.py            # library half of batch_eval.py
├── stats/                      # statistical analysis (rebuttal pipeline, consolidated)
│   ├── __init__.py
│   ├── load.py                 # xlsx → per-run normalization (was xlsx_to_per_run.py)
│   ├── merge.py                # supersession-merge per-run (was merge_per_run.py)
│   ├── bootstrap.py            # scene-clustered bootstrap CIs (from stats_analysis_full.py)
│   ├── permutation.py          # paired permutation tests
│   ├── spearman.py             # rank correlation
│   └── report.py               # report.md generator
├── baselines/
│   ├── __init__.py
│   ├── astar.py                # algorithmic body of astar_baseline.py
│   └── navid.py                # algorithmic body of navid_baseline.py
├── models/                     # BC model constructors
│   ├── __init__.py
│   ├── cnn.py
│   ├── resnet.py
│   └── dino.py
├── train/
│   ├── __init__.py
│   ├── dataset.py              # consolidated dataset (was navigation_bc/dataset.py + dataset_seq.py)
│   ├── controller.py           # from navigation_bc/controller.py (inference-time)
│   └── loop.py                 # unified train loop (was navigation_bc/train.py)
└── scripts/                    # thin CLI entry points
    ├── __init__.py
    ├── run_benchmark_grid.py   # outer orchestrator (subprocess-isolated cells)
    ├── run_benchmark_cell.py   # single-cell runner (was run_headless_benchmark.py)
    ├── collect_data.py         # human teleop (was run_headless_collect_data_assign_points_v2.py)
    ├── train_bc.py             # --base {dinov2,cnn,resnet50}
    ├── aggregate_eval.py       # batch_eval.py main()
    └── compile_stats.py        # stats pipeline driver

shs/
├── run_headless_benchmark.sh   # unified macOS/Linux entry → run_benchmark_grid.py
├── run_collect_data.sh         # → collect_data.py
├── train_bc.sh                 # → train_bc.py with --base flag
├── run_ablate_history_size.sh  # thin wrapper: run_headless_benchmark.sh --history-sizes ...
└── run_agent.ps1               # Windows equivalent of run_headless_benchmark.sh (parity, not perfect)

tests/                          # NEW — pytest, including IO-fixture-based tests
external/                       # was third_party/, ml-agents source install
deprecated/                     # stash, hard-deleted after refactor stabilizes
analysis/                       # aggregate stats (incl. grid_runs/ going forward)
outputs/                        # per-run telemetry only, untracked
tmp/                            # secrets.sh + .example, gitignored except for .example
unity_project/                  # mount point per issue #14
```

---

## PR sequence

The order keeps `main` runnable after every merge. Each PR ends with a fast smoke test: `shs/run_headless_benchmark.sh` for one cell on one scene must succeed.

### PR 0 — Pre-flight hygiene (small, low-risk warmup)

**Scope.** Things that should land before any code moves so the refactor PRs don't compound surprises.

- Read the existing `nav/{models,baselines}/*.py` stubs and confirm empty/placeholder status. If non-empty, fold their content into the appropriate later PR rather than letting it diverge.
- Verify what `unity_scripts/UnityWarehouseSceneHDRP/{PlayerController,PlayerControls}.cs` is vs. `c_sharp_scripts/{MinimapWorldMapper,TargetSideChannel,WarehouseAgent}.cs`. **Open question for Lichi**: is `unity_scripts/` the newer, canonical reference and `c_sharp_scripts/` the deprecated one? Assumption pending confirmation. If yes, `c_sharp_scripts/` → `deprecated/`.
- Add `__pycache__/` (already there), `tmp/*` with `!tmp/secrets.sh.example` un-ignore, `analysis/**/.DS_Store`, `outputs/`, `unity_log.txt` to `.gitignore`. Confirm `tmp/secrets.sh` is currently gitignored (it should be — its content is sensitive).
- `git rm --cached` for any `.pyc` or accidentally-tracked artifact files. Hard-delete `unity_log.txt`, `input_points-my.json`, `experiment_results_v6.csv`, `eval_results.xlsx`, `history_size_metrics.xlsx`, `__pycache__/`.
- Create `deprecated/` and `tests/` with a single `README.md` and `__init__.py` respectively.
- Commit `tmp/secrets.sh.example` with placeholder values; verify the real `tmp/secrets.sh` isn't accidentally in any commit.
- Stash `run_headless_collect_data_assign_points.py` (v1), `run_headless_depth_assign_points.py`, `run.sh`, `prompts/nav_ego_minimap_his_bev.txt` into `deprecated/` (these are clear-cut deletes per `cleanup.md`).

**No code moves yet.** Pure hygiene.

**Verification:** `shs/run_headless_benchmark_macos.sh` still works end-to-end on one cell.

---

### PR 1 — Foundation: `nav/__init__.py`, `nav/config.py`, `nav/utils.py`

**Scope.** Land the package and the two leaf modules that everything else imports. Doing these first means the rest of the refactor only ever has to update import paths to point at things that already exist.

**Moves.**
- `config.py` → `nav/config.py`
- `utils.py` → `nav/utils.py`

**What changes inside (not paste).**

- `config.py`:
  - Audit every constant. Anything that is currently defined inline in `run_headless_benchmark.py`, `run_async.py`, `run_sync.py`, `run_benchmark_grid.py`, or the shell scripts should be hoisted here. Today `config.py` is only ~3.5KB; it should be larger.
  - Group constants by domain (action space, scene IDs, env-param keys, model identifiers, retry policy, frame budgets). Use dataclasses where a constant is actually a structured config.
  - Add a `ModelConfig` and `BCModelConfig` dataclass so PR 7's `--base {dinov2,cnn,resnet50}` flag dispatches to typed config, not a dict.
  - Resolve the three `ACTION_SPACE` aliases (`ACTION_SPACE`, `ACTION_SPACE_AGENTS`, `ACTION_SPACE_ANNOTATION`) — these import-as-aliases scattered across files suggest there's one canonical action space being renamed at each call site. Pick the canonical name, delete the aliases, update consumers.
- `utils.py`:
  - It's 18KB — likely a grab-bag. Inventory functions; split into thematic sections (file I/O, JSON/CSV, encoding helpers like `data_url_png_from_rgb`, parsing like `parse_json_action`, logging). Move side-channel-coupled helpers *out* into `nav/harness/types.py` or similar — `utils.py` is for generic stateless code per the goals.
  - Drop unused helpers. Run `grep -rn "from utils import"` across the consumer set (10 files) to know what's actually called.
  - Absorb the body of `turn2gif.py` here as `make_gif_from_frames(...)`, since `cleanup.md` decided it folds into utils. Delete `turn2gif.py` after that.

**Import updates.** Every root-level `.py` that imports `utils` / `config` flips to `from nav.utils import ...` / `from nav.config import ...`. ~10 files (per the grep above).

**Tests added (PR 1).** `tests/test_utils.py` — pytest for pure functions: JSON/CSV roundtrip, action parsing, encoding helpers. Fixture-based test for `make_gif_from_frames` using a tempdir with a tiny synthetic frame sequence. Per item 16, both pure-function and IO-fixture tests.

**Risk.** Largest blast radius of any PR — every script imports `utils`/`config`. Mitigation: do this PR before all moves so subsequent PRs touch fewer files per move.

**Verification.** Full smoke test + `pytest tests/` green.

---

### PR 2 — Prompts + LLM provider

**Scope.** Move LLM-adjacent pieces. Small, contained.

**Moves.**
- `prompts/*.txt` → `nav/prompts/*.txt` (keep filenames; drop `nav_ego_minimap_his_bev.txt`, already stashed in PR 0).
- `llm_providers.py` → `nav/harness/llm_provider.py` (singular — there's currently only one).

**What changes inside.**

- `llm_providers.py` is 3.8KB and currently only exposes `llm_openrouter`. Add a small provider interface (Protocol or ABC) and confirm OpenAI / Gemini / OpenRouter all go through it. Currently the grep shows mixed `OPENAI_API_KEY`, `GEMINI`, `OPENROUTER_API_KEY` reads — centralize key reads into one place that asserts the right env var is set for the chosen provider, with a clear error message.
- Prompt loader: today prompts are likely loaded via `open(...)`-from-the-script-CWD. Add `nav.prompts.load(name: PromptName) -> str` with a typed enum / Literal of valid prompt names — kills magic strings at call sites.

**Tests added.** `tests/test_prompt_loader.py` — verify each prompt file loads and has the expected template variables.

**Risk.** Low.

**Verification.** Smoke test.

---

### PR 3 — Evaluation pipeline (`nav/eval/`)

**Scope.** Post-experiment metrics, no runtime path.

**Moves.**
- `warning_detect_mp.py` → `nav/eval/warning.py`
- `collision_analyzer.py` → `nav/eval/collision.py`
- `eval_metrics.py` → `nav/eval/metrics.py`
- `batch_eval.py` library half → `nav/eval/aggregate.py`; CLI half → `nav/scripts/aggregate_eval.py`

**What changes inside.**

- `warning_detect_mp.py`: the `_mp` suffix is multiprocessing. Move the pool setup out of module-import-time (if it currently does that) and into a `run(...)` function that accepts an optional `max_workers`. Module-level pool init is a classic bug source for parallel grid runs.
- `collision_analyzer.py` and `eval_metrics.py`: these together are ~27KB. Read carefully for shared helpers (trajectory loading, frame iteration). Likely some pixel-space / world-space coordinate code overlaps with `utils.py` and `harness/types.py`. Consolidate the trajectory I/O into one place.
- `batch_eval.py`: today it likely both *aggregates* (library work) and *writes xlsx* (entry-point work). Split: library functions in `nav/eval/aggregate.py`, the `if __name__ == "__main__":` becomes `nav/scripts/aggregate_eval.py`.
- Drop the side effect of writing into `eval_results.xlsx` at repo root — that file is gitignored / deleted in PR 0. Output goes under `analysis/` going forward.

**Tests added.** `tests/test_eval_metrics.py` with a fixture of a synthetic per-run output directory (a few JSON/PNG frames + a tiny `result.csv`); assert metrics match hand-computed expected values. This is the kind of fixture-based IO test from item 16(b).

**Risk.** Medium — `eval_metrics.py` is on the hot path of `compile_stats`. Test the pipeline end-to-end after the move using one existing run from `outputs/` as a smoke fixture.

**Verification.** Run the `compile_stats` skill end-to-end; output `analysis/nav1/` matches pre-PR baseline byte-for-byte (or, if any change, the change is intended and explained in the PR description).

---

### PR 4 — Stats pipeline consolidation (`nav/stats/`)

**Scope.** The rebuttal stats scripts, consolidated. This is the PR where the most aggressive deduplication happens — these grew under deadline pressure.

**Moves.**
- `xlsx_to_per_run.py` → `nav/stats/load.py`
- `merge_per_run.py` → `nav/stats/merge.py`
- `stats_analysis_full.py` → split into `nav/stats/{bootstrap,permutation,spearman,report}.py`
- `stats_analysis_partial.py` → folded into the above; "partial" was almost certainly a degenerate-mode flag that became its own script

**What changes inside.**

- `stats_analysis_full.py` is 37KB. Expected duplications: scene-clustered bootstrap functions, per-run loading, model-name normalization, leaderboard formatting. Identify and dedupe.
- `stats_analysis_partial.py` is 12KB. Assumption: ≥80% of its code is duplicated from `_full`. Verify by diffing the two files' AST-level function set during the PR; if true, kill it and pass a `mode: Literal["full","partial"]` flag to one entry point.
- One driver script: `nav/scripts/compile_stats.py` — replaces whatever incantation the `compile_stats` skill currently runs. Update the skill's `SKILL.md` (canonical at `.agents/skills/compile_stats/SKILL.md` plus the four wrappers per the cross-agent pattern) to point at the new entry.

**Tests added.** `tests/test_stats_bootstrap.py` with a small fixture per-run csv; verify the bootstrap CI bounds against scipy-computed reference values.

**Risk.** Medium-high — these scripts produce the headline numbers. Lock the test to known-good outputs from before the PR.

**Verification.** Re-run `compile_stats`; diff outputs against `main`-branch outputs; differences must be explained or zero.

---

### PR 5 — Harness restructure (`nav/harness/`)

**Scope.** This is the largest PR and the one where "not 1:1" matters most. The current shape (agents/ + red_detector.py + tangled runtime logic in run_async.py / run_headless_benchmark.py) needs a real architect's pass.

**Pre-PR investigation (deliverable: a 1-page sketch posted as a PR description draft before any code).**

Read in this order, taking notes:
1. `agents/{decision_maker, global_planner, local_planner, prompts, types}.py` (~33KB total)
2. `run_async.py` decision loop (the source of the async state machine that `run_headless_benchmark.py` backports)
3. `run_headless_benchmark.py` (40KB — has the env-param priming, side-channel ack handling, and frame archival)
4. `red_detector.py`
5. `agents/prompts.py` — this is **separate** from `prompts/*.txt`; figure out the relationship. My guess: it's prompt-assembly logic (filling templates with state), not the templates themselves. If so it belongs in `nav/harness/prompt_assembly.py`, not `nav/prompts/`.

**Moves (target shape, subject to revision after investigation).**
- `agents/decision_maker.py` → `nav/harness/decision_maker.py`
- `agents/global_planner.py` → `nav/harness/global_planner.py`
- `agents/local_planner.py` → `nav/harness/local_planner.py`
- `agents/prompts.py` → `nav/harness/prompt_assembly.py`
- `agents/types.py` → `nav/harness/types.py` (extend with side-channel types currently inline in scripts)
- `red_detector.py` → `nav/harness/perception/red_detector.py`
- Extract from `run_headless_benchmark.py` / `run_async.py`: the async state machine into `nav/harness/state_machine.py` + `nav/harness/decision_loop.py`. The router that picks LLM vs A* vs NaVid vs BC goes into `nav/harness/routing.py`.

**What changes inside.**

- Resolve `decision_maker.py` vs. the async loop inside `run_async.py` / `run_headless_benchmark.py`. These two are almost certainly redundant — one was the synchronous reference, the other the async re-implementation. Pick the canonical one (the async version, since it's what the unified client uses) and delete the other.
- The `vision_input` toggle (which prompt + which images get sent to the LLM) is currently scattered. Centralize in `nav/harness/routing.py` with one function that returns the prompt template + image set given the modality flag.
- The grid runner's `--vision_input off` mode currently swaps `prompts/nav_ego_state_history.txt` for `prompts/nav_state_history_no_vision.txt`. Bake into the same router.
- Types: `agents/types.py` is 3.5KB. Likely under-specified — add typed dataclasses for the side-channel acks (`TargetSideChannel` payload, spawn ack), the frame bundle (ego/depth/minimap), and the agent state (current/target world coords, history). Today these flow as untyped dicts/tuples through scripts.
- Multi-modal LLM input note from `AGENTS.md`: "the script sends only **one** image per call... depth and minimap are collected for tracking + frame archival but NOT sent to the LLM, even when other prompt templates suggest otherwise." Fix this inconsistency. Either make the router actually send the multi-modal bundle when the prompt expects it, or delete the prompts that promise multi-modal input. Don't leave the lie in.

**Tests added.** `tests/test_harness_routing.py` — for each `(modality, baseline)` combination, assert the router returns the right prompt path + image bundle shape. `tests/test_red_detector.py` with a fixture PNG that has a known red dot; assert detection coordinates.

**Risk.** High. The runtime path is the most fragile part of the codebase. Split this PR into 2 if it grows past ~2,000 LOC diff: (5a) move-and-rename only with `from nav.harness ...` imports working; (5b) the real consolidation.

**Verification.** Smoke test + at least 3 cells across 2 scenes pass; reproduce the most-recent saved `outputs/<scene>/<point>/<model>/result.csv` byte-for-byte within reason (LLM nondeterminism aside — pin seeds where possible).

---

### PR 6 — Baselines (`nav/baselines/`)

**Scope.** A* and NaVid algorithmic bodies; routing already handled in PR 5.

**Moves.**
- `astar_baseline.py` → `nav/baselines/astar.py` (algorithm) + the entry-point wiring is dropped (PR 5's router calls it).
- `navid_baseline.py` → `nav/baselines/navid.py`.

**What changes inside.**

- `astar_baseline.py` is 24KB — likely includes both the A* algorithm *and* a full standalone runner (env launch, frame loop, result writing). Strip out everything that isn't algorithm; the runner concerns are now `nav/scripts/run_benchmark_cell.py` (PR 8) plus `nav/harness/routing.py`.
- Same for `navid_baseline.py` (9.5KB).
- Confirm the `nav/baselines/{astar,navid}.py` stubs already on disk are empty; if not, reconcile.

**Tests added.** `tests/test_astar.py` — fixture grid with known shortest path; assert the planner returns it.

**Verification.** `shs/run_headless_benchmark.sh --baseline astar` (PR 8 spec) and `--baseline navid` both succeed on one cell.

---

### PR 7 — BC models + training (`nav/models/`, `nav/train/`)

**Scope.** Behavior cloning. Per `cleanup.md` line 119, kept as ML reference; modularize properly.

**Moves.**
- `navigation_bc/dataset.py` + `navigation_bc/dataset_seq.py` → consolidated `nav/train/dataset.py`. These two probably overlap heavily (one is the sequence variant); dedupe with a `sequence_length` parameter.
- `navigation_bc/controller.py` → `nav/train/controller.py` (used at inference time; consumed by harness routing for BC-mode runs).
- `navigation_bc/model.py` (19KB) → split per architecture into `nav/models/{cnn,resnet,dino}.py`. The existing stubs at those paths should be filled by this PR.
- `navigation_bc/train.py` (17KB) → `nav/train/loop.py` (the training loop, model-agnostic) + `nav/scripts/train_bc.py` (CLI entry, `--base {cnn,resnet50,dinov2}`).

**What changes inside.**

- `navigation_bc/model.py` likely has three model classes plus shared building blocks. Pull the shared blocks into `nav/models/_blocks.py`; per-arch files stay slim.
- Hyperparameters for each base (learning rate, epochs, batch size, image size, augmentations) move into `nav/config.py` as `BCModelConfig` dataclass instances. `--base dinov2` resolves to one such instance.
- The three shell scripts (`train_dinov2.sh`, `train_resnet50.sh`, `train_vanilla_cnn.sh`) collapse into one `shs/train_bc.sh` that takes `--base` and otherwise has identical body.

**Tests added.** `tests/test_models_construct.py` — instantiate each model with a config, run a single forward pass on a random tensor. Doesn't test correctness, just wiring.

**Risk.** Low (BC didn't work in the original; we're not chasing performance, only structural cleanliness).

**Verification.** `shs/train_bc.sh --base cnn` runs 1 epoch on a tiny synthetic dataset without crashing.

---

### PR 8 — Entry-point scripts + shell consolidation

**Scope.** The thinning of `run_*.py` into `nav/scripts/*.py` and the merge of the macOS/Linux shells.

**Moves.**
- `run_headless_benchmark.py` → `nav/scripts/run_benchmark_cell.py` (per-cell runner — its body should now be mostly orchestration, since harness/eval/baselines/routing have absorbed the actual logic).
- `run_benchmark_grid.py` → `nav/scripts/run_benchmark_grid.py` (outer orchestrator; two-tier per PR 4 of `cleanup.md`).
- `run_headless_collect_data_assign_points_v2.py` → `nav/scripts/collect_data.py`.
- `shs/run_headless_benchmark_macos.sh` + `shs/run_headless_benchmark_linux.sh` → merged `shs/run_headless_benchmark.sh` with an OS-detection branch (`uname -s`) that adds `xvfb-run -a -s "..."` only on Linux. The Python entry point is OS-agnostic.
- `shs/run_agent.sh`, `shs/run_bc_agent.sh`, `shs/run_human_eval.sh`, `shs/run_astar_all_scenes.sh` → folded into `shs/run_headless_benchmark.sh` via `--baseline {llm,bc,human,astar,navid}`. Each old shell becomes a thin wrapper (one-liner) that calls the unified shell with the right flags, *or* gets deleted if no longer referenced. Default: delete the wrappers, document the flag in `docs/run_benchmark.md`.
- `shs/run_ablate_history_size.sh` → thin wrapper calling `shs/run_headless_benchmark.sh --history-sizes 0 1 4 16`.
- `shs/run_agent.ps1` → updated to call into the new Python entry. Per the decision: keep Windows on parity.
- Stash to `deprecated/`: `run_async.py`, `run_sync.py`, `run_headless.py`, `play_mlagents_multiagent.py`, `play_mlagents_v6.py`.

**What changes inside.**

- After all the harness/eval/baselines work, `nav/scripts/run_benchmark_cell.py` should be **under 300 lines**. If it's larger, more logic still needs to come out into the harness. Today `run_headless_benchmark.py` is 40KB — the goal is to demolish that file's mass into the package.
- History size: in `nav/scripts/run_benchmark_grid.py`, add `history_size` to the sweep-axis enumeration. Currently the grid axes are model × scene × point × seed × vision; add this sixth axis.
- Grid-run aggregate outputs: write to `analysis/grid_runs/<timestamp>/{runs,failures}.csv` per `cleanup.md`, not `outputs/`. The `outputs/` directory stays for per-cell telemetry only.
- The OS-detection branch in the merged shell is the only place where OS matters; centralize so future scripts don't recreate the branch.

**Tests added.** `tests/test_grid_runner.py` — fake the cell runner with a stub subprocess that returns a known result; assert the grid CSV is written correctly, resume works, failures land in `grid_failures.csv`.

**Risk.** High — this is the user-facing entry point. Run a full grid (small: 1 model × 2 scenes × 1 point × 1 seed × 1 vision = 2 cells) before merging.

**Verification.** Full sweep on a known model+scene combination matches pre-refactor numbers within LLM-nondeterminism tolerance. Windows wrapper smoke-tested if a Windows machine is available; otherwise documented as "untested, parity intent only" in the PR.

---

### PR 9 — `third_party/` → `external/` + ml-agents reinstall

**Scope.** Pure cosmetic rename + reinstall + docs sweep.

**Steps in PR.**
1. `uv pip uninstall mlagents mlagents-envs`.
2. `git mv third_party/ external/`.
3. Update `pyproject.toml` `[tool.uv.sources]` paths from `third_party/ml-agents/...` to `external/ml-agents/...`.
4. `uv pip install -e external/ml-agents/ml-agents-envs && uv pip install -e external/ml-agents/ml-agents`.
5. `.gitignore`: flip `third_party/` line to `external/`.
6. Update canonical setup skills with the install commands (these were missing — per the original review):
   - `.agents/skills/install_dependencies_macos/SKILL.md`
   - `.agents/skills/install_dependencies_linux/SKILL.md`
   - `docs/python_env_options.md`

**Tests added.** None directly; smoke test covers it.

**Risk.** Low — purely cosmetic. The trap is forgetting one of the four wrapper files; the cross-agent pattern means `.claude/skills/`, `.codex/skills/`, `.cursor/rules/`, `.github/copilot-instructions.md`, and `AGENTS.md` all need to stay consistent with the canonical under `.agents/skills/`. Use `.agents/sync_check.py` after the edit to verify.

**Verification.** Fresh `uv sync` on a clean clone works; smoke test.

---

### PR 10 — Tests backfill

**Scope.** Tests that didn't fit into their source PR. By this point, most tests should already exist; this PR catches gaps.

**Likely gaps to fill.**
- Integration test: a full single-cell run against a recorded fixture (mock the Unity env via a stub `BaseEnv` that replays canned observations).
- Side-channel codec round-trip test.
- Config dataclass validation: pickle/unpickle, env-override behavior.

**Risk.** None — tests are additive.

---

### PR 11 — Docs sweep

**Scope.** README + docs/ alignment with the post-refactor layout.

- `README.md` and `README_AGENT_EVAL.md` / `README_DATA_COLLECTION.md` are stale per `AGENTS.md`. Replace `README.md` with a current quickstart pointing at `docs/run_benchmark.md` and the canonical skills; stash the two `README_*.md` to `deprecated/`.
- `docs/run_benchmark.md`: rewrite for the new entry (`shs/run_headless_benchmark.sh` with `--baseline` and `--history-sizes` flags).
- `docs/python_env_options.md`: incorporate the `external/ml-agents` install steps (done in PR 9, double-check here).
- New `docs/secrets.md`: document the `tmp/secrets.sh` workflow (or fold into `docs/python_env_options.md`, the user's call).
- `AGENTS.md`: update file path references that have moved (was/is the dev-agent entry). Cross-check that none of the canonical skill `SKILL.md` files reference old paths.
- Remove `docs/cleanup_clarification.md` (was always meant to be temporary).
- Decide fate of `docs/cleanup.md` and this `docs/refactor_plan.md`: either retain as historical record under a new `docs/history/` subfolder, or stash to `deprecated/` once everything has landed.

**Risk.** None functional; docs hygiene only.

---

## Open questions to resolve during execution (not blockers for starting)

1. **`unity_scripts/UnityWarehouseSceneHDRP/` vs `c_sharp_scripts/`** — which is canonical? Affects PR 0.
2. **`agents/prompts.py` purpose** — prompt-assembly logic vs. templates. Affects PR 5 placement.
3. **`run_async.py` vs `run_headless_benchmark.py` async loop** — which is the real source of truth, or are they actually different stages? Affects PR 5 scope.
4. **`navigation_bc/dataset.py` vs `dataset_seq.py`** — confirm overlap before consolidation. Affects PR 7.
5. **`stats_analysis_full.py` vs `stats_analysis_partial.py`** — confirm overlap. Affects PR 4 size.
6. **Windows `.ps1` testing** — do we have a Windows machine available for smoke testing in PR 8, or do we ship "parity intent, untested"?

---

## Consolidation hit-list (specific duplications I expect to find)

Concrete predictions; will revise as I read. Each one is an opportunity, not a guaranteed merge.

- Pixel↔world coordinate helpers in `utils.py` are likely dead code per `AGENTS.md` ("`pixel2world` / `world2pixel` helpers in legacy scripts are dead code for the unified client"). Verify and delete.
- Three `ACTION_SPACE` aliases (per PR 1 notes). Pick one.
- `run_sync.py` / `run_async.py` / `run_headless.py` / `run_headless_benchmark.py` — the older three are 100% replaceable by the newer one. The "spirit" of any unique logic they have should be absorbed first, then stash.
- `stats_analysis_full.py` and `stats_analysis_partial.py` likely share ≥80% bodies.
- `astar_baseline.py` and `navid_baseline.py` each likely contain their own copy of an env-launch + frame-loop preamble that duplicates `run_headless_benchmark.py`. Strip both to algorithm-only.
- `navigation_bc/dataset.py` and `dataset_seq.py` differ only in sequence handling.
- `train_dinov2.sh`, `train_resnet50.sh`, `train_vanilla_cnn.sh` differ only in `--base` flag.
- `shs/run_headless_benchmark_macos.sh` and `..._linux.sh` differ only in the `xvfb-run` wrapper. Already noted.

---

## Rules of engagement during execution

Stated previously, repeated here for the executing agent's reference:

1. **No 1:1 pasting.** Every move is an opportunity to dedupe, rename, simplify.
2. **Read consumers before moving.** Know who calls what before deciding the new module's surface.
3. **Push constants out of code into `nav/config.py`.** Dataclasses where the constant is structured.
4. **Don't change observable behavior** of working entry points without explicit user approval.
5. **Don't rename CLI flags or env params** that the Unity client / shell wrappers depend on.
6. **Fix-don't-flag** — if a function is buggy or confusingly named in the original, fix in the same PR as the move. Don't leave TODOs.
7. **`main` stays runnable** after every merge. Smoke test before opening each PR.
