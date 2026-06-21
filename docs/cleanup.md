# Refactor and cleanup

Currently the codebase has a super messy structure, some code should be deprecated + removed, the remaining of the codebase should be restructured and we should add unit tests, and then we will need to update the documentation under `docs/` accordingly.

# Goals

- Deprecate old files. **Interim policy:** stash them under a new root level folder `deprecated/` with no modifications (rather than hard-delete), so they remain handy as reference while the large refactor PRs are still landing. Once the refactor has stabilized, hard-delete the contents of `deprecated/` — git history still preserves them.
- For files we want to keep, we are keeping the functionality of them while refactoring the structure of the repo.
- Structure we are trying to get to: Instead of having lots of files scattered, we should shape the codebase similar to a python module package where its entry points are a few python scripts meant for job-style one time runs (i.e. no web services).
- Post-refactor, the code base should use a single root level folder `nav` (acting as a module) to house *ALL* python files.
- Output artifacts today are mostly (with some possible exceptions) under `outputs/` (which should stay as not-git-tracked as each run typically contains files too many / too large), but we should separate between statistical artifacts like post-experiment aggregative statistical summaries (which should be parked under `analysis/`, git tracked) versus per-run collected telemetry/logs (which are under `outputs/`).
- Post-refactor, the code should still be runnable at least on MacOS (26.4.1 or above) and Linux (you can assume Ubuntu >= 22.04 LTS), just like how it is able to run today on them. Windows 11 is optional but preferred to be supported if possible.
- Functions with overlapping/redundant intent or functionality should be considered for consolidation and overloading to reduce the codebase size and redundancy.
- Generic, non-data, non-ML, non-stats, simple, often stateless python utility functions (e.g. file operations, JSON load, JSON save, CSV write, logger construction) should be docked under `utils.py` which will be moved to `nav/utils.py`. This will be the centralized place to put common utilities that can easily be imported/reused in multiple other python files.
- *ALL* configurative hyperparameters, macros, env variables should be inside `config.py`, which should be docked into `nav/config.py`. The only exception is the `input_points.json` which we can keep it outside as a root level config json file for now as those are scene-level starting/destination hyperparameters manually specified for the benchmark experiment runs.
- After we modularize + refactor the codebase, most if not all stateless functions (as long as they are reasonably lightweight) should have new unit tests under `tests/` using python's pytest/unittest. Probably the LLM API call utilities won't need this.
- A* and NaVid baseline ML/algorithmic portions should be moved into `nav/baselines/astar.py` and `nav/baselines/navid.py`.
- For the behavioral cloning model bases (CNN, DINO V2, Resnet50), the model/ML constructor portions should be inside `nav/models/cnn.py`, `nav/models/resnet.py`, `nav/models/dino.py`. Their unified training loop main logic should be docked under `nav/train/`, but the final main thin wrapping entry point training script which imports them should be housed under `nav/scripts/`.
- LLM Prompts under `prompts/` should be migrated into under `nav/prompts/`.
- Post-experiment evaluation utilities (warning ratio, collision metric, final per-run stats calc, aggregator, and any vision-dependent post-processing such as depth estimation) should be docked under `nav/eval/`.
- Statistical analysis scripts (currently `merge_per_run.py`, `stats_analysis_full.py`, `stats_analysis_partial.py`, `xlsx_to_per_run.py`) should be consolidated and docked under `nav/stats/`. Aggressively dedupe overlapping logic during the move — these grew organically during the rebuttal and almost certainly have redundancy.
- The runtime agent harness (today scattered across `agents/` + `red_detector.py`) should be repackaged under `nav/harness/` with a senior-SWE-style restructure (not a 1:1 file copy). Suggested sub-modules: `harness/state_machine.py`, `harness/decision_loop.py`, `harness/local_planner.py`, `harness/routing.py` (baseline-vs-LLM input dispatch), and `harness/perception/` (e.g. `red_detector.py`).
- *ALL* post-refactor main entry point python scripts should be docked under `nav/scripts/`. Keep in mind after the refactor, these entry point python scripts should be somewhat thin, just being there to import all the submodules needed and glue them into the right control flow (in async or parallel), exposing plenty of configurable CLI arguments that will usually be specified by the `shs/` shell scripts hovering above them.
- Main entry-point level functionality to fully preserve:
  - Human operated data collection (whose sole entry point currently is `run_headless_collect_data_assign_points_v2.py`)
  - Benchmark experiment run (`shs/run_headless_benchmark_macos.sh` and `shs/run_headless_benchmark_linux.sh`). Ideally these two can be merged into one single shell script considering the OS's are both unix-based. If possible for a merge, the merged file should be called `shs/run_headless_benchmark.sh`. Python side, this shell drives a **two-tier** entry: `nav/scripts/run_benchmark_grid.py` (orchestrator) subprocesses into `nav/scripts/run_benchmark_cell.py` (per-cell runner, derived from today's `run_headless_benchmark.py`). Subprocess isolation is load-bearing — a Unity SIGKILL on one cell must not take down the rest of a long sweep. A 1×1 grid is the default for single-cell debugging.
  - Local behavior cloning model training (`shs/train_dinov2.sh`, `shs/train_resnet50.sh`, `shs/train_vanilla_cnn.sh`) should be consolidated into an overloaded code style. There should only be one `train_bc.sh` with CLI argument `base` to choose from `dinov2`, `cnn`, `resnet50` where all their model-specific hyperparameters and config should go under `nav/config.py` too.
  - `shs/run_ablate_history_size.sh` for history size related ablation run. Collapse history-size ablation into the grid runner as one more sweep axis (model × scene × point × seed × vision × **history_size**); this shell reduces to a thin wrapper that calls `shs/run_headless_benchmark.sh` with `--history-sizes 0 1 4 16` (or similar).
  - `shs/run_astar_all_scenes.sh` for A* classical baseline
  - `shs/run_bc_agent.sh` for inference of behavioral cloning models trained above, this should be consolidated into `shs/run_headless_benchmark_macos.sh` and `shs/run_headless_benchmark_linux.sh` or `shs/run_headless_benchmark.sh` if merged.
- Large dependency moving:
  - currently we docked the official Unity ml-agents giant package under `third_party/` and this `ml-agents` is currently installed into our UV python env. I'd like this root level folder renamed to `external` (which most likely means you should probably uninstall it, rename and then reinstall). Also this ml-agents clone+installation steps might have been accidentally absent from both our `docs/` documentation files and/or our AI dev agent skill markdown files, which we should check and fix.
  - another one is our super large Unity project folder (which contains all the editor-facing project files and assets, >>20GB total) currently under `/Users/lichili/dev/IndustryNav2/IndustryNav/`, and that whole Unity project collection of files to be docked into a root folder `unity_project/` of this repo. Now because I already loaded the project at a different location on my macos system here, you should give me instructions as to the safe way to let Unity Editor/Hub unmount the project from my current file location and then move them here -> remount it into Unity again. Obviously I am not trying to actually check in this giant Unity project into git, the point here is to permanently keep a project mounting spot so that when I myself or other collaborating devs use this repo, they can mount their copy of the same Unity project in the same relative path location, and all of us will be able to let local dev coding agents (Claude Code, Codex) perform file read/edits into the Unity project, which does closely relate to the code base in python/shell here.

# File level decisions, high level

Below we first outline the file level decisions. Here "keep" means keep the functionality of the file and all the sub functionality it depends on to run, NOT keep the file as-is. Delete/drop/deprecated literally means that file should be removed or moved into the `deprecated/` root folder and they are no longer of use to this project going forward (typically means it may have been superseded by another newer version file with similar purpose with improved logic or features, or that it was failed experimental feature that has no value).

## Data Collection

- v1 `run_headless_collect_data_assign_points.py` is based on old client (where 12 scenes were separated) -> deprecate, too old and not useful.
- v2 `run_headless_collect_data_assign_points_v2.py` new assets, client, map/scene coordinate fix -> main entry point file, must keep functionality
- `run_headless_depth_assign_points.py` -> Deprecate, drop

## Others

- Keep `utils.py`
- `config.py` -> contains all the runtime hyperparameter, Keep
- `llm_provider.py` -> keep, this is for API connectors, maybe agents/
- `turn2gif.py` -> now called by a shell script (turn2gif.sh, which should be integrated into post-experiment run as final step). Keep.
- `.env.example` -> delete entirely (file currently contains live-looking keys; coauthors have been notified to rotate). Replacement workflow: each developer manually creates `tmp/secrets.sh` containing `export OPENAI_API_KEY=...`, `export GEMINI=...`, `export OPENROUTER_API_KEY=...` (shell-source-able). The unified shell wrapper under `shs/` `source`s `tmp/secrets.sh` at startup. Commit a placeholder `tmp/secrets.sh.example` documenting the expected variable names. **`.gitignore` policy: ignore `tmp/*` but explicitly un-ignore `tmp/secrets.sh.example` (and `tmp/.gitkeep` if used) — be careful when editing `.gitignore` to not accidentally track other `tmp/` contents.** Setup instructions go in `docs/python_env_options.md` (or a new `docs/secrets.md`).
- `c_sharp_scripts/` -> super old copy of c sharp scripts when we first started the project. Deprecated & should be removed.

## Artifacts

- `unity_log.txt` -> generated by Unity engine, no need to keep (untrack+gitignore is fine, also fine to delete)
- `input_points-my.json` -> duplicate of `input_points.json`, delete. Keep `input_points.json`
- `experiment_results_v6.csv` -> delete, this is an outdated, old artifact that can be regenerated.
- `eval_results.xlsx` -> can be generated, delete
- `history_size_metrics.xlsx` -> delete, these are old deprecated results for ablation
- `analysis/` -> keep folder as the canonical git-tracked home for aggregate stats. Delete only the stale snapshots inside (e.g. `analysis/nav1/before_rebuttal/`); retain the current live `analysis/nav1/{bootstrap,paired_perm,per_run,…}.csv` consumed by the `compile_stats` skill.
- Grid-run aggregates `outputs/grid_runs.csv` and `outputs/grid_failures.csv` -> move under `analysis/grid_runs/` going forward. Update `run_benchmark_grid.py` so future runs write straight there instead of into `outputs/` (which stays untracked, per-run telemetry only).

## Evaluation

- `warning_detect_mp.py` -> keep. Warning-ratio metric. Move to `nav/eval/warning.py` (the `_mp` multiprocessing detail belongs inside the implementation, not the filename).
- `collision_analyzer.py` -> keep. Collision metric. Move to `nav/eval/collision.py`.
- `eval_metrics.py` -> keep. Final per-run stats calc. Move to `nav/eval/metrics.py`.
- `batch_eval.py` -> keep. Aggregator that compiles per-run evals into an xlsx. Move to `nav/eval/aggregate.py`; if its `main()` is a real entry point, split that out into `nav/scripts/aggregate_eval.py`.
- `run.sh` -> old entry shell script, deprecate
- `shs/run_agent.ps1` -> older windows shell script reference -> optional for keeping, can drop if we cannot support windows, ideally upgrade this script if windows 11 can still be supported somehow.
- `shs/run_agent.sh` -> keep but merge into headless benchmark shell
- `shs/run_human_eval.sh` -> human baseline, keep functionality but dont need to keep as dedicated file. -> migrate and delete (see which other sh+py files can absorb this)
- `merge_per_run.py` -> keep functionality but consolidate aggressively. Together with `stats_analysis_full.py`, `stats_analysis_partial.py`, and `xlsx_to_per_run.py`, move into `nav/stats/` and dedupe overlapping logic. The `compile_stats` skill should keep working end-to-end after this consolidation.

## Ablation

- `shs/run_ablate_history_size.sh` -> Ablation for history size, keep

## Prompts

- prompts/
  - `nav_ego_minimap_his_bev.txt` -> dont need, can delete
  - `nav_ego_minimap` -> actively used, keep
  - `nav_ego_state_history` -> for history ablation, keep
  - `nav_ego_state` -> keep (for non history ablation)
  - `nav_minimap_only` -> keep (for no history, no ego, minimap only)

## Baseline or Training

- `navigation_bc/` behavior cloning training - didn't work, but keep, maybe modularize
- `shs/train_*.sh` -> behavior cloning training - didn't work, maybe consolidate
- `navid_baseline.py` -> (this is a reproduction of NaVid https://arxiv.org/abs/2402.15852 as a non-VLLM classical baseline) keep, but consolidate this into our benchmark evaluation entry point
- `astar_baseline.py` -> (this is A* algorithm as a classical, non-VLLM baseline) keep, but consolidate this into our benchmark evaluation entry point.

## Method / Agents

- `agents/` contains most of the VLA/VLLM submodules called by run_* benchmarking main scripts. Keep functionality; reorganize + repackage under `nav/harness/` with a proper senior-SWE-style restructure (NOT a 1:1 file move). See the Goals section for the suggested sub-module layout.
- `agents/local_planner.py` contains BEV and egocentric. Keep. -> `nav/harness/local_planner.py`.
- `red_detector.py` is the red dot detector used during agent moving (runtime perception loop, not a post-hoc metric). Keep. -> `nav/harness/perception/red_detector.py`.

## Editor-based Unity interactions

- `play_mlagents_multiagent.py` -> stash in `deprecated/` (will be hard-deleted after the refactor stabilizes).
- `play_mlagents_v6.py` -> stash in `deprecated/` for now. Likely broken against the current Unity project + ml-agents version. Future task (separate from this refactor): revive editor-mode execution as an alternative to the built `scene_all` client; will likely require updates to the Unity project C# scripts too once Claude Code can observe the latest Unity project files.
