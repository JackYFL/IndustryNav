# Statistical analysis: limitations of the archived data

This document is the canonical reference for **what the existing
`experiment_results_v6.csv` does and does not support**, why, and how each of
the four reviewer statistics asks (numbered as in `industrynav_eccv2026_rebuttal/author_actions/TODO.md`)
maps onto the partial-value vs full-value scripts in this repo.

Use this when rewriting rebuttal responses so the framing of caveats matches
what the analysis code actually computes.

---

## TL;DR

- The archived per-run CSV (`experiment_results_v6.csv`, 620 rows across 16
  models) **does not record scene or point identity** for any run — the
  `exp_name` column is empty for all 620 rows.
- Without that metadata, **paired** model comparisons, **scene-clustered**
  bootstrap CIs, and **leaderboard correlation across independent target
  sets** are all impossible to compute on the archived data.
- We therefore implement weakened proxies in `nav/stats/partial.py`
  (unpaired permutation, non-clustered episode-level bootstrap, Spearman
  deferred entirely) and explicitly disclose the limitation.
- The proper paired/clustered/Spearman versions live in
  `nav/stats/full.py` and run against the new grid output tree
  (`nav.scripts.run_benchmark_grid`), which writes scene-labeled per-run records
  going forward.

---

## What the archived data actually contains

Source: `experiment_results_v6.csv` at the repo root.

Columns recorded per run:

| Field | Notes |
|---|---|
| `timestamp`, `exec_mode`, `provider`, `model` | model identity |
| `target_x`, `target_y`, `final_x`, `final_y`, `distance_px` | end-state |
| `stop_reason` | success indicator (`reached_vicinity` ⇒ success) |
| `steps_taken`, `max_steps`, `reach_px` | episode budget |
| `frame_sleep`, `modalities`, `sim_steps_per_decision` | run config |
| `exp_name` | **empty for all 620 rows** |

Crucially: there is **no scene name** (`yifan1`, `lichi2`, …) and **no point
id** (`point1`, …) on any row. Two runs from different scenes that share the
same `target_x/y` are indistinguishable in the CSV.

Run counts per model in the archive (n ≥ 1):

| Model | n |
|---|---:|
| `openai/gpt-4o-mini` | 385 |
| `google/gemini-2.5-flash` | 148 |
| `anthropic/claude-sonnet-4.6` | 15 |
| `qwen/qwen3.5-plus-02-15` | 13 |
| `google/gemini-3-flash-preview` | 12 |
| `anthropic/claude-sonnet-4.5` | 8 |
| `openai/gpt-5-image-mini` | 7 |
| `openai/gpt-5.2` | 7 |
| (8 others, n ≤ 4 each) | … |

Two practical caveats on top of the missing scene metadata:

1. **The archive is heterogeneous.** It pools all runs ever logged, including
   early-development runs with non-final prompts. Closed-source models
   (claude-sonnet-4.6, gemini-3-flash-preview) have 0% SR in this data — almost
   certainly an artefact of small-sample and pre-finalization runs, not a
   true performance estimate. **Do not present these archived numbers as
   the headline rebuttal numbers.** Use them only as historical reference;
   the actual rebuttal SR/dist values must come from the new grid runs.
2. **Sample sizes are highly imbalanced.** `gpt-4o-mini` has 385 runs while
   most other models have ≤15. Any unpaired test mixes models with very
   different effective N, which inflates the apparent precision of the larger-N
   side. Permutation p-values remain valid but interpretive weight is small.

---

## Item-by-item breakdown

### (1) Re-run 4 models with 3 seeds × 4 resampled start-target pairs/scene

This is **not a statistics ask** — it's the data-generation step that unblocks
the proper statistics in items (2)-(4). Implemented as `nav.scripts.run_benchmark_grid`.

Practical reduction for the rebuttal scope: 2 closed-source models
(`anthropic/claude-sonnet-4.6`, `google/gemini-3-flash-preview`) × 12 scenes × 4
existing points (from `input_points.json`, not newly resampled — see note
below) × 3 seeds × 2 vision modalities (vision-on, vision-off) = **576 cells**.
At ~5 min/cell wall-clock and `--max_concurrency 4`, ~12 h. Estimated cost
~$87 at current OpenRouter rates.

Note on "resampled" pairs: the rebuttal text asks for *new* start-target
pairs. We are reusing the original 4 points/scene from `input_points.json`
because (a) authoring 4 new pairs/scene by hand requires manual reachability
checks (~hours of work) and (b) the point of the re-run is to obtain
*scene-labeled, multi-seed* data, which the original points satisfy.
Reframe in the rebuttal as *"three independent runs over the existing four
start-target pairs per scene, yielding paired-model and scene-clustered
statistical tests not previously possible"* rather than *"resampled pairs"*.

### (2) Paired permutation test (10K perms)

**Asked:** paired permutation between each closed-vs-open and closed-vs-closed
pair; report p-values.

**On the archived CSV (`nav/stats/partial.py`):** weakened to an **unpaired
two-sample** permutation test on per-run distance and success indicator,
for every model pair where both have ≥`--min_n` runs (default 10). The
unpaired test is statistically valid but ignores the dominant variance source
(scene difficulty), so it is conservative and underpowered.

Output: `analysis/<subdir>/partial/permutation.csv`, summarized in
`analysis/<subdir>/partial/report.md`.

**Suggested rebuttal phrasing:**

> *Per-run distance and success indicators were compared via two-sided
> two-sample permutation tests (10,000 permutations) on the archived run
> log. Because that log does not record scene/point identity, paired tests
> at the (scene, start-target, seed) level are computed on the new grid
> in §X.Y rather than on the archive. The unpaired archive numbers are
> reported in Tab. Z as a baseline; we treat them as conservative.*

**On the new grid (`nav/stats/full.py`):** the proper **paired** version.
For each model pair, we pair runs on `(scene_name, point_id, seed_id)` and
do a sign-flip permutation on per-cell differences. This is the version that
should drive the headline rebuttal claims.

### (3) Spearman rank correlation across two independent start-target sets

**Asked:** rank-correlate the per-model leaderboard between two independently
resampled start-target sets — robustness of the leaderboard to the choice of
test points.

**On the archived CSV:** **deferred / cannot compute.** Without scene labels
there is no meaningful way to construct "two independent target sets" — the
only available split is by row index or by raw `target_x/y` quartile, neither
of which corresponds to the experimental concept the test is supposed to
measure. We do not produce a number; we record the deferral in the report.

**Suggested rebuttal phrasing:**

> *Leaderboard rank-correlation across independent start-target subsets is
> deferred until the grid in §X.Y completes; the archived run log does not
> retain the per-run scene/point identity required to construct two
> meaningfully-disjoint pair subsets after the fact. We commit to including
> Spearman ρ across two pair subsets in the camera-ready.*

**On the new grid:** computed by splitting the four points per scene into
two halves (`{point1, point2}` vs `{point3, point4}`, configurable via
`--spearman_split`), computing per-model SR leaderboards on each half, and
reporting Spearman ρ. Implemented in `nav/stats/full.py`.

### (4) 95% bootstrap CIs over the 12 scenes (clustered bootstrap)

**Asked:** clustered bootstrap (cluster = scene) for SR / DR / CR / WR.

**On the archived CSV:** weakened to **non-clustered episode-level**
bootstrap. We resample individual runs (with replacement) rather than whole
scenes. Because within-scene episodes share an unobserved difficulty term,
the true clustered CI is **wider**; the non-clustered numbers we report are
a *lower bound on uncertainty*. We disclose this in the report.

Output: `analysis/<subdir>/partial/bootstrap.csv`.

**Suggested rebuttal phrasing:**

> *95% confidence intervals on success rate and mean final distance are
> computed by 10,000-resample bootstrap. For runs lacking scene metadata
> (the archive set) we report a non-clustered episode-level bootstrap as
> a lower bound on uncertainty; for the new grid runs we use a
> scene-clustered bootstrap that resamples whole scenes with replacement
> (12 clusters), per Cameron & Miller (2015) for clustered data with
> within-cluster correlation.*

**On the new grid:** proper scene-clustered bootstrap — resample the 12 scenes
with replacement, recompute the metric on the pooled per-scene runs, repeat
10,000 times, take the 2.5/97.5 percentiles. Implemented in
`nav/stats/full.py`.

Metric coverage: the partial pipeline emits SR + mean distance only.
DR/CR (distance ratio, collision rate) require per-step data from
`agent_actions.csv`, which the archive lacks. The full pipeline
(`nav/stats/full.py`) computes SR/DR/CR for every (model, vision) group,
and WR additionally when the source carries it (xlsx-derived rows; grid
runs saved depth as PNG not raw NPY, so WR is NaN there).

---

## Proper-version path (after the grid runs land)

All stats stages are driven by one CLI — `nav.scripts.compile_stats` — with
subcommands (`grid`, `xlsx`, `per-run`, `merge`, `partial`, `all`). The
underlying logic lives in `nav/stats/` (see [`compile_stats` skill](../.agents/skills/compile_stats/SKILL.md)
for the full multi-source workflow). Note the CLI uses dash-style flags
(`--vision-input`, `--n-perm`), not the underscore style the pre-refactor
standalone scripts used.

Once `nav.scripts.run_benchmark_grid` produces scene-labeled per-run records:

1. `python -m nav.scripts.compile_stats grid --models anthropic/claude-sonnet-4.6 google/gemini-3-flash-preview --vision-input both`
2. Outputs: `analysis/<subdir>/{report.md, per_run.csv, bootstrap.csv, variance.csv, paired_perm.csv, spearman.csv}`
3. The full pipeline handles all three asks (paired permutation / Spearman /
   clustered bootstrap) and stratifies by vision-on vs vision-off so the
   modality ablation has its own rows.

The numbers from `nav/stats/full.py` are what should populate the
rebuttal's headline statistics. The numbers from `nav/stats/partial.py`
serve as a methodological bridge: *"we know the right tests; here is the
correct weakened version on the data we currently have; the proper version
is in §X.Y on the new runs."*

---

## What the rebuttal should NOT claim

- Do **not** report archived-CSV unpaired p-values as if they were paired.
- Do **not** report archived-CSV non-clustered CIs as if they were clustered.
- Do **not** quote the 0%-SR archived numbers for closed-source models as
  current performance — they are pre-finalization residue. Headline numbers
  must come from the new grid.
- Do **not** claim "two independently resampled start-target sets" without
  explicitly defining the split and noting that we used the original four
  points per scene (split into halves) rather than newly authored pairs.

---

## File map

| File | What it does |
|---|---|
| `experiment_results_v6.csv` | The archive. 620 rows, no scene labels. Read-only historical artifact. |
| `nav/stats/partial.py` | Partial-value tests on the archive. Implements (2)-unpaired and (4)-non-clustered; defers (3). |
| `analysis/<subdir>/partial/report.md` | Markdown summary of partial-value tests. |
| `nav.scripts.run_benchmark_grid` | Generates the new scene-labeled run records that unblock proper tests. |
| `nav/stats/full.py` | Proper-version tests on the new grid output tree. (2)-paired, (3)-Spearman, (4)-clustered. |
| `analysis/<subdir>/report.md` | Markdown summary of proper-version tests (created after grid completes). |
