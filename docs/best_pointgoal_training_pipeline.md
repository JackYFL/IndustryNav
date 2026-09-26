# Best PointGoal DAgger and PPO Pipeline

This workflow ports the provenance-checked training chain that produced the
52.08% DAgger checkpoint and the Mixed400 PPO checkpoint. The reference PPO
checkpoint scored 72/96 on action01 and 71/96 on an action30 repeat. The repeat,
not the larger count, is the recorded 73.96% result.

The chain has four distinct stages:

```text
all-scene A* + DAgger recovery data
  -> depth-only Transformer DAgger
  -> lossless DAgger-to-actor-critic conversion
  -> 100-update PPO continuation
  -> independent-point, map-reward, KL-regularized PPO updates 1-141
  -> optimizer-preserving resume for updates 142-400 (Mixed400)
```

Do not skip the 100-update continuation when reproducing Mixed400: its update
100 checkpoint, rather than the raw converted actor, initialized the final run.

## Portable provenance

Tracked paths:

- `configs/pointgoal_best_20260913/pipeline.json`: expected paths, hashes,
  task counts, model settings, and reference scores;
- `shs/bc/run_best_dagger_local.sh`: exact DAgger training settings;
- `shs/rl/prepare_best_dagger_ppo_local.sh`: lossless actor conversion;
- `shs/rl/run_best_ppo_stage1_local.sh`: the 100-update PPO continuation;
- `shs/rl/run_best_ppo_local.sh`: the final independent-point PPO run;
- `nav.scripts.tools.verify_best_pointgoal_pipeline`: local asset verification.

Large datasets and checkpoints remain gitignored. Synchronize them resumably:

```bash
bash shs/rl/sync_best_pointgoal_pipeline.sh
python -m nav.scripts.tools.verify_best_pointgoal_pipeline \
  --require-reference-outputs
```

The sync preserves the DAgger aggregate's relative symlinks. Copying only
`datasets/astar_pointgoal_dagger_long_v1_all24_round1` is insufficient: its
episodes point into the A* base and DAgger recovery roots.
It also retains the original experiment-local `nav/` snapshots under
`experiments/ppo_dagger52_20260912` and `experiments/ppo80_v8_20260913` for
source archaeology. The launchers use the current compatible package so local
fixes remain available; the frozen copies are not silently placed on
`PYTHONPATH`.

## 1. DAgger

Inspect the exact command without starting training:

```bash
DRY_RUN=1 bash shs/bc/run_best_dagger_local.sh
```

Run it in a new output directory:

```bash
OUTPUT_DIR=ckpts/astar_pointgoal_dagger_long_v1_all24_round1_local \
  bash shs/bc/run_best_dagger_local.sh
```

The reference configuration is depth-only ResNet-50, a 12-step three-layer
Transformer, polar relative PointGoal, three navigation actions without stop,
six epochs, and macro navigation accuracy model selection. On a Mac this BC
trainer currently uses CPU; the remote reference used CUDA.

## 2. Lossless actor conversion

The downloaded reference conversion is already available under
`analysis/ppo_dagger52_20260912/repro_full_update100/initialization_fp32`.
To audit a newly trained DAgger checkpoint independently:

```bash
CHECKPOINT=ckpts/astar_pointgoal_dagger_long_v1_all24_round1_local/best.pt \
OUTPUT_DIR=experiments/local_best_pointgoal/initialization \
  bash shs/rl/prepare_best_dagger_ppo_local.sh
```

The converter checks 20 synthetic observations and external action-history
updates, then constructs disjoint 720-task training and 48-task validation
manifests. It does not train or change the actor logits.

## 3. PPO continuation

Three profiles are explicit:

- `exact` (default): two CUDA ranks and 12 Unity environments per rank;
- `local`: one automatically selected CPU/CUDA rank and four environments;
- `smoke`: one CPU rank, one environment, one update, and two rollout steps.

Only `exact` reproduces the reference batch and interaction schedule. `local`
and `smoke` test portability and must not be reported as the 75% experiment.

```bash
DRY_RUN=1 PROFILE=exact bash shs/rl/run_best_ppo_stage1_local.sh
PROFILE=smoke OUTPUT_DIR=experiments/local_best_pointgoal/stage1_smoke \
  bash shs/rl/run_best_ppo_stage1_local.sh
```

For a fresh end-to-end run, pass the newly converted actor with
`INITIAL_ACTOR=.../initial_actor.pt`. The default uses the downloaded,
hash-verified reference actor and stops at update 100.

## 4. Final independent-point PPO

The final run uses 7,199 non-benchmark training pairs, 24 calibrated reward
maps, dynamic 80-320 step budgets, the training safety shield, map-distance
progress, and KL regularization against the converted DAgger actor. There are
no teacher actions, imitation loss, replay labels, minimap policy inputs, or
canonical `input_points.json` endpoints in training.

```bash
DRY_RUN=1 PROFILE=exact bash shs/rl/run_best_ppo_local.sh

PROFILE=exact \
USE_REFERENCE_ANCHOR=0 \
INITIAL_ACTOR=experiments/local_best_pointgoal/ppo_stage1/update_000100.pt \
OUTPUT_DIR=experiments/local_best_pointgoal/ppo_stage2 \
  bash shs/rl/run_best_ppo_local.sh
```

By default, `PROFILE=exact` seeds a new output directory with the downloaded
update-141 checkpoint and its Adam state, then resumes at update 142. This is
the actual lineage of Mixed400. Set `USE_REFERENCE_ANCHOR=0` to start a fresh
stage-2 run from `INITIAL_ACTOR`; that is an algorithmic retraining recipe, not
the exact winning trajectory. Existing output directories are never replaced.

The reference run continued through update 500, but update 400 is the selected
Mixed400 checkpoint. Always evaluate immutable checkpoints independently; do
not infer benchmark success from training reward.

On macOS, use `PROFILE=local` or `PROFILE=smoke`; ML-Agents receives the local
bundle path `unity_clients/scene_all.app` and resolves its executable. Because
the rank count, environment count, hardware, and dynamic-object trajectories
differ from the reference run, local training is a functional port rather than
a bitwise or score-equivalent reproduction.
