# Behavior Cloning Workflow

This document describes the current behavior-cloning (BC) path: human data collection, local training, and benchmark inference.

## Overview

BC is a supervised navigation baseline. It learns to imitate keyboard-controlled trajectories collected in the Unity runtime.

The pipeline is:

```text
human teleop episodes
    -> collect_data output folders
    -> nav.scripts.bc.train_bc checkpoint
    -> BASELINE=bc benchmark inference
```

Relevant code:

- `nav/scripts/bc/collect_data.py`: interactive keyboard data collection.
- `nav/train/dataset.py`: parses collected episodes into BC samples.
- `nav/scripts/bc/train_bc.py`: CLI training entry.
- `shs/bc/train_bc.sh`: thin shell wrapper around `nav.scripts.bc.train_bc`.
- `nav/train/controller.py`: inference-time controller used by `BASELINE=bc`.
- `nav/scripts/agent/run_benchmark_cell.py`: unified benchmark entry for LLM, A*, BC, and random baselines.

## 1. Data Collection

BC training consumes human-controlled episodes saved by:

```bash
# Set SCENE_ALL_APP to the local Unity runtime executable before running.
python -m nav.scripts.bc.collect_data \
  --file_name "$SCENE_ALL_APP" \
  --scene_id 0 \
  --frame_save_dir collect_data/scene1/point1 \
  --max_steps 100 \
  --ego_width 512 \
  --ego_height 512 \
  --minimap_width 862 \
  --dynamic_objects moving \
  --human_speed_min_mps 0.9 \
  --human_speed_max_mps 1.4 \
  --vehicle_speed_min_mps 2.0 \
  --vehicle_speed_max_mps 3.5 \
  --robot_speed_min_mps 1.0 \
  --robot_speed_max_mps 2.0 \
  --motion_random_seed 42 \
  --light_intensity_min 0.7 \
  --light_intensity_max 1.3 \
  --light_random_seed 42 \
  --modalities ego,minimap,depth \
  --marker_source vector
```

Set `--ego_width` and `--ego_height` to collect egocentric RGB and depth frames
at a different resolution. Set either `--minimap_width` or `--minimap_height`;
the missing dimension is derived automatically from the canonical `862:512`
aspect ratio. Canonical minimap coordinates and image-space parameters are
scaled to the selected runtime size automatically.
The Unity client must be rebuilt from the current project for these arguments
to take effect.

Use `--dynamic_objects static` to collect or evaluate a static-environment
variant. The navigation agent remains controllable; only environment motion is
frozen. Keep this value consistent between data collection and BC inference.

PointGoal A* collection and closed-loop policy evaluation both use the
scene-authored Unity lighting and exposure on macOS and Linux. Do not pass the
runtime `--light_*` overrides for these runs: fixed HDRP exposure can saturate
the minimap while darkening the egocentric camera. Photometric variation for
PointGoal training is applied by the dataset augmentation pipeline instead.

Human, vehicle, and robot speeds are controlled independently in meters/second.
Use `--human_speed_mps`, `--vehicle_speed_mps`, and `--robot_speed_mps` for fixed
values, or the corresponding `--*_speed_min_mps`/`--*_speed_max_mps` pairs for
reproducible variation. Use the same policy during collection and inference
when evaluating BC under matched dynamics.

During collection, use the OpenCV control window to drive the agent. The collector writes per-frame observations and the action log when the episode exits.

Each episode directory must contain:

```text
collect_data/
└── scene1/
    └── point1/
        ├── keyboard_actions.csv
        ├── keyboard_fp/
        │   ├── 0.png
        │   ├── 1.png
        │   └── ...
        ├── keyboard_depth/
        │   ├── 0.png
        │   ├── 1.png
        │   └── ...
        └── keyboard_minimap/        # useful for inspection; not required by current BC dataset
```

The training dataset discovers episodes by walking:

```text
<data_root>/<scene_code>/<point_id>/
```

An episode is usable only if it has:

- `keyboard_actions.csv`
- `keyboard_fp/`
- `keyboard_depth/` when `use_depth=True`

The important CSV columns are:

- `step`
- `action`
- `curr_world_x`, `curr_world_z`
- `curr_direction_y`
- `target_world_x`, `target_world_z`
- `distance_world`

Valid action labels are defined in `nav.config.BC_ACTION_TO_LABEL`:

```text
forward -> 0
stop -> 1
turn right -> 2
turn left -> 3
```

## 2. Training

The recommended training wrapper is:

```bash
bash shs/bc/train_bc.sh resnet50 --data_root collect_data --epochs 20
```

The first argument selects a preset:

```text
cnn
resnet50
dinov2
```

These presets live in `nav.config.BC_BASE_PRESETS`. Any explicit CLI flag overrides the preset.

Common examples:

```bash
# Default resnet50 transformer preset
bash shs/bc/train_bc.sh resnet50 \
  --data_root collect_data \
  --output_dir outputs/nav_bc_resnet50 \
  --epochs 20

# Small quick check
bash shs/bc/train_bc.sh resnet50 \
  --data_root collect_data \
  --output_dir outputs/nav_bc_debug \
  --epochs 1 \
  --num_workers 0

# Train without depth if the dataset lacks keyboard_depth/
bash shs/bc/train_bc.sh resnet50 \
  --data_root collect_data \
  --no-use_depth \
  --use_rgb
```

Training writes:

```text
outputs/nav_bc_resnet50/
├── config.json
├── best.pt
├── last.pt
└── metrics.json
```

`best.pt` is selected by validation accuracy. `last.pt` is the latest epoch. `config.json` is also embedded in the checkpoint and is used by inference to reconstruct the model.

### Preset Notes

The current `resnet50` preset is a transformer sequence policy:

```text
policy_type=transformer
seq_len=28
chunk_size=4
batch_size=4
use_depth=True
use_rgb=False
```

That means inference uses depth observations and a rolling goal/action history. If you train a checkpoint with `use_rgb=True`, make sure inference saves/provides ego RGB observations as well.

For the A* point-goal dataset, the improved training wrapper uses a one-step
action head, ImageNet initialization for the one-channel depth ResNet, normalized
polar goals, left/right mirror augmentation, and softened class balancing:

```bash
bash shs/bc/run_improved_pointgoal_train.sh
```

Set `TRAIN_EPOCHS`, `TRAIN_OUTPUT`, or `EVAL_OUTPUT` to override its defaults.
The wrapper validates the exported expert dataset before training and evaluates
one episode from each held-out scene after training.

New checkpoints use `goal_encoding=unity_egocentric_v2`: Unity yaw zero faces
world `+Z`, Cartesian goals are `[forward, right]`, and polar goals are
`[distance, signed bearing]` with positive bearing to the right. The encoding is
stored in the checkpoint so training and deployment cannot silently diverge;
untagged historical checkpoints retain their original `legacy_v1` transform.

For the expanded closed-loop-oriented run, use:

```bash
bash shs/bc/run_pointgoal_v3_pipeline.sh
```

This reuses existing canonical demonstrations, adds cross-route and additional
heading variants, corrupts a small fraction of previous-action tokens during
training, adds a goal-to-action residual and turn-direction loss, and selects
the checkpoint by macro navigation accuracy instead of forward-dominated
overall accuracy.

Historical v1-v4 checkpoints used the legacy goal transform and should not be
mixed with the corrected transform. Retrain and evaluate a corrected checkpoint
without recollecting the A* dataset with:

```bash
bash shs/bc/run_pointgoal_v5_train.sh
```

### DAgger recovery rounds

Pure A* demonstrations contain only states visited by the expert. A DAgger
round instead rolls out the current policy, queries A* at every visited state,
and stores the current RGB/depth/goal observation with the A* corrective action.
The behavior action is sampled from A* with probability `beta`; both controller
histories are updated with the action actually executed in Unity.

Only `split=train` is collected. Scene17-20 validation and scene21-24 testing
remain untouched, so DAgger cannot leak held-out layouts. Failed behavior
rollouts are useful and are exported as long as they contain valid A* labels.

Run one balanced round after the corrected v5 checkpoint is available:

```bash
BETA=0.5 ROUND_ID=round1 EPISODES_PER_SCENE=4 \
  bash shs/bc/run_pointgoal_dagger_round.sh
```

The wrapper collects resumably, exports oracle labels, creates a lightweight
symlink aggregate with the original expert dataset, fine-tunes from the input
checkpoint, and evaluates four held-out scenes. For later rounds, use a decay
such as `0.5 -> 0.25 -> 0.1` and point `INIT_CHECKPOINT` and `BASE_DATA_ROOT` to
the previous round. The A* minimap remains privileged supervision and is not a
policy input.

PPO and distributed PPO now have a dedicated guide. See
[`docs/rl_workflow.md`](rl_workflow.md) for architecture, safety rewards,
training, resume, evaluation, and troubleshooting.

## 3. Inference

BC inference uses the same benchmark runner as the other baselines:

```bash
BASELINE=bc bash shs/agent/run_headless_benchmark.sh scene1
```

By default, `run_benchmark_cell.py` looks for:

```text
ckpts/nav_bc_resnet50_causal_transformer_depth_aug_remove_stop_seq_32_bs4_num_layers3/best.pt
```

For a locally trained checkpoint, pass `--bc_ckpt` through the Python entry directly:

```bash
python -m nav.scripts.agent.run_benchmark_cell \
  --baseline bc \
  --file_name auto \
  --scene_id 0 \
  --scene_name scene1 \
  --point_id point1 \
  --max_steps 70 \
  --frame_save_dir outputs/scene1/point1/bc \
  --bc_ckpt outputs/nav_bc_resnet50/best.pt \
  --init_world_x 31.0 \
  --init_world_z 49.63 \
  --init_curr_direction 180 \
  --target_x 550 \
  --target_y 450
```

Useful BC inference flags:

```text
--bc_ckpt       checkpoint path
--bc_device     auto | cpu | cuda | cuda:0
--bc_seq_len    optional sequence-length override; 0 means use checkpoint config
```

The inference controller loads the checkpoint through `BCNavController`, keeps a rolling sequence of observations, predicts one of:

```text
forward
stop
turn right
turn left
```

and passes that action into the same Unity action execution path used by the other baselines.

## Debugging Checklist

If training fails with "No episodes found":

- Check that `--data_root` points to the parent folder containing `scene*/point*/`.
- Check that each episode has `keyboard_actions.csv`.
- Check that `keyboard_fp/` exists.
- If `use_depth=True`, check that `keyboard_depth/` exists.

If training fails with image or step mismatches:

- Confirm frame filenames are numeric step IDs like `0.png`, `1.png`, ...
- Confirm CSV `step` values match those filenames.
- Confirm every action string is one of `forward`, `stop`, `turn right`, `turn left`.

If inference fails while loading the checkpoint:

- Confirm `best.pt` contains a `model` state dict.
- Confirm `config.json` exists next to the checkpoint or is embedded in the checkpoint.
- Use `--bc_device cpu` to rule out CUDA/MPS issues.

If inference always stops or turns in place:

- Confirm `target_world_x/target_world_z` were present during collection.
- Confirm the inference scene/task distribution matches the collected data.
- Inspect `bc_actions.csv` under the output folder.
- Compare depth/RGB modality settings between training config and inference checkpoint.
