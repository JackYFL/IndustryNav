# Behavior Cloning Workflow

This document describes the current behavior-cloning (BC) path: human data collection, local training, and benchmark inference.

## Overview

BC is a supervised navigation baseline. It learns to imitate keyboard-controlled trajectories collected in the Unity runtime.

The pipeline is:

```text
human teleop episodes
    -> collect_data output folders
    -> nav.scripts.train_bc checkpoint
    -> BASELINE=bc benchmark inference
```

Relevant code:

- `nav/scripts/collect_data.py`: interactive keyboard data collection.
- `nav/train/dataset.py`: parses collected episodes into BC samples.
- `nav/scripts/train_bc.py`: CLI training entry.
- `shs/train_bc.sh`: thin shell wrapper around `nav.scripts.train_bc`.
- `nav/train/controller.py`: inference-time controller used by `BASELINE=bc`.
- `nav/scripts/run_benchmark_cell.py`: unified benchmark entry for LLM, A*, BC, and random baselines.

## 1. Data Collection

BC training consumes human-controlled episodes saved by:

```bash
# Set SCENE_ALL_APP to the local Unity runtime executable before running.
python -m nav.scripts.collect_data \
  --file_name "$SCENE_ALL_APP" \
  --scene_id 1 \
  --frame_save_dir collect_data/scene1/point1 \
  --max_steps 100 \
  --modalities ego,minimap,depth \
  --marker_source vector
```

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
bash shs/train_bc.sh resnet50 --data_root collect_data --epochs 20
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
bash shs/train_bc.sh resnet50 \
  --data_root collect_data \
  --output_dir outputs/nav_bc_resnet50 \
  --epochs 20

# Small quick check
bash shs/train_bc.sh resnet50 \
  --data_root collect_data \
  --output_dir outputs/nav_bc_debug \
  --epochs 1 \
  --num_workers 0

# Train without depth if the dataset lacks keyboard_depth/
bash shs/train_bc.sh resnet50 \
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

## 3. Inference

BC inference uses the same benchmark runner as the other baselines:

```bash
BASELINE=bc bash shs/run_headless_benchmark.sh scene1
```

By default, `run_benchmark_cell.py` looks for:

```text
ckpts/nav_bc_resnet50_causal_transformer_depth_aug_remove_stop_seq_32_bs4_num_layers3/best.pt
```

For a locally trained checkpoint, pass `--bc_ckpt` through the Python entry directly:

```bash
python -m nav.scripts.run_benchmark_cell \
  --baseline bc \
  --file_name auto \
  --scene_id 1 \
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
