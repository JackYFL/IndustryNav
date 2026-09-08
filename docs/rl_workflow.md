# Reinforcement Learning Baseline Workflow

This document describes the recurrent PointGoal PPO baseline, its safety-aware
reward, single-GPU training, synchronous distributed training, resume behavior,
and evaluation in IndustryNav.

## 1. Scope

The RL baseline is intended to improve a learned PointGoal policy through
closed-loop interaction with the Unity environment. It supports:

- PPO with multiple persistent Unity environments on one learner process;
- synchronous multi-GPU PPO through PyTorch DistributedDataParallel (DDP);
- optional initialization from the depth encoder of a BC checkpoint;
- safety-aware rewards based on collision and depth-warning signals;
- deterministic inference through the shared IndustryNav benchmark runner;
- checkpoint resume and held-out scene evaluation.

The implementation is under `nav/baselines/rl/`:

```text
nav/baselines/rl/
├── pointgoal_ppo.py        # Model, goal encoding, reward, and GAE
├── unity_pointgoal_env.py  # Persistent Unity environment wrapper
└── controller.py           # Deterministic benchmark inference
```

The main entry points are:

```text
nav/scripts/rl/train_pointgoal_ppo.py
nav/scripts/evaluation/evaluate_pointgoal_policy.py
shs/rl/run_pointgoal_ppo.sh
shs/rl/run_pointgoal_ddppo.sh
```

No Unity rebuild is required for the current RL workflow. Rewards and episode
termination are computed by Python from Unity observations and telemetry.

## 2. Policy Contract

### Observations

The policy receives only information available to a PointGoal agent:

| Input | Representation | Default size |
|---|---|---:|
| Unity depth | One-channel normalized encoding; converted to metric depth only for warning detection | 320 x 240 before model resize |
| Goal | `[distance / 50, sin(bearing), cos(bearing)]` | 3 |
| Previous action | Learned embedding, plus a BOS token | 16 |
| Recurrent state | GRU hidden state | 512 |

The bearing is egocentric: positive values indicate that the target is to the
agent's right. The current world pose and target world position are used to
construct this relative goal vector. RGB and minimap pixels are not policy
inputs. The minimap is used by the environment only for task initialization and
target-coordinate conversion.

### Actions

The RL policy has three actions:

```text
0  forward
1  turn right
2  turn left
```

There is deliberately no `stop` class. An episode ends automatically when the
agent enters the success radius or reaches the step limit.

The PointGoal action contract is shared with the corrected BC/DAgger policy:

| Action | Unity command | Approximate physical effect |
|---|---:|---|
| `forward` | `7.5` | 0.75 m theoretical displacement |
| `turn right` | `11.25` | about +22.5 degrees observed yaw |
| `turn left` | `-11.25` | about -22.5 degrees observed yaw |

Each policy decision advances Unity by one simulation step.

## 3. Network Architecture

```text
depth ──> 1-channel ResNet-50 ──> 256-D visual feature ─┐
                                                       │
goal ──> two-layer MLP ─────────> 64-D goal feature ───┼─> GRUCell(512)
                                                       │       │
previous action ──> embedding ──> 16-D feature ────────┘       ├─> actor logits
                                                               └─> state value
goal ──> trainable analytic PointGoal prior ───────────────────> + actor logits
```

The trainable goal prior initially favors forward motion when the target is in
front and favors the corresponding turn when it is off-axis. This avoids
starting an expensive on-policy run from a uniform random policy. The recurrent
visual branch then learns obstacle avoidance and corrections from experience.

When `--init-bc-checkpoint` is provided, only the matching BC depth encoder is
loaded. The actor, critic, GRU, goal branch, and action embedding remain PPO
components.

## 4. Environment and Task Sampling

The trainer reads JSONL records from `--manifest` and selects records matching
`--split` (`train` by default). Training scenes are assigned as follows:

1. scene names are sorted numerically;
2. DDP ranks receive disjoint scene slices;
3. each rank distributes its scenes across its local Unity environments;
4. an environment cycles through its assigned tasks;
5. Unity is reused when consecutive tasks are from the same scene and relaunched
   when the scene changes.

Use `--scene-limit` and `--episodes-per-scene` for smoke tests. The number of
selected scenes per rank must be at least `--num-envs`.

At Unity launch, each environment samples seeded dynamic-object speeds from the
same ranges used by the PointGoal data pipeline. Reused instances retain their
current scene simulation while cycling through same-scene tasks. Runtime
lighting overrides are disabled, so scene-authored lighting/exposure is
retained.

## 5. Reward and Safety Signals

For transition `t`, the default reward is:

```text
r_t = clip(d_(t-1) - d_t, -1, 1)
      - 0.01
      - 0.50 * collision_t
      - 0.05 * warning_t
      + 5.00 * success_t
      - 1.00 * timeout_t
```

The corresponding command-line flags are:

| Term | Flag | Default |
|---|---|---:|
| Progress scale | `--progress-scale` | 1.0 |
| Per-step penalty | `--step-penalty` | 0.01 |
| Success bonus | `--success-bonus` | 5.0 |
| Timeout penalty | `--timeout-penalty` | 1.0 |
| Collision penalty | `--collision-penalty` | 0.5 |
| Warning penalty | `--warning-penalty` | 0.05 |

The default success radius is 2 m and the default episode limit is 200 steps.

### Collision

Collision is a post-action physical-motion proxy. A forward action is marked as
a collision when:

```text
actual displacement < 0.95 x theoretical forward displacement
```

With the default PointGoal forward command, the theoretical displacement is
`7.5 x 0.1 m = 0.75 m`, so the collision boundary is 0.7125 m. Rotation actions
are not currently counted as collisions because this proxy is defined from
forward displacement.

### Warning

Warning is a pre-action depth-risk signal and is evaluated only before a
forward action. It uses the resolution-independent trapezoidal ROI shared with
benchmark evaluation. A warning requires at least 0.5% of valid ROI pixels to
be closer than:

```text
warning distance = 0.4 m + commanded forward distance
```

For the default forward command, this is `0.4 + 7.5 x 0.1 = 1.15 m`.

Warning and collision are complementary. Warning estimates imminent risk from
the current depth map; collision checks whether the commanded translation was
physically blocked. A depth warning is not treated as proof that a collision
occurred.

## 6. PPO Optimization

The trainer uses clipped PPO with generalized advantage estimation (GAE), a
clipped value loss, entropy regularization, gradient clipping, and truncated
backpropagation through the GRU.

| Setting | Default |
|---|---:|
| Rollout length | 32 steps/environment |
| BPTT length | 8 |
| PPO epochs | 2 |
| Chunks per minibatch | 8 |
| Learning rate | 2.5e-4 |
| Depth-encoder LR scale | 0.1 |
| PPO clip | 0.2 |
| Value-loss coefficient | 0.5 |
| Entropy coefficient | 0.01 |
| Gradient-norm limit | 0.5 |
| Discount `gamma` | 0.99 |
| GAE `lambda` | 0.95 |

Rollout tensors remain on CPU between collection and optimization to reduce GPU
memory pressure. In distributed mode, advantage moments and model gradients are
synchronized across ranks.

## 7. Single-GPU PPO

The default wrapper expects the resampled PointGoal manifest, the unified Linux
Unity client, and a corrected BC checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 \
OUTPUT_DIR=ckpts/pointgoal_ppo_resampled_v1 \
NUM_ENVS=4 \
TOTAL_UPDATES=200 \
  xvfb-run -a -s "-screen 0 1724x1024x24" \
  bash shs/rl/run_pointgoal_ppo.sh
```

On a machine with an active graphical display, `xvfb-run` may be omitted.

Run a small smoke test before a full experiment:

```bash
CUDA_VISIBLE_DEVICES=0 \
OUTPUT_DIR=ckpts/pointgoal_ppo_smoke \
NUM_ENVS=1 TOTAL_UPDATES=1 ROLLOUT_STEPS=4 \
SCENE_LIMIT=1 MAX_EPISODE_STEPS=4 \
  xvfb-run -a -s "-screen 0 1724x1024x24" \
  bash shs/rl/run_pointgoal_ppo.sh
```

The total number of collected transitions is:

```text
total updates x rollout steps x environments
```

## 8. Synchronous Distributed PPO

Use `run_pointgoal_ddppo.sh` for single-node, multi-GPU synchronous PPO:

```bash
CUDA_VISIBLE_DEVICES=4,5 \
INDUSTRYNAV_UNITY_DEVICE_IDS=4,5 \
OUTPUT_DIR=ckpts/pointgoal_ddppo_resampled_v1 \
NPROC_PER_NODE=2 \
NUM_ENVS_PER_RANK=2 \
TOTAL_UPDATES=200 \
BASE_PORT=47120 \
  xvfb-run -a -s "-screen 0 1724x1024x24" \
  bash shs/rl/run_pointgoal_ddppo.sh
```

`CUDA_VISIBLE_DEVICES` maps PyTorch local ranks to learner GPUs.
`INDUSTRYNAV_UNITY_DEVICE_IDS` contains the corresponding physical GPU indices
passed to Unity through `-force-device-index`.

Every Unity environment requires a distinct ML-Agents port. Reserve the range:

```text
BASE_PORT ... BASE_PORT + NPROC_PER_NODE x NUM_ENVS_PER_RANK - 1
```

For the command above, ports 47120-47123 must be free.

The global transitions per update are:

```text
rollout steps x environments per rank x number of ranks
```

The current implementation provides scene sharding, global advantage
normalization, DDP gradient synchronization, and rank-zero checkpoint writing.
It is best described as synchronous DDP-PPO. It does not yet implement the
straggler-aware rollout termination used by the large-scale DD-PPO paper.

## 9. Checkpoints and Resume

The output directory contains:

```text
OUTPUT_DIR/
├── latest.pt
├── update_000010.pt
├── update_000020.pt
├── metrics.jsonl
├── train_rank0.log
├── train_rank1.log       # distributed runs only
└── unity/
```

`latest.pt` contains model weights, optimizer state, model configuration,
training arguments, completed update, and global-step count. Numbered
checkpoints are written every ten updates by default.

Resume single-GPU PPO:

```bash
CUDA_VISIBLE_DEVICES=0 \
OUTPUT_DIR=ckpts/pointgoal_ppo_resampled_v1 \
TOTAL_UPDATES=400 \
  xvfb-run -a -s "-screen 0 1724x1024x24" \
  bash shs/rl/run_pointgoal_ppo.sh --resume
```

Resume distributed PPO with the same output directory and preferably the same
rank/environment layout:

```bash
CUDA_VISIBLE_DEVICES=4,5 \
INDUSTRYNAV_UNITY_DEVICE_IDS=4,5 \
OUTPUT_DIR=ckpts/pointgoal_ddppo_resampled_v1 \
NPROC_PER_NODE=2 NUM_ENVS_PER_RANK=2 TOTAL_UPDATES=400 \
BASE_PORT=47120 \
  xvfb-run -a -s "-screen 0 1724x1024x24" \
  bash shs/rl/run_pointgoal_ddppo.sh --resume
```

`TOTAL_UPDATES` is the final update number, not the number of additional
updates. Resume restores model and optimizer state, but it does not reproduce
the exact Unity process, task cursor, recurrent state, or random-number state
from the interrupted rollout. Resume therefore starts a new rollout from the
next saved update boundary.

## 10. Metrics

Rank zero appends one JSON object per update to `metrics.jsonl`:

| Field | Meaning |
|---|---|
| `update` | Completed optimizer update |
| `global_steps` | Transitions collected across all ranks |
| `mean_step_reward` | Mean reward in the current rollout |
| `episodes` | Episodes completed during the current rollout |
| `success_rate` | Success among those completed episodes |
| `mean_episode_return` | Mean completed-episode return |
| `collisions_per_episode` | Mean collision count for completed episodes |
| `warnings_per_episode` | Mean warning count for completed episodes |
| `loss` | Total PPO objective |
| `policy_loss` | Clipped actor loss |
| `value_loss` | Clipped critic loss |
| `entropy` | Mean action-distribution entropy |
| `approx_kl` | Approximate old/new policy KL diagnostic |

Per-update success can be noisy because some rollouts complete very few
episodes. Training success is also measured on training tasks. Model selection
and comparisons should use a fixed held-out evaluation set rather than the last
training row.

## 11. Held-Out Evaluation

Evaluate one task from each of four test scenes:

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m nav.scripts.evaluation.evaluate_pointgoal_policy \
  --baseline ppo \
  --manifest datasets/pointgoal_dagger_resampled_v1/collection_manifest.jsonl \
  --checkpoint ckpts/pointgoal_ppo_resampled_v1/latest.pt \
  --output-root outputs/pointgoal_ppo_eval \
  --unity clients/scene_all_24scenes_exact_pillar_markers_v14/scene_all.x86_64 \
  --scenes 4
```

Inference uses deterministic `argmax` actions while preserving the GRU state
and previous-action history. Evaluation runs through
`nav.scripts.agent.run_benchmark_cell`, so PPO produces the same action logs,
trajectories, safety metrics, and visualization inputs as other baselines.

For a fair comparison with BC and DAgger:

- use identical held-out scene/task records and motion seeds;
- use the same 2 m success threshold and dynamic step-budget settings;
- report Success@2m, Success@5m, Success@10m, collision, warning, SPL, and
  SoftSPL together;
- keep training metrics separate from held-out metrics.

## 12. Troubleshooting

### `Address already in use`

Another ML-Agents communicator is using one of the requested ports. Inspect the
entire port range, stop only the process belonging to the failed run, or choose
a new `BASE_PORT`.

### Unity exits with `SIGSEGV` before connecting

On a headless Linux server, run the wrapper under `xvfb-run`. Also avoid
starting a second experiment on the same Unity GPU while orphaned players from
a failed run are still active.

### DDP appears to hang

All ranks synchronize during optimization. If one rank fails to create a Unity
environment, the other rank may wait. Check every `train_rank*.log`, the main
launcher log, all assigned ports, and every Unity process before assuming the
learner itself is deadlocked.

### Unity and PyTorch use unexpected GPUs

Remember that PyTorch local ranks index into `CUDA_VISIBLE_DEVICES`, while Unity
needs physical device indices in `INDUSTRYNAV_UNITY_DEVICE_IDS`.

### BC initialization fails

The BC checkpoint must contain a matching `depth_encoder.*` state. Backbone
width and encoder architecture must agree with the PPO model configuration.

### Too few scenes for the requested environments

Each rank must receive at least one scene per local environment. Reduce
`NUM_ENVS`, increase `SCENE_LIMIT`, or use more training scenes.

## 13. Validation

Run the focused RL and PointGoal tests locally:

```bash
python -m unittest \
  nav.scripts.tests.test_pointgoal_ppo \
  nav.scripts.tests.test_astar_pointgoal_dataset -v
```

Check wrapper syntax:

```bash
bash -n shs/rl/run_pointgoal_ppo.sh shs/rl/run_pointgoal_ddppo.sh
```

Before committing a distributed change, run a two-rank smoke test with one
environment per rank, one update, and a short rollout.

## 14. Current Limitations

- Collision reward covers blocked forward translation, not rotation contact.
- Warning is derived from depth and cannot represent every physical collision.
- Resume occurs at an update boundary and is not an exact environment snapshot.
- A Unity crash currently terminates its rank instead of replacing that
  environment automatically.
- The distributed launcher is single-node and synchronous; it is not yet the
  full straggler-tolerant DD-PPO algorithm.
- Final checkpoint quality still requires held-out closed-loop evaluation.

References:

- [Decentralized Distributed PPO](https://arxiv.org/abs/1911.00357)
- [Habitat-Lab PointNav DD-PPO configuration](https://github.com/facebookresearch/habitat-lab/blob/main/habitat-baselines/habitat_baselines/config/pointnav/ddppo_pointnav.yaml)
