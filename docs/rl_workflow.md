# Reinforcement Learning Baseline Workflow

For the complete provenance-checked pipeline behind the best DAgger and
Mixed400 PPO checkpoints, see
[`best_pointgoal_training_pipeline.md`](best_pointgoal_training_pipeline.md).

This document describes the recurrent PointGoal PPO baseline, its safety-aware
reward, single-GPU training, synchronous distributed training, resume behavior,
and evaluation in IndustryNav.

## 1. Scope

The RL baseline is intended to improve a learned PointGoal policy through
closed-loop interaction with the Unity environment. It supports:

- PPO with multiple persistent Unity environments on one learner process;
- synchronous multi-GPU PPO through PyTorch DistributedDataParallel (DDP);
- optional initialization from the depth encoder of a BC checkpoint;
- PPO-compatible offline actor warm-up from exported A*/DAgger episodes;
- complete Transformer DAgger actor initialization with a new detached critic
  ([workflow](ppo_dagger_actor.md));
- safety-aware rewards based on collision and depth-warning signals;
- deterministic inference through the shared IndustryNav benchmark runner;
- checkpoint resume and held-out scene evaluation.

The implementation follows the shared package layers:

```text
nav/models/policies/pointgoal_actor_critic.py  # Network architecture
nav/baselines/rl/trainer.py                    # PPO/DDP-PPO optimization
nav/baselines/rl/imitation.py                  # Recurrent actor imitation warm-up
nav/baselines/rl/replay.py                     # Cross-update A* teacher replay
nav/baselines/rl/agent.py                      # Deterministic inference
nav/baselines/rl/reward.py                     # PPO reward definition
nav/envs/unity_pointgoal.py                    # Persistent Unity adapter
nav/safety/                                    # Shared warning/collision detectors
nav/train/advantages.py                        # Reusable GAE implementation
```

The main entry points are:

```text
nav/scripts/rl/train_pointgoal_ppo.py
nav/scripts/rl/pretrain_pointgoal_actor.py
nav/scripts/evaluation/evaluate_pointgoal_policy.py
shs/rl/run_pointgoal_ppo.sh
shs/rl/run_pointgoal_ddppo.sh
```

No Unity rebuild is required for the current RL workflow. Rewards and episode
termination are computed by Python from Unity observations and telemetry.

For pure-PPO experiments, `--policy-temperature` (default 1.0) optionally sharpens
the stochastic training distribution. Sampling, PPO likelihoods, entropy and
reference KL use the same temperature; deterministic evaluation is unchanged.
See the [experimental temperature contract](ppo_dagger_actor.md#experimental-stochastic-policy-temperature)
before using a non-default value. No performance gain is established by this option.

## 2. Policy Contract

### Observations

The policy receives only information available to a PointGoal agent:

| Input | Representation | Default size |
|---|---|---:|
| Unity depth | One-channel normalized encoding; converted to metric depth only for warning detection | 320 x 240 before model resize |
| Goal | `[distance / 50, sin(bearing), cos(bearing)]` | 3 |
| Action history | Last 5 actions, each with a learned embedding; BOS-padded at episode start | 5 x 16 |
| Visual history | Last 5 detached ResNet encoder features; zero-padded at episode start | 5 x 256 |
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
depth ──> ResNet-50 ──> current 256-D feature ───────────────┐
                                                            ├─> gated temporal visual feature ─┐
last 5 visual features ──> temporal GRU(256) ────────────────┘                                  │
                                                                                               ├─> main GRUCell(512)
goal ──> two-layer MLP ──> 64-D goal feature ───────────────────────────────────────────────────┤       │
last 5 actions ──> embeddings ──> 80-D feature ─────────────────────────────────────────────────┘       ├─> actor logits
                                                                                                       └─> state value
goal ──> trainable analytic PointGoal prior ───────────────────────────────────────────────────────────> + actor logits
```

Both histories are ordered from oldest to newest. Action history is reset to
BOS and visual history to zeros at every episode boundary. Historical visual
features are detached rollout caches: the temporal module learns from them,
while ResNet gradients are computed through the current frame. A zero-initialized
residual gate lets an older checkpoint initially retain its current-frame-only
behavior. The trainable goal prior initially favors forward motion when the target is in
front and favors the corresponding turn when it is off-axis. This avoids
starting an expensive on-policy run from a uniform random policy. The recurrent
visual branch then learns obstacle avoidance and corrections from experience.

For known-layout experiments, `--absolute-scene-state` optionally appends the
normalized current/target world coordinates, a 24-way scene one-hot vector,
normalized world-frame target delta, and explicit sine/cosine yaw to the
relative PointGoal. The final four planning features are appended after the
legacy 31-dimensional state, so prior checkpoints migrate with zero-initialized
new columns. This does not expose the minimap or any
`input_points.json` endpoint, and training still uses independently resampled
pairs. It gives the policy context for layout-specific detours; checkpoints
record the mode so evaluation and GIF runners construct the same state
automatically. Existing relative-only checkpoints migrate by zero-initializing
the added input columns.

`--coordinate-fourier-bands K` optionally appends `K` dyadic sine/cosine
bands for each normalized current/target coordinate. These features make
scene-specific route boundaries easier for the compact goal MLP to represent
while preserving the mapless PointGoal contract. They are appended at the end,
so zero-expanded legacy checkpoints remain numerically unchanged initially.

Older PPO checkpoints can initialize both temporal inputs: a single action is
aligned to the newest action slot, added action blocks start at zero, and the
new visual temporal branch starts behind its zero residual gate. Resume such a migration with
`--no-resume-optimizer` because the GRU parameter shape changed.

When `--init-bc-checkpoint` is provided, only the matching BC depth encoder is
loaded. The actor, critic, GRU, goal branch, and action embedding remain PPO
components. `--init-ppo-checkpoint` instead copies the complete policy/value
weights into a new run while resetting update count and optimizer state; the
two initialization modes are mutually exclusive. The shell wrappers expose
the latter through `INIT_PPO_CHECKPOINT`.

An optional DAgger/BC teacher can also supervise the PPO actor on the states
visited by the current stochastic policy. Set `--teacher-bc-checkpoint` and a
positive `--teacher-bc-coef`; the trainer adds cross-entropy to the clipped PPO
objective and aligns the teacher's action history with the action actually
executed by PPO. Guidance decays linearly over
`--teacher-bc-decay-updates`, down to `--teacher-bc-min-scale` of its initial
weight. This is policy guidance, not behavior replacement: rollouts and PPO
log-probabilities remain on-policy.

Alternatively, `--online-astar-teacher` queries the minimap A* planner for an
expert label at each state visited during training. The minimap is privileged
training information only: it is neither appended to the policy observation
nor stored as a checkpoint input, so evaluation remains depth + PointGoal.
`--teacher-action-prob` optionally executes some expert labels to expose the
learner to successful detours (DAgger-style). That probability decays to
`--teacher-action-min-prob`; mixed expert transitions train the critic and
imitation objective but are masked out of the PPO actor loss, preserving the
on-policy meaning of the remaining PPO samples. A BC checkpoint and the online
A* teacher are mutually exclusive.

Online labels can also be retained across PPO updates with
`--teacher-replay-capacity`. Each rank stores a bounded CPU replay of depth,
PointGoal, five-action/five-visual history, recurrent state, and the A* label;
the minimap itself is never stored or exposed to the policy. A small number of
`--teacher-replay-steps` after every PPO update prevents useful corrective
turns from disappearing after one rollout. `--teacher-replay-class-weight-power`
tempers inverse-frequency weighting so replay does not collapse to the more
common forward label. Replay is saved per rank beside model checkpoints and
restored by `--resume`.

For atomic PointGoal actions, `--astar-policy-forward-tolerance-deg` controls
the online teacher's heading deadband. Raising it moderately can prevent an
off-route rollout from being dominated by corrective turn labels; it does not
change policy observations or expose the minimap at test time.

Strict evaluation can optionally enable a near-goal PointGoal fallback with
`--shield-terminal-homing-distance-m`. It uses only the same goal bearing and
metric depth available to the policy: it aligns to the goal inside the chosen
radius, but refuses to override an avoidance turn with a depth-warned forward
step. The option is disabled by default and requires `--safety-shield`.

For conservative checkpoint ensembles, `--fallback-ppo-checkpoint` keeps a
second recurrent PPO policy synchronized with every executed action.
`--fallback-bc-checkpoint` provides the same recovery path for a BC/DAgger
policy; the two options are mutually exclusive. The evaluator activates the
recovery policy after `--fallback-collision-steps` cumulative physical-contact
steps. This allows a stable primary policy to handle normal motion while a
policy trained on expert corrections handles sustained off-route contact.
Both policies consume only the standard depth and PointGoal observations plus
their own synchronized history. `--fallback-stagnation-steps` can additionally
(or instead) trigger the same switch after that many consecutive steps without
improving the best goal distance.

A three-controller recovery cascade is also available for complementary
checkpoints: configure the first stage with `--fallback-ppo-checkpoint`, then
set `--second-fallback-bc-checkpoint` and a larger
`--second-fallback-collision-steps`. All three controllers remain synchronized
with the actions Unity actually executes; only the selected controller's
proposal is applied. The second threshold must exceed the first so the normal
order is primary PPO, recovery PPO, then final BC/DAgger recovery.
`--second-fallback-collision-stagnation-steps N` applies the same
collision-and-no-progress gate to that final BC/DAgger transition, allowing
the first PPO fallback to retain control while it is still making progress.
The final stage can be restricted further with
`--second-fallback-min-distance-m` and
`--second-fallback-min-step-fraction`: this avoids replacing a PPO recovery
that is already close to the target or has not yet had enough time to work.
`--second-fallback-far-distance-m` lets severe long-range failures bypass the
time gate. An optional late-budget close-recovery band is configured with
`--second-fallback-close-recovery-{min,max}-distance-m` and
`--second-fallback-close-recovery-min-step-fraction`. All of these gates use
only live PointGoal distance, collision/progress state, and the episode budget;
they do not inspect scene IDs, benchmark point IDs, or minimap pixels.

If a recovery policy tends to spin after taking control,
`--fallback-turn-streak N` enables the clear-depth forward escape only during
fallback stages. This leaves successful primary-PPO trajectories unchanged;
`--fallback-turn-escape-steps` controls the length of each escape burst.
For a PPO-to-PPO-to-BC cascade, add `--fallback-turn-final-stage-only` to keep
the first PPO fallback unchanged and apply turn-loop escape only to the final
BC/DAgger recovery stage.
`--fallback-abandon-turn-escapes N` treats repeated escape interventions as
failed recovery and permanently hands control back to the preceding
synchronized controller after `N` interventions. With
`--fallback-turn-final-stage-only`, a three-controller cascade therefore
returns from BC/DAgger to the first PPO fallback without changing the primary
PPO path or abandoning that first fallback.
For collision-heavy trajectories that are still making progress,
`--fallback-collision-stagnation-steps N` changes the collision trigger into an
AND gate: the cumulative collision threshold and `N` consecutive no-progress
steps must both be reached. The independent `--fallback-stagnation-steps`
option remains an OR trigger.

For long detours, `train_scene_waypoint.py` can learn a short-horizon world
waypoint from independent A* trajectories. The planner input is scene identity
plus current/final GPS coordinates; it predicts a local route offset and never
reads minimap pixels. Evaluation passes that learned waypoint to the recurrent
depth policy until the final goal is within the configured lookahead:

```bash
python -m nav.scripts.rl.train_scene_waypoint \
  --dataset-manifest datasets/astar_pointgoal_resampled_dagger_round1/dataset_manifest.jsonl \
  --dataset-manifest datasets/astar_pointgoal_resampled_dagger_round2/dataset_manifest.jsonl \
  --dataset-manifest datasets/astar_pointgoal_resampled_dagger_round3/dataset_manifest.jsonl \
  --output-dir ckpts/scene_waypoint_v1 \
  --lookahead-m 5

python -m nav.scripts.evaluation.evaluate_pointgoal_policy \
  ... --waypoint-planner-checkpoint ckpts/scene_waypoint_v1/best.pt
```

When a single coordinate regressor is too ambiguous at route branches, build a
topological route memory from the same independent demonstrations.  Nodes are
quantized visited GPS positions and edges are only transitions observed on a
successful A* trajectory.  The graph is not a simulator occupancy map and the
evaluator never reads minimap pixels.  It selects a graph waypoint while the
recurrent RL policy still performs all closed-loop depth-based control:

```bash
python -m nav.scripts.rl.build_route_graph \
  --dataset-manifest datasets/astar_pointgoal_resampled_dagger_round1/dataset_manifest.jsonl \
  --dataset-manifest datasets/astar_pointgoal_resampled_dagger_round2/dataset_manifest.jsonl \
  --dataset-manifest datasets/astar_pointgoal_resampled_dagger_round3/dataset_manifest.jsonl \
  --output ckpts/route_graph/graph.json \
  --grid-size-m 0.5 \
  --lookahead-m 8

python -m nav.scripts.evaluation.evaluate_pointgoal_policy \
  ... --route-graph-checkpoint ckpts/route_graph/graph.json \
  --route-graph-lookahead-m 8 \
  --route-graph-terminal-distance-m 8 \
  --route-graph-max-snap-distance-m 8
```

For a conservative hybrid, add
`--route-graph-activation-collision-steps 20`: the normal PointGoal remains
unchanged until the agent is physically stuck, after which the graph supplies
detour waypoints. This can be combined with a PPO fallback at the same
collision threshold.

Do not build either planner from `input_points.json` trajectories when that
file is the external benchmark.  Route-graph and neural-waypoint planners are
mutually exclusive in one evaluation.

### Offline recurrent actor warm-up

`pretrain_pointgoal_actor.py` trains the same `PointGoalActorCritic` used by
PPO, rather than a separate BC network. It replays complete exported
A*/DAgger episodes, constructs the same five-action and five-visual-feature
histories used online, and applies truncated BPTT to the main recurrent state.
DAgger rows use oracle `action` as the target and the actually executed
`behavior_action` to update history.

The depth encoder is frozen during this stage. This preserves the initialized
visual representation and permits batched frame encoding; the
goal/scene branch, action embeddings, visual-history GRU, main GRU, and actor
are optimized. The critic is retained and updated later by PPO. The resulting
`best.pt` and `latest.pt` use the normal PPO checkpoint schema and can be
passed directly to `--init-ppo-checkpoint` or the evaluator.

There are two mutually exclusive initialization paths. `--init-checkpoint`
loads an existing PPO-compatible actor. `--init-bc-checkpoint` transfers the
depth encoder from a BC/DAgger checkpoint, then fits the PPO recurrent actor on
the exported expert actions. The latter is the appropriate bridge when the
source policy uses an incompatible head such as a transformer: PPO training
can then start from the full imitation-trained actor instead of retaining only
the shared encoder.

```bash
python -m nav.scripts.rl.pretrain_pointgoal_actor \
  --data-root datasets/astar_pointgoal_resampled_dagger_round3 \
  --output-dir ckpts/pointgoal_actor_imitation \
  --init-checkpoint ckpts/pointgoal_ppo_v5/update_000100.pt \
  --absolute-scene-state \
  --coordinate-fourier-bands 4 \
  --epochs 12 \
  --batch-size 8
```

For a DAgger-to-pure-PPO run, first create the compatible actor and then pass
its best checkpoint to PPO without enabling any teacher options:

```bash
python -m nav.scripts.rl.pretrain_pointgoal_actor \
  --data-root datasets/astar_pointgoal_dagger_all24 \
  --output-dir ckpts/pointgoal_actor_from_dagger \
  --init-bc-checkpoint ckpts/dagger_all24/best.pt \
  --epochs 12

INIT_PPO_CHECKPOINT=ckpts/pointgoal_actor_from_dagger/best.pt \
TEACHER_BC_COEF=0 TEACHER_ACTION_PROB=0 TEACHER_REPLAY_CAPACITY=0 \
ONLINE_ASTAR_TEACHER=0 SPLIT=all SCENE_LIMIT=24 \
  bash shs/rl/run_pointgoal_ddppo.sh
```

This is one learned actor-critic during PPO and evaluation; it does not switch
between DAgger and PPO controllers.

To add a genuine on-policy DAgger round before PPO, use the PPO-compatible
actor as `INIT_PPO_CHECKPOINT` with `run_pointgoal_dagger_init.sh`. The learned
policy executes every action (`teacher-action-prob=0`), while online A* labels
exactly the states that policy visits. The actor is optimized only by the
DAgger cross-entropy objective (PPO policy, value, and entropy losses are zero
in this stage); the final checkpoint is still a normal
`PointGoalActorCritic` checkpoint and can initialize pure PPO directly.

```bash
INIT_PPO_CHECKPOINT=ckpts/pointgoal_actor_from_dagger/best.pt \
MANIFEST=datasets/pointgoal_dagger_dense_v2/collection_manifest.jsonl \
SPLIT=all SCENE_LIMIT=24 \
OUTPUT_DIR=ckpts/pointgoal_dagger_init_all24 \
  bash shs/rl/run_pointgoal_dagger_init.sh
```

`run_pointgoal_ppo_after_dagger.sh` can wait for that job, verify the expected
final update, and then start the pure-PPO stage. It force-disables every
teacher and A* option before launching PPO, so an incomplete DAgger run cannot
silently seed the next stage.

Use only independently resampled exports when `input_points.json` is the
external holdout. `--episode-limit 2 --epochs 1 --num-workers 0` provides a
real-data smoke test; `--resume` continues from `latest.pt` with the same model
configuration.

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
When `input_points.json` is the external held-out benchmark, `--split all` may
be used to train on every split of a separately resampled manifest. This covers
all 24 scene layouts without using any benchmark endpoint; the resampled
dataset spec records the benchmark exclusion hash and clearance constraints.

At Unity launch, each environment samples seeded dynamic-object speeds from the
same ranges used by the PointGoal data pipeline. By default, point pairs are
drawn without replacement from a shuffled per-scene bag. The sampler randomly
changes scene after four episodes, which mixes short and long tasks throughout
training while amortizing Unity scene relaunches. Use `--task-sampling
sequential` only for deterministic debugging. Runtime lighting overrides are
disabled, so scene-authored lighting/exposure is retained.

Unity launch failures are retried within a task. If all launch attempts fail,
the random sampler abandons that scene block and tries another task instead of
leaving the other DDP rank blocked at a collective. Retry counts are controlled
by `INDUSTRYNAV_UNITY_LAUNCH_ATTEMPTS` and
`INDUSTRYNAV_TASK_RESET_ATTEMPTS`.

## 5. Reward and Safety Signals

For transition `t`, the default reward is:

```text
r_t = 1.50 * clip(d_(t-1) - d_t, -1, 1)
      - 0.01
      - 1.00 * collision_onset_t
      - 0.00 * collision_step_t
      - 0.02 * warning_t
      - 0.005 * rotation_t
      - 0.05 * turn_reversal_t
      - 0.02 * stagnant_t
      + 0.02 * safe_forward_t
      + 10.00 * success_t
      - 2.00 * timeout_t
      - 2.00 * stuck_recovery_t
```

`turn_reversal` detects consecutive left/right or right/left actions. A step is
`stagnant` after 12 consecutive steps fail to improve the episode's best goal
distance by at least 0.05 m. `safe_forward` requires a forward action with
neither a pre-action warning nor a post-action collision. These terms prevent
the policy from lowering its collision cost by oscillating in place.
`collision_onset` is true only on the first step of a contiguous physical
contact, so one blocked motion does not contribute hundreds of full collision
penalties. Raw collision steps and independent collision events are both kept
in training metrics. `--collision-step-penalty` can add a smaller cost to each
blocked forward step when contact loops remain common; it is separate from the
larger onset penalty so it can be tuned without overwhelming progress reward.

With the training-only online A* teacher enabled,
`--geodesic-progress-reward` replaces straight-line progress with progress
along the current feasible A* path. This removes the misleading negative
reward that a policy otherwise receives while correctly detouring around an
obstacle. The minimap and A* path remain privileged training signals and are
not policy inputs or checkpoint dependencies. In this mode, stagnation and
stuck recovery use the same path-progress signal, so a necessary detour is not
terminated merely because its Euclidean goal distance temporarily increases.

For **regularized PPO without any action teacher**, use `--navigation-map-dir`
instead. This computes approximate shortest-path progress on a fixed calibrated
occupancy map and tracks stagnation with the same distance. The new map provider
is independent of the older online-teacher switch; the two reward modes are
exclusive. Missing map coverage receives zero progress and is recorded instead
of silently reverting to straight-line reward. Only the relative point goal,
depth features, and action/state history enter the actor. See
[independent-point PPO](ppo_dagger_actor.md#independent-point-map-reward-ppo)
for sampling exclusions, physical validation, reference-policy KL, concurrent
environment steps, and the fixed 96-point / independent-validation protocol.

An episode that fails to improve its best goal distance for 48 consecutive
steps is ended as a failed stuck-recovery episode and immediately resampled.
This prevents long collision or rotation loops from consuming the remaining
episode budget while keeping PPO strictly on-policy; no heuristic action is
substituted for the policy output.

The corresponding command-line flags are:

| Term | Flag | Default |
|---|---|---:|
| Progress scale | `--progress-scale` | 1.5 |
| A*-path progress | `--geodesic-progress-reward` | off |
| Per-step penalty | `--step-penalty` | 0.01 |
| Success bonus | `--success-bonus` | 10.0 |
| Timeout penalty | `--timeout-penalty` | 2.0 |
| Collision penalty | `--collision-penalty` | 1.0 |
| Persistent collision-step penalty | `--collision-step-penalty` | 0.0 |
| Warning penalty | `--warning-penalty` | 0.02 |
| Rotation penalty | `--rotation-penalty` | 0.005 |
| Turn-reversal penalty | `--turn-reversal-penalty` | 0.05 |
| Stagnation penalty | `--stagnation-penalty` | 0.02 |
| Stagnation onset | `--stagnation-start-steps` | 12 |
| Stagnation progress epsilon | `--stagnation-progress-epsilon-m` | 0.05 m |
| Safe-forward bonus | `--safe-forward-bonus` | 0.02 |
| Stuck-recovery onset | `--stuck-recovery-steps` | 48 steps |
| Stuck-recovery penalty | `--stuck-recovery-penalty` | 2.0 |

For mixed short/long-distance training without using benchmark
`input_points.json`, combine the independently resampled manifests:

```bash
EXTRA_MANIFEST=datasets/pointgoal_dagger_long_v1/collection_manifest.jsonl \
  bash shs/rl/run_pointgoal_ppo.sh
```

The primary resampled training split has a 17.1 m median Euclidean distance;
the long-distance split has a 32.8 m median. When changing reward or
curriculum during a resume, pass `--no-resume-optimizer` so the new learning
rate and a fresh optimizer state take effect while model weights are retained.

The default success radius is 2 m and the default episode limit is 200 steps.
Training defaults to CUDA. For a minimal single-process local integration test,
pass `--device cpu`; CPU mode is not intended for full training, and DDP remains
CUDA-only.

Unity worker initialization is retried up to three times when scene setup or
sensor warm-up fails transiently. Set `INDUSTRYNAV_UNITY_LAUNCH_ATTEMPTS` to a
different positive integer when needed; exhausting the limit remains a hard
failure.

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
| Action-history length | 5 |
| Visual-feature-history length | 5 |
| BPTT length | 8 |
| PPO epochs | 2 |
| Chunks per minibatch | 8 |
| Critic/base learning rate | 1.0e-4 |
| Depth-encoder LR scale | 0.1 |
| Actor/temporal LR scale | 2.5 |
| PPO clip | 0.2 |
| Value-loss coefficient | 0.5 |
| Entropy coefficient | 0.01 |
| Gradient-norm limit | 0.5 |
| Discount `gamma` | 0.99 |
| GAE `lambda` | 0.95 |
| BC-teacher coefficient | 0 (disabled) |
| BC-teacher decay | 200 updates |
| BC-teacher minimum scale | 0.1 |
| Teacher action probability | 0 (disabled) |
| Teacher action decay | 100 updates |
| Teacher action minimum probability | 0 |

Rollout tensors remain on CPU between collection and optimization to reduce GPU
memory pressure. In distributed mode, advantage moments and model gradients are
synchronized across ranks.
The resulting default rates are `1e-5` for the BC-initialized depth encoder,
`2.5e-4` for actor/recurrent/goal/action-history/visual-history parameters, and
`1e-4` for the value head.

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
| `teacher_loss` | Auxiliary BC-teacher cross-entropy (zero when disabled) |
| `teacher_agreement` | PPO/teacher argmax agreement on rollout states |
| `teacher_coefficient` | Effective decayed teacher-loss coefficient |
| `teacher_action_probability` | Scheduled probability of executing the teacher action |
| `teacher_action_fraction` | Actual rollout fraction executed by the teacher |
| `teacher_source` | `bc`, `astar`, or null |
| `policy_argmax_action_rates` | Deterministic forward/right/left proposal fractions |
| `executed_action_rates` | Sampled or teacher-mixed forward/right/left fractions |
| `teacher_action_rates` | Teacher forward/right/left label fractions |
| `teacher_class_weights` | Effective per-action imitation weights, or null |
| `teacher_replay_size_per_rank` | Retained labeled states on each rank |
| `teacher_replay_loss` | Cross-update A* replay cross-entropy |
| `teacher_replay_agreement` | Policy/replayed-A* argmax agreement |
| `teacher_replay_class_weights` | Soft inverse-frequency replay weights |

For A* datasets dominated by forward labels, enable
`--teacher-class-balance` (or `TEACHER_CLASS_BALANCE=1` in the DDP wrapper).
It applies bounded inverse-frequency weights within each rollout so turn
errors cannot be hidden by a high aggregate agreement on forward actions.
During an imitation-heavy recovery phase, `--policy-loss-coef` can reduce the
PPO actor gradient without disabling the critic or the on-policy rollout. Its
default remains 1.0; the coefficient is an ablation control, not a policy
input.
For pure on-policy DAgger, `--rollout-action-mode argmax` makes collection
match deterministic evaluation. It is accepted only with
`--policy-loss-coef 0`, because deterministic argmax trajectories are not
valid samples for a PPO actor update; A* cross-entropy training remains valid.

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

For the canonical 96-point benchmark, load each model only once per worker,
restart Unity for every task so moving-object state is isolated, use the shared
dynamic budget `ceil(2.2 * direct_distance_m + 60)` clamped to 80--320, and
resume completed worker records after interruptions:

```bash
CUDA_VISIBLE_DEVICES=1 \
xvfb-run -a -s "-screen 0 1724x1024x24" \
python -m nav.scripts.evaluation.evaluate_pointgoal_policy \
  --baseline ppo \
  --input-points input_points.json \
  --checkpoint ckpts/pointgoal_ppo_resampled_v1/latest.pt \
  --output-root analysis/rl_input_points_eval/ppo_seed0 \
  --unity clients/scene_all_24scenes_exact_pillar_markers_v14/scene_all.x86_64 \
  --device cuda --eval-seed 0 \
  --scenes 24 --episodes-per-scene 4 --workers 4 \
  --persistent-policy --resume
```

`--safety-shield` optionally starts recovery only after eight consecutive
physically blocked forward steps. `--shield-recovery-turns` controls its
consistent clearance turn (default two). Optionally,
`--shield-recovery-forward-steps` follows it with a short escape translation;
those forward steps run only while the depth warning ROI is clear. The shield
uses no minimap and never blocks ordinary forward actions outside recovery.
When it overrides PPO or BC, the controller's action history is updated with
the action Unity actually executed.

For a proactive depth-only ablation, `--shield-warning-streak N` starts the
same clearance recovery after `N` consecutive warned forward proposals,
before contact. `--shield-warning-turns` controls its turn duration. The
historical behavior remains the default (`N=0`), and every episode records its
`proactive_warning_recoveries` count.

`--shield-turn-streak N` optionally breaks a deterministic turn loop after
`N` consecutive executed turns by requesting up to
`--shield-turn-escape-steps` forward actions. Each forced translation is
executed only while the forward warning ROI is clear; collision recovery has
higher priority. The default `N=0` disables this ablation.

Inference uses deterministic `argmax` actions while preserving the main GRU
state plus explicit five-action and five-visual-feature histories. The standard
mode runs through `nav.scripts.agent.run_benchmark_cell` and retains full action
logs, trajectories, and frames. Persistent mode is the faster checkpoint-
selection path and writes compact per-task success, distance, step-budget,
collision, warning, and shield-intervention records.

Use `--model-id` for full-frame evaluations that will be exported to a shared
gallery, so PPO, PPO-architecture DAgger, and later fine-tuned variants retain
distinct cards. `shs/gallery/export_pointgoal_dagger_after_training.sh` waits
for a requested DAgger update, evaluates all 96 `input_points.json` tasks, and
builds both an annotated GIF gallery and a top-down trajectory gallery.

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
- Mid-episode Unity process crashes still terminate the affected rank; failed
  task initialization is retried and skipped automatically.
- The distributed launcher is single-node and synchronous; it is not yet the
  full straggler-tolerant DD-PPO algorithm.
- Final checkpoint quality still requires held-out closed-loop evaluation.

References:

- [Decentralized Distributed PPO](https://arxiv.org/abs/1911.00357)
- [Habitat-Lab PointNav DD-PPO configuration](https://github.com/facebookresearch/habitat-lab/blob/main/habitat-baselines/habitat_baselines/config/pointnav/ddppo_pointnav.yaml)
