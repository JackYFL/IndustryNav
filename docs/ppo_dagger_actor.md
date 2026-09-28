# PPO from the complete DAgger Transformer

The DAgger evaluation with **50/96 (52.08%) Success@2m** used
`astar_pointgoal_dagger_long_v1_all24_round1/best.pt`: a depth-only ResNet-50,
12-frame, three-layer Transformer actor. It is not a GRU PPO checkpoint.

`DaggerTransformerActorCritic` retains all actor weights, the original polar
point goal, BOS/first-frame padding, and executed-action history. It adds a
value head to the last Transformer feature. The depth encoder is frozen and
its features are cached; goal/action embeddings and the Transformer actor
remain trainable. Value gradients are detached from the actor. Dropout and
BatchNorm stay in evaluation mode during both sampling and PPO updates.

The implementation reuses the existing PPO trainer, GAE, Unity environment,
checkpoint format, inference controller, and benchmark evaluator. PPO samples
actions stochastically; the shield is applied during evaluation. Training has
no expert action replacement or imitation objective.

With `--training-safety-shield`, rollouts also use the benchmark's fixed
contact/warning/turn-escape state machine. This is a fixed action mapping in
the environment, not expert imitation. PPO stores the policy's proposed
action and its log-probability for optimization, while action history,
physical rewards, and executed-action statistics use the shield's actual
action. `shield_intervention_fraction` measures its rollout use. The default
is disabled to preserve earlier experiments. Tests compare the new training
wrapper against 1,500 transitions through the live, unchanged evaluator's
shield branches and exercise the complete trainer with forced overrides.

## Initialization and task audit

```bash
python -m nav.scripts.rl.prepare_dagger_ppo \
  --checkpoint ckpts/astar_pointgoal_dagger_long_v1_all24_round1/best.pt \
  --manifest datasets/pointgoal_dagger_resampled_v1/collection_manifest.jsonl \
  --manifest datasets/pointgoal_dagger_long_v1/collection_manifest.jsonl \
  --manifest datasets/pointgoal_dagger_dense_v2/collection_manifest.jsonl \
  --input-points input_points.json --output-dir initialization --device cuda
```

This verifies the original and converted actors over 20 observations,
including resets and external action overrides. It records checkpoint and
manifest hashes, rejects starts within 3 m of benchmark starts and targets
within 20 canonical pixels of benchmark targets, removes duplicate pairs,
and reserves two validation pairs per scene. The September 12 audit yielded
720 training pairs and 48 validation pairs across all 24 scenes.

## Training

```bash
CUDA_VISIBLE_DEVICES=0,1 INDUSTRYNAV_UNITY_DEVICE_IDS=0,1 \
INITIAL_ACTOR=initialization/initial_actor.pt \
MANIFEST=initialization/train_manifest.jsonl \
UNITY_CLIENT=/path/to/scene_all.x86_64 \
OUTPUT_DIR=ckpts/ppo_from_full_dagger_actor \
xvfb-run -a bash shs/rl/run_pointgoal_ppo_dagger_actor.sh
```

Defaults use two GPUs, 12 environments per rank, a frozen encoder, actor LR
`1e-6`, critic LR `1e-4`, four critic-only warm-up updates, PPO clip `0.1`,
and KL early stopping at `0.01`. Add `--resume` to continue from `latest.pt`.
Since the Transformer reconstructs its complete window from fixed encoder
features and raw goal/action history, one-step minibatch chunks still retain
all 12 observations. Each immutable snapshot can be evaluated independently.

`--freeze-dagger-transformer` enables a more conservative fine-tuning mode:
the original Transformer, goal/action embeddings, and goal-residual branch
remain fixed; PPO updates only the action head and critic. The checkpoint
records this setting. This is still a single PPO policy initialized from
DAgger, with no online expert actions or imitation loss.

`--dynamic-step-budget` also enables distance-dependent limits during
training: `clip(ceil(2.2 * initial_distance_m + 60), 80, 320)` with the
benchmark's default coefficients. Without this flag, training retains the
fixed `--max-episode-steps` limit for backward compatibility. This setting
only changes training episodes; benchmark budgets remain unchanged.

## Independent-point, map-reward PPO

The independent-point workflow samples new endpoints from calibrated scene
occupancy maps, not the interiors of previously successful A* trajectories:

```bash
python -m nav.scripts.tools.capture_navigation_maps \
  --manifest initialization/train_manifest.jsonl --unity /path/to/scene_all.x86_64 \
  --output-dir navigation_maps
python -m nav.scripts.tools.sample_independent_pointgoal_pairs \
  --map-dir navigation_maps --input-points input_points.json \
  --output-dir independent_candidates --train-per-scene 300 --val-per-scene 12
python -m nav.scripts.tools.validate_independent_pointgoal_pairs \
  --manifest independent_candidates/train_manifest.jsonl \
  --manifest independent_candidates/val_manifest.jsonl \
  --map-dir navigation_maps --unity /path/to/scene_all.x86_64 --output-dir physical_checks
python -m nav.scripts.tools.finalize_independent_pointgoal_dataset \
  --source-dir independent_candidates --physics-dir physical_checks \
  --output-dir independent_final
python -m nav.scripts.tools.refine_reward_navigation_maps \
  --source-dir navigation_maps --output-dir reward_maps
```

Every scene/split has balanced short, medium, long, and detour categories;
headings are random. Canonical starts **and** goals are excluded in **both**
roles in world coordinates. Default sampling clearances are 4m from canonical
endpoints and 3m between train/validation endpoints, including a 1m margin for
bounded runtime placement/calibration differences. Finalization rejects only
invalid physical spawn placement, never policy failures or collision outcomes.
The fixed quick validation set has one task per difficulty per scene; full
validation has twelve tasks per scene. These are distinct from the canonical 96.

For a later distance-matched curriculum, retain the **same** validation cohort
instead of resampling an easier one. The sampler supports a minimum training
distance and distance quotas independently of the detour label:

```bash
python -m nav.scripts.tools.sample_independent_pointgoal_pairs \
  --map-dir navigation_maps --input-points input_points.json \
  --output-dir long_candidates --train-per-scene 240 --val-per-scene 12 \
  --reserved-validation-manifest independent_final/val_manifest.jsonl \
  --excluded-training-manifest independent_final/train_manifest.jsonl \
  --training-min-distance-m 30 --training-long-fraction 0.6 \
  --training-episode-prefix long_train --seed 20260913
python -m nav.scripts.tools.validate_independent_pointgoal_pairs \
  --manifest long_candidates/train_manifest.jsonl --map-dir reward_maps \
  --unity /path/to/scene_all.x86_64 --output-dir long_physical_checks
python -m nav.scripts.tools.finalize_independent_pointgoal_dataset \
  --source-dir long_candidates --physics-dir long_physical_checks \
  --reuse-validation-dir independent_final --validation-physics-dir physical_checks \
  --output-dir long_final
```

In this example every scene has 96 pairs in [30,40)m and 144 pairs in [40,70]m;
detour cases may occur in either distance group. Existing unordered training
pairs are excluded (`--excluded-training-manifest` can be repeated for several
prior datasets, whose individual hashes are recorded), and both roles of the reserved validation endpoints retain
the configured separation. Reused validation files must match the original
finalized dataset byte-for-byte, and prior physics evidence hashes are checked.
No point is filtered by policy success, collision, warning, or map coverage.

For a training-only near-obstacle start stratum, optionally add
`--training-start-clearance-m 0.25 0.75`. This selects start cells whose distance
to occupied cells on the **already inflated** sampling map lies in that meter
range. It is an approximate grid-center distance scaled by the smallest affine
cell scale, not physical body clearance or distance to the original obstacle
surface. Map inflation is recorded separately. The option is disabled by
default, preserving the original sampler and seeded selection. Unless the
separate goal option below is set, goals keep their original endpoint pool;
validation always does. Canonical cross-role exclusions,
reserved validation separation, prior-pair exclusion, distance quotas and step
budget checks remain active. An empty or unachievable stratum fails without
relaxing its constraints. Newly sampled tasks still require the same Unity
placement/depth checks before training; forward collisions are retained.

For the analogous training-goal stratum, use
`--training-goal-clearance-m 0.25 0.75`. It is independently opt-in and leaves
the default start pool unchanged; both endpoint options can also be combined.
The same approximate, already-inflated grid-distance caveat applies. Goal
pixels must still project back to the selected grid cell, and both actual
projected goals and start cells retain all benchmark/validation separation
checks. Unattainable ranges fail rather than widening the band. Validation
records are not resampled or moved by either option. Newly sampled goal pairs
remain candidates until Unity placement/depth validation passes; never append
them to a running experiment's frozen manifest.

`--navigation-map-dir` enables training-only occupancy shortest-path distances
and consistent best-distance stagnation tracking, without enabling an A* teacher.
Sampling retains conservative 0.45m clearance; reward maps use a finer 0.3m grid
and 0.1m inflation to cover physical states near obstacle boundaries. Distances
cannot cut blocked corners; local snapping is bounded in **meters** and may not
prefer a farther reachable component over a closer unreachable component.
Missing coverage earns zero progress and is logged, including at episode reset;
it never causes a physically valid task to be rejected. The first observation
after entering mapped space establishes a baseline without a progress jump.
Scene/physical calibration checks remain strict. These image-derived maps
are approximations, **not** Unity's physics/NavMesh ground truth. Online physical
collision and warning costs remain active; moving objects are not baked into
the policy's inputs. No map or path is used for inference.

`--reference-policy-checkpoint` and `--reference-kl-coef` enable a decaying
categorical KL(current policy || frozen initialization) on on-policy states.
This is **regularized PPO**, not a vanilla-PPO ablation; it uses neither expert
action mixing nor action-label/BC supervision. It is separate from PPO's
old-policy KL stopping criterion. The reference and current frozen visual
encoders must match exactly so cached visual/action/goal histories are valid.

`--parallel-env-steps` steps independent players concurrently and preserves
result ordering. Resets stay on the main thread. `--resample-invalid-spawns`
logs displaced training spawns and retries another sampled task at most four
times; it is disabled by default and is **never enabled in benchmark evaluation**.

`--fresh-unity-per-episode` is an opt-in training lifecycle control. It restarts
the Unity player for every task, including consecutive tasks in one scene,
matching the canonical evaluator's fresh-player lifecycle. The default remains
same-scene player reuse for compatibility and speed. Scene/task sampling,
episode budgets, reward settings, policy history resets, and evaluation are
unchanged. Fresh players are slower and do not guarantee bitwise-identical
physics. A reused-world spawn can differ from a fresh-world spawn as moving
objects evolve; this option is an ablation, not a demonstrated success gain or
a reason to filter tasks by their navigation outcomes.

Repeated player launches use an idempotent observation-decoder patch. Older
snapshots that re-wrap the decoder on every launch can accumulate a recursive
call chain and eventually fail even when individual reset smokes pass. The
current implementation installs one guarded wrapper; repeated installation
does not change normal or shape-tolerant depth/RGB decoding. A long-running
old process must be checkpointed and resumed with the fixed source rather
than having its source files changed underneath it.

The shared launcher `shs/rl/run_pointgoal_ppo_independent.sh` composes these
settings with the complete-actor launcher. In addition to its existing variables,
set `NAVIGATION_MAP_DIR` and `REFERENCE_ACTOR`. Defaults are 500 updates of
128 rollout steps (1,536,000 transitions with 24 environments), eight critic-only
warm-up updates, actor LR 1e-6, critic LR 1e-4, GAE lambda 0.98, and reference KL
0.05 decaying to 0.01 over 400 updates. Episode budgets and training shield match
the benchmark protocol; no success claim follows from training statistics alone.

`nav.scripts.rl.evaluate_ppo_curriculum` evaluates independent validation first
and then fixed-protocol benchmark milestones or validation-best candidates.
Its >80% candidate threshold is **77/96**, with validation non-regression and
collision-rate gates. Candidate detection does not establish completion:
independent benchmark repetition, full validation, and provenance audits remain
required. The earlier `evaluate_ppo_checkpoints` retains its historical 48-pair
validation / >60% workflow for reproducibility.

## Experimental stochastic-policy temperature

`--policy-temperature 0.5` parameterizes the **training** policy as
`softmax(logits / 0.5)`. Rollout sampling, saved behavior likelihoods, PPO replay
likelihoods, entropy, and both sides of reference KL use the same temperature.
This remains stochastic on-policy PPO; it is not deterministic-action PPO or an
off-policy sampling shortcut. Non-default temperatures are restricted to pure
PPO without teacher actions, BC supervision, or teacher replay.

The default is 1.0 and returns unmodified logits. Temperature is recorded in
training configuration/checkpoints and each update's metrics, not the model
architecture. Evaluation still takes the unmodified actor's deterministic
argmax with the same action space, step budget, and safety shield. Positive
temperature scaling alone does not change argmax for a fixed checkpoint.

This option supports a controlled investigation of stochastic-training versus
greedy-evaluation loops. Sharpening reduces exploration and can hurt performance;
it is **not** an established improvement. Use an immutable new source snapshot
and a finite ablation, then compare independent validation, canonical96 and
safety metrics before adopting it. Do not change temperature mid-rollout or
mix likelihoods from different temperatures within one PPO update.

## Fixed evaluation horizons under reset jitter

For repeated persistent evaluations, `--step-budget-reference prior/summary.json
--step-budget-reference-sha256 <sha256>` pins every selected task to that report's
existing budget. This avoids a one-step `ceil` change when the physical reset
pose moves by centimetres. The default remains the original dynamic calculation.
The reference's identity and protocol must be audited separately; pinning an
arbitrary report does not certify benchmark compliance. Only IDs and budgets
are read, not success labels. This option is not available to nonpersistent runs.

The evaluator verifies the hash and task coverage before launching, applies
the reference horizon immediately after every reset, and records both the
reset-computed budget and reference hash. It never changes policy inputs,
spawn/goal coordinates, rewards, success radius or the fixed shield. Completed
resume rows with different budgets are rejected rather than silently merged.
Keep an original mismatched run intact; any correction must use the same model
and original per-task budget, and retain explicit repair provenance. Do not
repeatedly rerun a task until it succeeds.

For new curriculum-monitor output directories, `--pin-initial-budgets` uses the
supplied initialization benchmark as the canonical budget reference. It also
pins later validation evaluations to that monitor's first validation report.
The mode and benchmark-reference hash are part of monitor identity; do not
enable it midway through an existing monitor. Generated command records contain
the exact budget-reference paths and hashes. The default remains disabled.

## Experimental persistent residual memory

The preserved DAgger Transformer consumes a rolling window (12 frames in the
current checkpoint), but does not retain learned state beyond that window.
`MemoryDaggerTransformerActorCritic` adds a GRU state across windows without
changing the original actor at initialization. Its inputs remain depth features,
relative PointGoal, and executed-action history; no scene IDs, absolute
positions, maps, expert actions, or planner outputs enter this branch.

```bash
python -m nav.scripts.rl.prepare_memory_ppo \
  --checkpoint preserved_ppo/update_000025.pt \
  --output-dir memory_initialization --memory-size 256 --freeze-base-actor
```

The conversion preserves every original actor and critic tensor. Zero-initialized
residual action/value heads give exactly matching initial logits and values;
the audit records the source/checkpoint hashes and reset-aware parity probes.
The optimizer and update counter start fresh. `--freeze-base-actor` freezes the
original policy, while the new GRU/action residual and both value heads remain
trainable. Without this flag the original actor can also be optimized.

Train with the same independent-data launcher, but append `--bptt-len 8
--chunks-per-minibatch 8 --adaptation-lr-scale 1.0`. The new memory/action head
use the base learning rate (1e-4 in the launcher); the original actor retains
its separate LR group if unfrozen. Learned memory is detached between rollout
chunks, not between the steps of a BPTT chunk, and resets at episode boundaries.
Raw visual-window features remain detached. Value losses do not update either
actor or learned memory; critic-only warmup includes both value heads. Frozen
reference KL sees the raw window, not the current policy's recurrent state.

This is an experimental architecture, not an established performance gain.
Validate initialization in the real evaluator and run a finite multi-GPU smoke
test before a full training run. Use a new source snapshot: older evaluators do
not understand the explicit `dagger_transformer_memory` checkpoint format.

An opt-in `--motion-features` initializer adds six relative-observation cues to
the GRU: transition validity, the previous **executed** action's three-way
one-hot, signed goal-distance progress, and displacement estimated during a
forward action. The latter two use metres, clipped to [-2,2] and [0,2]. These
come from adjacent goals already stored in the actor's raw window; they do not
read absolute poses, collision labels, rewards, maps, or future observations.
Forward displacement assumes unchanged heading during forward; it is zero for
turn actions and is not a rotation-collision detector. Localization errors or
clipped goal inputs can affect the estimate. Reset/BOS cues are zero, including
when stale state contains NaNs. Rollout and PPO replay use the same function.

For a paired memory ablation, prepare context-only and motion-enabled models
from the same source and seed. Motion initialization copies the context-only
GRU exactly and appends six **zero** input columns; all other tensors match.
Both residual heads also start at zero, so both actors/critics initially match
the original PPO model. This feature is disabled by default, preserving old
checkpoint shapes. Only PPO gradients train the new branch; its navigation
benefit must be measured against the matched context-memory control.

For resource/stability diagnostics only, `INDUSTRYNAV_UNITY_JOB_WORKERS=2`
adds Unity's `-job-worker-count 2`; optionally
`INDUSTRYNAV_UNITY_SINGLE_THREADED_RENDER=1` adds `-force-gfx-direct`.
Both are opt-in and are recorded in the launch log. The defaults do not change
historical training or evaluation. Unity documents the [worker limit](https://docs.unity3d.com/6000.3/Documentation/ScriptReference/Unity.Jobs.LowLevel.Unsafe.JobsUtility.JobWorkerMaximumCount.html)
and [single-threaded rendering flag](https://docs.unity3d.com/6000.3/Documentation/Manual/PlayerCommandLineArguments.html).

If a completed training run waits minutes per player during shutdown, optionally
set `INDUSTRYNAV_UNITY_CLOSE_TIMEOUT_SECONDS=5`. This only shortens the owned
player's exit wait after ML-Agents sends its normal shutdown message, then reaps
the process. It does not reduce the connection/RPC timeout or change navigation
termination, rewards or step budgets. The default is unset and uses the original
public `close()` path. Keep this training runtime override unset in fixed-protocol
benchmark launches. An invalid configured value fails before player creation.
Neither flag is an established fix for a particular native crash. Verify actual
thread counts, valid camera observations, step behavior and sustained training
before adopting a new execution setting. Keep benchmark launch settings fixed.

## Optimizer resume and update diagnostics

By default, `--resume --resume-optimizer` restores the checkpoint's learning
rates as well as its Adam moments. Merely changing `--lr` or
`--policy-lr-scale` does not override those saved rates. For a rate-only
continuation, add `--resume-optimizer-lr-mode config`: this preserves moments,
step counters, weight decay, betas and epsilon, but applies the newly configured
rates to matching named parameter groups. It requires an existing `latest.pt`
and rejects incompatible groups. `--no-resume-optimizer` remains a separate
choice that resets optimizer state. Logs and `effective_optimizer_lrs` report
the actual rates after restore.

`sampled_kl_k3` logs the same KL estimator used by early stopping;
`maximum_checked_kl` also includes a rejected minibatch and the maximum across
ranks. The old `approx_kl` field is retained as the noisier signed sample mean;
small negative values do not imply a negative true KL. Float roundoff can also
put k3 a few ulps below zero near identity. `ppo_clip_fraction` and pre-clipping
`gradient_norm_total`, `gradient_norm_policy`, `gradient_norm_value` help diagnose
update strength. These are measurements only: neither gradient clipping nor
the existing KL stopping rule is changed. Diagnostic averages cover accepted
PPO minibatches, not any optional teacher replay updates.

`--gradient-clip-mode global` remains the default and performs the exact original
single PyTorch norm-clipping operation. The experimental `actor_value` mode is
restricted to the detached-critic DAgger PPO architectures: all actor/encoder/
adaptation parameters are clipped together, and `critic` plus `critic_memory`
parameters are clipped separately. Each group receives `--max-grad-norm`, so
the combined norm can reach sqrt(2) times that limit. This is an explicit
algorithm ablation, not merely a diagnostic or a promised navigation improvement.
It prevents a large value gradient from shrinking an otherwise small actor
gradient through the shared norm. Adam normalization can reduce that effect;
the impact must be measured, not inferred from pre-clipping norms alone.

The objective, detached value features, PPO/KL guards, optimizer state and
learning rates remain unchanged. Critic warmup still clears actor gradients.
`gradient_clip_mode`, `gradient_norm_policy_after_clip` and
`gradient_norm_value_after_clip` record the applied mode and accepted-minibatch
post-clipping norms. Both PPO and optional replay paths use the configured
mode, although the detached-critic runs here remain teacher/replay-free. Invalid
separate-mode limits or architectures fail before creating Unity environments;
nonfinite gradients fail before either group is clipped. Existing checkpoint
formats and inference behavior are unchanged.

## Evaluation

### Experimental checkpoint averaging

`nav.scripts.rl.average_ppo_checkpoints` creates one uniformly averaged policy
from at least two trusted PPO snapshots with a common initialization. Supply
`--checkpoints`, matching `--expected-sha256` values, and a new `--output-dir`.
All model/observation/history configurations and frozen depth-encoder tensors
must match; non-floating buffers must also be identical. Other floating tensors
(including the critic) are averaged in float64 and restored to their original
dtype. Sources remain unchanged, and the output records their hashes and equal
weights. This is a single-model artifact: no per-task model choice, logit voting,
teacher labels, or action overrides are introduced.

The output is for evaluation or fresh `--init-ppo-checkpoint` initialization,
**not** optimizer resume: optimizer state is intentionally omitted and counters
reset. If fine-tuning it, initialize a new optimizer and validate the value head
with critic warmup. Functional temporal probes do not establish navigation
performance. Evaluate the original protocol and independent validation before
adopting any average; do not report the union of its parents' successes as a score.
The idea follows [Model Soups](https://proceedings.mlr.press/v162/wortsman22a.html),
whose classification results do not establish effectiveness for this PPO task.

### Fixed navigation protocol

`nav.scripts.rl.evaluate_ppo_checkpoints` evaluates saved snapshots on the 96
canonical pairs and the 48 reserved validation pairs. It uses Success@2m,
seed 0, the shared dynamic 80–320 step budget, and the same shield settings
as the 52.08% reference: collision `8`, recovery turns `6`, forward escape `4`,
bounded clearance `8`, warning streak `8`/turns `2`, turn streak `12`/escape `4`.
Terminal homing is disabled. It checks all expected episodes and return codes
before reporting a score. More than 60% requires at least **58/96** successes.

The canonical points never enter gradient updates. If their scores guide
checkpoint or hyperparameter selection, they are benchmark-selection results,
not an untouched estimate of generalization. Report the PPO-reserved
validation score alongside them. These pairs are excluded from this PPO run;
overlap with historical DAgger training has not been audited. No improvement
or target attainment is assumed before the corresponding evaluations complete.

For Linux persistent-policy evaluation, launch the monitor inside
`xvfb-run -a -s "-screen 0 1724x1024x24"`; unlike the per-episode subprocess
path, persistent evaluation does not create a virtual display automatically.
Use `CUDA_VISIBLE_DEVICES` for the policy and
`INDUSTRYNAV_UNITY_DEVICE_INDEX` for Unity's physical GPU. The plural
`INDUSTRYNAV_UNITY_DEVICE_IDS` is a training-launcher setting mapped per DDP
rank; the evaluation process does not interpret it.

`nav.scripts.rl.audit_pointgoal_ppo_result` performs a separate read-only audit
of a completed 96-point report against initialization. It verifies all 96
task IDs, success versus final distance, unchanged per-point step budgets,
normal termination, action/collision counters, fixed shield settings and the
absence of planner/fallback/terminal-homing overrides. It also verifies the
provided canonical-points SHA-256 and reports gained/lost tasks and safety
counts. A passing report audit alone does not prove checkpoint provenance or
repeatability; retain the checkpoint/source/data hashes and a repeated run.
