# Python Architecture

IndustryNav separates reusable contracts, model structure, algorithms, runtime
adapters, and experiment tooling. The dependency direction is intentionally
one-way:

```text
scripts ──> baselines ──> models
   │            │           │
   ├────────> train <────────┘
   ├────────> envs ──────> safety
   └────────> eval ──────> safety

core <── data / models / baselines / envs / train / eval
```

## Package Responsibilities

| Package | Owns | Must not own |
|---|---|---|
| `nav/core` | Stable agent contracts, shared types, PointGoal geometry | Torch models, Unity setup, experiment IO |
| `nav/models` | Encoders and policy network architectures | Preprocessing, rewards, rollouts, trainers, evaluation |
| `nav/baselines` | A*, BC, PPO, DAgger method logic and concrete agents/trainers | Generic metric or environment infrastructure |
| `nav/data` | Dataset discovery, transforms, and samples | Optimization loops |
| `nav/envs` | Unity and future simulator adapters | Policy architecture or metric aggregation |
| `nav/safety` | Online warning/collision primitives | Reading output directories or reporting tables |
| `nav/train` | Generic trainer lifecycle, GAE, checkpoint initialization | BC/PPO objectives and method-specific schedules |
| `nav/eval` | Metric lifecycle and aggregation over run artifacts | Simulator control or policy inference |
| `nav/scripts` | Thin CLI entry points that assemble the layers above | Model, algorithm, or environment implementations |

Compatibility modules keep older imports working (`nav.models.policy`,
`nav.train.dataset`, `nav.train.loop`, and the original PPO module paths), but
new code should use the owning package shown above.

## Extension Contracts

Stateful learned methods implement `nav.core.NavigationAgent`: call `reset()`
at each episode boundary and `predict_action(...)` for each decision. Trainers
derive from `nav.train.BaseTrainer`, which guarantees the shared
`setup -> fit -> teardown` lifecycle. Environments derive from
`nav.envs.NavigationEnvironment`, streaming metrics derive from
`nav.eval.BaseMetric`, and repeated evaluators derive from
`nav.eval.BaseEvaluator`.

Keep variation compositional. A new method should reuse an encoder or policy
from `nav.models`, a simulator adapter from `nav.envs`, and online detectors
from `nav.safety`, while placing only its action selection, loss/reward, and
training behavior under `nav/baselines/<method>/`.

For example, a new learned PointGoal baseline normally consists of:

```text
nav/models/policies/my_policy.py       # torch architecture
nav/baselines/my_method/agent.py       # checkpoint + inference state
nav/baselines/my_method/trainer.py     # objective and optimization logic
nav/scripts/my_method/train.py         # argument parsing / assembly only
```

Safety intentionally remains separate from `eval`: PPO reward calculation,
live visualization, and post-hoc evaluation all need identical event
definitions. `nav.eval` aggregates those shared detector decisions into rates
and reports.
