# RoboTwin-ICIL

**A one-demonstration in-context imitation learning benchmark built on
[RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin).**

Release: [v0.1.0](https://github.com/robotensor/RoboTwin-ICIL/releases/tag/v0.1.0)
([release notes](docs/releases/v0.1.0.md)).

This repository supports **ICIL duels** through the competition plugin and a **Robotensor
standard benchmark** through `standard-eval`: RoboTwin 2.0's own evaluation — all 50 tasks, its
evaluation scenes, Easy and Hard settings — with the expert's demonstration as the policy's input.
Training happens outside the evaluator. Every evaluated episode generates its own expert
demonstration *at evaluation time*, hands that single demonstration to a **frozen** policy as
context, and asks the policy to reproduce the behaviour from the **exact same initial scene** the
expert started from:

```
RoboTwin task
      │
      ▼
generate scene (seed S)  ──►  RoboTwin expert  ──►  ONE successful demonstration
      │                                                      │
      │  reset to the exact same scene (seed S)              │
      ▼                                                      ▼
  observations  ─────────────►   frozen ICIL policy   ◄──── demonstration
                                        │
                                     actions
                                        ▼
                              RoboTwin success check
```

The demonstration is **not** training data — no gradients are taken during a benchmark run. It is
context supplied to the model moments before execution. The question the benchmark asks is
deliberately narrow:

> Can a model watch one successful robot demonstration and reproduce that behaviour from the same
> starting configuration?

By holding the scene fixed between demonstration and rollout, the benchmark removes scene generalization from
the measurement and isolates the model's ability to consume an in-context robot trajectory.
Generalization becomes a separate, explicitly named setting later — never a "difficulty" knob.

## The protocol

| | |
| --- | --- |
| Demonstrations per episode | exactly **1** |
| Evaluation setting | **Same Scene** (demonstration and rollout share task, seed, objects, poses, cameras, lighting, background and initial robot state) |
| Policy | frozen — `policy.reset()` between episodes, no `backward()`, no optimizer, no parameter write |
| Demonstration source | RoboTwin's own scripted expert (`play_once`), captured on the fly |
| Robot | `--embodiment aloha-agilex` (one dual-arm URDF, 14-dim qpos) or `franka-panda` (two Franka arms, 16-dim qpos); without the flag, the task config's own robot, aloha-agilex in every shipped config; recorded with every episode |
| Success | RoboTwin's own per-task `check_success()`, binary |
| Score | **Same Scene 1-Demo Success Rate**, reported overall, by skill category and by task; the standard profile averages the 50 tasks per setting |

An episode is only scored if the expert actually solved the scene. Planning failures, invalid
placements, collisions and unstable simulation are *benchmark generation* failures: the episode is
rejected, another seed is drawn, and the rejection is recorded in a separate statistic. They never
enter the model's denominator.

$$SR_{\text{same-scene}} = \frac{\text{successful policy rollouts}}{\text{valid evaluated Same Scene episodes}}$$

There is no zero-shot score, no ICL gain, no multi-shot setting and no difficulty tiering.

## Status

The Same Scene 1-Demo protocol runs end to end on RoboTwin 2.0.

The replay oracle plays each demonstration's own actions back from the rebuilt scene. It is the
harness's ceiling — what a perfect imitator scores here. A harness check on aloha-agilex, with the
nine tasks below, two episodes per task and global seed 42, scored **18/18**:

| Skill | Task | Replay oracle | Expert success ([survey](docs/survey.md)) |
| --- | --- | ---: | ---: |
| Pick and Place | place_a2b_left | 2/2 | 85% |
| | place_a2b_right | 2/2 | 75% |
| | place_empty_cup | 2/2 | 90% |
| | place_container_plate | 2/2 | 75% |
| Stacking | stack_blocks_two | 2/2 | 100% |
| | stack_bowls_two | 2/2 | 75% |
| Press / Push | click_bell | 2/2 | 100% |
| | click_alarmclock | 2/2 | 80% |
| | press_stapler | 2/2 | 95% |

Every rebuilt scene matched its demonstration's fingerprint (0 invalid). The expert needed 24
attempts for 18 demonstrations; its 6 failures were recorded as generation rejections and never
touched a score. The run took 35 minutes on an RTX A6000 shared with a 40 GiB training job.

The standard profile (below) has not yet completed a simulator run, and no learned-policy result
is published. Next: a real
ICIL policy (#12), and scene-generalization settings beyond Same Scene (#13).

## Skill categories and arms

RoboTwin 2.0's 50 tasks are mapped to manipulation skill categories in
[`src/robotwin_icil/tasks.yml`](src/robotwin_icil/tasks.yml) — Pick and Place, Stacking,
Press / Push, Open / Close, Insertion, Bimanual and Articulated. `eval` and `survey` select all
50 tasks by default; `--task NAME` selects one, and `--arms 1` filters to strictly one-arm tasks.
`--episodes` is the total evaluation count, distributed round-robin over the selected tasks:
50 episodes runs each task once, and 500 runs each ten times. Expert success depends on the task,
scene and robot; catalog membership does not guarantee a successful demonstration.

Measured expert success rates are in [`docs/survey.md`](docs/survey.md).

The same table says how many arms each task's expert needs: `arms: 1` for the 26 whose expert
drives one arm per episode (chosen once from the scene, or fixed), `switching` for the 6 stacking
and ranking tasks that pick an arm per object, and `2` for the 18 that use both. The value is a
static read of the task's `play_once` — `src/robotwin_icil/arms.py` re-derives it from the pinned
checkout, following an arm through the helpers, parameters and attributes the expert passes it
through, and a test fails naming any task whose entry disagrees. Where the read is unsure it
errs towards more arms, so a wrong entry fails that test rather than admit a two-arm expert to a
one-arm run. `--arms 1` on `eval`, `survey` and `tasks` keeps only the one-arm tasks; without it
a run is unchanged.

## Quick start

The standalone benchmark supports local policy adapters and the replay oracle without the
competition orchestrator. Served policies and the optional `competition/` plugin additionally
require `icil-policy`, currently distributed in Robotensor's private orchestrator repository.
Those features require access to that package; the standalone commands below do not.

The Python distribution and command remain `robotwin-icil`, and the import package remains
`robotwin_icil`.

```bash
git clone --recurse-submodules https://github.com/robotensor/RoboTwin-ICIL.git
cd RoboTwin-ICIL

# host tools only (report, records, task table) — no simulator needed
uv venv --python 3.10 .venv && uv pip install -e ".[dev]"
pytest -m "not sim"

# simulator: RoboTwin 2.0's own environment and assets (see docs/install.md)
bash scripts/install_robotwin.sh

# one episode, replay-oracle policy, to prove the harness end to end
robotwin-icil eval --policy replay --task click_bell --episodes 1 --seed 42 --run-dir runs/smoke

# the same on two Franka arms instead of the task config's aloha-agilex
robotwin-icil eval --policy replay --embodiment franka-panda --task click_bell --episodes 1 --seed 42 --run-dir runs/smoke-franka

# how often RoboTwin's own expert solves every cataloged task
robotwin-icil survey --seeds 20 --json runs/survey.json

# the competition's shape: build one demonstration and save it, then evaluate from the file
robotwin-icil materialize --task click_bell --scene-seed 42 --scene-seed 43 --out runs/unit/prompt
robotwin-icil run-unit --prompt runs/unit/prompt/prompt.npz --policy replay --out runs/unit/run

# the same unit against a policy served in a process of its own (needs icil-policy installed)
export ICIL_POLICY_AUTHKEY=$(python -c "import secrets; print(secrets.token_hex(32))")
python -m icil_policy.serve --manifest <icil-policy>/examples/replay_policy/icil.yaml \
    --address /tmp/policy.sock --authkey-env ICIL_POLICY_AUTHKEY --log-file /tmp/policy.log &
robotwin-icil run-unit --prompt runs/unit/prompt/prompt.npz --policy-address /tmp/policy.sock \
    --authkey-env ICIL_POLICY_AUTHKEY --policy-log /tmp/policy.log --out runs/unit/served

# all 50 tasks, ten episodes per task
robotwin-icil eval --policy <adapter> --episodes 500 --seed 42 --run-dir runs/all
robotwin-icil report runs/all

# one-arm task selection: only the tasks whose expert uses one arm (26 of 50; `tasks --arms 1` lists them)
robotwin-icil eval --policy <adapter> --arms 1 --episodes 260 --seed 42 --run-dir runs/one-arm

# the robot and the task selection combine: every one-arm task's expert, on two Franka arms
robotwin-icil survey --embodiment franka-panda --arms 1 --seeds 20 --json runs/survey-franka.json
```

The `replay` policy ignores its observations and plays the demonstration's actions back verbatim.
Because Same Scene means the rollout starts from the identical state, it is the harness's own
upper bound: if it does not succeed, the bug is in the benchmark, not in the model.

`materialize` and `run-unit` are the same episode in two processes, which is how a competition
runs it: both policies in a duel are handed the identical `prompt.npz`, and a third party can
check it by hash. `materialize` tries its candidate `--scene-seed`s in order and writes
`prompt.npz` and `demonstration.mp4` for the first one the expert solves; its `result.json` names
the chosen seed and every attempt, and is void when the expert was rejected on all of them. It
exits 0 whenever it wrote `result.json`. `run-unit` rebuilds the scene from the prompt's
privileged `meta`, verifies its fingerprint (a tampered meta or a drifted scene voids the unit),
rolls the policy out and writes `result.json` and `evaluation.mp4`; it exits 0 once the unit has a
result, whatever it is, and 1 only on a harness error before the unit starts (no simulator, a
policy that will not load, a served policy whose `--authkey-env` holds no usable key, icil-policy
not installed). Given `--expect-source-sha256`, either command exits 1 before writing anything
unless the benchmark's own source (`robotwin_icil.source_sha256()`) digests to it, and both
results record `source_sha256`; `--denoiser oidn|none` sets `ROBOTWIN_ICIL_DENOISER` for a caller
that passes no such variable. Both commands' `result.json` carry `success`, `void`, `steps` and
`error`; a policy that raises or returns an invalid action fails its unit, and only what the
harness could not give it (a prompt, a scene, a GPU, a fault of its own) voids one. `prompt.npz`
holds `frames_<camera>`, `qpos`, `endpose`, `actions`, `times` and `frequency` under the channel
map `prompt.CHANNELS` publishes, plus `meta`, which never reaches a policy.

A competition does not import the policy at all. `run-unit --policy-address ADDR --authkey-env
NAME` drives one served by `python -m icil_policy.serve` in a process of its own: the
demonstration crosses the socket as `prompt.npz`'s own arrays, each observation as
`frames_<camera>`, `qpos` and `endpose`, and nothing of `meta`. A served policy that answers a
call with an error fails its unit; one that cannot be spoken to (nothing listening, `hello`
refused, a timeout, a hang-up, a malformed reply) or whose calls use up its time budget for the
unit (`--policy-budget-s`, 300 s) voids it with `void_cause` "policy". A unit that runs out of
`--unit-timeout-s` while the policy is within its budget, and every other void, carries "harness"
— see [`docs/policies.md`](docs/policies.md).

The competition orchestrator runs the benchmark through [`competition/`](competition/README.md),
a distribution of its own that it imports without a simulator: the catalogue, the unit list of
a duel, prompt verification and results, and the argv of `materialize` and `run-unit`.

## ICIL standard benchmark

`robotwin-icil-standard` measures a policy the way RoboTwin 2.0 does, with a demonstration as
input. For each of the 50 tasks, RoboTwin's evaluation walks scene seeds up from
`100000 * (1 + seed)`, runs its expert on each, skips seeds the expert cannot solve, rebuilds the
same scene and rolls the policy out until `check_success()` or the task's step limit — 100 times,
under `demo_clean` (Easy) and `demo_randomized` (Hard). The standard profile does exactly that on
aloha-agilex, and hands the expert's successful trajectory to the frozen policy as its one
demonstration, verifying the rebuilt scene's fingerprint first.

```bash
# Full evaluation: 50 tasks x 100 episodes, clean and randomized; needs the simulator.
robotwin-icil standard-eval --policy module:FrozenPolicy --policy-arg checkpoint=/absolute/model \
  --run-dir runs/standard/model
robotwin-icil standard-report runs/standard/model

# A reduced-budget smoke run is explicitly provisional, not an official result.
robotwin-icil standard-eval --policy replay --setting clean --episodes-per-task 1 \
  --run-dir runs/standard/smoke
```

The score per setting is the mean of the 50 tasks' success rates, with a breakdown by skill
category and task. It is official when every one of the 5,000 planned episodes per setting was
scored at seed group 0. Expert rejections and scene mismatches never count against a model, but
they leave the score provisional. There is no train, validation or test split: as in RoboTwin, a
model is scored in simulation, not on stored trajectories. Duel task sets and unit derivation do
not change.

Training data: [robotensor/robotwin-icil-aloha-clean](https://huggingface.co/datasets/robotensor/robotwin-icil-aloha-clean)
republishes RoboTwin's 2,500 clean Aloha trajectories (50 per task) with head, left-wrist and
right-wrist 320x240 videos aligned to each HDF5's rows. All of them are training data; they come
from RoboTwin's collection seeds, not its evaluation seeds. The commands pin its verified release
revision.

```bash
pip install -e ".[dataset]"
robotwin-icil dataset download --revision f0332a22c7e3a7b537f9eb4eba1ba70432aa1785 --out /absolute/path/to/dataset
robotwin-icil dataset verify /absolute/path/to/dataset --videos
robotwin-icil dataset sample /absolute/path/to/dataset --row 10

# Rebuild from RoboTwin's pinned source without a simulator; publishing needs write access.
robotwin-icil dataset build --out /absolute/dataset --cache-dir /absolute/source-cache --workers 4
robotwin-icil dataset upload /absolute/dataset
```

See [the standard benchmark guide](docs/standard-benchmark.md) for the protocol beside
RoboTwin's, scoring rules, the data format, the training sample API and what to report.

## Layout

```
src/robotwin_icil/
  tasks.yml tasks.py        task -> skill category and arms table, --task/--arms selection
  arms.py                   static read of each expert's play_once: 1, switching or 2 arms
  demo.py                   model-independent demonstration container
  prompt.py                 a demonstration on disk: prompt.npz and its channel map
  scene.py                  initial-state fingerprint, Same Scene verification, its digest
  policy.py                 the policy interface, dummy and replay policies
  generate.py               on-demand expert demonstrations, seed streams, rejections
  episode.py                one episode: expert -> demo -> exact reset -> rollout -> success
  unit.py                   the episode in two files: materialize a prompt, run a unit from it
  remote.py                 a policy served at an address, driven through icil-policy's client
  runner.py                 episode loop, seed streams (independent or RoboTwin's), resume
  standard.py               RoboTwin's evaluation on 50 tasks per setting, plan and scoring
  dataset.py                HDF5 training loader and causal training sample construction
  dataset_release.py        resumable source copy, three-camera conversion, verify/upload/download
  policies/nearest.py       observation-conditioned reference adapter (no training)
  records.py report.py      episode records, aggregation to overall/skill/task
  video.py                  demonstration and evaluation clips per episode
  survey.py                 the expert's own success rate per task, and which arms it moved
  robotwin.py               the only module that imports RoboTwin
  cli.py
competition/                the orchestrator plugin, its own distribution (competition/README.md)
docs/                       installation, policy adapters, the expert survey
vendor/RoboTwin             RoboTwin 2.0, pinned as a git submodule
```

`robotwin.py` is the single seam onto the simulator; everything else is importable and testable
without SAPIEN, assets or a GPU.

## Adding a policy

A policy needs four methods — reset, accept one demonstration, take an observation, return an
action. Nothing in the benchmark core is specific to any model; per-model tensor conversion lives
in an adapter under `policies/`. The policy never receives the scene seed, the success condition,
target object identities, ground-truth task state or RoboTwin actor handles. If a model requires a
language input it is given the neutral string `"Follow the demonstrated behavior."`, so that what
is measured is the demonstration and not the prompt.

See [`docs/policies.md`](docs/policies.md).

## Reproducibility

A run is reproducible from its global seed. Each run directory records the benchmark and RoboTwin
git commits, both configs, the robot (`embodiment`), whether it asked for one-arm tasks only
(`arms`), and per episode: the task, skill category, scene seed, robot, number of expert
generation attempts, rollout length and outcome. The manifest's `tasks` records the exact selected
task list; a default `--arms 1` run records all 26 one-arm tasks and `arms: "1"`. Manifests from
earlier versions that carry a `suite` label still load; the label is ignored. With `--video`,
demonstration and evaluation clips are saved side by side (`episode_00015/demonstration.mp4`, `evaluation_same_scene.mp4`) — the fastest way to
confirm by eye that the rollout really did start where the expert started.

## Citation

If you use Robotensor's RoboTwin-ICIL, please cite this project:

```bibtex
@misc{robotensor2026robotwinicil,
  author = {{Robotensor}},
  title = {{RoboTwin-ICIL}},
  year = {2026},
  url = {https://github.com/robotensor/RoboTwin-ICIL},
  note = {Same Scene one-demonstration in-context imitation learning benchmark}
}
```

Citation metadata is also available in [`CITATION.cff`](CITATION.cff). For reproducibility,
include the benchmark commit used in your experiments.

## Acknowledgements

We thank the [**RoboTwin team**](https://github.com/RoboTwin-Platform/RoboTwin) for developing
and openly releasing RoboTwin 2.0. This Robotensor benchmark uses their simulated environments,
manipulation tasks, scene generation and domain randomization, scripted experts, motion planning,
and per-task success checks to implement its Same Scene one-demonstration ICIL evaluation protocol.
Their work makes this benchmark possible, and we appreciate their contribution to the robotics
research community.

We also acknowledge [SAPIEN](https://sapien.ucsd.edu/),
[mplib](https://github.com/haosulab/MPlib), and the wider open-source robotics ecosystem that
RoboTwin builds on.

## License

This benchmark is licensed under Apache-2.0. RoboTwin 2.0 is vendored as a submodule and remains
under its own MIT license, © 2025 Tianxing Chen.
