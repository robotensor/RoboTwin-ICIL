# RoboTwin-ICIL

**A one-demonstration in-context imitation learning benchmark built on
[RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin).**

Standard imitation-learning benchmarks collect a dataset, train a policy on it, and evaluate the
result. This benchmark does none of those things. Every episode generates its own expert
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

By holding the scene fixed between demonstration and rollout, V1 removes scene generalization from
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
| Official score | **Same Scene 1-Demo Success Rate**, reported overall, by skill category and by task |

An episode is only scored if the expert actually solved the scene. Planning failures, invalid
placements, collisions and unstable simulation are *benchmark generation* failures: the episode is
rejected, another seed is drawn, and the rejection is recorded in a separate statistic. They never
enter the model's denominator.

$$SR_{\text{same-scene}} = \frac{\text{successful policy rollouts}}{\text{valid evaluated Same Scene episodes}}$$

There is no zero-shot score, no ICL gain, no multi-shot setting and no difficulty tiering in V1.

## Status

V1 — the Same Scene 1-Demo protocol — is complete: every V1 issue is closed, and the whole loop
runs end to end on RoboTwin 2.0.

The replay oracle plays each demonstration's own actions back from the rebuilt scene. It is the
harness's ceiling — what a perfect imitator scores here — and on the V1 suite, on aloha-agilex, it
scores **18/18**:

This historical run used the nine tasks below, with two episodes per task and global seed 42.
The current CLI selects all tasks by default or one task with `--task`.

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

Next: a real ICIL policy (#12), and scene-generalization settings beyond Same Scene (#13).

## Skill categories and arms

RoboTwin 2.0's 50 tasks are mapped to manipulation skill categories in
[`src/robotwin_icil/tasks.yml`](src/robotwin_icil/tasks.yml) — Pick and Place, Stacking,
Press / Push, Open / Close, Insertion, Bimanual and Articulated. `eval` and `survey` select all
50 tasks by default; `--task NAME` selects one, and `--arms 1` filters to strictly one-arm tasks.
`--episodes` is the total evaluation count, distributed round-robin over the selected tasks:
50 episodes runs each task once, and 500 runs each ten times. Expert success depends on the task,
scene and robot; catalog membership does not guarantee a successful demonstration.

Named task sets remain internal data for the competition plugin and historical results. The
historical `v1` set contains nine surveyed tasks; the competition's `franka_1arm` set contains four.
They are not standalone CLI options. See [`docs/survey.md`](docs/survey.md).

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
robotwin-icil survey --embodiment franka-panda --arms 1 --seeds 20 --json runs/survey-franka-1arm.json
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
  runner.py                 episode loop, seed drawing, rejection accounting
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
task list; a default `--arms 1` run records all 26 one-arm tasks and `arms: "1"`. New CLI runs
record `suite: null`; the field remains readable for older results and Python callers. Use a new
run directory when moving from an old named selection to the full catalog. With `--video`,
demonstration and evaluation clips
are saved side by side (`episode_00015/demonstration.mp4`, `evaluation_same_scene.mp4`) — the fastest way to
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
