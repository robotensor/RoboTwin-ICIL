# RoboTwin ICIL Benchmark

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

```bash
robotwin-icil eval --policy replay --suite v1 --episodes 18 --seed 42 --video
```

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
Press / Push, Open / Close, Insertion, Bimanual and Articulated. The official V1 suite is nine
short-horizon tasks across Pick and Place, Stacking and Press / Push, each kept because RoboTwin's
expert solves at least 70% of surveyed seeds — see [`docs/survey.md`](docs/survey.md).

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

```bash
git clone --recurse-submodules https://github.com/robotensor/robotwin-icil-benchmark.git
cd robotwin-icil-benchmark

# host tools only (report, records, task table) — no simulator needed
uv venv --python 3.10 .venv && uv pip install -e ".[dev]"
pytest -m "not sim"

# simulator: RoboTwin 2.0's own environment and assets (see docs/install.md)
bash scripts/install_robotwin.sh

# one episode, replay-oracle policy, to prove the harness end to end
robotwin-icil eval --policy replay --task click_bell --episodes 1 --seed 42 --run-dir runs/smoke

# the same on two Franka arms instead of the task config's aloha-agilex
robotwin-icil eval --policy replay --embodiment franka-panda --task click_bell --episodes 1 --seed 42 --run-dir runs/smoke-franka

# how often RoboTwin's own expert solves each task (decides suite membership)
robotwin-icil survey --suite v1 --seeds 20 --json runs/survey.json

# the competition's shape: build one demonstration and save it, then evaluate from the file
robotwin-icil materialize --task click_bell --scene-seed 42 --scene-seed 43 --out runs/unit/prompt
robotwin-icil run-unit --prompt runs/unit/prompt/prompt.npz --policy replay --out runs/unit/run

# the same unit against a policy served in a process of its own (needs icil-policy installed)
export ICIL_POLICY_AUTHKEY=$(python -c "import secrets; print(secrets.token_hex(32))")
python -m icil_policy.serve --manifest <icil-policy>/examples/replay_policy/icil.yaml \
    --address /tmp/policy.sock --authkey-env ICIL_POLICY_AUTHKEY --log-file /tmp/policy.log &
robotwin-icil run-unit --prompt runs/unit/prompt/prompt.npz --policy-address /tmp/policy.sock \
    --authkey-env ICIL_POLICY_AUTHKEY --policy-log /tmp/policy.log --out runs/unit/served

# the official V1 suite
robotwin-icil eval --policy <adapter> --suite v1 --episodes 500 --seed 42 --run-dir runs/v1
robotwin-icil report runs/v1

# one-arm robots: only the tasks whose expert uses one arm (26 of 50; `tasks --arms 1` lists them)
robotwin-icil eval --policy <adapter> --suite v1 --arms 1 --episodes 500 --seed 42 --run-dir runs/v1-one-arm

# the robot and the task selection combine: every one-arm task's expert, on two Franka arms
robotwin-icil survey --suite all --embodiment franka-panda --arms 1 --seeds 20 --json runs/survey-franka-1arm.json
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
policy that will not load). Both commands' `result.json` carry `success`, `void`, `steps` and
`error`; a policy that raises or returns an invalid action fails its unit, and only what the
harness could not give it (a prompt, a scene, a GPU, a fault of its own) voids one. `prompt.npz`
holds `frames_<camera>`, `qpos`, `endpose`, `actions`, `times` and `frequency` under the channel
map `prompt.CHANNELS` publishes, plus `meta`, which never reaches a policy.

A competition does not import the policy at all. `run-unit --policy-address ADDR --authkey-env
NAME` drives one served by `python -m icil_policy.serve` in a process of its own: the
demonstration crosses the socket as `prompt.npz`'s own arrays, each observation as
`frames_<camera>`, `qpos` and `endpose`, and nothing of `meta`. A served policy that answers a
call with an error fails its unit; one that cannot be spoken to (nothing listening, `hello`
refused, a timeout, a hang-up, a malformed reply) voids it with `void_cause` "policy", and every
other void carries "harness" — see [`docs/policies.md`](docs/policies.md).

## Layout

```
src/robotwin_icil/
  tasks.yml tasks.py        task -> skill category and arms table, suites, --arms selection
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
generation attempts, rollout length and outcome. In the manifest, `suite` is what was asked for
and `tasks` what ran: a `--arms 1` run of `v1` records `suite: v1`, `arms: "1"` and the seven
one-arm tasks, so read `arms` or `tasks` with `suite`, never `suite` alone. With `--video`, demonstration and evaluation clips
are saved side by side (`episode_00015/demonstration.mp4`, `evaluation_same_scene.mp4`) — the fastest way to
confirm by eye that the rollout really did start where the expert started.

## Acknowledgements

This benchmark is a thin evaluation protocol layered on top of
[**RoboTwin 2.0**](https://github.com/RoboTwin-Platform/RoboTwin), which does the substantial work:
the simulated bimanual environments, the 50 manipulation tasks, the scene generation and domain
randomization, the scripted expert and its motion planning, and the per-task success conditions.
We reuse those mechanisms as they are rather than reimplementing them, and we are grateful to the
RoboTwin authors for releasing them openly. RoboTwin builds in turn on
[SAPIEN](https://sapien.ucsd.edu/), [mplib](https://github.com/haosulab/MPlib) and the wider
open-source robotics ecosystem.

If you use this benchmark, please cite RoboTwin:

```bibtex
@article{chen2025robotwin,
  title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization
         for Robust Bimanual Robotic Manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and
          Li, Zixuan and Liang, Qiwei and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}

@InProceedings{Mu_2025_CVPR,
  author    = {Mu, Yao and Chen, Tianxing and Chen, Zanxin and Peng, Shijia and Lan, Zhiqian and
               Gao, Zeyu and Liang, Zhixuan and Yu, Qiaojun and Zou, Yude and Xu, Mingkun and
               Lin, Lunkai and Xie, Zhiqiang and Ding, Mingyu and Luo, Ping},
  title     = {RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins},
  booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
  year      = {2025},
  pages     = {27649--27660}
}
```

## License

This benchmark is licensed under Apache-2.0. RoboTwin 2.0 is vendored as a submodule and remains
under its own MIT license, © 2025 Tianxing Chen.
