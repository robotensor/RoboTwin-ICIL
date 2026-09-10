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
| Success | RoboTwin's own per-task `check_success()`, binary |
| Official score | **Same Scene 1-Demo Success Rate**, reported overall, by skill category and by task |

An episode is only scored if the expert actually solved the scene. Planning failures, invalid
placements, collisions and unstable simulation are *benchmark generation* failures: the episode is
rejected, another seed is drawn, and the rejection is recorded in a separate statistic. They never
enter the model's denominator.

$$SR_{\text{same-scene}} = \frac{\text{successful policy rollouts}}{\text{valid evaluated Same Scene episodes}}$$

There is no zero-shot score, no ICL gain, no multi-shot setting and no difficulty tiering in V1.

## Skill categories

RoboTwin 2.0's 50 tasks are mapped to manipulation skill categories in
[`src/robotwin_icil/tasks.yml`](src/robotwin_icil/tasks.yml) — Pick and Place, Stacking,
Press / Push, Open / Close, Insertion, Bimanual and Articulated. The official V1 suite is a
subset chosen for short horizons and high expert success rates, spanning Pick and Place, Stacking
and Press / Push.

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
robotwin-icil eval --policy replay --task place_object_basket --episodes 1 --seed 42 \
    --run-dir runs/smoke

# the official V1 suite
robotwin-icil eval --policy <adapter> --suite v1 --episodes 500 --seed 42 --run-dir runs/v1
robotwin-icil report runs/v1
```

The `replay` policy ignores its observations and plays the demonstration's actions back verbatim.
Because Same Scene means the rollout starts from the identical state, it is the harness's own
upper bound: if it does not succeed, the bug is in the benchmark, not in the model.

## Layout

```
src/robotwin_icil/
  tasks.yml tasks.py        task -> skill category table and suites
  config.py                 benchmark + RoboTwin configuration
  demo.py                   model-independent demonstration container
  scene.py                  initial-state fingerprint and Same Scene verification
  policy.py                 the policy interface, dummy and replay policies
  generate.py               on-demand expert demonstrations, seed streams, rejections
  episode.py                one episode: expert -> demo -> exact reset -> rollout -> success
  runner.py                 episode loop, seed drawing, rejection accounting
  records.py report.py      episode records, aggregation to overall/skill/task
  video.py                  demonstration and evaluation clips per episode
  robotwin.py               the only module that imports RoboTwin
  cli.py
docs/                       installation, policy adapters
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
git commits, both configs, and per episode: the task, skill category, scene seed, number of expert
generation attempts, rollout length and outcome. With `--video`, demonstration and evaluation clips
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
