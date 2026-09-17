# robotwin-icil — Same Scene 1-demo ICIL benchmark on RoboTwin 2.0

Python 3.10, package `robotwin_icil` under `src/`. A separate benchmark that uses RoboTwin 2.0 for
simulation, tasks, scene generation, the expert, execution and success checking. Every episode
generates its own demonstration at evaluation time: pick a task and scene seed, run RoboTwin's
existing expert until one demonstration succeeds, recreate that exact scene, hand the frozen policy
that one demonstration, roll out, score with RoboTwin's own `check_success()`.
The duel workflow is unchanged. `standard.py` adds the `robotwin-icil-standard` profile —
RoboTwin's own evaluation on all 50 tasks (its seed stream, `demo_clean` and `demo_randomized`,
100 episodes per task) with the expert's demonstration as input — and
`dataset.py`/`dataset_release.py` provide a separate training-data release and loader. Training
never happens in the evaluator. The score is Same Scene 1-Demo Success Rate, reported overall, by
skill category and by task; the standard profile's primary metric is the mean of the 50 tasks'
success rates, per setting.

## RoboTwin, as it actually works

Read before touching `robotwin.py`; all of it lives in `vendor/RoboTwin`.

- `envs/<task>.py` defines one class per task (50 of them) over `envs/_base_task.py:Base_Task`.
  `load_actors()` builds the scene, `play_once()` is the scripted expert, `check_success()` is the
  binary outcome, `plan_success` records whether motion planning held.
- `setup_demo(now_ep_num=…, seed=S, is_test=True, **args)` -> `_init_task_env_`, which calls
  `np.random.seed(S)` and `torch.manual_seed(S)` before building anything. Scene generation is
  therefore deterministic in `(task, seed, config)`. No task uses the `random` module — python's
  RNG is deliberately *not* seeded there, so a task that started to use it would silently break
  Same Scene. `check_stable()` raises `UnStableError` on a bad placement.
- Demonstrations are recorded by `_take_picture()`, called from `take_dense_action()` every
  `save_freq` control steps when `save_data` is set; it pickles `get_obs()` frames to a cache dir.
  `get_obs()` returns per-camera rgb, `endpose`, and `joint_action` (`vector` is the left arm's
  joint state followed by the right's — joints then a gripper per arm, so 14-dim on aloha-agilex
  and 16-dim on two Frankas). We capture frames in memory instead of via that cache.
  `capture(images=False)`, the survey's default, reads that `vector` and `endpose` straight from
  the robot (`robot_state`) and never calls `get_obs`, so no camera ray-traces: no expert at the
  pinned commit reads an observation or a camera, and the one `get_obs` side effect an expert
  could see, `_update_render`'s `crazy_random_light` RNG draw, is kept. A policy's demonstration
  always captures with images.
- `env_cfg/task_config/_embodiment_config.yml` names the embodiments, and `robot.py:_init_robot_`
  always builds a left and a right arm: a one-entry `embodiment` (`[aloha-agilex]`) is one URDF
  holding both arms (`dual_arm_embodied`); a single-arm robot such as franka-panda must be given
  as `[franka-panda, franka-panda, <distance>]`, one URDF per arm, bases `distance` metres apart
  (0.8 m, from RoboTwin's configuration guide). `SceneConfig.embodiment` picks the form;
  `robotwin.action_dims` reads each arm's live joint count, which is also how `take_action`
  splits a qpos action, so 16-dim actions need no upstream change.
- Rollout is `take_action(action, action_type='qpos'|'ee')`; it stops at `step_lim`
  (`env_cfg/task_config/_eval_step_limit.yml`, per task, needs `eval_mode`) or once `eval_success`
  is set.
- `scripts/eval_policy_xpolicylab.py:run_one_batch_episode` already does expert-check -> same-seed
  re-`setup_demo` -> policy rollout. It throws the expert trajectory away; this benchmark is that
  loop with the trajectory kept and given to the policy. Read it before writing a new loop.
- `setup_demo(is_test=True)` mirrors upstream's evaluator, but no task in the pinned checkout reads
  `is_test` (15 accept it in their signature and ignore it). It is not a held-out object split.
- Physics steps every `scene.get_timestep()` (1/250 s) and a frame is recorded every `save_freq`
  of those steps, so a demonstration runs at 250/`save_freq` fps. `Demonstration.frequency` is that
  rate; upstream passes `save_freq` where it means a frame rate, and so did we once. The frames
  are not evenly spaced, though: `take_dense_action` (and `together_move_to_pose`, with a loop of
  its own) records one frame before its first step, one after every `save_freq`-th step counted
  from that first step, and one after its last, and the counter restarts per motion primitive. So
  `robotwin.clock` counts `scene.step()` calls and every captured frame carries `time_s`: a
  generated or saved demonstration's `times()` is real, and only one built without a clock (in
  tests) falls back to `index / frequency`.
- RoboTwin renders with SAPIEN's ray tracer and asks for the OIDN denoiser, which cannot run on
  Blackwell GPUs: it leaves images untouched and, under GPU contention, hangs camera reads.
  `robotwin.py` turns it off at compute capability 10.0 and above (`ROBOTWIN_ICIL_DENOISER`).
- Assets load from `./assets/...` relative to the working directory, so `robotwin.py` chdirs into
  `vendor/RoboTwin`. Resolve any path (run dir, checkpoint) to absolute before calling into it.

## Commands

- Host env (pure, no simulator): `uv venv --python 3.10 .venv && uv pip install -e ".[dev]"`; `ruff check . && ruff format --check .`; `pytest -m "not sim"`.
- Simulator env: `bash scripts/install_robotwin.sh` (conda env `robotwin` under `/root/miniforge3`, python 3.10, RoboTwin's own pins + assets); `PYTHONPATH=src $RT -m pytest -m sim` with `RT=/root/miniforge3/envs/robotwin/bin/python`. Run it from the main checkout: git worktrees have no `vendor/RoboTwin` checkout or assets. One simulator
  process per GPU at a time: two processes rendering at once can hang in SAPIEN's camera read.
- Smoke: `robotwin-icil eval --policy replay --task click_bell --episodes 1 --seed 42 --run-dir runs/smoke`, then `robotwin-icil report runs/smoke`. `eval`, `survey` and `materialize` take `--embodiment aloha-agilex` or `franka-panda`; without it the task config's own robot runs (aloha-agilex in every shipped config). A run directory is tied to its robot. `--arms 1` on `eval`, `survey` and `tasks` keeps only the one-arm tasks, and combines with `--embodiment` (`survey --embodiment franka-panda --arms 1`).
  `survey` renders no camera unless given `--images`, and its JSON records which. Pass `--images` when its rejections must predict `eval`'s on a GPU short of memory: rendering holds memory, and RoboTwin's CuRobo batch planner reports a CUDA out-of-memory error as a failed plan.
- The competition's shape, one process per half: `robotwin-icil materialize --task click_bell --scene-seed S [--scene-seed S2 ...] --out DIR` tries the candidate seeds in order and writes `prompt.npz` and `demonstration.mp4` for the first the expert solves, and `result.json` with the chosen `scene_seed` and every attempt (exit 0 whenever `result.json` was written, every candidate rejected included; 1 on a harness error before it: a task it does not have, a seed given twice, no simulator); `robotwin-icil run-unit --prompt DIR/prompt.npz --policy replay --out DIR2` rebuilds the scene from the prompt's `meta`, verifies it, rolls out and writes `result.json` and `evaluation.mp4` into a directory other than the prompt's (exit 0 once the unit has a result, void or not; 1 on a harness error before it starts: no simulator, a policy that will not load, a served policy without a usable key). Both results carry `success`, `void`, `steps` and `error`, the fields the orchestrator reads; every candidate rejected is a void materialize, `void_cause` "harness". In `run-unit` a policy at fault (raising from `reset`/`set_demonstration`, a wrong-width or non-finite action) is a failure, never void — void is for the harness: an unreadable or tampered prompt, scene drift, a GPU lost mid-rollout, any other harness fault while evaluating (its traceback goes to stderr). `--policy-arg key=value` reaches the policy's constructor, which runs before the simulator chdirs into `vendor/RoboTwin`. `run-unit --policy-address ADDR --authkey-env NAME` (instead of `--policy`) drives a policy served by `python -m icil_policy.serve` (`remote.py`; icil-policy is not on PyPI: `uv pip install -e <orchestrator>/packages/icil-policy`, and the tests needing it skip without it): an error reply to reset/prompt/act fails the unit, any other remote failure (no listener, hello refused, a timeout, a hang-up, a malformed reply, the unit's calls past `--policy-budget-s`, 300 s) is `PolicyUnreachable`, void with `void_cause` "policy"; every other void carries "harness", including a unit that ran out of `--unit-timeout-s` (calls stop 30 s short of it) with the policy within budget. The policy's log is opened at the path given, never resolved (a swapped link must stay refusable), `close` gets 5 s, an action is checked before it is copied and what a server says is bounded. Both commands take `--expect-source-sha256` (exit 1 before writing under other `robotwin_icil.source_sha256()`) and `--denoiser`.
- Orchestrator plugin: `competition/` is its own distribution (`robotwin-icil-competition`, package `icil_benchmark_robotwin`, entry point `icil.benchmarks: robotwin`); `uv pip install -e ./competition`, then `cd competition && pytest -m "not sim"`. It never imports `icil_orchestrator`, its pure half imports no simulator, and its argv run `python -m robotwin_icil.cli` under `$ROBOTWIN_ICIL_PYTHON` (default `/root/miniforge3/envs/robotwin/bin/python`). A change to how `derive_units` draws changes every duel's units: bump `units.DERIVATION` and the pinned test together. A change to the task table changes `units.catalogue_sha256`, and `derive_units` refuses to draw until `units.CATALOGUE_SHA256` is updated with it. Suites exist only in the plugin (`catalogue.SUITES`), because the orchestrator's `spec.json` names one: `franka_1arm`, chosen from the Franka survey (`docs/results/survey-franka-seed0.json`, #83); stack_bowls_two is in it as the arm-switching stand-in for stacking, which has no one-arm task, and `info()["franka_1arm"]` says so. The pinned `franka_1arm` unit hashes must not move unless `units.DERIVATION` is bumped.
- RoboTwin is a pinned submodule at `vendor/RoboTwin`; never commit changes inside it.

## Rules

- `eval` and `survey` select all cataloged tasks without `--task`; `--task NAME` selects one.
  The core benchmark has no named task sets; do not add them to `tasks.yml`.
  `standard-eval` is separate: all 50 tasks on RoboTwin's evaluation seed stream (`--seed-stream
  robotwin`: from `100000 * (1 + seed)`, each episode continuing after the seeds its task's earlier
  episodes tried), aloha-agilex, save_freq=15, `demo_clean` and `demo_randomized` scored
  separately, no arm/task filters or scene overrides. Keep it matching
  `vendor/RoboTwin/scripts/eval_policy_xpolicylab.py`. `standard-report` aggregates per setting;
  only seed group 0 with 100 episodes/task, every episode scored, is an official score.
  There is no train/validation/test split, in the dataset or the evaluator.
  Dataset commands remain simulator-free and optional dependencies are loaded lazily. Preserve
  original HDF5 bytes, standard RGB JPEG decoding, three-view alignment, unknown physical timing,
  and every trajectory as training data. Task/seed/instruction/calibration metadata
  is bookkeeping, not policy input. Videos are lossy convenience views, HDF5 is authoritative.

- The protocol is fixed: exactly one demonstration per episode, demonstration and rollout from the
  identical initial scene. No K, no multi-shot, no zero-shot score, no ICL gain, no difficulty tiers.
- Never continue the rollout from the expert's final state. Close the env, `setup_demo` again with
  the same seed and the same `args`, and verify the fingerprint (actor names and poses, robot qpos,
  camera extrinsics) matches before the policy acts. A mismatch fails the episode loudly — same-seed
  drift from unseeded RNG is the bug this benchmark exists to not have.
- Expert failure is a generation failure, never a model failure: `UnStableError`, `plan_success`
  false, `check_success()` false, or any exception out of `play_once()`. Reject the episode, draw
  the next seed, count it in the generator statistics; it never enters the score's denominator.
- The policy is frozen. `policy.reset()` before every episode; no `backward()`, optimizer or
  parameter write anywhere in the evaluator. Inference-time state (KV cache, history) is fine.
- No privileged state reaches the policy: scene seed, success condition, target object or
  destination id, `info` from `play_once()`, ground-truth task state, actor handles, planner
  internals. If a model needs language, it gets `"Follow the demonstrated behavior."`.
- Demonstrations are model-independent (rgb per camera, endpose, qpos, frame times, control
  frequency). Per-model conversion lives in a `policies/` adapter; nothing in the core knows about
  ICRT. On disk a demonstration is `prompt.npz` (`prompt.py`): the same arrays under the published
  channel map, plus a `meta` JSON string that is privileged — task, seed, config, the initial
  scene's fingerprint and its digest — and is read only by the harness, never handed to a policy.
- Reuse RoboTwin's task setup, expert, success check and reset as they are. The benchmark exposes
  that expert as an on-demand demonstration generator; it does not reimplement or fork it. In-memory
  capture is a `_take_picture` override, not a patch to the submodule.
- `robotwin.py` is the only module that may import from `vendor/RoboTwin`; every other module stays
  importable without SAPIEN, assets or a GPU, and is covered by tests that run in CI.
- Skill categories and arm counts are data: `tasks.yml` maps every upstream task exactly once to a
  `category` and an `arms` value (`1`, `switching`, `2`), and tests fail when
  `vendor/RoboTwin/envs/` and the table disagree — on the task set, or on `arms` against `arms.py`'s
  static read of `play_once`. The table holds no task sets; the competition plugin's own track list
  lives in `competition/`.
- Evaluation settings are explicitly named (`same_scene`), and the setting is the seam future
  settings drop into (`different_object_pose`, …). Only `same_scene` is implemented.
- Scores are fractions `[0, 1]` over valid evaluated episodes; formatting to percent happens once,
  at report time.
- Every episode records episode id, setting, skill category, task, scene seed, embodiment, expert
  generation attempts, success, rollout steps and model/checkpoint; every run also records the
  global seed, the embodiment, both configs, both git commits and whether it asked for one-arm
  tasks only (`arms`, part of the run's identity). Videos go to
  `episode_NNNNN/{demonstration.mp4,evaluation_same_scene.mp4}`.

## Conventions

- Commit at completed, validated deliverables. Default to one commit per task or issue, keeping
  related implementation, tests, documentation and mechanical changes together. During ongoing
  work, batch edits and fixes in the working tree; do not commit after each file, plan step or
  test run. Split only when changes are independent deliverables that need separate review or
  reverts, or when explicitly requested. Each commit must pass the relevant checks and leave the
  repository in a working state.
- Commit title: `(feat): …`, `(fix): …`, `(refactor): …`, `(docs): …`, `(test): …`, `(chore): …`;
  imperative, lower-case after the prefix, under 72 characters, no trailing period. Body: why the
  change, not what the diff shows; short bullets; `Refs #N` for the issue it advances,
  `Closes #N` only on the commit that finishes it.
- Issues stand on their own: someone who was not in the conversation that produced one must be
  able to act on it. The title is the outcome in plain words - what is true once it closes
  ("Rebuild the identical scene and verify it matches") - not a component name or a plan step; a
  bug's title is its symptom. The body, in this order:
  - `## Why`: the problem, and what goes wrong without the change. No "see the plan", no "as
    discussed".
  - `## Scope`: the deliverable as concrete bullets (behaviour, files, commands), then
    `Out of scope:` for what a reader might expect and will not get.
  - `## Acceptance criteria`: a `- [ ]` checklist of things that can be checked - a test, a
    command and its result, an observable behaviour. Never "works well".
  - `## Notes`, optional: constraints, pitfalls, upstream references with paths, `Depends on #N`.
  - A bug has `## What happened` (the command, the commit, the evidence), `## Expected` and, once
    known, `## Cause`, in place of Why and Scope.
  - On closing, add `## Outcome`: what shipped and in which commits, the measured result, and anything
    that differs from the scope. A criterion that was dropped or changed is said, not silently
    ticked.
- One issue is one deliverable. Label it with its area, add `bug` for a defect, and put it in a
  milestone when the work belongs to one; split anything that will not land in one go and link the
  parts with `Depends on #N`.
- Commit directly to `main` and push to `origin/main`. Do not create branches or pull requests
  unless the user explicitly requests them. Include relevant tests in the same commit.
- Keep only `main` locally and on the remote unless the user explicitly requests another branch.
- Reports and evaluation results are plain files in the repository or run directory, not hosted
  artifacts.
