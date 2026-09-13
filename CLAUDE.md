# robotwin-icil — Same Scene 1-demo ICIL benchmark on RoboTwin 2.0

Python 3.10, package `robotwin_icil` under `src/`. A separate benchmark that uses RoboTwin 2.0 for
simulation, tasks, scene generation, the expert, execution and success checking. Every episode
generates its own demonstration at evaluation time: pick a task and scene seed, run RoboTwin's
existing expert until one demonstration succeeds, recreate that exact scene, hand the frozen policy
that one demonstration, roll out, score with RoboTwin's own `check_success()`. No dataset, no
training, no train/eval split. The only V1 score is Same Scene 1-Demo Success Rate, reported
overall, by skill category and by task. Follows the conventions in `../CLAUDE.md`.

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
  rate; upstream passes `save_freq` where it means a frame rate, and so did we once.
- RoboTwin renders with SAPIEN's ray tracer and asks for the OIDN denoiser, which cannot run on
  Blackwell GPUs: it leaves images untouched and, under GPU contention, hangs camera reads.
  `robotwin.py` turns it off at compute capability 10.0 and above (`ROBOTWIN_ICIL_DENOISER`).
- Assets load from `./assets/...` relative to the working directory, so `robotwin.py` chdirs into
  `vendor/RoboTwin`. Resolve any path (run dir, checkpoint) to absolute before calling into it.

## Commands

- Host env (pure, no simulator): `uv venv --python 3.10 .venv && uv pip install -e ".[dev]"`; `ruff check . && ruff format --check .`; `pytest -m "not sim"`.
- Simulator env: `bash scripts/install_robotwin.sh` (conda env `robotwin` under `/root/miniforge3`, python 3.10, RoboTwin's own pins + assets); `PYTHONPATH=src $RT -m pytest -m sim` with `RT=/root/miniforge3/envs/robotwin/bin/python`. Run it from the main checkout: git worktrees have no `vendor/RoboTwin` checkout or assets. One simulator
  process per GPU at a time: two processes rendering at once can hang in SAPIEN's camera read.
- Smoke: `robotwin-icil eval --policy replay --task click_bell --episodes 1 --seed 42 --run-dir runs/smoke`, then `robotwin-icil report runs/smoke`. `eval` and `survey` take `--embodiment aloha-agilex` or `franka-panda`; without it the task config's own robot runs (aloha-agilex in every shipped config). A run directory is tied to its robot.
- RoboTwin is a pinned submodule at `vendor/RoboTwin`; never commit changes inside it.

## Rules

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
- Demonstrations are model-independent (rgb per camera, endpose, qpos, control frequency). Per-model
  conversion lives in a `policies/` adapter; nothing in the core knows about ICRT.
- Reuse RoboTwin's task setup, expert, success check and reset as they are. The benchmark exposes
  that expert as an on-demand demonstration generator; it does not reimplement or fork it. In-memory
  capture is a `_take_picture` override, not a patch to the submodule.
- `robotwin.py` is the only module that may import from `vendor/RoboTwin`; every other module stays
  importable without SAPIEN, assets or a GPU, and is covered by tests that run in CI.
- Skill categories are data: `tasks.yml` maps every upstream task exactly once and a test fails when
  `vendor/RoboTwin/envs/` and the table disagree. Suites are named there too; V1 is `v1`.
- Evaluation settings are explicitly named (`same_scene`), and the setting is the seam future
  settings drop into (`different_object_pose`, …). V1 implements only `same_scene`.
- Scores are fractions `[0, 1]` over valid evaluated episodes; formatting to percent happens once,
  at report time.
- Every episode records episode id, setting, skill category, task, scene seed, embodiment, expert
  generation attempts, success, rollout steps and model/checkpoint; every run also records the
  global seed, the embodiment, both configs and both git commits. Videos go to
  `episode_NNNNN/{demonstration.mp4,evaluation_same_scene.mp4}`.

## Conventions

- Small commits. One concern per commit (a rename, a schema change, a new stage, a doc update),
  never a whole issue in one commit. Each commit builds and passes the pure tests on its own, so
  the history bisects and reverts cleanly. Split mechanical moves from behaviour changes.
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
  - On closing, add `## Outcome`: what shipped and in which PRs, the measured result, and anything
    that differs from the scope. A criterion that was dropped or changed is said, not silently
    ticked.
- One issue is one deliverable. Label it with its area, add `bug` for a defect, and put it in a
  milestone when the work belongs to one; split anything that will not land in one go and link the
  parts with `Depends on #N`.
- A branch carries a theme, not an issue number: related issues that touch the same code ship on
  one branch (`short-slug`, or `issue-N-short-slug` when it really is a single issue) and land in
  one PR, which says `Closes #N` for every issue it finishes and `Refs #N` for the ones it only
  advances. Tests and a CHANGELOG entry land with it. Rebase, do not merge `main` into the branch.
- Small changes go straight to `main`: a typo, a comment, a doc line, a version bump, a one-line
  fix that comes with its test. Anything that changes behaviour a reader would need explained,
  touches a published contract, or wants a second pair of eyes takes a branch and a PR.
- When a branch is merged, delete it locally and on the remote, so only `main`, long-lived
  `milestone-*` branches and deliberate `archive/*` refs remain.
- Reports and evaluation results are plain files in the repository or run directory, not hosted
  artifacts.
