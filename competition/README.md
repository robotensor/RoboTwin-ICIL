# robotwin-icil-competition

The RoboTwin-ICIL benchmark (`robotwin-icil`, the repository root) as a plugin of the Robotensor
ICIL competition orchestrator. It is its own distribution, found through the `icil.benchmarks`
entry point group under the name `robotwin`, and it never imports the orchestrator.

## Installing

The orchestrator pins the plugin by distribution, version and wheel sha256 in its `spec.json`, and
`icil-orchestrator benchmarks check robotwin` can confirm a wheel's sha256 only for a distribution
installed from that wheel: an editable install records none. On the orchestrator host, build the
wheels and install them (none of the three is on a package index):

```bash
uv build --wheel --out-dir dist <this repository>
uv build --wheel --out-dir dist <this repository>/competition
uv pip install dist/robotwin_icil-*.whl dist/robotwin_icil_competition-*.whl \
    <orchestrator repository>/packages/icil-policy
icil-orchestrator benchmarks check robotwin
```

The plugin depends on `robotwin-icil` at exactly the version released with it. For development,
editable installs do (`uv pip install -e <this repository> -e <this repository>/competition -e
<orchestrator repository>/packages/icil-policy`), and `benchmarks check` then notes the pin it
cannot confirm.

The simulator environment (`$ROBOTWIN_ICIL_PYTHON`) must run the same `robotwin_icil` source as the
plugin imports: every command the plugin builds carries `--expect-source-sha256` with
`robotwin_icil.source_sha256()`, and exits 1 before writing anything under any other.

## What it exposes

Pure — no simulator, no assets, no GPU:

- `info()`: the id and ABI version; `benchmark`, with `source_sha256`, the digest of the
  `robotwin_icil` source the commands must run, and `plugin`; the benchmark and RoboTwin commits,
  null unless `robotwin_icil` is imported from the top of a git checkout (a wheel has none); the
  robots and their action widths, the observed cameras, the protocol (`same_initial_state`, view
  `sensorimotor`), the prompt's channel map; `catalogue_sha256`, `pinned_catalogue_sha256` and
  `derivation`; the `limits` a served policy gets and the `run_extra` keys that set them; the
  command line; and `franka_1arm`: the survey that chose it, each category's tasks and arms, and
  its `stand_ins`.
- `catalogue()`: `{"suites": {suite: [task]}, "categories": {id: label}, "tasks": {task:
  {"category", "arms", "label"}}, "embodiments": {suite: robot}}`: tasks, categories and arms from
  `robotwin_icil.tasks`, suites from this plugin's `catalogue.SUITES`.
- `derive_units(seed_material, count, suite, category)`: `count` units from a sha256 counter over
  `seed_material`, tasks spread evenly over the suite (within `category`). Each unit's
  `instance_params` holds `SCENE_SEED_CANDIDATES` candidate `scene_seeds`, its `embodiment`, and
  `scene_seed` null: which candidate becomes the scene is known only once `materialize` has run,
  and its `result.json`, the prompt's `meta` and `verify_prompt`'s verdict name it. Units are drawn
  only from the task catalogue they were pinned on (`units.CATALOGUE_SHA256`): `robotwin-icil` is
  not in the plugin's wheel, and another table would derive other units under the same pin.
- `verify_prompt(path, unit)`: reads `prompt.npz`, never unpickling it, and holds it to the unit —
  its task, robot and a candidate scene seed; an expert record that tried the unit's candidates in
  order up to that seed and rejected every earlier one; the scene config every unit is built with
  (`demo_clean`, `save_freq` 15, no head camera or override, `same_scene`); arrays of the published
  channel map holding a demonstration of the robot's width, from the head and wrist cameras only;
  and the scene digest.
- `read_result(out_dir)`: either command's `result.json`, with `void_cause`. A result that is
  missing, unreadable or inconsistent, or that other benchmark source wrote (`source_sha256`), is
  void on the harness.

Commands — the argv of `python -m robotwin_icil.cli` in the simulator's environment
(`ROBOTWIN_ICIL_PYTHON`, default `/root/miniforge3/envs/robotwin/bin/python`), each with
`--expect-source-sha256` and, when `$ROBOTWIN_ICIL_DENOISER` is set where the argv is built,
`--denoiser`, since the orchestrator does not hand the subprocess that variable:

- `materialize_command(unit, out_dir)`: `materialize` with every candidate seed.
- `run_command(unit, prompt, out_dir, policy_address, authkey_env, **extra)`: `run-unit
  --policy-address`, with `--act-timeout-s`, `--policy-budget-s`, `--unit-timeout-s` and
  `--policy-log` when `extra` carries `act_timeout_s`, `policy_budget_s`, `unit_timeout_s` and
  `policy_log`. A time limit that is not a positive, finite number is refused here rather than
  voiding the unit when run-unit refuses it.

## What the orchestrator should pass and provide

- `unit_timeout_s`: the seconds it gives run-unit before killing it. With it, a unit whose harness
  runs long is written void on the harness 30 s before the kill instead of leaving no result; it
  must leave the harness its own time beyond the policy's budget.
- `policy_budget_s`, unless 300 s is right: how long all of one unit's calls to a policy may take.
  Running out is void with `void_cause` "policy", the side's loss, so a policy cannot run a unit it
  is losing into the kill that would void it for both sides.
- `policy_log`: a copy of the policy server's log in a directory the policy cannot write.
  run-unit never follows a link at the log's path, but only the last path component is guarded.
- A GPU the policy does not share, or a cap on its memory: a GPU that runs out or is lost
  mid-rollout voids the unit on the harness. The result then lists `gpu_processes` (pid and MiB,
  from `nvidia-smi`) and `run_unit_pid`, from which the orchestrator can tell whether the policy's
  processes held the memory.

## The franka_1arm suite

Suites belong to this plugin alone: the orchestrator's spec names one, and `derive_units` draws
from it. The benchmark itself has no named task sets; the plugin invokes `materialize --task NAME`
and `run-unit`. Moving the list here from the benchmark's task table left the `franka_1arm` units
and their pinned hashes unchanged.

`franka_1arm`, the suite the one-arm Franka track draws from (`catalogue.SUITES`), is
place_empty_cup (pick_and_place), stack_bowls_two (stacking), click_bell and press_stapler
(press_push). The Franka survey chose it (robotensor/RoboTwin-ICIL#83, measurements in
[`docs/results/survey-franka-seed0.json`](../docs/results/survey-franka-seed0.json)):
a task is in it when its expert, on two Franka arms, solved at least 2 of 3 surveyed seeds and
moved one arm in every successful demonstration. Stacking has no one-arm task, so
stack_bowls_two, an arm-switching task, stands in for it, with a limit: a scene the survey did not
see can still make its expert switch arms, and neither `materialize` nor `verify_prompt` refuses a
demonstration that moved both. `derive_units` draws a skill's units only from its category of the
suite. `info()["franka_1arm"]` names the survey, each category's tasks and arms, and the stand-in
in `stand_ins`; `benchmarks check` prints only its pin notes and `ok`.

## Tests

```bash
cd competition && pytest -q -m "not sim"
```
