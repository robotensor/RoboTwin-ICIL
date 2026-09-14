# robotwin-icil-competition

The RoboTwin ICIL benchmark (`robotwin-icil`, the repository root) as a plugin of the RoboTensor
ICIL competition orchestrator. It is its own distribution, found through the `icil.benchmarks`
entry point group under the name `robotwin`, and it never imports the orchestrator.

```bash
# beside icil-orchestrator, on the orchestrator host: none of the three is on a package index
uv pip install -e <this repository> -e <this repository>/competition \
    -e <orchestrator repository>/packages/icil-policy
icil-orchestrator benchmarks check robotwin
```

The orchestrator pins it by distribution, version and wheel sha256 in its `spec.json`.

## What it exposes

Pure — no simulator, no assets, no GPU:

- `info()`: id, ABI version, the robots and their action widths, the observed cameras, the
  protocol (`same_initial_state`, view `sensorimotor`), the prompt's channel map, the benchmark
  and RoboTwin commits, the command line it builds, and what is provisional.
- `catalogue()`: `{"suites": {suite: [task]}, "categories": {id: label}, "tasks": {task:
  {"category", "arms"}}}` from `robotwin_icil.tasks`.
- `derive_units(seed_material, count, suite, category)`: `count` units from a sha256 counter over
  `seed_material`, tasks spread evenly over the suite (within `category`), each unit's
  `instance_params` holding `SCENE_SEED_CANDIDATES` candidate `scene_seeds` and its `embodiment`.
- `verify_prompt(path, unit)`: reads `prompt.npz` (never unpickling) and checks its task, scene
  seed, robot, arrays and scene digest against the unit.
- `read_result(out_dir)`: either command's `result.json`, with `void_cause`.

Commands — the argv of `python -m robotwin_icil.cli` in the simulator's environment
(`ROBOTWIN_ICIL_PYTHON`, default `/root/miniforge3/envs/robotwin/bin/python`):

- `materialize_command(unit, out_dir)`: `materialize` with every candidate seed.
- `run_command(unit, prompt, out_dir, policy_address, authkey_env, **extra)`: `run-unit
  --policy-address`, with `--act-timeout-s` and `--policy-log` when `extra` carries
  `act_timeout_s` and `policy_log`.

## Provisional

The `franka_1arm` suite is named by the Franka survey (robotensor/ICIL-robotwin-benchmark#83),
which has not run. Until then the plugin serves a provisional one: the one-arm tasks of Pick and
Place and Press / Push, and — since no stacking expert uses a single arm, each picks its arm per
object — the stacking tasks whose expert switches arms. The arms each task needs come from
`robotwin_icil.tasks` once its table records them; until this branch carries that table, from a
copy in `catalogue.py`. `info()["provisional"]` says so.

## Tests

```bash
cd competition && pytest -q -m "not sim"
```
