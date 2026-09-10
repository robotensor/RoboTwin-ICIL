# Installing the simulator

The pure parts of the benchmark — task table, records, reporting — need only the host tools:

```bash
uv venv --python 3.10 .venv && uv pip install -e ".[dev]"
pytest -m "not sim"
```

Anything that builds a scene needs RoboTwin 2.0's own stack, which `scripts/install_robotwin.sh`
builds in a dedicated conda env. It follows upstream's `scripts/_install.sh` step for step, and is
idempotent per stage, so an interrupted download or build resumes:

```bash
git submodule update --init --recursive
bash scripts/install_robotwin.sh
RT=/root/miniforge3/envs/robotwin/bin/python
PYTHONPATH=src $RT -m pytest -m sim
```

## What it does, and why

| Stage | What | Why |
| --- | --- | --- |
| conda | Miniforge, env `robotwin`, python 3.10 | RoboTwin's native extensions pin python 3.10 and the numpy 1.x ABI. |
| setuptools | `setuptools==69.5.1` | Upstream's final pin. sapien 3.0.0b1 does `import pkg_resources`, removed in setuptools 81, and conda-forge ships 84 — so it goes first. |
| deps | `vendor/RoboTwin/scripts/requirements.txt` | RoboTwin's own pins, unmodified. |
| sapien patch | `urdf_loader.py`: utf-8 reads, `.srdf` suffix | Upstream's edit; the embodiment URDFs need it. |
| mplib patch | drop `or collide` from the screw-plan failure test | Upstream's edit. The expert's approach motions depend on it; without it the expert fails more often and its demonstrations are no longer RoboTwin's. |
| CUDA toolchain | `cuda-toolkit` 12.1.1 (`nvidia/label/cuda-12.1.1`) and gcc 12 (conda-forge), inside the env | Only to build CuRobo: nvcc must match torch's cu121, and nvcc 12.1 rejects host compilers newer than gcc 12. |
| CuRobo | v0.7.8 cloned into `vendor/RoboTwin/envs/curobo`, built for this GPU's compute capability, plus `warp-lang==1.12.0` | Every RoboTwin embodiment plans with CuRobo (`planner: "curobo"`), and `envs/robot/robot.py` imports `CuroboPlanner` at module level — without it no task imports at all. |
| assets | `TianxingChen/RoboTwin2.0` on Hugging Face, then upstream's `update_embodiment_config_path.py` | Objects, embodiments and background textures (~15 GB); ignored by RoboTwin's `.gitignore`. |

Deliberately skipped:

- **pytorch3d.** RoboTwin imports it in a `try/except` in `envs/camera/camera.py`, only for
  farthest-point sampling of point clouds. The benchmark records rgb and joint state, so a CUDA
  source build buys nothing.
- **XPolicyLab.** Upstream's `_install.sh` pulls it for its own policy server; the benchmark has its
  own policy interface.

## Reference install

What the script produced on the machine the V1 numbers come from:

| | |
| --- | --- |
| GPU / driver | NVIDIA RTX A6000 (48 GB, compute capability 8.6), driver 595.71.05 |
| python | 3.10 (conda-forge) |
| torch | 2.4.1+cu121, CUDA available |
| numpy | 1.26.4 |
| sapien | 3.0.0b1, urdf loader patched |
| mplib | 0.2.1, screw-plan test patched |
| CuRobo | v0.7.8, built with nvcc 12.1 and gcc 12 for sm_86; warp-lang 1.12.0 |
| assets | `background_texture.zip` 10.97 GB, `objects.zip` 3.74 GB, `embodiments.zip` 0.22 GB |

## Troubleshooting

- **`cannot import name 'CuroboPlanner'`.** CuRobo is missing or failed to build. Upstream's
  `planner.py` swallows the import error, prints a warning, and leaves the name undefined, so the
  failure surfaces later in `robot.py`. Rerun the script and read the CuRobo build output.
- **`unsupported GNU version`** during the CuRobo build. nvcc picked up the system gcc; the
  script points `CC`/`CXX` at the env's gcc 12.
- **`Failed to find Vulkan ICD file`** at import. SAPIEN then ships its own ICD; on the reference
  machine rendering works regardless (upstream's `scripts/test_render.py` reports "Render Well").
  If rendering fails, install the NVIDIA Vulkan driver package matching the kernel driver.
- **`ModuleNotFoundError: pkg_resources`.** setuptools is too new; rerun the script, which pins it.
- **Embodiment `config.yml` or `curobo_left.yml` missing.** The asset stage did not finish; rerun
  the script.
- **Sim tests skipped or `vendor/RoboTwin` empty inside a git worktree.** Worktrees carry neither
  the submodule nor its assets; run sim tests from the main checkout.
