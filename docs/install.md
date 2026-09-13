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
git submodule update --init vendor/RoboTwin
bash scripts/install_robotwin.sh
RT=/root/miniforge3/envs/robotwin/bin/python
PYTHONPATH=src $RT -m pytest -m sim
```

The script picks one of two GPU paths from `nvidia-smi`'s compute capability, and the simulator runs
on either. `ROBOTWIN_GPU_PATH=reference` or `blackwell` overrides the choice.

| Path | GPUs | torch | CUDA toolkit for CuRobo | Host compiler |
| --- | --- | --- | --- | --- |
| reference | compute capability below 10.0: Ampere, Ada, Hopper, and the A6000 the V1 numbers come from | 2.4.1+cu121, RoboTwin's pin | 12.1.1 | gcc 12 |
| blackwell | 10.0 and above; an RTX 5090 is 12.0 | 2.8.0+cu128 | 12.8.1 | gcc 13 |

The reference stack cannot run on Blackwell GPUs at all: torch 2.4.1 ships no kernels for them, its
extension builder rejects arch 12.0, and nvcc 12.1 cannot target sm_120.

## What it does, and why

| Stage | What | Why |
| --- | --- | --- |
| conda | Miniforge, env `robotwin`, python 3.10 | RoboTwin's native extensions pin python 3.10 and the numpy 1.x ABI. |
| setuptools | `setuptools==69.5.1` | Upstream's final pin. sapien 3.0.0b1 does `import pkg_resources`, removed in setuptools 81, and conda-forge ships 84 — so it goes first. |
| deps | `vendor/RoboTwin/scripts/requirements.txt`; on the blackwell path torch 2.8.0 and torchvision 0.23.0 from the cu128 index first, then the requirements without their torch lines | RoboTwin's own pins, unmodified on the reference path; on the blackwell path pip must not pull torch 2.4.1 back in. |
| sapien patch | `urdf_loader.py`: utf-8 reads, `.srdf` suffix | Upstream's edit; the embodiment URDFs need it. |
| mplib patch | drop `or collide` from the screw-plan failure test | Upstream's edit. The expert's approach motions depend on it; without it the expert fails more often and its demonstrations are no longer RoboTwin's. |
| rendering | `libegl1`, then an NVIDIA Vulkan ICD manifest (`nvidia_icd.json`) and EGL vendor manifest (`10_nvidia.json`) when the system has none; upstream's `scripts/test_render.py` must print "Render Well" | SAPIEN renders through NVIDIA's Vulkan driver, which needs glvnd's `libEGL.so.1` and both manifests. Containers made by the NVIDIA container toolkit often inject only the driver libraries. As root the manifests go where a driver package puts them; otherwise into the env, where `robotwin_icil` points SAPIEN at them. |
| CUDA toolchain | `cuda-toolkit` from `nvidia/label/cuda-12.1.1` with gcc 12 (reference) or `nvidia/label/cuda-12.8.1` with gcc 13 (blackwell), from conda, inside the env | Only to build CuRobo: nvcc must match torch's CUDA. nvcc 12.1 rejects host compilers newer than gcc 12, and CUDA 12.8 is the first toolkit that targets sm_100 and sm_120. |
| CuRobo | v0.7.8 cloned into `vendor/RoboTwin/envs/curobo`, built for this GPU's compute capability with `TORCH_CUDA_ARCH_LIST` set explicitly (torch's autodetection misreads some Blackwell cards, curobo#596), plus `warp-lang==1.12.0` | Every RoboTwin embodiment plans with CuRobo (`planner: "curobo"`), and `envs/robot/robot.py` imports `CuroboPlanner` at module level — without it no task imports at all. |
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
  script points `CC`/`CXX` at the env's gcc (12 on the reference path, 13 on blackwell).
- **`no kernel image is available for execution on the device`** or **`Unknown CUDA arch (12.0) or
  GPU not supported`.** torch 2.4.1 on a Blackwell GPU: the env was built on the reference path.
  Remove the env and rerun the script, which detects compute capability 10.0 and above, or set
  `ROBOTWIN_GPU_PATH=blackwell`.
- **`CUDA out of memory` while building a scene.** Each RoboTwin robot builds two CuRobo motion
  planners on the GPU when its task env is created, a few GiB each. The runner keeps one task env
  alive at a time and releases it before the next task. If the GPU is shared with other jobs and
  still runs out, the run stops with the error rather than recording rejections: free GPU memory,
  then rerun `robotwin-icil eval` with the same arguments, which resumes where it stopped.
- **`FileNotFoundError: '/usr/share/glvnd/egl_vendor.d'`** at `import sapien`, or **`failed to find a
  rendering device`** when a scene is built. The machine has the NVIDIA driver libraries but not
  glvnd's `libEGL.so.1` or the NVIDIA EGL vendor and Vulkan ICD manifests, which is common in
  containers; the NVIDIA Vulkan driver cannot start without them. Rerun the script: its rendering
  stage installs `libegl1` (as root, or tells you to) and writes the manifests. Setting
  `VK_ICD_FILENAMES` alone does not help, because the driver fails to initialise without
  `libEGL.so.1`. Do not reinstall the driver's GL packages inside a container over the libraries the
  toolkit mounts. SAPIEN's `Failed to find Vulkan ICD file` warning at import is harmless once
  `scripts/test_render.py` reports "Render Well".
- **A run stops making progress, its process asleep in `get_obs`.** SAPIEN's OIDN denoiser, which
  RoboTwin requests, cannot run on Blackwell GPUs: it logs `OIDN Error: unsupported device type:
  CUDA` and `invalid handle` and leaves each image as rendered, and once another process loads the
  GPU (a training job, or a second simulator) its failing path hangs the camera read for good.
  `robotwin_icil` therefore turns the denoiser off at compute capability 10.0 and above, which
  changes no pixel there (renders compared on the RTX 5090); set `ROBOTWIN_ICIL_DENOISER=oidn` or
  `none` to override. With the denoiser off a render still hangs occasionally on a long,
  render-heavy run: an end-effector replay stopped after nine episodes with the GPU otherwise idle,
  in the same camera read, and with a training job holding the GPU a one-episode click_bell smoke
  hung twice in the expert's first frames before a third attempt scored in 51 s. So run long jobs
  under `scripts/simwatch.py`, which kills an attempt whose CPU time stops advancing or that
  outlives a wall-clock cap, and reruns it. A stopped `robotwin-icil eval` resumes from
  `episodes.jsonl`, so the rerun picks up where it stopped.
- **`ModuleNotFoundError: pkg_resources`.** setuptools is too new; rerun the script, which pins it.
- **Embodiment `config.yml` or `curobo_left.yml` missing.** The asset stage did not finish; rerun
  the script.
- **Sim tests skipped or `vendor/RoboTwin` empty inside a git worktree.** Worktrees carry neither
  the submodule nor its assets; run sim tests from the main checkout.
