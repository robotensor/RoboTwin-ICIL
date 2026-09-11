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

## Model environments

Model adapters (`policies/`) run in environments of their own, never in the simulator env:
BPP's and UniSkill's stacks conflict with RoboTwin's pins (`huggingface_hub`, `transformers`,
`diffusers`) and with each other. `scripts/install_policy_env.sh` builds one with
[uv](https://docs.astral.sh/uv/) from the lockfiles in `policies/envs/<name>/`:

```bash
bash scripts/install_policy_env.sh bpp        # $ICIL_HOME/envs/icil-bpp
bash scripts/install_policy_env.sh uniskill   # $ICIL_HOME/envs/icil-uniskill
```

`ICIL_HOME` defaults to `~/.cache/robotwin-icil`, and model sources are checked out under
`$ICIL_HOME/src`. uv must be on `PATH` (the verified install used uv 0.12.13, from
`pip install uv`); it fetches its own python 3.10, so no conda is involved. Like
`install_robotwin.sh`, the script is idempotent per stage: checkouts and the backbone download
are skipped when present, and each install stage records a hash of its pins under
`$ENV/.icil-stages` and is skipped while they are unchanged. Delete the env directory to rebuild
it from scratch. It never touches the conda `robotwin` env.

| file in `policies/envs/<name>/` | what |
| --- | --- |
| `pins.env` | the model repository and commit, submodules, the torch index and pins, and each later stage's own pins |
| `requirements.lock` | the core dependencies, each one an exact pin |
| `constraints.txt` | torch, torchvision and numpy, held fixed while later stages install |

### `icil-bpp`

The stages codify the recipe verified on 2026-09-11 on an RTX 5090 (driver 580.173.02, compute
capability 12.0) in a container:

| Stage | What | Why |
| --- | --- | --- |
| source | `real-stanford/behavior_prompting` at `ec29e62`, submodules `deps/LIBERO` and `deps/icrt` | the model code and the LIBERO it imports |
| env | `uv venv --seed --python 3.10` | python 3.10.21, managed by uv |
| torch | `torch==2.8.0 torchvision==0.23.0` from the cu128 index | sm_120 kernels, which BPP's default 2.7.1+cu118 lacks; reports CUDA available, capability (12, 0) |
| core | `requirements.lock`: numpy 1.26.4, hydra-core 1.2.0, dill 0.3.7, timm 1.0.20, diffusers 0.35.1, transformers 4.57.1, huggingface_hub 0.35.3, ... | BPP's own stack |
| libero | mujoco 3.3.0, robosuite 1.4.0 and bddl 1.0.1 with uv; gym 0.21.0 with pip after `pip==24.0 setuptools==65.5.0 wheel==0.38.4`; `deps/LIBERO` editable | gym 0.21's setup.py fails under current build tools |
| robomimic | cmake, then the `austinapatel/robomimic` fork (branch `behavior_prompting_fixes`, pinned to `f4357ed`) with `CMAKE_POLICY_VERSION_MINIMUM=3.5`, under `constraints.txt` | BPP's LIBERO runner imports it; its egl-probe dependency needs cmake, and CMake 4 rejects egl-probe's old minimum |
| model | `behavior_prompting` editable, `--no-deps` | its own requirements would undo the stack above |
| benchmark | `robotwin-icil` and `policies[bpp,pure]` editable, under `constraints.txt` | the adapter, the core types it converts, and pytest for its contract tests; keyed on the checkout path and the extras, so the stage re-runs once for an env built before them |
| backbone | `timm.create_model("vit_base_patch16_clip_224.openai", pretrained=True)` | BPP builds its model with `pretrained=true` (with `false`, its weight init rejects the CLIP ViT's bias-free patch layer) and the checkpoint overwrites the weights, but timm downloads them at construction: 576 MB into `$HF_HOME/hub` once, so servers run with `HF_HUB_OFFLINE=1` |
| smoke | torch and CUDA, BPP's policy class, the benchmark, the backbone with `HF_HUB_OFFLINE=1` | |

On that environment BPP's own Hydra composition (`libero_policy_dunetp` with
`task=liberogen_spatial_combination`) built a 518.8M-parameter `DiffusionUnetPolicy`, the
LIBERO-Gen Combination checkpoint (6.9 GB) loaded with every key matched strictly, and a
16-step chunk took 0.09 s and 2.2 GB of GPU memory to predict.

### `icil-uniskill`

Built and verified on 2026-09-11 on the same RTX 5090; `policies/tests/test_uniskill_env.py` is
its contract (the fork builds `DiffusionPolicyUNet` from the adapter's config with diffusers'
current `EMAModel`, and the skill extractor runs on the GPU with the released weights):

| Stage | What | Why |
| --- | --- | --- |
| source | `KimHanjung/UniSkill` at `eca49f0` (the skill encoder); `kang-jaehyun/UniSkill-Policy` at `2803ad6` (the robomimic fork), without its LIBERO, robosuite and robocasa submodules | only the fork's LIBERO training and evaluation scripts import those, never the adapter, and robocasa pins numpy 1.23.3 |
| env | `uv venv --seed --python 3.10` | python 3.10.21, managed by uv |
| torch | `torch==2.8.0 torchvision==0.23.0` from the cu128 index | sm_120 kernels; the fork pins torch 2.0.1 |
| core | `requirements.lock`: transformers 4.57.1 (Depth-Anything-V2 needs at least 4.48), diffusers 0.35.1, huggingface_hub 0.35.3, numpy 1.26.4, ... | in place of the fork's transformers 4.36.2, diffusers 0.23.0 and huggingface-hub < 0.25; xformers and flash-attn, in the encoder's requirements, are left out because nothing imports them |
| model | the fork editable, `--no-deps`; the encoder's checkout on the env's path (`icil_uniskill_isd.pth`) | the fork's own pins would undo the stack above; the encoder is a script repository |
| benchmark | `robotwin-icil` and `policies[uniskill,pure]` editable, under `constraints.txt` | the adapter, and pytest for the contract test |
| smoke | torch and CUDA, `robomimic.algo`, `dynamics.idm`, `diffusers.training_utils`, transformers | |

The weights are not downloaded by the installer: `idm.pth` from `HanjungKim/UniSkill` and
Depth-Anything-V2-Small go under `$ICIL_HOME/models` ([models/uniskill.md](models/uniskill.md)).

## Troubleshooting

- **`cannot import name 'CuroboPlanner'`.** CuRobo is missing or failed to build. Upstream's
  `planner.py` swallows the import error, prints a warning, and leaves the name undefined, so the
  failure surfaces later in `robot.py`. Rerun the script and read the CuRobo build output.
- **`unsupported GNU version`** during the CuRobo build. nvcc picked up the system gcc; the
  script points `CC`/`CXX` at the env's gcc 12.
- **`CUDA out of memory` while building a scene.** Each RoboTwin robot builds two CuRobo motion
  planners on the GPU when its task env is created, a few GiB each. The runner keeps one task env
  alive at a time and releases it before the next task. If the GPU is shared with other jobs and
  still runs out, the run stops with the error rather than recording rejections: free GPU memory,
  then rerun `robotwin-icil eval` with the same arguments, which resumes where it stopped.
- **`Failed to find Vulkan ICD file`** at import. SAPIEN then ships its own ICD; on the reference
  machine rendering works regardless (upstream's `scripts/test_render.py` reports "Render Well").
  If rendering fails, install the NVIDIA Vulkan driver package matching the kernel driver.
- **`ModuleNotFoundError: pkg_resources`.** setuptools is too new; rerun the script, which pins it.
- **Embodiment `config.yml` or `curobo_left.yml` missing.** The asset stage did not finish; rerun
  the script.
- **Sim tests skipped or `vendor/RoboTwin` empty inside a git worktree.** Worktrees carry neither
  the submodule nor its assets; run sim tests from the main checkout.
