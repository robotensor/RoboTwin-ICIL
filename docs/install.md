# Installing the simulator

The pure parts of the benchmark — task table, records, reporting — need only the host tools:

```bash
uv venv --python 3.10 .venv && uv pip install -e ".[dev]"
pytest -m "not sim"
```

Anything that builds a scene needs RoboTwin 2.0's own stack, which `scripts/install_robotwin.sh`
builds in a dedicated conda env. It is idempotent per stage, so a failed download resumes:

```bash
git submodule update --init --recursive
bash scripts/install_robotwin.sh             # env, deps, sapien patch, assets
RT=/root/miniforge3/envs/robotwin/bin/python
$RT -m pytest -m sim
```

## What it does, and why

| Stage | What | Why |
| --- | --- | --- |
| conda | Miniforge, env `robotwin`, python 3.10 | RoboTwin's native extensions pin python 3.10 and the numpy 1.x ABI. |
| setuptools | `setuptools<81` | sapien 3.0.0b1 does `import pkg_resources`, removed in setuptools 81; conda-forge ships 84. |
| deps | `vendor/RoboTwin/scripts/requirements.txt` | RoboTwin's own pins, unmodified. |
| sapien patch | `urdf_loader.py`: utf-8 reads, `.srdf` suffix | The same edit upstream `_install.sh` applies; the embodiment URDFs need it. |
| assets | `TianxingChen/RoboTwin2.0` on Hugging Face | Objects, embodiments and background textures; ignored by RoboTwin's `.gitignore`. |

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
| GPU / driver | NVIDIA RTX A6000 (48 GB), driver 595.71.05 |
| python | 3.10 (conda-forge) |
| torch | 2.4.1+cu121, CUDA available |
| numpy | 1.26.4 |
| sapien | 3.0.0b1, urdf loader patched |
| assets | `background_texture.zip` 10.97 GB, `objects.zip` 3.74 GB, `embodiments.zip` 0.22 GB |

## Troubleshooting

- **`Failed to find Vulkan ICD file`** at import. SAPIEN then ships its own ICD; on the reference
  machine rendering works regardless (upstream's `scripts/test_render.py` reports "Render Well").
  If rendering fails, install the NVIDIA Vulkan driver package matching the kernel driver.
- **`ModuleNotFoundError: pkg_resources`.** setuptools is too new; rerun the script, which pins it.
- **Embodiment `config.yml` missing.** The asset stage did not finish; rerun the script.
