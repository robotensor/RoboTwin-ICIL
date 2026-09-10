#!/usr/bin/env bash
# Build the simulator environment the benchmark's `-m sim` tests and runs need.
#
# RoboTwin 2.0 pins python 3.10 with the numpy 1.x ABI, and its native extensions (sapien, mplib,
# toppra, CuRobo) will not resolve against a newer stack, so this creates a dedicated conda env
# rather than reusing the host interpreter. It follows upstream's `scripts/_install.sh` step for
# step, minus XPolicyLab and pytorch3d. Idempotent: every stage is skipped when already done.
#
#   bash scripts/install_robotwin.sh              # env, deps, patches, CuRobo, assets
#   ROBOTWIN_SKIP_ASSETS=1 bash scripts/...       # everything but the multi-GB asset download
#   ROBOTWIN_ROOT=/path/to/RoboTwin bash ...      # a RoboTwin checkout other than vendor/RoboTwin
#
# pytorch3d is deliberately not installed. RoboTwin imports it in a try/except in
# `envs/camera/camera.py` for farthest-point sampling of point clouds; the benchmark records rgb
# and joint state only, so paying for a CUDA source build buys nothing.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${REPO_ROOT}/vendor/RoboTwin}"
CONDA_ROOT="${CONDA_ROOT:-/root/miniforge3}"
ENV_NAME="${ROBOTWIN_ENV_NAME:-robotwin}"
PY_VERSION="3.10"
CUROBO_VERSION="v0.7.8"
# CuRobo is built against torch's CUDA; RoboTwin pins torch 2.4.1, whose default wheel is cu121.
CUDA_CHANNEL="nvidia/label/cuda-12.1.1"

log() { printf '\033[95m[install]\033[0m %s\n' "$*"; }

if [[ ! -f "${ROBOTWIN_ROOT}/envs/_base_task.py" ]]; then
    echo "${ROBOTWIN_ROOT} is empty; run: git submodule update --init --recursive" >&2
    exit 1
fi

# --- conda ------------------------------------------------------------------
if [[ ! -x "${CONDA_ROOT}/bin/conda" ]]; then
    log "installing miniforge into ${CONDA_ROOT}"
    installer="$(mktemp -d)/miniforge.sh"
    curl -fsSL -o "${installer}" \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    bash "${installer}" -b -p "${CONDA_ROOT}"
fi
export PATH="${CONDA_ROOT}/bin:${PATH}"

ENV_PREFIX="${CONDA_ROOT}/envs/${ENV_NAME}"
if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    log "creating env ${ENV_NAME} (python ${PY_VERSION})"
    conda create -y -q -n "${ENV_NAME}" "python=${PY_VERSION}"
fi
PY="${ENV_PREFIX}/bin/python"

# --- setuptools -------------------------------------------------------------
# Upstream's final pin. sapien 3.0.0b1 does `import pkg_resources`, which setuptools 81 removed,
# and conda-forge's python 3.10 ships a newer setuptools, so the pin goes first, not last.
if [[ "$("${PY}" -c 'import setuptools; print(setuptools.__version__)')" != "69.5.1" ]]; then
    log "pinning setuptools 69.5.1"
    "${PY}" -m pip install -q "setuptools==69.5.1"
fi

# --- RoboTwin dependencies --------------------------------------------------
if ! "${PY}" -c "import sapien, mplib, toppra" 2>/dev/null; then
    log "installing RoboTwin requirements"
    "${PY}" -m pip install -q -r "${ROBOTWIN_ROOT}/scripts/requirements.txt"
fi

# --- sapien urdf loader patch ----------------------------------------------
# From upstream's `_install.sh`: the embodiment URDFs carry non-ASCII bytes and name their
# companion file `.srdf`, neither of which stock sapien 3.0.0b1 handles.
URDF_LOADER="$("${PY}" -c 'import sapien, pathlib; print(pathlib.Path(sapien.__file__).parent)')/wrapper/urdf_loader.py"
if grep -q 'with open(urdf_file, "r") as f' "${URDF_LOADER}"; then
    log "patching sapien urdf_loader.py"
    sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "${URDF_LOADER}"
    sed -i 's/urdf_file\[:-4\] + "srdf"/urdf_file[:-4] + ".srdf"/' "${URDF_LOADER}"
fi

# --- mplib screw-plan patch -------------------------------------------------
# From upstream's `_install.sh`: drop `or collide` from mplib's screw-plan failure test. The
# expert's approach motions rely on it; without it, plans that brush an object fail, the expert's
# success rate drops, and the demonstrations are no longer RoboTwin's.
MPLIB_PLANNER="$("${PY}" -c 'import mplib, pathlib; print(pathlib.Path(mplib.__file__).parent / "planner.py")')"
if grep -q "< 1e-4 or collide or not within_joint_limit" "${MPLIB_PLANNER}"; then
    log "patching mplib planner.py"
    sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "${MPLIB_PLANNER}"
fi

# --- CuRobo -----------------------------------------------------------------
# Every RoboTwin embodiment plans with CuRobo (`planner: "curobo"`), and `envs/robot/robot.py`
# imports `CuroboPlanner` at module level, so no task even imports without it. Upstream clones
# it into `envs/curobo`, which RoboTwin's .gitignore covers. Its CUDA kernels need an nvcc that
# matches torch's CUDA and a host compiler nvcc 12.1 accepts (gcc <= 12); a driver-only machine
# has neither, so both go into the env.
if ! "${PY}" -c "import curobo" 2>/dev/null; then
    TORCH_CUDA="$("${PY}" -c 'import torch; print(torch.version.cuda)')"
    if [[ "${TORCH_CUDA}" != "12.1" ]]; then
        echo "torch reports CUDA ${TORCH_CUDA}; this script builds CuRobo for 12.1 only" >&2
        exit 1
    fi
    if [[ ! -x "${ENV_PREFIX}/bin/nvcc" ]]; then
        log "installing the CUDA 12.1 toolkit into ${ENV_NAME} for the CuRobo build"
        conda install -y -q -n "${ENV_NAME}" -c "${CUDA_CHANNEL}" cuda-toolkit
    fi
    if [[ ! -x "${ENV_PREFIX}/bin/x86_64-conda-linux-gnu-g++" ]]; then
        log "installing gcc 12 into ${ENV_NAME} (nvcc 12.1 rejects newer host compilers)"
        conda install -y -q -n "${ENV_NAME}" -c conda-forge "gcc_linux-64=12" "gxx_linux-64=12"
    fi
    CUROBO_DIR="${ROBOTWIN_ROOT}/envs/curobo"
    if [[ ! -d "${CUROBO_DIR}/.git" ]]; then
        git clone -q --branch "${CUROBO_VERSION}" --depth 1 https://github.com/NVlabs/curobo.git "${CUROBO_DIR}"
    fi
    "${PY}" -m pip install -q "setuptools-scm==8.1.0" wheel
    ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
    log "building CuRobo ${CUROBO_VERSION} for compute capability ${ARCH} (several minutes)"
    (
        cd "${CUROBO_DIR}"
        export CUDA_HOME="${ENV_PREFIX}"
        export CC="${ENV_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
        export CXX="${ENV_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
        export TORCH_CUDA_ARCH_LIST="${ARCH}"
        export PATH="${ENV_PREFIX}/bin:${PATH}"
        "${PY}" -m pip install -q -e . --no-build-isolation
    )
    "${PY}" -m pip install -q "warp-lang==1.12.0"
fi

# --- benchmark package ------------------------------------------------------
if ! "${PY}" -c "import robotwin_icil" 2>/dev/null; then
    log "installing robotwin_icil into ${ENV_NAME}"
    "${PY}" -m pip install -q -e "${REPO_ROOT}" --no-deps
    "${PY}" -m pip install -q pytest "imageio-ffmpeg>=0.5"
fi

# --- assets -----------------------------------------------------------------
if [[ "${ROBOTWIN_SKIP_ASSETS:-0}" != "1" ]]; then
    if [[ ! -d "${ROBOTWIN_ROOT}/assets/objects" ]]; then
        log "downloading assets (~15 GB, resumable)"
        (cd "${ROBOTWIN_ROOT}/assets" && "${PY}" _download.py)
        for zip in background_texture embodiments objects; do
            [[ -f "${ROBOTWIN_ROOT}/assets/${zip}.zip" ]] || continue
            (cd "${ROBOTWIN_ROOT}/assets" && unzip -q -o "${zip}.zip" && rm -f "${zip}.zip")
        done
    fi
    if [[ ! -f "${ROBOTWIN_ROOT}/assets/embodiments/aloha-agilex/curobo_left.yml" ]]; then
        log "configuring embodiment paths"
        (cd "${ROBOTWIN_ROOT}" && "${PY}" ./scripts/update_embodiment_config_path.py)
    fi
fi

log "done. interpreter: ${PY}"
