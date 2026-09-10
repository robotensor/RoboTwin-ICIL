#!/usr/bin/env bash
# Build the simulator environment the benchmark's `-m sim` tests and runs need.
#
# RoboTwin 2.0 pins python 3.10 with the numpy 1.x ABI, and its native extensions (sapien, mplib,
# toppra) will not resolve against a newer stack, so this creates a dedicated conda env rather
# than reusing the host interpreter. Idempotent: every stage is skipped when already done.
#
#   bash scripts/install_robotwin.sh              # env, deps, sapien patch, assets
#   ROBOTWIN_SKIP_ASSETS=1 bash scripts/...       # everything but the multi-GB asset download
#
# pytorch3d is deliberately not installed. RoboTwin imports it in a try/except in
# `envs/camera/camera.py` for farthest-point sampling of point clouds; the benchmark records rgb
# and joint state only, so paying for a CUDA source build buys nothing.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOTWIN_ROOT="${REPO_ROOT}/vendor/RoboTwin"
CONDA_ROOT="${CONDA_ROOT:-/root/miniforge3}"
ENV_NAME="${ROBOTWIN_ENV_NAME:-robotwin}"
PY_VERSION="3.10"

log() { printf '\033[95m[install]\033[0m %s\n' "$*"; }

if [[ ! -f "${ROBOTWIN_ROOT}/envs/_base_task.py" ]]; then
    echo "vendor/RoboTwin is empty; run: git submodule update --init --recursive" >&2
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

# --- RoboTwin dependencies --------------------------------------------------
if ! "${PY}" -c "import sapien, mplib, toppra" 2>/dev/null; then
    log "installing RoboTwin requirements"
    "${PY}" -m pip install -q --upgrade pip
    "${PY}" -m pip install -q -r "${ROBOTWIN_ROOT}/scripts/requirements.txt"
fi

# --- sapien urdf loader patch ----------------------------------------------
# RoboTwin's own `scripts/_install.sh` applies this; the embodiment URDFs carry non-ASCII bytes
# and name their companion file `.srdf`, neither of which stock sapien 3.0.0b1 handles.
URDF_LOADER="$("${PY}" -c 'import sapien, pathlib; print(pathlib.Path(sapien.__file__).parent)')/wrapper/urdf_loader.py"
if grep -q 'with open(urdf_file, "r") as f' "${URDF_LOADER}"; then
    log "patching $(basename "${URDF_LOADER}")"
    sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "${URDF_LOADER}"
    sed -i 's/urdf_file\[:-4\] + "srdf"/urdf_file[:-4] + ".srdf"/' "${URDF_LOADER}"
fi

# --- benchmark package ------------------------------------------------------
if ! "${PY}" -c "import robotwin_icil" 2>/dev/null; then
    log "installing robotwin_icil into ${ENV_NAME}"
    "${PY}" -m pip install -q -e "${REPO_ROOT}[dev]" --no-deps
    "${PY}" -m pip install -q pytest
fi

# --- assets -----------------------------------------------------------------
if [[ "${ROBOTWIN_SKIP_ASSETS:-0}" != "1" ]]; then
    if [[ ! -d "${ROBOTWIN_ROOT}/assets/objects" ]]; then
        log "downloading assets (several GB, resumable)"
        (cd "${ROBOTWIN_ROOT}/assets" && "${PY}" _download.py)
        for zip in background_texture embodiments objects; do
            [[ -f "${ROBOTWIN_ROOT}/assets/${zip}.zip" ]] || continue
            (cd "${ROBOTWIN_ROOT}/assets" && unzip -q -o "${zip}.zip" && rm -f "${zip}.zip")
        done
        log "configuring embodiment paths"
        (cd "${ROBOTWIN_ROOT}" && "${PY}" ./scripts/update_embodiment_config_path.py)
    fi
fi

log "done. interpreter: ${PY}"
