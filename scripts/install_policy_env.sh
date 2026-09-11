#!/usr/bin/env bash
# Build a model environment for an adapter under policies/: `bpp` or `uniskill`.
#
#   bash scripts/install_policy_env.sh bpp
#   ICIL_HOME=/data/icil bash scripts/install_policy_env.sh uniskill
#
# Each model's stack conflicts with RoboTwin's (huggingface_hub, transformers, diffusers) and with
# the other model's, so each gets its own virtualenv, built with uv from the lockfiles in
# policies/envs/<name>/ (pins.env, requirements.lock, constraints.txt): the env at
# $ICIL_HOME/envs/icil-<name>, the model's source at $ICIL_HOME/src. ICIL_HOME defaults to
# ~/.cache/robotwin-icil. The simulator env (conda `robotwin`, scripts/install_robotwin.sh) is
# never touched; the benchmark reaches a model env through a remote policy (#39).
#
# Idempotent per stage: checkouts and downloads are skipped when present, and each install stage
# records a hash of its pins under $ENV/.icil-stages and is skipped while they are unchanged.
# An existing env on a python other than PYTHON_VERSION is refused. Delete the env directory to
# rebuild it from scratch.
set -euo pipefail

usage() {
    echo "usage: $0 bpp|uniskill" >&2
    exit 2
}
[[ $# -eq 1 ]] || usage
NAME="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCKS="${REPO_ROOT}/policies/envs/${NAME}"
[[ -f "${LOCKS}/pins.env" ]] || usage

ICIL_HOME="${ICIL_HOME:-${HOME}/.cache/robotwin-icil}"
ENV_DIR="${ICIL_HOME}/envs/icil-${NAME}"
SRC_ROOT="${ICIL_HOME}/src"
STAMPS="${ENV_DIR}/.icil-stages"
PY="${ENV_DIR}/bin/python"

log() { printf '\033[95m[icil-%s]\033[0m %s\n' "${NAME}" "$*"; }
die() {
    printf '[icil-%s] %s\n' "${NAME}" "$*" >&2
    exit 1
}

UV="${UV:-$(command -v uv || true)}"
[[ -n "${UV}" && -x "${UV}" ]] || die "uv not found: pip install uv==0.12.13, or set UV=/path/to/uv"

# shellcheck source=/dev/null
source "${LOCKS}/pins.env"

hash_of() { sha256sum | cut -c1-16; }
key() { printf '%s\n' "$@" | hash_of; }
file_key() { cat "$@" | hash_of; }

# stage NAME KEY COMMAND...: run COMMAND unless the stamp for NAME already holds KEY.
stage() {
    local name="$1" stamp_key="$2"
    shift 2
    if [[ -f "${STAMPS}/${name}" && "$(cat "${STAMPS}/${name}")" == "${stamp_key}" ]]; then
        log "stage ${name}: done, skipped"
        return 0
    fi
    log "stage ${name}"
    "$@"
    mkdir -p "${STAMPS}"
    printf '%s\n' "${stamp_key}" >"${STAMPS}/${name}"
}

uvpip() {
    log "uv pip install $*"
    "${UV}" pip install --python "${PY}" "$@"
}

# checkout DIR URL COMMIT [SUBMODULE...]: clone once, then hold DIR at COMMIT.
checkout() {
    local dir="$1" url="$2" commit="$3"
    shift 3
    mkdir -p "$(dirname "${dir}")"
    if [[ ! -d "${dir}/.git" ]]; then
        log "cloning ${url} into ${dir}"
        git clone -q "${url}" "${dir}"
    fi
    if [[ "$(git -C "${dir}" rev-parse HEAD)" == "${commit}" ]]; then
        log "${dir} is at ${commit}, kept"
    else
        git -C "${dir}" cat-file -e "${commit}^{commit}" 2>/dev/null || git -C "${dir}" fetch -q origin
        log "checking out ${commit} in ${dir}"
        git -C "${dir}" checkout -q "${commit}"
    fi
    if [[ $# -gt 0 ]]; then
        log "submodules of ${dir}: $*"
        git -C "${dir}" submodule update --init "$@"
    fi
}

# The stage keys leave PYTHON_VERSION out, so an env on another python is refused, not reused.
make_venv() {
    local have
    if [[ -x "${PY}" ]]; then
        have="$("${PY}" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')"
        [[ "${have}" == "${PYTHON_VERSION}" || "${have}" == "${PYTHON_VERSION}".* ]] ||
            die "${ENV_DIR} has python ${have}, pins want ${PYTHON_VERSION}: delete it to rebuild"
        log "env ${ENV_DIR} exists (python ${have}), kept"
    else
        log "creating ${ENV_DIR}: python ${PYTHON_VERSION}, managed by uv"
        mkdir -p "$(dirname "${ENV_DIR}")"
        "${UV}" venv --seed --python "${PYTHON_VERSION}" "${ENV_DIR}"
    fi
}

install_torch() { uvpip "${TORCH[@]}" --index-url "${TORCH_INDEX}"; }

# The model's extra, and `pure` for pytest, so the env can run its own contract tests. The
# installs are editable, so the stage's key includes the checkout they point at.
BENCHMARK_EXTRAS="${NAME},pure"

install_benchmark() {
    uvpip -c "${LOCKS}/constraints.txt" -e "${REPO_ROOT}" -e "${REPO_ROOT}/policies[${BENCHMARK_EXTRAS}]"
}

smoke_common() {
    "${PY}" - <<'PY'
import torch

print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, available {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU {torch.cuda.get_device_name(0)}, capability {torch.cuda.get_device_capability(0)}")
PY
    "${PY}" -c "import robotwin_icil.policy, icil_policies.common.rotations"
    local module
    for module in "$@"; do
        log "smoke: import ${module}"
        "${PY}" -c "import importlib, sys; importlib.import_module(sys.argv[1])" "${module}"
    done
}

# --- bpp ----------------------------------------------------------------------
# Codifies the recipe verified on an RTX 5090 on 2026-09-11 (docs/install.md).

bpp_libero() {
    local src="$1"
    uvpip "${LIBERO_STACK[@]}"
    log "pip install ${LEGACY_BUILD_TOOLS[*]}, then ${GYM[*]}: gym 0.21 builds only with old tools"
    "${PY}" -m pip install -q "${LEGACY_BUILD_TOOLS[@]}"
    "${PY}" -m pip install -q "${GYM[@]}"
    uvpip -e "${src}/deps/LIBERO"
    # LIBERO's `libero` directory has no `__init__.py`, so setuptools' editable install writes a
    # finder whose MAPPING is empty and `import libero` fails, though the distribution is there.
    # Put the checkout itself on the path, as an editable install of a plain package would.
    if ! "${PY}" -c "import libero" 2>/dev/null; then
        local site
        site="$("${PY}" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")"
        printf '%s\n' "${src}/deps/LIBERO" > "${site}/libero_checkout.pth"
        log "libero: the editable install maps no package; wrote ${site}/libero_checkout.pth"
        "${PY}" -c "import libero" || die "libero still does not import after the path file"
    fi
}

bpp_robomimic() {
    uvpip "${CMAKE[@]}"
    log "building the robomimic fork: egl-probe needs cmake on PATH and CMAKE_POLICY_VERSION_MINIMUM=3.5"
    PATH="${ENV_DIR}/bin:${PATH}" CMAKE_POLICY_VERSION_MINIMUM=3.5 \
        "${UV}" pip install --python "${PY}" -c "${LOCKS}/constraints.txt" "${ROBOMIMIC[@]}"
}

bpp_backbone() {
    local hub="${HF_HOME:-${HOME}/.cache/huggingface}/hub/models--timm--${BACKBONE}"
    if compgen -G "${hub}/snapshots/*/*" >/dev/null; then
        log "backbone ${BACKBONE} is in ${hub}, kept"
    else
        log "prefetching timm's ${BACKBONE} into the Hugging Face cache, so servers run offline"
        "${PY}" -c "import sys, timm; timm.create_model(sys.argv[1], pretrained=True)" "${BACKBONE}"
    fi
}

build_bpp() {
    local src="${SRC_ROOT}/${REPO_DIR}"
    checkout "${src}" "${REPO_URL}" "${REPO_COMMIT}" "${SUBMODULES[@]}"
    make_venv
    stage torch "$(key "${TORCH_INDEX}" "${TORCH[@]}")" install_torch
    stage core "$(file_key "${LOCKS}/requirements.lock")" uvpip -r "${LOCKS}/requirements.lock"
    stage libero "$(key "${REPO_COMMIT}" "${LIBERO_STACK[@]}" "${LEGACY_BUILD_TOOLS[@]}" "${GYM[@]}")" \
        bpp_libero "${src}"
    stage robomimic "$(key "${CMAKE[@]}" "${ROBOMIMIC[@]}" "$(file_key "${LOCKS}/constraints.txt")")" \
        bpp_robomimic
    stage model "$(key "${REPO_COMMIT}")" uvpip --no-deps -e "${src}"
    stage benchmark "$(key "${REPO_ROOT}" "${BENCHMARK_EXTRAS}" "$(file_key \
        "${REPO_ROOT}/pyproject.toml" "${REPO_ROOT}/policies/pyproject.toml" \
        "${LOCKS}/constraints.txt")")" install_benchmark
    bpp_backbone
    smoke_common "${SMOKE_IMPORT}"
    log "smoke: the backbone with HF_HUB_OFFLINE=1"
    HF_HUB_OFFLINE=1 "${PY}" -c "import sys, timm; timm.create_model(sys.argv[1], pretrained=True)" \
        "${BACKBONE}"
}

# --- uniskill -----------------------------------------------------------------
# Verified on an RTX 5090 on 2026-09-11 (#43, docs/models/uniskill.md): the fork without its
# simulator submodules, the skill encoder on the path, torch 2.8.0+cu128.

uniskill_model() {
    local fork="$1" isd="$2" site
    # The fork's own pins (torch 2.0.1, diffusers 0.23, transformers 4.36) are the ones replaced.
    uvpip --no-deps -e "${fork}"
    # The skill encoder is a script repository, not a package: put its checkout on the path.
    site="$("${PY}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    log "adding ${isd} to the env's path (${site}/icil_uniskill_isd.pth)"
    printf '%s\n' "${isd}" >"${site}/icil_uniskill_isd.pth"
}

build_uniskill() {
    local isd="${SRC_ROOT}/${ISD_DIR}" fork="${SRC_ROOT}/${POLICY_DIR}"
    checkout "${isd}" "${ISD_URL}" "${ISD_COMMIT}"
    # Not its LIBERO, robosuite and robocasa submodules: only the fork's LIBERO training and
    # evaluation scripts import them, never the adapter, and robocasa pins numpy 1.23.3.
    checkout "${fork}" "${POLICY_URL}" "${POLICY_COMMIT}"
    make_venv
    stage torch "$(key "${TORCH_INDEX}" "${TORCH[@]}")" install_torch
    stage core "$(file_key "${LOCKS}/requirements.lock" "${LOCKS}/constraints.txt")" \
        uvpip -c "${LOCKS}/constraints.txt" -r "${LOCKS}/requirements.lock"
    stage model "$(key "${POLICY_COMMIT}" "${ISD_COMMIT}")" uniskill_model "${fork}" "${isd}"
    stage benchmark "$(key "${REPO_ROOT}" "${BENCHMARK_EXTRAS}" "$(file_key \
        "${REPO_ROOT}/pyproject.toml" "${REPO_ROOT}/policies/pyproject.toml" \
        "${LOCKS}/constraints.txt")")" install_benchmark
    smoke_common "${SMOKE_IMPORTS[@]}"
}

log "building ${ENV_DIR} from ${LOCKS}"
case "${NAME}" in
bpp) build_bpp ;;
uniskill) build_uniskill ;;
*) usage ;;
esac
log "done. interpreter: ${PY}"
