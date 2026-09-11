# Source this in an interactive shell before running RoboTwin tools directly.
#
# install_robotwin.sh writes the NVIDIA Vulkan ICD and EGL vendor manifests into the conda env
# when it cannot write them system-wide (not root). SAPIEN only finds them through these
# variables; `robotwin_icil` sets the same defaults itself, so benchmark commands need nothing.
_rt_prefix="${CONDA_ROOT:-/root/miniforge3}/envs/${ROBOTWIN_ENV_NAME:-robotwin}/share/robotwin-icil"
if [[ -f "${_rt_prefix}/vulkan/icd.d/nvidia_icd.json" && -z "${VK_ICD_FILENAMES:-}" ]]; then
    export VK_ICD_FILENAMES="${_rt_prefix}/vulkan/icd.d/nvidia_icd.json"
fi
if [[ -f "${_rt_prefix}/glvnd/egl_vendor.d/10_nvidia.json" && -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ]]; then
    export __EGL_VENDOR_LIBRARY_FILENAMES="${_rt_prefix}/glvnd/egl_vendor.d/10_nvidia.json"
fi
unset _rt_prefix
