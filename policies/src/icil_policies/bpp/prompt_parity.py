"""Pre-flight gate 1: this adapter's prompt, built from a LIBERO demonstration (issue #42).

`BPPPolicy` builds its prompt by converting RoboTwin quantities into LIBERO's and then running
BPP's own `PromptActionChunker`. Nothing in the benchmark can check the first half against
LIBERO ground truth — RoboTwin is not LIBERO — but the second half must be exactly what BPP's
training loader produces, or the model is prompted with tensors it never saw in that shape.

So this module runs the adapter's own prompt path with a LIBERO demonstration's arrays as its
input: the hdf5's `ee_pos`, `ee_ori` (axis-angle) and `gripper_states` in place of converted
proprioception, its stored frames through BPP's own loader function, and its recorded actions
encoded as BPP's dataset encodes them. `icil-bpp preflight --libero-data` then compares the
result with `LiberoReplayImageDataset(only_prompt=True)`'s, which reads the same demonstration
through the zarr cache, and the two must agree within 1e-5.

Read against `dataset/libero_replay_image_dataset.py` (`_convert_actions:353`,
`_receding_rgb_numpy_thwc_to_float_chw:39`), `common/sampler.py:903-990`
(`sample_prompting_data` with `downsample_obs_by_chunk=True`) and
`utils/prompt_util.py:41-139` (`chunk_prompt`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..common.rotations import axis_angle_to_matrix, matrix_to_rot6d
from .settings import PROMPT_CHUNK_N_ACTIONS

OBSERVATION_KEYS = ("agentview_rgb", "eye_in_hand_rgb", "ee_pos", "ee_ori", "gripper_states")


def encode_actions(raw: np.ndarray) -> np.ndarray:
    """(T, 7) robosuite actions as BPP's (T, 10): position, rot6d of the axis-angle, gripper.

    `_convert_actions` runs the recorded action's three rotation numbers — themselves in [-1, 1],
    units of 0.5 rad — through `RotationTransformer(axis_angle -> rotation_6d)`, so the rot6d
    encodes the *scaled* axis-angle, not a rotation in radians. `conversion.prompt_actions`
    encodes RoboTwin's deltas the same way.
    """
    raw = np.asarray(raw, dtype=np.float64)
    return np.concatenate(
        [raw[:, :3], matrix_to_rot6d(axis_angle_to_matrix(raw[:, 3:6])), raw[:, 6:7]], axis=1
    ).astype(np.float32)


def adapter_prompt_from_hdf5(
    demo: Any,
    shape_meta: Any,
    rgb_to_float_chw: Callable[[np.ndarray], np.ndarray],
    chunker: Any | None = None,
) -> dict[str, Any]:
    """The prompt the adapter's chunking builds from one LIBERO demonstration group.

    `rgb_to_float_chw` is BPP's own loader function, passed in so the image path is theirs and
    only the prompt's assembly is ours.
    """
    from behavior_prompting.common.np_util import add_batch_dim, remove_batch_dim
    from behavior_prompting.train_network.utils.prompt_util import PromptActionChunker

    chunker = chunker if chunker is not None else PromptActionChunker(shape_meta)
    observations = demo["obs"]
    stride = PROMPT_CHUNK_N_ACTIONS
    state = {
        "agentview_rgb": rgb_to_float_chw(observations["agentview_rgb"][::stride]),
        "eye_in_hand_rgb": rgb_to_float_chw(observations["eye_in_hand_rgb"][::stride]),
        "ee_pos": observations["ee_pos"][::stride].astype(np.float32),
        "ee_ori": matrix_to_rot6d(axis_angle_to_matrix(observations["ee_ori"][::stride])).astype(
            np.float32
        ),
        "gripper_states": observations["gripper_states"][::stride].astype(np.float32),
    }
    sample = {"obs": state, "action": encode_actions(demo["actions"][:]), "metadata": {}}
    return remove_batch_dim(chunker.chunk_prompt(add_batch_dim(sample), obs_predownsampled=True))
