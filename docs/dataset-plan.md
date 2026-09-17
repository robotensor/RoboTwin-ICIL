# Training dataset format audit

Status: dataset built and verified; loader implemented.
Dataset repo: [robotensor/robotwin-icil-aloha-clean](https://huggingface.co/datasets/robotensor/robotwin-icil-aloha-clean).
See [the standard benchmark guide](standard-benchmark.md) for commands, format and scoring.
Format audit date: 2026-09-16.

## Decision

Use the official RoboTwin `demo_clean.zip` trajectories as training data. Publish the original
XPolicyLab-format HDF5 files, three aligned RGB MP4 streams per trajectory, and provenance
manifests on Hugging Face. Do not make LeRobot, model-specific video latents, or `prompt.npz` the
canonical training format. A later model adapter can convert this release to LeRobot if needed.

The release targets `aloha-agilex`, matching the official clean data and the repository's default
robot. Use its three D435 policy views at 320 x 240. Do not substitute Franka trajectories or mix
robots into this release: Franka has a different joint action width and must have its own
compatible data.

All 2,500 trajectories are training data. As in RoboTwin 2.0, evaluation happens in simulation on
RoboTwin's evaluation scene seeds (from 100000), with the current one-demonstration,
same-initial-scene, frozen-policy protocol; there is no stored validation or test split.

## What was actually inspected

Raw results are in [dataset-format-audit.json](results/dataset-format-audit.json).

| Source | Pinned revision | Inspection |
| --- | --- | --- |
| `TianxingChen/RoboTwin2.0` | `981c92aa34d8f94d4cff47e0d5bc2f7d4e0af042` | Full `click_bell/demo_clean.zip`: all 50 HDF5s and 50 MP4 metadata records |
| Same source | Same revision | `place_empty_cup/demo_clean.zip`: archive directory and first HDF5, downloaded using byte ranges |
| `Robbyant-Research/HumanGen` | `971d12e942bf101c0d47716f928b1c044d02ac71` | `click_bell` metadata, listed episode files, episode 47 Parquet and head video |
| This repository's RoboTwin submodule | `96c1feab536306b50c26af200044fcdf126e8904` | Config, camera definitions, observation capture, joint packing, prompt serialization and video output |

The official source tree contains 50 `demo_clean.zip` files matching all 50
catalog task names. Their total compressed size is 23,375,659,439 bytes
(23.38 decimal GB). This is archive size, not expanded or converted size.

The initial format audit inspected two official archives. The implementation subsequently
downloaded and validated all 50 archives: each contains 50 aligned Aloha episodes, 2,500 in all.
Full release receipt: [dataset-release-preparation.json](results/dataset-release-preparation.json).

## Confirmed upstream format

`demo_clean.zip` is a per-task ZIP archive containing:

```text
<task>/aloha_agilex/
  data/episode_0000000.hdf5
  video/episode_0000000.mp4
  instruction/episode_0000000.json
  scene_info.json
  seed.txt
```

The inspected HDF5 has `data_format_version = "v1.0"`, with
`source_format = "RoboTwin"` and `source_path = "native_collection"` attributes.
Its schema is XPolicyLab trajectory data, not the legacy raw RoboTwin
`observation/.../rgb` + `joint_action/vector` layout and not LeRobot Parquet.
Do not use a legacy HDF5 reader on these refreshed clean archives.

Let `N` be the number of observation/action pairs. Episode zero of `click_bell`
has `N = 80`; episode zero of `place_empty_cup` has `N = 179`.

| Actual HDF5 key | Shape | Type | Meaning |
| --- | --- | --- | --- |
| `vision/cam_head/colors` | `(N,)` | Fixed-width byte strings | Encoded head-camera JPEG frames |
| `vision/cam_left_wrist/colors` | `(N,)` | Fixed-width byte strings | Encoded left-wrist JPEG frames |
| `vision/cam_right_wrist/colors` | `(N,)` | Fixed-width byte strings | Encoded right-wrist JPEG frames |
| `vision/cam_third_view/colors` | `(N,)` | Fixed-width byte strings | Extra third-person view, outside the default benchmark input |
| `vision/<camera>/shape` | `(3,)` | `int32` | `[240, 320, 3]` for each inspected view |
| `vision/<camera>/intrinsic_matrix` | `(N, 3, 3)` | `float32` | Camera calibration |
| `vision/<camera>/extrinsics_matrix` | `(N, 4, 4)` | `float32` | Camera calibration |
| `state/left_arm_joint_states` | `(N, 6)` | `float32` | Left joint command state |
| `state/left_ee_joint_states` | `(N, 1)` | `float32` | Left normalized gripper state |
| `state/right_arm_joint_states` | `(N, 6)` | `float32` | Right joint command state |
| `state/right_ee_joint_states` | `(N, 1)` | `float32` | Right normalized gripper state |
| `state/left_ee_poses`, `state/right_ee_poses` | Each `(N, 7)` | `float32` | End-effector position and quaternion |
| `action/<same six field names>` | Same shapes as `state` | `float32` | Next sampled joint/gripper/pose targets |
| `additional_info/frequency` | Scalar | `int32` | Value `15`; do not assume this is physical Hz |
| `instructions` | Scalar | String | JSON list of language instructions |

Pack joint state and action in this order:

```text
[left joints (6), left gripper (1), right joints (6), right gripper (1)]
```

This produces `(N, 14)` state and `(N, 14)` absolute joint-position targets.
These are not joint deltas, velocities, torques, or measured encoder positions.
The pinned upstream `get_*_arm_jointState()` reads `joint.get_drive_target()`;
its separate `get_*_arm_real_jointState()` reads measured articulation qpos.
Preserve the command-state semantics to match the current benchmark.

Pack end-effector state and optional EE action in this order:

```text
[left x,y,z,qw,qx,qy,qz, left gripper,
 right x,y,z,qw,qx,qy,qz, right gripper]
```

The result is `(N, 16)`. A 16-value EE vector on Aloha is not a 16-joint
Franka action; action representation must be recorded explicitly.

Across all 50 inspected `click_bell` HDF5s, state, action and camera rows align,
joint values are finite, and `action_qpos[:-1] == state_qpos[1:]` exactly.
Keep the provided action rows. Do not shift them again during conversion.

## Required data versus optional data

The benchmark is model-independent and contains no training implementation.
The following is the proposed complete training data contract for policies
compatible with its default inputs. Individual models may consume a subset.

| Data | Training release | Current benchmark behavior |
| --- | --- | --- |
| Head RGB | Required | `images["head_camera"]`; prompt `frames_head_camera` |
| Left wrist RGB | Required | `images["left_camera"]`; prompt `frames_left_camera` |
| Right wrist RGB | Required | `images["right_camera"]`; prompt `frames_right_camera` |
| Joint command state + grippers | Required, 14 values | `Observation.qpos` and demonstration `qpos` |
| Joint target actions | Required supervision, 14 values | Demonstration `actions`; default policy returns `qpos` targets |
| EE poses + grippers | Retain, 16 values | Present in observations and demonstration `endpose` |
| EE target actions | Retain as optional supervision | Policy may select `action_type = "ee"`, width 16 |
| Frame index and timing provenance | Required | Generated demonstrations contain simulated `times` and nominal `frequency` |
| Task ID, episode ID, robot and source hashes | Required bookkeeping | Task and scene metadata remain harness-only |
| Task-specific language | Retain as metadata, not required ICIL input | Policy instruction is only `Follow the demonstrated behavior.` |
| Camera intrinsics/extrinsics | Preserve in original HDF5; optional model input | Not supplied in policy observations; cameras participate in harness fingerprinting |
| Third-person RGB | Optional upstream evidence | Not a policy view under default `demo_clean` |
| Depth, point clouds, segmentation | Not required | Disabled by the default config and not exposed by the policy interface |
| Scene seed, object identity, expert `info` | Preserve separately for provenance | Never supplied to the policy |

The benchmark's default camera contract comes from
`vendor/RoboTwin/env_cfg/task_config/demo_clean.yml` and `_camera_config.yml`:
head and wrist cameras are D435, each 320 x 240. At runtime the core accepts
the RGB cameras actually returned by the config; these three are defaults,
not hard-coded requirements of every possible custom configuration.
The downloaded `click_bell` head and wrist intrinsics have focal lengths
358.64218 pixels and principal point `(160, 120)`, giving a vertical FOV of
37 degrees, matching the local D435 definition. Full camera pose equality
across source and current simulation still needs the pilot in step 5.

## Why videos alone are insufficient

The upstream MP4s are H.264 head-camera previews at 30 FPS. Each of the 50
inspected `click_bell` previews contains `N + 1` frames, while its HDF5 has
`N` RGB/state/action rows. Episode zero is 81 preview frames / 80 transitions.
The previews do not provide wrist streams or action labels.

For the release, encode three MP4s from each HDF5's `colors` arrays. Every
output stream must contain exactly `N` frames; frame `i` matches state/action
row `i`. Encode the head view too so all streams share an explicit alignment.
Retain the upstream `N + 1` preview separately if desired, never substitute it
for an aligned training stream without checking and trimming its terminal frame.

No new physics simulation is necessary to produce these three RGB videos.
Use recorded images. Recreating the exact old trajectory would be extra work
and is not justified merely to obtain MP4 containers.

### Color compatibility

The source revision's clean archives were refreshed with standard RGB JPEGs.
Direct inspection confirms that the pinned repository decoder reverses the
standard JPEG RGB channels on the downloaded sample. Its decoded head frame
is exactly the channel-reversed standard RGB decode; mean pixel error against
the upstream preview is 3.13 versus 1.14 for standard RGB decoding.

Implement conversion with an explicit source-revision/encoding profile. For
this pinned refreshed release, use standard JPEG-to-RGB decoding. For older
archives, verify their documented encoding before selecting a decoder. Do not
change the simulator's already-decoded RGB or globally apply a color swap.
`data_format_version = "v1.0"` alone cannot identify this encoding change.

### Timing compatibility

The official HDF5s inspected contain no per-frame simulator timestamps.
Upstream's pinned collection writer stores `save_freq` as `frequency`, and
encodes the preview at 30 FPS. These are distinct quantities.

The current benchmark uses 250 Hz physics and `save_freq = 15`, so its nominal
demonstration rate is `250 / 15 = 16.6666667 Hz`. It also counts actual physics
steps and records uneven frame times, including motion-primitive boundaries.
The old HDF5s do not contain enough information to recover those exact times.

Use frame-based observation/action pairing for this release. Set aligned MP4
playback to 30 FPS and record `video_fps = 30`, `source_frequency_value = 15`,
`simulated_frame_times = null`, and `timing_source = "unavailable"` in the
manifest. Video timestamps are playback timestamps, not a physical clock.
A model needing physical-time inputs may use a documented nominal assumption
in its adapter; the dataset must not mislabel inferred times as measurements.
Exact-time training data requires a separate recording pass with the clock
enabled, not merely re-encoding these JPEGs.

## Current evaluation format is different

`materialize` creates an evaluation demonstration directly from the simulator.
`run-unit` reads `prompt.npz`, not HDF5, MP4, or LeRobot data.
Let `T` be the full captured demonstration frame count:

| Prompt array | Shape | Type |
| --- | --- | --- |
| `frames_head_camera`, `frames_left_camera`, `frames_right_camera` | Each `(T, 240, 320, 3)` for defaults | `uint8`, RGB |
| `qpos` | `(T, 14)` on Aloha; `(T, 16)` on dual Franka | `float64` |
| `endpose` | `(T, 16)` | `float64` |
| `actions` | `(T - 1, D)` and exactly `qpos[1:]` | `float64` |
| `times` | `(T,)`, actual simulated times for generated prompts | `float64` |
| `frequency` | Scalar nominal frame rate | `float64` |
| `meta` | Scalar JSON string | Task, seed, config, robot, initial fingerprint/digest, source commits and expert attempts |

Local policies receive a `Demonstration` with RGB, qpos, EE state and actions.
Remote policies receive equivalent arrays through `remote.py`. The base
benchmark exposes the full demonstration; this plan does not silently turn
it into a video-only prompt track. Prompt channel restrictions belong in an
explicit protocol/profile and must apply equally to every model.

Live observations contain the current RGB frames, qpos, endpose and neutral
instruction. Local adapters also have the observation step and public episode
robot/action dimensions. Remote array names are `frames_<camera>`, `qpos`
and `endpose`. No privileged scene metadata is sent to the policy.

The benchmark's `demonstration.mp4` and rollout MP4 are head-camera diagnostic
clips, not the serialized policy input. A correct training download is not
automatically a valid `run-unit --prompt` input: the HDF5 omits terminal wrist
RGB and exact times and has no benchmark initial-scene fingerprint. Do not
fabricate those fields to pass `read_prompt` validation. Continue generating
evaluation prompts with `materialize` and the pinned simulator.

## HumanGen is an alternative, not a drop-in replacement

Its inspected `click_bell` directory uses LeRobot v2.1: per-episode Parquet,
per-camera MP4s, and `meta/info.json`. Episode 47 has 74 rows. It contains
three RGB streams (`cam_high`, `cam_left_wrist`, `cam_right_wrist`) at
640 x 480 / 50 FPS / AV1. The head sample was decoded successfully with FFmpeg.

`observation.state` and `action` each have width 16, with EE position,
quaternion and grippers. There is no 14-value joint-state column in the
sample Parquet. Therefore, it does not supply the full default qpos contract.
Its `timestamp` advances by approximately 0.02 s; that verifies stored timing,
not independently the original simulation clock. The copied task metadata
says 1,000 episodes, while the published task directory lists 50 Parquet files
with non-contiguous original episode indices. Count actual files, not that
metadata, when selecting data. Camera pose equality with this repo is unverified.

Use the official HDF5 release for this plan. HumanGen remains useful for a
separately validated EE/human-video experiment, but importing it would add
state, camera, timing and metadata changes that the current training release
does not need.

## Sources

- [Official RoboTwin dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset)
- [Official collection and conversion documentation](https://robotwin-platform.github.io/doc/usage/collect-data.html)
- [Zero-WAM repository and HumanGen release instructions](https://github.com/robbyant-research/Zero-WAM)
- Local contracts: `src/robotwin_icil/prompt.py`, `policy.py`, `robotwin.py`,
  `remote.py`, `video.py`, `unit.py`, and the pinned upstream configuration files.
