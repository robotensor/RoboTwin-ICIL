# BPP on RoboTwin: the LIBERO-Gen Combination transfer adapter

[Behavior Prompting](https://github.com/real-stanford/behavior_prompting) (BPP) is prompted with
one demonstration and predicts chunks of 20 Hz robosuite OSC deltas for a single Panda arm in
LIBERO. This adapter runs its released **LIBERO-Gen spatial combination** checkpoint, unchanged
and frozen, on RoboTwin's bimanual aloha-agilex: it converts one RoboTwin demonstration into
BPP's prompt, each observation into BPP's observation, and BPP's actions back into RoboTwin
actions. It is a *transfer* row — the checkpoint has never seen RoboTwin, aloha, or any V1 task —
so it is read next to the oracles, never on its own.

- Adapter: `icil_policies.bpp` (`policies/src/icil_policies/bpp/`), `ADAPTER_VERSION` 1.
- Config: [`policies/configs/bpp_liberogen_combination.yaml`](../../policies/configs/bpp_liberogen_combination.yaml) — every constant below.
- Environment: `icil-bpp` (`bash scripts/install_policy_env.sh bpp`), BPP at `ec29e62`.
- Camera profile: **`far_side`**, required; the adapter refuses any other.

## Running it

```bash
# once: 6.9 GB of dill -> a 2.8 GB inference checkpoint, its config and its normalizer
$ICIL_HOME/envs/icil-bpp/bin/icil-bpp slim \
    --source /root/checkpoints/bpp/liberogen_spatial_combination_behavior_prompting.ckpt \
    --out $ICIL_HOME/models/bpp/liberogen_spatial_combination \
    --config policies/configs/bpp_liberogen_combination.yaml

# the model, in its own environment, behind the built-in `remote`
robotwin-icil eval --policy remote \
    --policy-arg policy=icil_policies.bpp:BPPPolicy \
    --policy-arg python=$ICIL_HOME/envs/icil-bpp/bin/python \
    --policy-arg config=policies/configs/bpp_liberogen_combination.yaml \
    --camera-profile far_side --suite v1 --episodes 90 --seed 42 --run-dir runs/bpp

# the conversion oracle (O3), in the simulator's own environment, no model
robotwin-icil eval --policy icil_policies.bpp:BPPConversionReplay \
    --policy-arg config=policies/configs/bpp_liberogen_combination.yaml \
    --camera-profile far_side --suite v1 --episodes 90 --seed 1000 --run-dir runs/bpp-o3
```

The calibration seed (1000) is disjoint from the reported seed (42): the execution mode and the
bounds are chosen on it, never on a seed whose score is published.

## The checkpoint

| | |
| --- | --- |
| Hugging Face | `austinpatel/liberogen_spatial_combination/liberogen_spatial_combination_behavior_prompting.ckpt` |
| revision | `0e4c1fdf496592583acfbde51ce82880006aa0f3`, 6,915,257,998 B |
| sha256 | `74e0f841f3b86f1383adda23dab59fddee8806a4bcb5685ed7f826afb320c097` (the Hub's LFS oid) |
| slimmed | 967 tensors, 690,455,718 state-dict elements (parameters, buffers and the normalizer), 2.76 GB |
| slimmed sha256 | `2a0bbeb90502f21d0bb86d29a79ca033a31008853990139bde10eaa208e83039` |

**The model is built from the repository's own Hydra composition**, not from the config stored in
the checkpoint: `libero_policy_dunetp` with `task=liberogen_spatial_combination` and
`+modifiers=libero/liberogen_spatial_combination`, the `eval` resolver registered first. The
stored config lacks `use_pool_modality_pos_embed`, whose code default would add a parameter and
break the strict load. The vision backbone stays `pretrained=true`: with it false, BPP's
`init_weights` rejects the CLIP ViT's bias-free patch convolution. timm downloads
`vit_base_patch16_clip_224.openai` once (the installer prefetches it, so servers run with
`HF_HUB_OFFLINE=1`) and the checkpoint overwrites those weights. Every key matches strictly.

**Deviation from the plan.** The plan says `slim` writes "the EMA state dict". The release
payload's `state_dicts` holds exactly `model` and `optimizer` — there is no EMA copy — so the
model weights *are* the inference weights, and `SOURCE.json` records `"ema": false`. `slim`
raises rather than guess if a future checkpoint carries both.

## The conversion

### Frames and proprioception (plan 3.4, 3.6)

LIBERO's world axes are its Panda base's; a LIBERO vector `v` is `M v` in RoboTwin's world, with
`M = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]` — aloha's root turned a quarter turn about z. The tool
centre point is 0.12 m along the flange's own +x (`gripper_bias`), which is where robosuite
applies its position deltas, so the adapter integrates there and converts back to the flange.

- `ee_pos = M.T (p_world - b_arm) + b_Panda`, `b_arm` the active arm's base
  (about `(∓0.30, -0.42, 0.78)`), `b_Panda = (-0.66, 0, 0.90)`.
- `ee_ori = rot6d(M.T R_world C)`, `C` a quarter turn about the flange's y, taking robosuite's
  +z approach onto aloha's +x. **To be confirmed from renders in #47.**
- `gripper_states = [f, -f]`, `f` the *measured* finger joint mapped affinely from aloha's
  `gripper_scale` (-0.01 to 0.045 m) onto LIBERO's [0, 0.04]. RoboTwin's own gripper value is
  the command, which reads closed on an object; #36's measured joints are what LIBERO records.

`b_Panda` is robosuite's `PandaRobot.base_xpos_offset["table"](1.0) = (-0.66, 0, 0)` with
LIBERO's table offset `(0, 0, 0.90)`; the z is the table top rather than the mount's own frame,
an **assumption**. It is cross-checked against the checkpoint's own normalizer, whose training
range is `ee_pos` x −0.355…0.179, y −0.353…0.508, z 0.910…1.343 m and `gripper_states`
−0.0004…0.0411 m: the adapter's mapped positions and fingers land inside both. Every episode
reports the fraction that does not (`proprio_out_of_range`).

### Cameras (plan 3.1)

`far_side_camera` replaces `front_camera` in its slot: an L515 at `(0, 0.36, 1.20)`, 320x180 with
a 45-degree vertical field, looking back at the robot 38.9 degrees down, image left being the
robot's right. The focal length follows from the field, `f = 90 / tan(22.5°) = 217.3 px`.

The adapter crops a 180x180 square centred on the projection of the **active arm's workspace
centre** — its base plus `(0, 0.25, -0.04)` in world axes, a constant of the profile and the
embodiment, never scene information — then area-resizes to 128 and bilinear-resizes to 224
(`align_corners=False`), reproducing the 128-pixel renders BPP trained on rather than a sharper
224 image. Those centres project to columns **65 (right arm) and 252 (left arm)**, which clamp to
windows 0–179 and 140–319; the #36 render analysis put the arms' own pixel centres at 63 and 251.
The wrist view is the centre 240x240 of the active arm's 320x240 wrist camera, rolled by a
multiple of 90 degrees and resized the same way. Images reach the model upright and float 0..1,
as BPP's loader leaves them (it flips its stored LIBERO frames, which are upside down).

**The wrist roll (0) and `C` take the plan's defaults and are confirmed from renders in #47.**

### The prompt (plan 3.4, 3.5)

The demonstration is resampled on its own `time_s` at `stretch * 20` Hz (frames are not evenly
spaced), tool-centre deltas are taken in LIBERO's frame, divided by the gains, clipped to
[-1, 1], and re-encoded as BPP's 10-dim `[dx, dy, dz, rot6d, gripper]` exactly as its dataset
encodes recorded actions. Gripper labels follow the onset of the commanded change: +1 (close)
while the commanded value falls or is closed, -1 while it rises or is open, so a label never lags
RoboTwin's 300-step gripper ramp. BPP's own `PromptActionChunker` and `collate_prompts` then
build the prompt, one observation per 20-action chunk, the last chunk zero-padded
(`pad_end_prompt_actions: zeros`). At most 50 chunks, or the episode fails loudly; V1
demonstrations need 5–20.

### Tracking gains

`icil-bpp calibrate` fits them by least squares of achieved on commanded tool motion over the
held-out LIBERO-Gen demonstrations (`libero_spatial_selected_combinations_view`, 10 files x 10
demonstrations, 19,552 steps):

| | fitted | per axis | R² |
| --- | --- | --- | --- |
| `alpha_p` | **0.241057** | x 0.2347, y 0.2514, z 0.2385 | 0.951 |
| `alpha_r` | **0.203565** | x 0.2359, y 0.1424, z 0.2361 | 0.732 |

An OSC delta is a goal offset, not a displacement: with robosuite's kp the arm covers about
`alpha` of it in one 20 Hz step. Prompt deltas divide by it, executed deltas multiply by it. The
rotation fit is much the weaker of the two — one gain per axis misrepresents fast or contact
phases (plan 3.4, a known risk). LIBERO's own steps move the tool 5.2 mm (median) and 12.8 mm
(99th percentile), and none of its recorded actions clip, so `stretch` stays **1.0**: whether
RoboTwin's expert outruns that is answered by the clipped-action fraction the conversion oracle
reports, not by this command, which never touches a RoboTwin demonstration.

### Execution (plan 3.3, 3.4)

One action per `take_action` call, from a queue of `exec_action_horizon` 12 out of the 16
`predict_action` returns, with a history of the last 2 observations padded by repetition at the
start. No warm-up: grippers already start open, and BPP's own ten opening steps would come out
of the task's step budget.

A **virtual tool-centre target** integrates the deltas, so motion below CuRobo's 5 mm goal
tolerance is not thrown away; it is re-anchored to the measured pose only when tracking breaks
down. At the fitted gain a full-scale action commands **12.05 mm** of tool motion, so the
re-anchoring bounds are **25 mm and 0.25 rad**, about twice one step: a bound below one step
would re-anchor at every call and the target could never accumulate anything, and the config
refuses such a setting rather than running it. A stall detector reports calls where the tool was
told to move and did not.

| mode | `action_type` | calls per chunk of 12 |
| --- | --- | --- |
| `ee_step` (default) | `ee` | 12 |
| `ee_grouped` | `ee` | 4, 4, 3, 1, split again wherever the gripper command flips, and wherever a group's composed rotation would outgrow what one action encodes (`pi * alpha_r * 0.5` = 0.32 rad, past which the re-encoding wraps and the group would turn the other way) |
| `qpos_ik` | `qpos` | 12, through the aloha kinematics in `icil_policies.common` |

The mode sets `action_type` **per instance**, and `remote` carries it from the server to the
simulator. The idle arm holds the fixed target taken from the episode's first observation — pose
and commanded gripper — which #36 measured as drifting at most 2.8e-5 rad over ten calls, where
echoing the current pose grew every call. The mode is chosen by the conversion oracle on the
calibration seed.

## The oracles

`BPPConversionReplay` replays the prompt's own converted actions through the whole conversion and
execution chain — same resampling, gains, labels, virtual target, mode and idle hold — with no
network. **Its V1 fraction is the ceiling for any model behind this adapter**, and the gap to
`replay` and `replay_ee` is the structural loss of driving one aloha arm with LIBERO-frame 20 Hz
deltas. It also reports the RoboTwin-side numbers `calibrate` cannot: the clipped-action fraction
and the proprio out-of-range fraction, per episode, without exporting anything. Past the end of
the prompt it holds: no motion, and the gripper the prompt's last action commanded, so a task
that ends in a release is not re-grasped by the conversion itself. `actions_held_past_the_end`
counts those calls.

## Gates

| gate | state |
| --- | --- |
| Model builds and loads strictly | **passed** — 518.8M parameters in 967 tensors (690.5M state-dict elements), every key matched, `pretrained=true`, repo composition |
| Prompt chunking equals BPP's own | **passed** — `policies/tests/test_bpp_model_env.py`, BPP's `PromptActionChunker` and `collate_prompts`, shapes `(1, L, 20, 10)` and `(1, L, 3, 224, 224)`, mask all false, last chunk zero-padded |
| Action encoding equals BPP's dataset | **passed** — against BPP's own `RotationTransformer` on random actions |
| Gate 2: served actions equal a direct `predict_action` | **passed** — `icil-bpp preflight`, max absolute difference **0.0** |
| Gate 1: prompt tensors equal `LiberoReplayImageDataset(only_prompt=True)` | **passed** — `icil-bpp preflight --libero-data`, max absolute difference **2.1e-7** (`ee_pos`, `gripper_states` and both images exactly 0, `ee_ori` 2.1e-7, the actions 1.5e-8) on `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_cookie_box_demo.hdf5`, and 2.4e-7 on `pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_ramekin_demo.hdf5` |
| LIBERO sanity rollout (≥ 5 of 10) | **passed** — 6 of 10 on the unseen `pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate`, `icil-bpp libero-sanity --episodes 10` on an RTX 5090; the paper reports about 71% |
| O3 on the calibration seed, V1 run | **not run**: the RoboTwin simulator is not installed in this worktree (#42 is a pure-adapter issue; the sim test is `policies/tests/sim/test_bpp_execution.py`) |

**Running the two LIBERO gates.** Both need BPP's LIBERO stack inside the `icil-bpp`
environment — `LiberoReplayImageDataset` imports `libero.libero` and BPP's `libero_util` imports
`open3d` — which the installer now provides (`future`, `open3d`, and LIBERO's checkout on the
path). They also need `~/.libero/config.yaml` pointing `datasets` at the LIBERO-Gen root and
`bddl_files` / `init_states` at BPP's own `train_network/env/libero/{bddl_files,init_files}`
trees, which carry the split; and the adapter registers LIBERO-Gen's splits itself before
building either the dataset or the runner (`discover_and_register_benchmarks`, which BPP calls
only from `utils/load_env.py`, never from the dataset or the env runner, so both raised
`KeyError` on the split without it).

```bash
icil-bpp --json gate1.json preflight \
    --config policies/configs/bpp_liberogen_combination.yaml \
    --libero-data /root/datasets/libero_gen/demonstration_data/libero_spatial_selected_combinations_view/<one>.hdf5 \
    --cache-dir <a cache outside the repository>
icil-bpp libero-sanity --config policies/configs/bpp_liberogen_combination.yaml \
    --dataset <another one>.hdf5 --episodes 10
```

Gate 1 builds the dataset's zarr cache on the CPU and needs no GPU; gate 2 and the sanity
rollout load the checkpoint and must run under the machine's shared GPU lock. The rollout logs
through wandb (`WANDB_MODE=offline` is enough), renders with `MUJOCO_GL=egl`, and writes one
video into `<--out>/media`, which `icil-bpp` creates: BPP's runner names a visualisation from
the first environment's video whatever you ask of it.

## What a run records

`describe()` carries the adapter and its version, the source checkpoint and its sha256, the
required camera profile, a `parameter_checksum` over every tensor (the frozen-policy audit
compares it at both ends of a run), `training_tasks: []` with `training_domains` naming
LIBERO-Gen spatial combinations, and every constant above under `settings`.

`episode_info()` adds, per episode: the active arm and why it was chosen, the demonstration's arm
set, the prompt's chunks, steps and rate, the clipped-action fraction, the proprio out-of-range
fraction per key — for the prompt and, as `proprio_out_of_range_rollout`, for the observations
the model was actually conditioned on — the number of plans and calls, re-anchors by cause, stall
events and unreachable IK targets.

## Sampling and the frozen policy

BPP's `predict_action` takes no generator: `conditional_sample` draws `torch.randn` from the
process's global RNG. `seed()` therefore seeds that global RNG per episode, which is safe **only
out of process** — RoboTwin reseeds torch's global RNG with the *scene* seed whenever it builds a
scene, so an in-process adapter drawing from it would sample as a function of privileged state.
In the model server nothing else touches torch's RNG and the seed comes from the policy's own
stream. `BPPPolicy.__init__` refuses outright where SAPIEN is importable, naming `remote`, so
that "only out of process" is enforced and not merely documented (`allow_in_process=True` is for
a caller that has checked, such as BPP's own LIBERO runner).

The model is loaded in eval mode with gradients off, nothing in the adapter writes a parameter,
and `describe()` recomputes the `parameter_checksum` every time it is called — a cached value
would have made the audit compare a constant with itself.
