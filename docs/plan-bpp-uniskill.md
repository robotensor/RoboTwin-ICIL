# BPP and UniSkill on robotwin-icil-benchmark: plan

Plan, 2026-09-11. Read against RoboTwin `96c1feab` (the submodule pin), behavior_prompting `ec29e62`,
UniSkill `eca49f0`, UniSkill-Policy `2803ad6`, RoboTwin 2.0's `embodiments.zip` on Hugging Face
(`TianxingChen/RoboTwin2.0`), and the papers (arXiv 2606.30457, 2505.08787). Every number below was
read in source, computed from source, or measured by a check recorded in the research notes;
**ASSUMPTION** marks the rest.

## Answers

- **Cameras.** BPP reads 2: one third-person view and one wrist view, in the prompt and in every
  observation. UniSkill's skill encoder reads 1 (third-person only) and its released policy 2. The
  RoboTwin-trained BPP and UniSkill in this plan read 3: `far_side_camera` and both wrists. RoboTwin
  renders 4 today (head, front, two wrists); the plan swaps the front camera for `far_side_camera`,
  so it stays 4 and every seed keeps its scene.
- **Demonstrations.** Both models take exactly 1 at inference, the benchmark's protocol unchanged.
  For training, the BPP checkpoint used about 50 per task over 150 LIBERO-Gen tasks, UniSkill's
  policy 50 per task over 8 LIBERO-90 tasks, and the RoboTwin-trained models here need at least 30
  per task over at least 25 held-out tasks (M3 gate; **ASSUMPTION** thresholds).
- **Camera positions.** LIBERO's `agentview` sits across the table at (0.659, 0, 1.610), looking back
  at the robot 38.9° down with a 45° field; its wrist camera sits on the Panda hand with 75°.
  RoboTwin's `head_camera` sits over the robot's shoulders at (-0.032, -0.45, 1.35), 53° down, 37°;
  `front_camera` at (0, -0.45, 0.85) looks at the wall; the wrist cameras sit on link 6 with 37°. The
  new `far_side_camera` goes at (0, 0.36, 1.20), 38.9° down, 45° (§3.1 has the variants and gates).
- **What to build.** Core C1-C7 (camera profiles, physics clock, end-effector views and measured
  fingers, policy hooks, records and report, out-of-process policies, training export), adapters
  A1-A4, model-side B1-B4, infrastructure I1-I4 and docs (§4), in the order of §5.
- **When each can be scored.** The released BPP checkpoint after M0-M2, as a cross-embodiment
  transfer expected near zero. UniSkill, and a meaningful BPP number, only after training on
  RoboTwin exports (M3-M5).

## 1. Summary

| | BPP, LIBERO-Gen Combination checkpoint | UniSkill |
| --- | --- | --- |
| Cameras the model reads | 2: third-person `agentview` and wrist `eye_in_hand`, in the prompt and in every observation | Skill encoder (ISD): 1 third-person. Released policy: 2 (`agentview`, wrist) |
| Demonstrations at inference | exactly 1 (`max_prompt_full_demos: 1`), all of it | exactly 1 prompt video; frames only, actions unused |
| Training data behind the checkpoint | the 150 training-view tasks of LIBERO-Gen Combination, about 50 demos each; evaluated on 10 held-out combinations | 8 LIBERO-90 tasks, 50 demos each |
| Camera position it was trained with | LIBERO `agentview`: 1.32 m in front of the robot, 0.71 m above the table, looking back at it 38.9° down, 45° field of view | the same LIBERO renders |
| What RoboTwin has | `head_camera` over the robot's shoulders (0.61 m up, 53° down, 37°), `front_camera` near the table looking away from the robot, two wrist cameras; nothing across the table | the same |
| Runs as released | yes, through an adapter; expected success 0.00-0.05 | no: the policy checkpoint link returns 404; only the ISD is reusable |

What gets built:

- **Core** (`src/robotwin_icil`) gains model-agnostic seams: camera profiles that keep every seed's
  scene, a physics clock on frames, end-effector views of a demonstration and an `ee` replay
  oracle, policy configuration, seed, provenance and close hooks, a generic out-of-process policy,
  and a training-data export. Nothing in the core names BPP or UniSkill.
- **Adapters** (`policies/`, outside the core as `docs/policies.md` already requires) gain a BPP
  adapter that runs the released checkpoint as a single-arm end-effector policy on aloha-agilex,
  and a UniSkill adapter (frozen ISD plus a policy trained on exports). Each runs in its own
  environment and ships a conversion oracle that measures the ceiling of its conversion.
- **Infrastructure**: the simulator environment moves to CUDA 12.8 and torch 2.8 for the RTX 5090,
  and the container gains the three files SAPIEN needs to render.

Two expectations to set now:

- The released BPP checkpoint on aloha-agilex is a cross-embodiment transfer: different arm,
  gripper, viewpoint, renderer and action frame. Its score will be near zero and says little about
  BPP's in-context ability, so it is always printed beside its conversion oracle. BPP was also trained with prompts recorded
  from a different initial state; here the prompt starts in the rollout's own scene, a case it never
  saw. CLAUDE.md
  reserves "zero-shot" for a no-demonstration score, so this run is not called zero-shot.
- A meaningful number for either model needs its architecture trained on RoboTwin data outside
  the benchmark and then frozen. That fits the protocol: at evaluation the policy adapts only
  through the one demonstration.

The protocol does not change: exactly one demonstration, the same scene, a frozen policy, RoboTwin's
own success check, fractions over valid episodes.

## 2. Requirements against what RoboTwin offers

| | BPP | UniSkill (released) | RoboTwin / benchmark today |
| --- | --- | --- | --- |
| Images | `agentview_rgb`, `eye_in_hand_rgb`: 128x128 upright renders resized to 224; at eval CenterCrop 0.95 (212) then 224, CLIP normalization | ISD: one third-person frame pair resized to 224 (no crop) plus Depth-Anything-V2-Small depth. Policy: two cameras at 128, center crop 116 | `head_camera`, `front_camera`, `left_camera`, `right_camera`: D435 320x240 uint8, ray traced at 32 samples per pixel |
| Proprio | `ee_pos` (3), `ee_ori` rot6d (6, first two rows of R), `gripper_states` (2 finger positions, ≈[0.04, -0.04] open) | `ee_pos`, `ee_ori` axis-angle, `gripper_states`, `joint_states` (7) | `qpos` (14 commanded drive targets: left arm 6, left gripper, right arm 6, right gripper), `endpose` per arm (measured flange pose, xyz + wxyz, world) and gripper (commanded, 0 closed, 1 open); measured finger joints are not exposed today (C3 adds them) |
| History | 2 steps for every key | 2 | one observation per `act()` |
| Prompt | the whole demo: both images and proprio every 20 steps, all actions in chunks of 20, last chunk zero-padded, mask (B, L) with True = padding, at most 50 chunks | z_t = ISD(I_t, I_min(t+k, T-1)), k = 20 frames at 20 Hz (1.0 s), one row per frame | `Demonstration`: frames of images per camera, qpos, endpose; `frequency` 250/15 ≈ 16.7 fps |
| Actions | predicts (16, 10) = dpos, rot6d, gripper; executes 12 as 7-dim robosuite OSC_POSE deltas (±0.05 m, ±0.5 rad per unit; gripper -1 open, +1 close) | 7-dim LIBERO deltas, same convention; executes 8 of 16 | `qpos`: 14 absolute targets (TOPP). `ee`: 16 = per arm absolute flange pose (xyz + wxyz) and gripper, one CuRobo plan per arm per call |
| Rate and budget | 20 Hz; 550 steps | 20 Hz; 500 steps | step limit counts `take_action` calls: 400 for seven V1 tasks, 500 place_empty_cup, 800 stack_blocks_two, 900 stack_bowls_two |
| Warm-up | 12 open-gripper steps | 5 zero-action steps | robot starts at home with both grippers open |
| Checkpoint | `austinpatel/liberogen_spatial_combination` / `liberogen_spatial_combination_behavior_prompting.ckpt`, 6,915,257,998 B dill payload (config, EMA weights, optimizer), 518.8M parameters (loaded and counted) | `HanjungKim/UniSkill` `UniSkill_final_weight/idm.pth` (80,338,210 B) and `depth-anything/Depth-Anything-V2-Small-hf` (99 MB); policy `.pth` 404 | replay oracle |
| Environment | python 3.10, torch 2.8.0+cu128 (the cu128 variant BPP documents; its default 2.7.1+cu118 has no sm_120), diffusers 0.35.1, transformers 4.57.1, timm, hydra 1.2, dill | ISD repo: torch 2.3.0, transformers 4.48. Policy fork: torch 2.0.1, diffusers 0.23, transformers 4.36, huggingface-hub < 0.25 | python 3.10, torch 2.4.1+cu121, sapien 3.0.0b1, mplib 0.2.1, CuRobo 0.7.8, huggingface_hub 0.25.0 |

## 3. Gaps and how they close

### 3.1 Viewpoint

- **LIBERO `agentview`** (LIBERO_Tabletop_Manipulation, the domain of every LIBERO-Spatial and
  LIBERO-Gen Combination task): position (0.659, 0, 1.610), quaternion wxyz (0.638, 0.305, 0.305,
  0.638), table top z = 0.90, Panda base x = -0.66. It meets the table 0.44 m in front of the base at
  1.13 m range; fovy 45° (MuJoCo default) at 128x128. The robot enters the image from the top, and
  image left is the Panda's right.
- **RoboTwin `head_camera`**: (-0.032, -0.45, 1.35), forward (0, 0.6, -0.8): 0.61 m above the 0.74 m
  table, 53.1° down, 0.76 m range, 37°. An over-the-shoulder view with the arms entering from the
  bottom, mirrored relative to `agentview`. `front_camera` (0, -0.45, 0.85), forward (0, 1, -0.1),
  sits 0.11 m above the table looking at the wall; it is rendered and returned today, so every frame
  carries four cameras, not the three the docs name. No policy uses it.
- **Wrist**: aloha's `left_camera` / `right_camera` sit on `fl_link6` / `fr_link6` at xyz
  (0.07, 0.032, 0.065), rpy (0, 0.4, 0), looking along the gripper like robosuite's eye-in-hand, but
  at 37° against robosuite's 75°. No RoboTwin camera type is wider than 45°; the profile may switch
  the wrist cameras to `L515` (45°, centre 180x180 crop), and wrist cameras draw no random numbers,
  so scenes stay the same. The wrist image's roll (a multiple of 90°) is set once from renders.
- **The new camera** replaces `front_camera` (see 3.2) and is named `far_side_camera`: the core
  names geometry, not models. Type `L515`: 320x180 at 45° vertical, so a 180x180 square is LIBERO's 45° field and
  already exceeds BPP's 128-pixel training resolution (`Large_L515` is the same field at 640x360
  and costs four times the ray tracing). Left vector (1, 0, 0) in every variant, which keeps
  LIBERO's handedness (image left = the robot's right).

  | Variant | Position | Forward | Down | Range to the table along the axis | Arm bases in a centred 45° square crop (vertical, horizontal; fraction of half-size) |
  | --- | --- | --- | --- | --- | --- |
  | LIBERO reference | (0.659, 0, 1.610) | (-0.778, 0, -0.628) | 38.9° | 1.13 m | Panda base +0.47, 0.00 |
  | A (default) | (0, 0.36, 1.20) | (0, -0.778, -0.628) | 38.9° | 0.73 m | +0.45, ±0.83 |
  | B | (0, 0.36, 1.42) | (0, -0.643, -0.766) | 50.0° | 0.89 m | +0.45, ±0.73 |
  | C | (0, 0.90, 1.45) | (0, -0.778, -0.628) | 38.9° | 1.13 m | +0.51, ±0.50 |

  Every variant is referenced to the arm-base line (world y ≈ -0.42, arm bases ≈ (±0.30, -0.42,
  0.78)), the analogue of LIBERO's Panda base; the robot root at y = -0.65 is not. A and B sit 4 cm in
  front of the wall (front face y = 0.40). A keeps LIBERO's 38.9° angle and puts the arm bases at the
  same image height as LIBERO's Panda base (+0.45 against +0.47 of the half-height), but it aims
  0.21 m in front of the arm bases instead of 0.44 m and is 0.73 m from the table instead of 1.13 m,
  so objects appear 1.57x larger. C reproduces LIBERO's geometry relative to the arm bases: 1.32 m in
  front of them and 0.71 m above the table, aiming 0.44 m in front of them at 1.13 m range. It sits
  inside the wall volume and relies on the ray tracer ignoring the wall's back faces, which was read
  in SAPIEN's shader code and never rendered.
- **Arm-centred crop.** aloha's arms sit ±0.30 m off the centre line, so a centred square crop puts
  the active arm near the image edge while LIBERO's Panda is centred. The BPP adapter crops 180x180
  from the 320-wide frame, centred on the projection of the active arm's workspace centre (computed
  from the intrinsics, f = 217.3 px, clamped to columns 90-230 and checked on renders; a constant of
  the profile, not scene information), then area-resizes to 128 and bilinear-resizes to 224, reproducing the
  128-pixel training renders rather than a sharper 224 image.
- The variant and the wrist roll are chosen from renders (`robotwin-icil cameras`, C1) next to
  LIBERO frames in M1. The same gate projects every V1 task's spawn region (range corners plus object
  extents) into the full frame and the arm-centred crop. At variant A the image's bottom edge meets
  the table at y ≈ +0.11, while V1 spawn ranges reach y = +0.05 and, for `stack_bowls_two`, +0.15;
  variant C's bottom edge falls beyond the table's far edge. The camera is raised or tilted, or C is
  adopted, if any region falls outside.

### 3.2 Adding a camera changes every scene

`Camera.load_camera` calls `create_camera` for every static camera, and `create_camera` draws
`np.random.randn(3)` and `np.random.uniform(0, random_head_camera_dis)` even when the bound is 0.
`_init_task_env_` seeds numpy, then loads the robot and the cameras, and only then `load_actors`.
The state reaching `load_actors` depends only on how many static cameras were created (not on names,
types or poses); `head_camera` is created only while `camera.collect_head_camera` is true. Adding a
camera therefore moves every object of every seed (a verifier saw the first placement draw move from
0.1855 to 0.1702), which would silently invalidate the survey, the replay 18/18 and every per-seed
comparison, and the Same Scene check cannot notice because both builds of an episode use the same
profile. Replacing one camera by another keeps every scene. Names must also stay unique:
`get_config()` and `get_rgba()` key by name, so a second `front_camera` silently overwrites the first.

### 3.3 One arm of a bimanual robot

- RoboTwin is bimanual everywhere; single-arm embodiments come as pairs. Switching embodiment would
  change the expert, the step limits and the survey, so it is out of scope; the released models
  drive one aloha arm.
- All nine V1 experts pick the arm by object x (x > 0 right, else left) over symmetric spawn ranges;
  `stack_blocks_two` and `stack_bowls_two` use both arms when the objects fall on opposite sides.
- The adapter picks the arm whose tool centre travels further in the demonstration, ties broken by
  whichever arm moves first: observable, not privileged. The chosen arm and the demonstration's arm
  set go into `policy_info`; switching arms per demonstration segment is a possible ablation, not the
  default. Two-arm demonstrations are scored like any other; filtering them would make the
  denominator model-dependent. The conversion oracle shows the structural loss.
- The idle arm gets a fixed hold: its endpose from the episode's first observation plus its commanded
  gripper value. The endpose round trip is exact on aloha (the flange pose; `global_trans_matrix`
  diag(1, -1, -1) cancels SAPIEN's joint-6 anchor, `gripper_bias` 0.12 cancels the offset, and the
  CuRobo base transform inverts the URDF base joints), and it is the expert's own convention
  (`back_to_origin`). A fixed target rather than the current pose, because each successful plan
  re-bases on the measured joints within CuRobo's 5 mm / 0.05 rad tolerance, which could let a loaded arm
  creep (**ASSUMPTION**; the C3 hold test measures drift under both the fixed and the echoed hold). The idle gripper stays open, which six V1 success checks require.

### 3.4 From 20 Hz OSC deltas to RoboTwin actions

- **What one `ee` call costs.** `take_action(..., 'ee')` plans both arms with CuRobo `plan_single`
  and runs max(left_n, right_n) physics steps of 1/250 s. A successful plan is at least 31 steps
  (0.124 s) even for an unchanged target; a failed plan costs 50 steps with that arm not commanded.
  A 2-10 mm delta is about 31-70 steps (analytic estimate from aloha's CuRobo limits).
- **Frames.** robosuite deltas act in LIBERO's world frame, whose axes are the Panda base axes
  (x forward, y left, z up). On aloha (root at (0, -0.65, 0), facing +y) a LIBERO vector v is M v in
  the world, M = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]. robosuite applies position deltas at the tool
  centre point and rotations as world-frame rotations multiplied on the left; aloha's tool centre
  point is 0.12 m along the flange's +x axis, so the adapter integrates there and converts back to
  the flange. Rotating about the flange would add about 0.12 m of tool motion per radian.
- **Gain.** An OSC delta is a goal offset, not a displacement: with kp 150 the arm covers only part
  of it in one 50 ms step. On one LIBERO-90 task (50 demos, 5821 steps) a least-squares fit of achieved
  to commanded tool motion gives 0.21 (per axis 0.16 / 0.26 / 0.19); an independent re-fit gave
  0.215, unchanged at lags of 1-3 steps, so it is persistent under-tracking, not lag. That data is
  LIBERO-90 human teleoperation, not the scripted LIBERO-Gen demonstrations the checkpoint trained on,
  so the value is an **ASSUMPTION** until `calibrate` refits it. Feeding 0.05·a
  straight into RoboTwin, whose planner reaches its targets, would move about 4.7 times too far, and
  prompt actions built without the gain would be 4.7 times too small. The adapter fits α_p and
  α_r by least squares on the checkpoint's own LIBERO-Gen training data (`icil-bpp calibrate`,
  which also records a time-stretch factor from step-speed medians, default 1.0), multiplies executed
  deltas by them and divides prompt deltas by them, so prompt and rollout stay consistent.
  **ASSUMPTION** that one constant per axis is adequate; the fit quality is recorded.
- **Prompt speed.** One action unit is α_p · 0.05 ≈ 10.7 mm of achieved motion per 20 Hz step, and
  RoboTwin's CuRobo expert may move faster than that (unmeasured). `calibrate` reports the fraction
  of prompt actions that clip at ±1 and the prompt-action range after the checkpoint's normalizer. If
  clipping is common, the adapter time-stretches instead of clipping: it resamples the demonstration
  more densely (stretch × 20 samples per demonstration second, each treated as one 20 Hz step), with
  the stretch chosen so the 99th-percentile displacement fits, and records it in `describe()`. The
  budget table below is then recomputed; a stretch above about 2 makes `ee_grouped` necessary on
  the 400-call tasks.
- **Cadence.** One 20 Hz delta per `take_action` (`act()` returns k = 1 from an internal queue of 12).
  The 2-step history then matches LIBERO's spacing in commanded displacement (through α), not in
  time (two observations are 0.124-0.28 s of simulated time apart instead of 0.05 s), re-planning
  happens every 12 calls, and the call
  budget matches LIBERO's step budget. Budgets at one delta per call, V1 demos at 20 Hz:

  | Task | Demo steps at 20 Hz | Step limit |
  | --- | ---: | ---: |
  | click_bell | 94 | 400 |
  | click_alarmclock | 106 | 400 |
  | press_stapler | 144 | 400 |
  | place_a2b_left / right | 179 | 400 |
  | place_container_plate | 192 | 400 |
  | place_empty_cup | 208 | 500 |
  | stack_blocks_two | 380 | 800 |
  | stack_bowls_two | 384 | 900 |

- **Tolerance and execution modes.** At about 1 mm per step the motion sits under CuRobo's 5 mm
  goal tolerance, so re-basing each call on the measured pose could stall. The adapter integrates on
  a virtual target that keeps sub-tolerance residuals: it is re-anchored to the measured pose only
  when the tracking error exceeds a bound (for example twice CuRobo's tolerance) or a plan fails,
  with the bound sized from the C3 probe, and a stall detector (no tool motion across N calls)
  reports in `policy_info` and in O3. Three execution
  modes, recorded in `describe()`: `ee_step` sends one delta's target per call through
  `take_action('ee')`; `ee_grouped` sends a chunk's 12 deltas as calls of 4, 4, 3 and 1 (split again
  at gripper flips), which saves budget while the trailing single delta keeps the last two
  observations one 20 Hz step apart; `qpos_ik` solves the aloha 6-joint chain in the adapter with a
  standard kinematics library loading the vendor URDF and sends joint targets through
  `take_action('qpos')`, the path the replay oracle validates. The conversion oracle on a
  calibration global seed (1000, audited disjoint from 42; **ASSUMPTION** choice), never a reported
  seed, picks the mode. `qpos_ik` is the fallback, because it
  duplicates kinematics RoboTwin already owns.
- **Gripper**: `gripper_states` comes from the measured finger joints (C3), mapped affinely from
  aloha's range (`gripper_scale` -0.01 to 0.045 m) onto LIBERO's [0, 0.04] and [-0.04, 0]. RoboTwin's
  reported gripper value is the command, which reads fully closed on an object whose width LIBERO's
  measured fingers would show. The execution target is 1 (open) when the action is < 0, else 0. Prompt labels follow
  the onset of the commanded change: +1 (close) while the commanded value is falling or closed, -1
  while it is rising or open, so labels do not lag RoboTwin's 300-step gripper ramp the way a 0.5
  threshold would.
- **Rotations**: RoboTwin quaternions are wxyz and go straight into BPP's `quaternion_to_matrix`;
  the runner's xyzw reorder would corrupt them. rot6d is the first two rows of R.
- **Warm-up** is skipped: grippers start open, and 12 calls would come out of the budget. The first
  history is padded by repetition, as BPP's own wrapper does.

### 3.5 Demonstration timing

Frames are not evenly spaced: `take_dense_action` records before the first physics step, after
steps 1, 16, 31, ..., and after the last step, so every primitive adds a 1-step near-duplicate, a
0-14-step remainder and an exact duplicate at the next boundary; a gripper open or close is 300 steps
recorded as 22 frames with the arm still. Frame index x 15 as time distorts a 20 Hz resampling by up
to about 0.2 s per primitive. RoboTwin builds its scene with `Engine.create_scene()`, which returns
SAPIEN's Python `Scene` wrapper whose `step()` is a plain Python method, so the core can count
physics steps with an instance-level override like the existing `_take_picture` one, applied to
the scene object rather than the task env.
The counter sees every physics step, including those of `together_move_to_pose`, which has its own
step-and-picture loop outside `take_dense_action`; wrapping `take_dense_action` instead would miss
them.

### 3.6 Proprioception

The checkpoint's normalizer (min/max of `ee_pos`, `gripper_states`, delta position) is
LIBERO-specific. The adapter maps the tool-centre position as p_L = Mᵀ (p_world − b_arm) + b_Panda
(b_arm the active arm's base, b_Panda LIBERO's Panda base) and orientations as R_L = Mᵀ R_world C,
where C is a fixed tool-frame correction (aloha approaches along +x, robosuite's eef along +z) set
once and checked on renders. The fraction of normalized proprio outside [-1, 1] is recorded per
episode.

### 3.7 Environments and the RTX 5090

- **GPU.** torch 2.4.1+cu121 has no sm_120 kernels, its extension builder rejects arch 12.0 (so the
  CuRobo build fails), and `install_robotwin.sh` exits unless torch reports CUDA 12.1. torch 2.8.0
  cu128 ships sm_120. CuRobo v0.7.8 on nvcc 12.8 has community reports of success with
  `TORCH_CUDA_ARCH_LIST=12.0`, none from the maintainers.
- **Rendering does not work in this container as it stands.** The NVIDIA Vulkan driver loads
  `libEGL.so.1`, which is missing, so Vulkan finds no device (RoboTwin issue #259, the same on RTX
  5080/5090), and `import sapien` already fails on the missing `/usr/share/glvnd/egl_vendor.d`. A
  probe on this machine brought Vulkan up, with the RTX 5090 and `VK_KHR_ray_tracing_pipeline`, after
  providing three things: `libegl1`, an NVIDIA EGL vendor manifest (`10_nvidia.json`), and a Vulkan
  ICD manifest pointing at `libGLX_nvidia.so.0`. The install doc's advice for this error is wrong for
  containers.
- **Conflicts that force separate processes.** The BPP and UniSkill stacks conflict with each other
  (transformers, diffusers, huggingface-hub) and with RoboTwin's `huggingface_hub==0.25.0`, and each
  installs a different package named `robomimic`. UniSkill's robomimic calls `torch.load` without
  `weights_only`, which fails on torch 2.6 and later. BPP's workspace class imports LIBERO, so the
  adapter builds the model directly.

## 4. Benchmark changes

Each item is one issue and a series of one-concern commits, and each commit passes
`pytest -m "not sim"`.

### Core (`src/robotwin_icil/`)

**C1. Camera profiles.** `cameras.yml` (data, like `tasks.yml`) and `camera_profiles.py`;
`SceneConfig.camera_profile = "stock"`; `--camera-profile` on `eval`, `survey` and `export`.
- A profile may replace a static camera in its slot or change a camera type; it may not add, remove
  or rename `head_camera`, toggle `collect_head_camera`, or duplicate a name. A pure guard counts the
  static cameras that will call `create_camera` and refuses any profile that changes the count.
- `far_side` replaces `front_camera` with `far_side_camera` (variant A until M1 decides).
- The manifest records the profile name, version and sha256 plus the resolved static camera list
  (a derived entry in `runner._ROBOTWIN_CONFIG_KEYS`, which today omits `left_embodiment_config`).
- `EpisodeVideo` takes its camera from the profile. `robotwin-icil cameras --profile P --task T
  --seed S --out DIR` writes one PNG per camera; it is the gate for any new or changed profile.
- Tests: pure guard and slot-preserving resolve; sim test that `stock` and `far_side` give identical
  actor, articulation, robot and extras fingerprints for three seeds in each V1 category, and that
  `far_side_camera` renders 180x320x3 without being dominated by the wall.

**C2. Physics clock.** `robotwin.clock(env)` installs the `scene.step` counter after `setup_demo`
and the fingerprint (so RoboTwin's 2500-step stability settle is excluded) and removes it before
`close`. `Frame.time_s`, `Observation.time_s`, `EpisodeRecord.physics_steps`; `Demonstration`
requires non-decreasing times (ties allowed); `timestamps()` falls back to collapsing duplicates for
data without times. Tests: sim frame gaps follow the primitive pattern above; fingerprints and the
replay 18/18 are unchanged with the clock on.

**C3. End-effector views and the `ee` oracle.** `Demonstration.endposes() -> (T, 16)` in
`take_action('ee')` layout, `ee_actions() -> (T-1, 16)`, `arms_moved()`; measured finger joint positions per arm as
`Frame.gripper_joints` and `Observation.gripper_joints`, read in `robotwin.py` from the arm entities
(proprioception a real robot has, unlike RoboTwin's commanded gripper value); a built-in `replay_ee`
(O2). `docs/policies.md` documents the `ee` semantics: world frame, wxyz, flange pose, at least 31
steps per call or 50 on failure, and step limits count calls. Sim tests: joint drift over 10 calls is measured
under both hold rules (first-observation pose and echoed current pose), and the fixed hold must stay
under 1e-3 rad; a probe logs CuRobo's
status, plan length and wall-clock for no-op, 1, 5, 10 and 20 mm tool-centre targets on one arm,
plus the wall-clock of one four-camera `get_obs()`, which measures
the tolerance question in 3.4 before any adapter depends on it.

**C4. Policy configuration and hooks.**
- `eval --policy-arg key=value` (repeatable) passes kwargs to `make_policy`, which already accepts
  them; an adapter's many constants live in its own YAML, passed as `config=PATH`. Adapters stay
  zero-argument constructible (`tests/test_policy.py`).
- `ICILPolicy.seed(seed)`, called before `reset()` with `default_rng([global_seed, episode,
  POLICY_STREAM])`, never the scene seed: diffusion sampling becomes reproducible per episode and
  independent of resume order. The adapter builds a dedicated `torch.Generator` from it for the
  diffusion noise; RoboTwin seeds torch's global RNG when it builds a scene, so in-process adapters
  must not draw from it, and torch models run through `remote` by default.
- `ICILPolicy.episode_info() -> dict` fills `EpisodeRecord.policy_info` (active arm, prompt chunks,
  proprio out-of-range fraction, clipped actions).
- `ICILPolicy.close()`, called by `runner.run` in `finally`.
- A `describe()` convention, checked by a test when present: `adapter` and `adapter_version`,
  `checkpoint` and its sha256, `training_tasks` (or "unknown"), `camera_profile_required`, and a
  parameter checksum taken at the start and end of the run; a mismatch raises `PolicyError` (a
  frozen-policy audit).
- The manifest records the config; a resume with a different config or adapter version is refused.
  `RunManifest.policy_environment` (python, torch, CUDA, GPU, model-repo commits) is recorded and
  excluded from identity like `environment`.

**C5. Records and report.**
- New `EpisodeRecord` fields, all defaulted so old runs still load: `action_type`,
  `physics_steps`, `demonstration_arms`, `policy_info`. `demonstration_arms` is metadata for
  analysis, not a report slice, because CLAUDE.md forbids tiers.
- `records.environment()` adds GPU, driver, torch, CUDA and CuRobo versions: the Blackwell stack
  can change expert plans.
- `report --reference RUN_DIR ...` prints each reference run's per-task fractions beside the model's
  and refuses references with a different global seed, suite, benchmark config or profile. The report
  labels the profile, adapter version and training regime, and prints "evaluation tasks seen in
  training: k/9". No new official score.

**C6. Out-of-process policies.** `protocol.py`, `serve.py`, `remote.py`.
- Adapters are ordinary `ICILPolicy` subclasses that run in-process for development or behind
  `python -m robotwin_icil.serve --policy module:Class --config FILE` in their own environment;
  `RemotePolicy` (built-in `remote`) is the generic client. The simulator environment never imports
  model code.
- Transport: standard-library `multiprocessing.connection` over a Unix socket in the run directory,
  or TCP for a model on another machine, with an authkey. It adds no dependency to the simulator environment and has no size cap. ZMQ
  REQ/REP is stuck after a timeout, and websockets defaults to a 1 MiB message limit.
- Messages: a JSON header plus raw ndarray buffers, no pickle. Operations `hello` (protocol
  version, adapter, action type), `describe`, `seed`, `reset`, `demo_begin` / `demo_frames` /
  `demo_end` (the demonstration streamed, up to 300 MB), `act`, `info`, `ping`, `shutdown`, each with a
  timeout (900 s at startup for the 6.9 GB checkpoint).
- Every error reply, timeout or dead server becomes a `PolicyError` naming the server log, the
  server is killed, and the runner resumes from `episodes.jsonl` as it does today.
- Tests: fake servers (replay, crash, hang, wrong shape, bad version) launched with
  `sys.executable`; parity: served `replay` produces records identical to in-process `replay` on
  the fake environment.

**C7. Training-data export.** `export.py` and `robotwin-icil export --suite S --episodes N --seed G
--camera-profile P --out DIR`.
- `export` produces data for work outside the benchmark and never feeds a score. It needs the
  CLAUDE.md amendment in Docs, which rewrites the "No dataset, no training, no train/eval split."
  sentence, signed off by the maintainer first.
- Reuses `generate.generate` unchanged: same expert, capture, clock, profile and rejection
  accounting.
- Seeds from `default_rng([global_seed, episode, EXPORT_STREAM])`, filtered against a reserved
  evaluation seed set (every seed the official global seeds can draw for up to 10,000 episodes); the
  filter is recorded. `robotwin-icil audit-overlap RUN_DIR EXPORT_DIR` fails on any shared
  (task, scene seed).
- One `.npz` per episode mirroring `Demonstration` (images per camera, qpos, endpose, gripper,
  times, frequency) with a JSON header (task, scene seed, profile, both commits, schema version).
  NumPy keeps the core free of an HDF5 dependency. `export.load(path) -> Demonstration` is the only
  way training code reads it, so the adapters' own conversion functions run on the same type at
  training and evaluation.
- `tasks.yml` gains `heldout_v1`: the tasks outside `v1` (41 in the table) whose expert solves at
  least half of the surveyed seeds under the profile (**ASSUMPTION** threshold; `survey --suite all`
  runs first), with a disjointness test. `export`
  refuses `v1` tasks unless `--allow-eval-tasks`, and records the flag.
- Test: an exported episode loads bit-equal to the `Demonstration` that episode hands a policy.

### Adapters (`policies/`, a separate distribution `robotwin-icil-policies`)

Extras `pure` (CI), `bpp`, `uniskill`. Each adapter declares `ADAPTER_VERSION`; changing the
conversion math bumps it.

**A1. `icil_policies.common`** (NumPy only, tested in CI): wxyz quaternion, matrix, axis-angle and
rot6d conversions; the aloha base frame M and arm bases; time-based resampling (linear position,
slerp rotation, held gripper, nearest image); demonstration arm choice and fixed hold; image path
(arm-centred crop, area resize to 128, bilinear to 224 with `align_corners=False`); `ChunkExecutor`
(history of `To` padded by repetition, queue of `Ta`, one action per `act()`); aloha forward and
inverse kinematics built from the vendor URDF with a standard kinematics library (used only by
`qpos_ik`), checked against RoboTwin's own endposes in a sim test.

**A2. BPP transfer adapter** (`icil_policies.bpp`).
- `icil-bpp slim` verifies the checkpoint's sha256 and writes the EMA state dict (normalizer
  included), the composed config and `SOURCE.json`, so servers do not load 6.9 GB of dill each start.
- The model comes from the repo's own Hydra composition (`libero_policy_dunetp`,
  `task=liberogen_spatial_combination`, `+modifiers=libero/liberogen_spatial_combination`) with the
  vision backbone left at `pretrained=true`, then a strict load through `BasePolicy.load_state_dict`,
  which drops `_extra_training_split_info`. With `pretrained=false`, BPP runs its own weight
  initialisation, which rejects the CLIP ViT's bias-free patch layer; `pretrained=true` downloads the
  backbone once and the checkpoint overwrites it, so the environment installer prefetches it and servers
  start offline. Verified on the RTX 5090: every key matches strictly. Not from the checkpoint's
  embedded config, which lacks `use_pool_modality_pos_embed`; the code default would add a parameter
  and strict loading would fail.
- Cameras: `agentview_rgb` from `far_side_camera` (arm-centred crop), `eye_in_hand_rgb` from the
  active arm's wrist camera (centre 240x240, fixed roll from a render); the adapter raises
  `PolicyError` if the profile is not `far_side`.
- Prompt: resample the demonstration to 20 Hz by time, base-frame proprio, gain-divided deltas
  clipped to [-1, 1], BPP's own `PromptActionChunker` and `collate_prompts`; L ≤ 50 or `PolicyError`
  (V1 needs 5-20). `reset()`, then `prompt()` once, then `predict_action()`.
- Execution as in 3.4; `mode` (`ee` or `qpos_ik`), gains, wrist roll and C recorded in `describe()`.
- `icil-bpp calibrate` fits α_p and α_r on the LIBERO-Gen training hdf5s and reports the proprio
  out-of-range fraction and the angle between a downward aloha grasp under C and LIBERO's median
  initial orientation.
- Conversion oracle `BPPConversionReplay` (O3): the prompt's own converted actions replayed through
  the whole adapter chain. Its V1 fraction is the ceiling for any model behind this adapter.
- Pre-flight gates before any RoboTwin number: the server's prompt tensors equal
  `LiberoReplayImageDataset(only_prompt=True)` on one LIBERO-Gen demonstration within 1e-5, and its
  actions equal a direct `predict_action` call for a fixed seed; the checkpoint, run by BPP's own
  LIBERO runner in a separate LIBERO environment, solves at least 5 of 10 episodes of one unseen
  LIBERO-Gen task (the paper reports about 71%).
- `robotensor/bpp-genesis` is not used until its tensors are hash-compared with the Hugging Face
  checkpoint (its README and config name different sources).

**A3. UniSkill adapter** (`icil_policies.uniskill`).
- `SkillExtractor`: frozen `idm.pth` and Depth-Anything-V2-Small, third-person frames upright,
  on the demonstration resampled to 20 Hz by `time_s`, k = 20 steps (1.0 s, the paper's setting), one
  row per step with the future frame clamped at the end: the released `skills.zip` convention, not
  `extract_skill.py`'s T-k rows. Recomputing the released skills matched upright frames (cosine
  0.964) far better than flipped ones (0.84). The camera is `far_side_camera`, the closest match to the ISD's LIBERO pretraining view and the
  profile every model row uses, unless M4's separation check prefers `head_camera`; Depth-Anything
  runs on the same crop at training and evaluation.
- Policy server: robomimic-fork `DiffusionPolicyUNet` trained on exports (B3), 14-dim absolute
  next-frame qpos, `To` 2, `Ta` 8, `Tp` 16, DDIM 10, EMA weights baked into the exported checkpoint at
  conversion and loaded once before the start checksum (the fork's rollout used live weights;
  recorded as a deviation), so the evaluator never calls `copy_to`, which would be a parameter write. At each re-plan the skill row is the number of actions executed so far,
  held past the end; one qpos action per resampled step keeps the step count aligned with the skill rows. No warm-up.
- Its conversion oracle replays the 20 Hz resampled demonstration through `take_action('qpos')`,
  the same oracle as BPP-RoboTwin's; it must come out close to the native replay's 18/18.
- Environment: torch 2.8+cu128, transformers ≥ 4.48 for Depth-Anything, current diffusers (its
  `EMAModel` still accepts the fork's `model_cls`), xformers and flash-attn dropped because nothing
  imports them, `torch.load(..., weights_only=False)`; validated by the environment's contract test.

**A4. Training exporters** call the adapters' own conversion functions: BPP-RoboTwin data (B2) and
UniSkill robomimic hdf5 plus `skills/<task>/demo_i/{base,aug_0..4}.npy` (B3), with an export
manifest (ISD sha, k, augmentation parameters). Golden tests: each model's own loader reading the
export equals what the adapter feeds the network for the same demonstration (teacher forcing), on a
committed, downscaled real demonstration.

### Model-side

- **B1. BPP transfer**: no code changes; config composition and the α fit.
- **B2. BPP-RoboTwin**, the proper BPP entry: a task config `robotwin_aloha_qpos` with rgb
  `far_side_camera`, `left_camera`, `right_camera` at 224 and `qpos` (14); the demonstration is
  resampled to 20 Hz on `time_s` (native frames are unevenly spaced) for training data and prompts
  alike; actions are the next step's 14-dim absolute qpos, horizon 16, execute 12;
  `prompt_chunk_n_actions: 20` reuses the checkpoint's chunk size and positional embeddings;
  `max_sequence_length` at least the longest demo; a
  dataset class over `export.load`. Initialize the ViTs, prompt decoder and UNet trunk from the LIBERO
  checkpoint; re-initialize the per-modality projections, position embeddings (the current-observation one
  shrinks from 10 to 8 tokens), action MLP and UNet input and output layers; refit the normalizer on
  the export. Its conversion oracle is the qpos replay of the resampled demonstration. BPP reports 28 h on 4x A6000 for the Combination domain.
- **B3. UniSkill-RoboTwin**: config `uniskill_robotwin` (rgb `far_side_camera`, `left_camera`,
  `right_camera`; `qpos`; `ac_dim` 14 with min-max normalization), lazy LIBERO imports,
  `weights_only=False`, EMA weights baked in at export, skill augmentation (`aug_0..4`; the paper and code do not
  say how the released ones were made, so the choice is documented and ablated; **ASSUMPTION**
  colour jitter and small crops). The trained checkpoint goes to a pinned Hugging Face repo with the
  export manifest in its model card.
- **B4. BPP bridge fine-tune (optional ablation)**: the released shape_meta fine-tuned on
  bridge-converted single-arm exports (LIBERO-shaped hdf5, images stored flipped because BPP's hdf5
  loader flips them back). It shares A2 unchanged, so its row is comparable with the transfer row.
- **Training pairs and what the headline measures.** At evaluation the demonstration starts in the
  rollout's own scene, so the prompt's first frame equals the first observation; BPP never saw that
  in training, where its runner pairs each rollout with another demonstration of the same task from
  a different initial state. Each model follows its authors' recipe. BPP-RoboTwin trains on
  cross-seed pairs within a task (BPP's pair mode), and a same-episode variant (prompt equals target,
  which mostly teaches copying) is a labelled ablation. UniSkill trains, by design, on skills
  extracted from the same trajectory whose actions it imitates, with skill augmentation standing in
  for prompt and rollout differences. The regime goes into `describe()` and the model card, and every
  score prints next to the replay and conversion oracles, so a policy that learned to copy reads as
  near-replay rather than as in-context learning.
- **Training regime** for both: `heldout_v1` by default, so V1 measures use of the demonstration
  rather than task memory, as BPP's own held-out combinations do. A seen-task variant is allowed and
  labelled via `training_tasks`; only a future `different_object_pose` setting separates the two.

### Infrastructure

- **I1. Rendering prerequisites** in the container image: `libegl1`,
  `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`, `/usr/share/vulkan/icd.d/nvidia_icd.json` pointing
  at `libGLX_nvidia.so.0` (or the `__EGL_VENDOR_LIBRARY_FILENAMES` and `VK_ICD_FILENAMES` exports in a
  `scripts/robotwin_env.sh`). Gate: RoboTwin's `scripts/test_render.py` prints "Render Well".
- **I2. `install_robotwin.sh` for sm_120**: a branch on compute capability keeps today's
  reference path for GPUs below 10.0, so A6000 results stay comparable. On sm_120: torch 2.8.0+cu128 installed before RoboTwin's
  requirements with its `torch==2.4.1` pin overridden; conda `nvidia/label/cuda-12.8.1`;
  `TORCH_CUDA_ARCH_LIST` from `nvidia-smi` (12.0); CuRobo v0.7.8 from source; the CUDA guard becomes
  "toolkit matches torch"; the gcc 12 stage becomes optional. sapien, mplib and numpy 1.26 pins stay.
- **I3. Re-baseline**: the survey and replay 18/18 re-run on the new stack under `stock` and
  `far_side` before any model number; a different result means the stack changed the benchmark.
- **I4. `scripts/install_policy_env.sh bpp|uniskill`** builds the model environments from lockfiles
  in `policies/envs/`:

  | Environment | Contents |
  | --- | --- |
  | `robotwin` | the simulator and `robotwin_icil`; never imports model code |
  | `icil-bpp` | torch 2.8.0+cu128, diffusers 0.35.1, transformers 4.57.1, huggingface_hub 0.35.3, the BPP repo, `robotwin-icil` and `policies[bpp]` |
  | `icil-uniskill` | torch ≥ 2.7+cu128, the modernized policy fork and the ISD dependencies, `robotwin-icil` and `policies[uniskill]` |

### Docs

- `docs/policies.md`: four cameras today; `ee` semantics; profiles; remote policies;
  `--policy-arg`; oracles and `--reference`; the held-out training rule; provenance keys.
- `docs/cameras.md`; `docs/models/bpp.md` and `docs/models/uniskill.md` with every constant
  (α, C, k, crops, mode, wrist roll).
- `docs/install.md`: the Blackwell path and the container rendering prerequisites.
- `CLAUDE.md`: profiles are replace-only; `policies/` is a separate distribution; and an amendment
  the maintainer must sign off before any export or training, replacing the sentence "No dataset, no
  training, no train/eval split." with: "The evaluator uses no dataset and trains nothing. Adapters
  may train outside it on exported expert demonstrations of non-evaluation tasks or seeds, drawn
  from a separate seed stream and audited disjoint from evaluation; every adapter records its
  training tasks in `describe()`." 
- Upstream issues: the UniSkill policy checkpoint's dead link, and the BPP checkpoint's config
  lacking `use_pool_modality_pos_embed`.
- README oracle rows; a CHANGELOG entry per issue.

## 5. Milestones

| Milestone | Items | Go / no-go |
| --- | --- | --- |
| M0 Stack | I1, I2, I3 | "Render Well"; `pytest -m sim` passes; replay 18/18 on `--suite v1 --episodes 18 --seed 42`; survey within ±15 points per task of `docs/results/survey-v1-seed0.json`, committed as a second reference |
| M1 Seams | C1-C6 | `far_side` keeps every stock scene; `replay_ee` (O2) measured; served replay equals in-process replay; camera variant, crop columns, wrist type and wrist roll chosen from renders; if O2 falls far below replay, the `ee` path is fixed or documented before any adapter number |
| M2 BPP transfer | A1, A2, B1 | the pre-flight gates pass; O3 measured on a calibration seed, at least 0.70 on single-arm demonstrations, and the execution mode chosen; a 90-episode V1 run (10 per task, seed 42, `far_side`) completes with no `PolicyError` and the frozen checksum intact; videos show the active arm moving and the idle arm still; the report prints O1, O2 and O3 beside the model; a near-zero score carries its Wilson 95% bound (0 of 90 is at most 4.1%) |
| M3 Data | C7, A4 | the CLAUDE.md amendment is signed off; at least 25 held-out tasks with at least 30 demonstrations each (**ASSUMPTION** thresholds); export equals evaluation demonstrations; golden tests pass; overlap audit clean |
| M4 Training | A3, B2, B3 | ISD skills from RoboTwin frames separate tasks before policy training (camera chosen here); each project's own validation looks sane |
| M5 Scores | runs | BPP-RoboTwin and UniSkill-RoboTwin held-out checkpoints on 450 V1 episodes (50 per task, about ±4.6 points overall at 95%), seed 42, `far_side`, videos on a sample; seen-task variants on 180 episodes, labelled; every table uses one profile and prints the reference rows |
| M6 Hardening | CI, docs | a `policies` CI job per environment; docs and CHANGELOG complete |

Issues C1-C6 are independent of each other once M0 lands; A2 needs C1, C2, C4 and C6; A3 and A4 need
C7. Compute: these estimates come from the qpos replay run (expert 18-89 s per seed; about 70-160 s per
scored episode, so 90 episodes about 2-4 hours and 450 about 9-20 hours). End-effector modes add two
CuRobo plans per call and, at one action per `act()`, a four-camera ray-traced observation per call,
up to 400-900 per failed episode, so the C3 probe also times one `take_action('ee')` and one
`get_obs()`, and M2 and M5 are re-budgeted from those numbers. Model inference is cheap by comparison: BPP encodes an 8-chunk prompt in 0.45 s and predicts a
16-step chunk in 0.09 s on the RTX 5090, using 2.2 GB of GPU memory.

## 6. Risks and open questions

- The container rendering fix was proven by a Vulkan probe, not yet by SAPIEN end to end (M0). If
  SAPIEN ray tracing or CuRobo will not run on sm_120, the simulator stays on the A6000 reference
  machine and the model servers run on the 5090 over TCP.
- CuRobo on nvcc 12.8 and sm_120 has only community reports (M0).
- One gain per axis may misrepresent fast or contact phases (M2, recorded fit quality).
- CuRobo's 5 mm tolerance against 1 mm steps may force `qpos_ik`, which needs checked-in aloha
  kinematics (M2).
- `far_side_camera` shows objects 1.57x larger than LIBERO; variant C depends on untested
  back-face behaviour; the wrist field of view (37° against 75°) cannot be widened without
  patching the submodule's camera table (M1).
- The BPP transfer score will be near zero and must not be read as evidence about BPP's in-context
  ability; the oracles printed beside it make that visible.
- `robotensor/bpp-genesis` names two different sources; decide which file is the competition's BPP
  before reusing it. This plan uses the Hugging Face LIBERO-Gen Combination file.
- UniSkill's policy checkpoint is gone; ask the authors. Its skill augmentation is unspecified, and
  the ISD has never seen SAPIEN renders or aloha (M4 check).
- Held-out training on 41 tasks, several with low expert success or two-arm structure, may give low
  V1 scores; the seen-task variant is tempting but confounded.
- Single-arm transfer fails two-arm stacking seeds by construction; per-task tables and O3 show it.
- The clock override must not change behaviour; the sim test with the clock on guards it.
- The CLAUDE.md amendment may not be accepted; then only the transfer row ships.
- UniSkill's step-count skill index drifts if the policy leads or lags the demonstration (the
  paper's fixed-k speed limit); the adapter records the skill index against progress in
  `policy_info`, with no re-anchoring in V1.
- Fine-tuning memory and time on one 32 GB GPU are unknown; a short profiling run picks batch size and
  precision.
- New manifest and record fields change `RunManifest.identity()`; old runs must keep loading and
  resumes must refuse changed settings.
- RoboTwin's expert may move faster than the largest LIBERO step, saturating prompt actions;
  `calibrate` measures it, and a time stretch then raises the step budget (M2).
- Retrained models may learn to copy a same-scene prompt; the training-pair regime, the oracles
  printed beside every score and the later `different_object_pose` setting keep that visible.

Open questions, in the order they need answers: the CLAUDE.md amendment; whether the simulator stays
on the 5090 or moves to the A6000 (M0); camera variant A or C (M1); the execution mode (M2); the ISD
camera (M4); which checkpoint the competition treats as its BPP (`bpp-genesis`).
