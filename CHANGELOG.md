# Changelog

## Unreleased

- (feat): end-effector views, measured finger joints and a `replay_ee` oracle.
  `Demonstration.endposes()` gives each frame as the 16 numbers `take_action('ee')` reads — per
  arm the world-frame flange pose, wxyz, then the commanded gripper — `ee_actions()` the next
  one per transition, and `arms_moved()` the arms whose flange travels more than 2 cm
  (provisional). RoboTwin's gripper value is the command, which reads closed on an object, so
  frames and observations gain `gripper_joints`, the finger joint positions measured off the
  robot; it defaults to None, so older demonstrations still load. `replay_ee` plays
  `ee_actions()` back, the ceiling of the `ee` path for any adapter. `docs/policies.md`
  documents the path: at least 31 physics steps per call, 50 on a failed plan, step limits
  counting calls. Sim tests measure an idle arm's drift under a fixed and an echoed hold, what
  CuRobo does with 0-20 mm targets, and run `replay_ee` on click_bell (#36).
- (feat): a physics clock. Demonstration frames are not evenly spaced — every motion primitive
  adds a one-step gap, a remainder and an exact duplicate where the next one starts — so frame
  index over `frequency` is not time. `robotwin.clock` counts `scene.step()` through an instance
  override of SAPIEN's Python `Scene.step`, from after the fingerprint (RoboTwin's settle is not
  counted) to before `close`, and sees `take_dense_action` and `together_move_to_pose` alike.
  Frames and observations carry `time_s` in simulated seconds; `Demonstration.times()` returns
  the frames' times, which never decrease, or for untimed data collapses exact duplicates and
  spaces the rest at `1 / frequency`, and `duration_s` is the span of those times rather than
  frame count over `frequency`; records gain `physics_steps`. All three default, so older
  demonstrations and runs still load (#35).
- (feat): camera profiles that keep every seed's scene. `--camera-profile` on `eval` and
  `survey` picks one from `cameras.yml`; `far_side` puts a 45° L515 across the table in
  `front_camera`'s slot, looking back at the robot 38.9° down. Every static camera draws from
  numpy's RNG before the scene is built, so profiles only replace one: a pure guard refuses a
  change in the camera count, toggling `collect_head_camera`, touching `head_camera`,
  duplicate names and a static camera named like a wrist camera, and holds
  `SceneConfig.overrides` to the same check. The manifest records the profile's sha256 and the
  static cameras (older manifests read as `stock`), survey results name their profile, clips
  film the profile's `video_camera`, and `robotwin-icil cameras` writes one PNG per camera of a
  scene (#34).
- (docs): the README reports V1's status — the replay oracle scores 18/18 on the nine-task
  suite — points at the survey, and uses click_bell, the fastest expert, as its smoke task.
- (fix): a multi-task run no longer collapses once the GPU fills up. The runner keeps one RoboTwin
  env — and its CuRobo planners — alive at a time; a scene that fails to build, or a GPU that runs
  out of memory, stops the run instead of being recorded as rejected seeds; records and reports
  say why seeds were rejected (#30).
- (feat): `robotwin-icil survey` measures RoboTwin's expert per task. Over 20 seeds it solves
  75–100% of every `v1` task but place_object_basket (45%), which leaves the suite; `v1` is
  nine tasks across Pick and Place, Stacking and Press / Push (`docs/survey.md`) (#4).
- (fix): both scenes of an episode are built under the task's name, so rollouts get the
  task's own step limit instead of RoboTwin's silent 1000-step fallback (#7).
- (docs): `docs/policies.md`, the policy adapter guide; the README layout gains `generate.py`,
  `video.py` and `--video`; CLAUDE.md records the frame rate, the working-directory move and that
  RoboTwin's `is_test` selects nothing (#12).
- (fix): a demonstration's `frequency` is frames per second — the sim rate over RoboTwin's
  `save_freq`, ~16.7 fps at the default — not `save_freq` itself, which upstream passes where it
  means a frame rate (#3).
- (feat): `--video` writes `demonstration.mp4` and `evaluation_same_scene.mp4` per episode, from
  frames the episode already has, so it cannot change a scene or a score (#10).
- (feat): Same Scene episodes end to end. `run_episode` generates the demonstration, rebuilds the
  seed, verifies the scene fingerprint, resets the policy, hands it the one demonstration, and
  rolls out to RoboTwin's own success check; a resumable runner assigns tasks round-robin and
  writes the run directory; `robotwin-icil eval | report | tasks`. The replay oracle scores 3/3
  on place_object_basket (#7, #8, #11).
- (feat): one expert demonstration per episode, generated on demand by RoboTwin's own expert.
  Scene seeds are a pure function of (global seed, episode); a seed whose scene is unstable, whose
  plan fails, or whose expert misses or raises is a recorded rejection — never a model failure —
  and the next seed in the stream is tried (#4).
- (feat): `scripts/install_robotwin.sh` builds RoboTwin 2.0's simulator env reproducibly, following
  upstream's `_install.sh`: setuptools 69.5.1, the sapien and mplib patches, and CuRobo v0.7.8
  built against a CUDA 12.1 toolkit and gcc 12 inside the env; pytorch3d and XPolicyLab are
  skipped. Documented in `docs/install.md` with the reference install (#1).
- (feat): every episode's initial scene is fingerprinted before anyone acts — actor and
  articulation poses, joints, camera extrinsics, robot qpos, texture and light draws — and
  compared against the demonstration's; a sim test rebuilds three seeds per V1 category and
  requires identical fingerprints, and different seeds to differ (#5).
- (fix): a RoboTwin task whose imports fail is reported as a broken install, not as an unknown
  task (#20).
- (feat): report the Same Scene 1-Demo Success Rate overall, by skill category and by task, with
  expert rejections, simulation failures and rollout lengths kept apart as benchmark
  diagnostics; fractions throughout, percent only in the rendered text (#9).
- (feat): episode records and append-only run directories — `manifest.json` plus
  `episodes.jsonl`, fsynced per line, resumable, and refusing to merge two different runs (#8).
- (feat): `ICILPolicy` enforces reset -> exactly one demonstration -> act, with `DummyPolicy` and
  the `ReplayPolicy` Same Scene upper bound; `make_policy` takes a built-in name or
  `module:Class` (#6).
- (fix): the demonstration no longer carries the task name, which told the policy which task it
  was solving (#6).
- (feat): model-independent demonstrations captured in memory by intercepting
  `Base_Task._take_picture`; `robotwin.py` is the single seam onto `vendor/RoboTwin` (#3).
- (feat): every RoboTwin task mapped once to a skill category in `tasks.yml`, checked against the
  pinned checkout; provisional `v1` suite (#2).
- (chore): repository scaffolding — Apache-2.0, `robotwin_icil` package skeleton, ruff/pytest
  configuration, CI, and RoboTwin 2.0 pinned as a submodule at `vendor/RoboTwin`.
