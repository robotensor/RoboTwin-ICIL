# Changelog

## Unreleased

- (feat): every task declares how many arms its expert needs. `tasks.yml` entries become
  `{category, arms}` with `arms: 1 | switching | 2` (26 / 6 / 18 tasks), `arms.py` re-derives the
  value from a static read of each `play_once` in the pinned checkout, and a test fails naming any
  task whose entry disagrees. `--arms 1` on `eval`, `survey` and `tasks` keeps only one-arm tasks
  and refuses a task that needs more; `2` is the default and changes nothing; the run manifest
  records `arms`. Two values differ from the issue's list, decided from the source:
  `put_bottles_dustbin` hands right-side bottles to the left arm (2), `shake_bottle_horizontally`
  drives one arm like `shake_bottle` (1) (#82).
- (feat): a run chooses its robot. `eval` and `survey` take `--embodiment aloha-agilex` (one
  dual-arm URDF) or `franka-panda` (two Franka arms 0.8 m apart, the distance RoboTwin's
  configuration guide gives), resolved into RoboTwin's one-entry or `[left, right, distance]`
  form by `SceneConfig.embodiment`; without the flag the task config's own robot runs, aloha-agilex
  in every shipped config. Frame and action widths come from
  the live robot — `Demonstration.qpos_dim`, `robotwin.action_dims` — instead of the 14/16
  constants that fitted only aloha-agilex, so a dual Franka's 16-wide qpos runs end to end. Every
  episode record, the run manifest and the scene fingerprint name the robot — by its flag name, or
  by its arms and their distance (`franka-panda@0.6`) for any other list a task config gives — and
  a run directory refuses to continue on another one. Verified on an RTX 5090: the click_bell
  replay scores 1/1 on both robots, seed 42 (aloha 60 steps, Franka 36 steps from a 44- or
  45-frame, 16-wide demonstration: the Franka expert's trajectory length varies between runs of
  the same scene), and the aloha record is unchanged (#81).
- (feat): the simulator installs and runs on Blackwell GPUs. `scripts/install_robotwin.sh` picks
  a GPU path from the compute capability — the reference torch 2.4.1 + CUDA 12.1 below 10.0, torch
  2.8 + CUDA 12.8 from 10.0 up — builds CuRobo for it, and makes SAPIEN render in containers that
  ship no glvnd manifests; `robotwin_icil` turns off the OIDN denoiser where it cannot run and
  stops a run when the renderer loses the GPU. `scripts/simwatch.py` reruns a job whose camera
  read hangs. Verified on an RTX 5090: `pytest -m sim` passes and the click_bell replay smoke
  scores (#80).
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
