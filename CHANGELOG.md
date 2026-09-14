# Changelog

## Unreleased

- (feat): an episode runs against a policy served at an address, and a lost policy voids it.
  `robotwin-icil run-unit --policy-address ADDR --authkey-env NAME [--act-timeout-s S]
  [--policy-log PATH]`, in place of `--policy`, drives a policy served by `python -m
  icil_policy.serve` through `icil_policy.client.RemotePolicy` (`robotwin_icil.remote`, which
  imports it only when one is built). The key is read as hex from the named variable and removed
  from the environment, never taken from argv. The demonstration is sent as `prompt.arrays_from`'s
  arrays with `info` `{frequency, cameras, embodiment, action_dims}`, each observation as
  `frames_<camera>`, `qpos` and `endpose`, `meta` never; `reset` gets a seed drawn from the prompt's
  bytes, never the scene seed. An error reply to `reset`, `prompt` or `act` fails the unit; any
  other remote failure — nothing listening, `hello` refused or unanswered, a timeout, a hang-up, a
  malformed reply — is `PolicyUnreachable` and voids it with `void_cause` "policy" and the server
  log's tail in `error`. Every other void of either command carries `void_cause` "harness", a field
  added to both results. Every policy is now reset with a public `EpisodeInfo` (robot, live action
  widths, seed) and closed once its unit is over (#85).
- (feat): a demonstration is built once and saved, and an episode is evaluated from the saved file.
  `robotwin-icil materialize --task T --scene-seed S [--scene-seed S2 ...] --out DIR` tries the
  candidate seeds in order — one scene built and one expert run each — and writes `prompt.npz` and
  `demonstration.mp4` for the first the expert solves, and `result.json` either way, with the
  chosen `scene_seed` and every attempt (seed, rejection, detail), which the prompt's `meta`
  records too; every candidate rejected is a void result with `void_cause` "harness", and a seed
  given twice is refused. It exits 0 whenever `result.json` was written, a rejection included, so
  the caller reads the reason instead of a log tail; `robotwin-icil run-unit --prompt DIR/prompt.npz --policy P --out
  DIR` rebuilds the scene from the prompt's privileged `meta`, refuses a meta whose digest is not
  its own fingerprint's, voids on scene drift with the mismatches, rolls the policy out and writes
  `result.json` (`success` and `steps` null exactly when `void`, the rebuilt scene's
  `live_scene_sha256` and its `scene_max_error`, the checkpoint and both commits) and
  `evaluation.mp4`. Both commands' `result.json` carry `success`, `void`, `steps` and `error`, the
  fields the orchestrator reads: every candidate rejected is a void materialize. In `run-unit` a policy at
  fault — raising from `reset` or `set_demonstration`, or a wrong-width or non-finite action —
  fails its unit and never voids it; void is kept for an unreadable, mistyped or tampered prompt,
  scene drift, a config RoboTwin refuses, any other harness fault while evaluating (traceback to
  stderr) and a GPU lost or full mid-rollout, which in `eval` now stops the run as it does in the
  expert. `run-unit` exits 1 only when the unit cannot start. Both commands clear their outputs
  before anything can fail, and `run-unit` refuses to write into its prompt's directory; an
  adapter's paths are resolved in its constructor or passed absolute, since the simulator runs
  from `vendor/RoboTwin`. `prompt.npz` holds
  `frames_<camera>`, `qpos`, `endpose` (per arm, left then right: pose then gripper, 16 wide),
  `actions`, `times`, `frequency` and `meta`; `prompt.CHANNELS` publishes which arrays are video,
  proprioception and actions. Every frame records the simulated time it was taken at
  (`Frame.time_s`, `Demonstration.times()`), counted in physics steps by `robotwin.clock`, because
  RoboTwin's frames are not evenly spaced. `eval` records are otherwise unchanged: `run_episode` is
  `generate.attempt` then `episode.evaluate`, and a fixture pins its records. That fixture, not a
  second GPU run, is the comparison: on any commit a scene's `demonstration_frames` and replay
  `steps` can differ between runs, because the expert's CuRobo trajectories are not the same
  length every time and a primitive one physics step longer can record one more frame (click_bell,
  seed 42, aloha-agilex, before this change: 77 frames and 60 steps, then 78 and 59, on one commit
  with identical manifests) (#84).
- (feat): the survey renders no camera unless asked. Every frame called `get_obs`, which
  ray-traces every camera, although the survey reads only joints; `robotwin.capture(images=False)`
  now reads the joint vector and endpose straight from the robot (`robot_state`) and never calls
  `get_obs`, `attempt` and `survey_task` pass the choice through, and `survey --images` renders
  as before. The survey JSON becomes an object, `{"images": …, "tasks": [...]}`, and every task
  entry records `images`. At the pinned RoboTwin no expert reads an observation or a camera, and
  the one `get_obs` side effect an expert could see, the `crazy_random_light` RNG draw, is kept;
  a sim test compares attempts with and without images on dual Franka. Rendering still holds GPU
  memory, and RoboTwin's CuRobo batch planner reports a CUDA out-of-memory error as a failed
  plan, so on a GPU short of memory a survey without images can keep seeds `eval` would reject;
  `--images` measures what `eval` sees. `eval` always renders (#83).
- (feat): the survey records every seed, and for every demonstration which arms moved and by how
  much. Each task's JSON entry gains `seeds_detail` — seed, outcome (`ok` or the rejection),
  frames, `arms_moved`, `displacement`, seconds — and the counts `one_arm_demonstrations`,
  `two_arm_demonstrations` and `no_arm_demonstrations`; the table gains a *one-arm* column.
  `demo.arm_displacements` reads each arm's largest departure from its first-frame value off the
  joint trajectory (the qpos row splits into a left and a right half on every embodiment RoboTwin
  ships), and `demo.arms_moved` calls an arm moved when that exceeds 0.05 — radians for a joint,
  a fraction of full travel for the gripper's normalised opening. A task's `arms: 1` is thereby
  measured on what the expert did, not only read from its source, and the threshold can be
  re-judged from the JSON after the run (#83).
- (fix): the survey's `--json` file is rewritten whole after every task — written beside itself
  and renamed into place — so a kill that lands mid-write leaves the previous complete file, not
  a torn one; and a survey on an embodiment whose qpos does not split into two equal arms stops
  with one line on stderr instead of a traceback (#83).
- (fix): a frame whose qpos holds NaN or inf is refused, so such a seed is an `expert_error`
  rejection with the reason on record rather than a demonstration in which the arm would have
  read as still (#83).
- (feat): every task declares how many arms its expert needs. `tasks.yml` entries become
  `{category, arms}` with `arms: 1 | switching | 2` (26 / 6 / 18 tasks), `arms.py` re-derives the
  value from a static read of each `play_once` in the pinned checkout, and a test fails naming any
  task whose entry disagrees. The read follows an arm through a helper's parameter, a nested
  def, an `if`/`else`, a lookup made per object, an attribute set in `load_actors` and
  `Base_Task`'s `together_*` helpers, and errs towards more arms where it is unsure. `--arms 1`
  on `eval`, `survey` and `tasks` keeps only one-arm tasks and refuses a task that needs more;
  `2` is the default and changes nothing; the run manifest records `arms`, and a spec or
  manifest refuses an `arms` it cannot honour. Two values differ from the issue's list, decided
  from the source: `put_bottles_dustbin` hands right-side bottles to the left arm (2),
  `shake_bottle_horizontally` drives one arm like `shake_bottle` (1) (#82).
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
