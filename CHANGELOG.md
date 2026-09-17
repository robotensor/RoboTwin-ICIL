# Changelog

## Unreleased

- (docs): feature the RoboTwin-ICIL announcement video near the top of the README.

- (feat): add the `robotwin-icil-standard` profile, RoboTwin 2.0's own evaluation with the
  expert's demonstration as the policy's input: all 50 tasks, 100 episodes each, RoboTwin's
  evaluation seed stream (from `100000 * (1 + seed)`, skipping seeds its expert cannot solve),
  aloha-agilex, and `demo_clean` and `demo_randomized` scored separately as `clean` and
  `randomized`. `standard-eval` runs and resumes a plan under `<setting>/seed_<seed>/`;
  `standard-report` gives each setting's mean of 50 task success rates, a skill-category and task
  breakdown and diagnostics, and an official score only at seed group 0 with every episode scored.
  Duel APIs and unit derivation are unchanged.
- (feat): `eval --seed-stream robotwin` draws RoboTwin's evaluation seeds per task and resumes the
  stream exactly; manifests record `seed_stream` (older ones read as `independent`).
- (feat): add optional dataset build/verify/upload/download/sample commands for
  `robotensor/robotwin-icil-aloha-clean`: RoboTwin's 2,500 clean Aloha trajectories with original
  HDF5 bytes, aligned three-camera videos, provenance and checksums, all of them training data
  with no splits. Add causal training samples and a three-view nearest reference adapter. Preserve
  unknown physical timing instead of inventing timestamps from video playback rates. `upload`
  removes Hub files the release no longer inventories.
- (refactor): remove named task sets from the benchmark core. `tasks.yml` has no `suites`,
  `TaskTable` has no `suites`/`suite()` and `select()` takes only `task` and `arms`. The historical
  `v1` and `all` sets are gone. The competition plugin keeps its one track list, `franka_1arm`, in
  `catalogue.SUITES` because the orchestrator's spec names it; its units and pinned hashes are
  unchanged and `units.CATALOGUE_SHA256` is re-pinned for the smaller catalogue. Survey results are
  renamed `docs/results/survey-aloha-seed0.json` and `survey-franka-seed0.json`.

## 0.1.0 — 2026-09-16

- (chore): release both Python distributions as 0.1.0, update citation metadata, and
  add release notes. Limit the benchmark source distribution
  to its own package, tests, scripts and documentation; the pinned RoboTwin submodule
  and simulator assets are installed through a recursive Git checkout.
- (feat): remove standalone CLI suite selection. `eval` and `survey` select the full task
  catalog by default, `--task` selects one task, and `--arms 1` filters to strictly one-arm tasks.
  Task listings and reports show task/category/arm data without suite tags. Keep internal named
  selections and the competition plugin API so the orchestrator's unit derivation remains stable;
  remove suite metadata from `RunSpec` and new manifests. Older manifests remain readable by
  ignoring their legacy suite label; resume identity still checks the exact task list.

- (chore): ignore local crash dumps and environment files while allowing environment examples.
- (docs): rename the project to RoboTwin-ICIL, update repository and citation links, fix the
  clone instructions, and document the private dependency needed for served policies and the
  competition plugin. Python package and command names remain unchanged.
- (docs): cite Robotensor's benchmark in the README and `CITATION.cff`, with a separate
  acknowledgement thanking the RoboTwin team for the simulation and task infrastructure.
- (feat): the one-arm Franka suite is chosen by measuring RoboTwin's expert on two Franka arms.
  `tasks.yml` gains `franka_1arm`: place_empty_cup, stack_bowls_two, click_bell and press_stapler,
  the tasks whose expert solved at least 2 of 3 surveyed seeds with every successful
  demonstration moving one arm. The survey (`docs/results/survey-franka-1arm.json`,
  `docs/survey.md`) ran without images on six tasks at 3 seeds each, cut by decision from every
  one-arm task at 20 seeds to finish the milestone sooner: place_a2b_left solved 1 of 3 and
  stack_blocks_two moved both arms in 2 of 3, so both are out. Stacking has no one-arm task, so
  stack_bowls_two, an arm-switching task, stands in for it, with a stated limit: a scene the
  survey did not see can still make its expert switch arms, and nothing refuses that
  demonstration. The plugin serves the table's suite instead of deriving a provisional one from
  each task's arms, drops `provisional` from `info()` and `catalogue()`, and names the stand-in in
  `info()["franka_1arm"]["stand_ins"]`. `units.CATALOGUE_SHA256` is the new table's and
  `units.DERIVATION` becomes `units/2`, so every unit changes; a test pins `franka_1arm`'s units
  beside `v1`'s (#83).
- (fix): a served policy can no longer void the units it is losing, or reach past its socket. All
  of a unit's calls to it share `run-unit --policy-budget-s` (300 s): the call it runs out in is
  cut short and the unit is void on the policy, where a policy answering every call just in time
  used to run its unit into the orchestrator's kill, void for both sides. `--unit-timeout-s` gives
  run-unit its caller's kill: no call runs within 30 s of it, and one cut short there while the
  policy is within its budget is void on the harness, written before the kill. `close` gets 5 s,
  not whatever its last call had. `--policy-log` is opened at the path given, never resolved, so a
  log the policy swapped for a link to a host file stays refused. An action's width and dtype are
  checked before it is copied, and rows past 4096 dropped, so a gigabyte of booleans no longer
  costs run-unit eight. What a server says is bounded before it reaches `result.json`: a
  200-character name, a failure's first 4000 and last 8192 characters. `result.json` records
  `policy_wall_s` and `policy_budget_s`, and after a GPU failure every process `nvidia-smi` saw
  holding GPU memory (`gpu_processes`) and `run_unit_pid`, since a policy sharing the GPU could
  have filled it (#85).
- (feat): the plugin and the benchmark code it runs are held to each other. `materialize` and
  `run-unit` take `--expect-source-sha256` and exit 1 before writing anything unless
  `robotwin_icil.source_sha256()` is that digest, and both results record theirs; the plugin passes
  its own, reports it in `info()` and voids a result other source wrote. `derive_units` refuses a
  task catalogue whose digest is not the pinned `units.CATALOGUE_SHA256`, and `robotwin-icil` is
  pinned to the plugin's version. `verify_prompt` also holds a prompt's scene config (`demo_clean`,
  `save_freq` 15, no head camera or override, `same_scene`), its cameras (head and wrists only) and
  its expert record (the unit's candidates tried in order up to the scene seed) to the unit. A
  commit is recorded only for the top of a git checkout, never for a repository a wheel sits in.
  `run_command` passes `policy_budget_s` and `unit_timeout_s` on and refuses a time limit run-unit
  would refuse; `--denoiser` carries `ROBOTWIN_ICIL_DENOISER`, which the orchestrator does not hand
  a benchmark subprocess (#86).
- (feat): the benchmark plugs into the competition orchestrator as its own distribution.
  `competition/` is `robotwin-icil-competition`, package `icil_benchmark_robotwin`, found through
  the `icil.benchmarks` entry point `robotwin` and never importing the orchestrator. Its pure half
  needs no simulator: `info` (robots and action widths, cameras, protocol, channel map, commits,
  command line), `catalogue` (suites, categories, each task's category and arms), `derive_units`
  (a sha256 counter over the seed material, pinned by a test and run on Python 3.10 and 3.12 in
  CI; each unit holds four candidate `scene_seeds` and its robot), `verify_prompt` (reads
  `prompt.npz` without unpickling and holds its task, seed, robot, arrays and scene digest to the
  unit) and `read_result` (with `void_cause`). `materialize_command` and `run_command` return the
  argv of `python -m robotwin_icil.cli` under `$ROBOTWIN_ICIL_PYTHON`. Units of the task table's
  `franka_1arm` suite run on two Franka arms, every other suite's on aloha-agilex (#83).
  `icil-orchestrator benchmarks check robotwin` reports it ok, noting the unpinned version and
  wheel (#86).
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
  malformed reply — is `PolicyUnreachable` and voids it with `void_cause` "policy", its `error`
  ending with the server log's tail when `--policy-log` names the log. Every other void of either
  command carries `void_cause` "harness", a field added to both results. Every policy is now reset
  with a public `EpisodeInfo` (robot, live action widths, seed) and closed once its unit is over
  (#85).
- (feat): a demonstration is built once and saved, and an episode is evaluated from the saved file.
  `robotwin-icil materialize --task T --scene-seed S [--scene-seed S2 ...] --out DIR` tries the
  candidate seeds in order — one scene built and one expert run each — and writes `prompt.npz` and
  `demonstration.mp4` for the first the expert solves, and `result.json` either way, with the
  chosen `scene_seed` and every attempt (seed, rejection, detail), which the prompt's `meta`
  records too; every candidate rejected is a void result with `void_cause` "harness", and a seed
  given twice is refused. It exits 0 whenever `result.json` was written, a rejection included, so
  the caller reads the reason instead of a log tail. `robotwin-icil run-unit --prompt
  DIR/prompt.npz --policy P --out DIR` rebuilds the scene from the prompt's privileged `meta`,
  refuses a meta whose digest is not its own fingerprint's, voids on scene drift with the
  mismatches, rolls the policy out and writes `result.json` (`success` and `steps` null exactly
  when `void`, the rebuilt scene's `live_scene_sha256` and its `scene_max_error`, the checkpoint
  and both commits) and `evaluation.mp4`. Both commands' `result.json` carry `success`, `void`,
  `steps` and `error`, the fields the orchestrator reads: every candidate rejected is a void
  materialize. In `run-unit` a policy at fault — raising from `reset` or `set_demonstration`, or a
  wrong-width or non-finite action — fails its unit and never voids it; void is kept for an
  unreadable, mistyped or tampered prompt, scene drift, a config RoboTwin refuses, any other
  harness fault while evaluating (traceback to stderr) and a GPU lost or full mid-rollout, which in
  `eval` now stops the run as it does in the expert. `run-unit` exits 1 only when the unit cannot
  start. Both commands clear their outputs before anything can fail, and `run-unit` refuses to
  write into its prompt's directory; an adapter's paths are resolved in its constructor or passed
  absolute, since the simulator runs from `vendor/RoboTwin`. `prompt.npz` holds
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
- (fix): `scripts/simwatch.py` reruns the two failures it missed. An attempt is stalled when its
  process tree averaged under `--min-cpu-rate` cores (default 0.25) over the last `--stall`
  seconds, so a camera-read hang that still trickles CPU is caught after the window instead of at
  `--max-wall`; an attempt that exits non-zero with `ErrorDeviceLost` or `VK_ERROR_DEVICE_LOST`
  (or any `--rerun-on` regex) anywhere in its own output is rerun, and any other exit ends the
  watch with its code. Every process of an attempt, workers it orphaned included, is killed and
  gone before its output is searched or the next attempt starts; a `--log` that is not a regular
  file is refused (#89).
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
