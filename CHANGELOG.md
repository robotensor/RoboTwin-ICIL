# Changelog

## Unreleased

### Competition plugin

- (feat): `competition/` is a third distribution, `robotwin-icil-competition` (package
  `icil_benchmark_robotwin`), that makes this benchmark a plugin of the ICIL competition. It is
  found through the `icilval.benchmarks` entry point group, depends on the core, and never imports
  `icilval` - a benchmark that stands on its own cannot depend on the competition that scores it.
- (feat): the plugin splits in two. `info`, `catalogue`, `derive_units`, `verify_prompt` and
  `read_result` run with no simulator, no assets and no GPU, because the validator host, CI and a
  third party all need them; everything that does need a simulator is returned as an argv, so the
  orchestrator never imports SAPIEN and the simulator side can run in another image or on another
  host. A test asserts the argv the plugin builds is the argv the CLI parses.
- (feat): a unit here is a task and a scene seed, derived from a sha256 counter rather than
  numpy's RNG so a third party holding the published record can reproduce it. A prompt is written
  in the array convention the orchestrator already reads, grouped by channel - no new file format,
  and no change to `Demonstration`: what a policy may see of a demonstration stays the
  orchestrator's decision, which is what makes it hold for every benchmark.
- (feat): only the `video_only` view is offered. Same Scene makes the sensorimotor view
  degenerate, since replaying the demonstration's own actions solves the episode.
- (docs): `docs/competition.md`, and the CLAUDE.md amendment - the evaluator still uses no dataset
  and trains nothing; a competition built on this benchmark may materialise and publish what it
  runs, and that lives in `competition/`. A test checks neither the core nor the plugin imports a
  network client, rather than trusting the rule.
- (feat): `run-unit --policy-address` runs a unit against a policy the orchestrator is serving,
  instead of saying it cannot. That is the only way a model whose pins conflict with RoboTwin's
  runs at all - the BPP adapter refuses to be built where `sapien` is importable, because seeding
  torch's global RNG in the simulator's process would make its noise a function of the scene seed.
  The transport comes from the branch that built it: `protocol.py`, `serve.py` and `remote.py`,
  with the wire protocol unchanged - `send_bytes`/`recv_bytes`, a JSON header and raw bool,
  integer and float arrays, never a pickle - plus the `ICILPolicy` hooks the served path calls
  (`seed`, `episode_info`, `close`, `environment`), `parse_policy_arg`/`format_policy_arg`, and
  `Frame`/`Demonstration`'s `time_s`, `times()`, `gripper_joints`, `endposes()` and
  `ee_actions()`, which `competition/prompt.py` already called. A run closes its policy exactly
  once, so no policy server outlives its client. `--policy-address` and `--policy` are
  alternatives: a unit runs one policy, and the record has to say which (#74).
- (feat): a prompt carries the **measured** gripper joints, one `(T, 2, J)` array in the `proprio`
  channel. RoboTwin's gripper value in `qpos` and `endpose` is the command, which reads closed
  while the fingers rest on an object; where the fingers are is proprioception, and it is what a
  model trained on LIBERO's `gripper_states` reads. Without it every prompt round-tripped to
  `gripper_joints=None` and the BPP adapter refused to build a prompt from any of them. A prompt
  written before the array does not have it and rebuilds without one, so the schema does not move
  (#74).
- (fix): the `competition` CI job installs pytest, which it ran without installing (#74).
## Unreleased

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
