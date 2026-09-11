# Changelog

## Unreleased

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
