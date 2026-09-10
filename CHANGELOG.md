# Changelog

## Unreleased

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
