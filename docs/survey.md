# Expert survey

A task can only be scored if RoboTwin's own expert can produce its demonstration. Every seed the
expert fails is rejected and costs a full expert run, so a task whose expert rarely succeeds
dominates a run's wall-clock and its rejection statistics without telling us anything about the
model. `robotwin-icil survey` runs the expert alone — through the same `attempt` an episode's
generator uses — over a fixed seed stream per task, and reports how often it succeeds and why not.

```bash
robotwin-icil survey --suite v1 --seeds 20 --seed 0 --json docs/results/survey-v1-seed0.json
```

**No camera renders.** Everything the survey keeps is read from the robot's joints, so by default
it records its demonstrations without images: each frame reads the joint vector and the endpose
straight from the robot, from the accessors `get_obs` uses, and `get_obs` is never called.
RoboTwin ray-traces every camera at 32 samples per pixel for each frame, which is most of a seed's
cost, and SAPIEN's camera read is where a run hangs on a shared GPU. Nothing the survey measures
changes: at the pinned commit no expert (`play_once`, `check_success` or a helper they call) reads
an observation, an image or a camera, and the one side effect of `get_obs` an expert could see —
the light colours `crazy_random_light` draws from numpy's RNG — is kept (`robotwin.robot_state`
says how this was checked). `tests/sim/test_capture_without_images.py` runs the check on two
Frankas: with and without images, a seed ends the same way, in as many frames, with the same
joints and the same arms moved. The Franka expert does not always repeat itself, so a mismatch
is run again with images; a seed whose two rendered runs differ is only an expected failure, and
only while the run without images ends as one of them does, in a frame count between theirs.
`--images` renders every frame as `eval` does; `eval` always
renders, since the demonstration it hands a policy needs its images. The JSON says which: it is
`{"images": false, "tasks": [...]}`, and every task entry carries `images` too.

The expert's rate is a property of the task *and* the robot: `--embodiment franka-panda` surveys
the same seeds on two Franka arms, and the table's `robot` column and the JSON's `embodiment`
field name the robot each row was measured on. `--arms 1` keeps only the tasks whose expert uses
one arm, and the two combine — every one-arm task, on two Frankas:

```bash
robotwin-icil survey --suite all --embodiment franka-panda --arms 1 --seeds 20 --seed 0 \
  --json docs/results/survey-franka-1arm.json
```

## V1 survey

20 seeds per task from global seed 0, `demo_clean` config, aloha-agilex, on the reference machine
in [install.md](install.md) (RTX A6000, CuRobo 0.7.8). Raw results:
[`results/survey-v1-seed0.json`](results/survey-v1-seed0.json), a plain list of task entries
from before the file said whether it rendered. It did: that survey predates capturing without
images, so its *s / seed* includes rendering every camera at every frame and does not compare
with a survey run without `--images`.

| task | category | expert success | demo frames | s / seed | rejections |
| --- | --- | ---: | ---: | ---: | --- |
| place_object_basket | Pick and Place | **45%** (9/20) | 249 | 40.3 | missed 8, plan failed 3 |
| place_a2b_left | Pick and Place | 85% (17/20) | 149 | 20.9 | error 2, plan failed 1 |
| place_a2b_right | Pick and Place | 75% (15/20) | 149 | 19.7 | error 2, plan failed 3 |
| place_empty_cup | Pick and Place | 90% (18/20) | 173 | 25.0 | plan failed 2 |
| place_container_plate | Pick and Place | 75% (15/20) | 160 | 23.6 | error 3, plan failed 2 |
| stack_blocks_two | Stacking | 100% (20/20) | 317 | 89.1 | — |
| stack_bowls_two | Stacking | 75% (15/20) | 320 | 65.5 | missed 1, plan failed 4 |
| click_bell | Press / Push | 100% (20/20) | 78 | 18.4 | — |
| click_alarmclock | Press / Push | 80% (16/20) | 88 | 20.8 | missed 2, plan failed 2 |
| press_stapler | Press / Push | 95% (19/20) | 120 | 25.5 | error 1 |

**Timing caveat.** For most of the survey the GPU was shared with an unrelated training job
holding about 40 GiB, so the *s / seed* column is inflated by contention. Success rates are not
affected: each seed's scene and expert are deterministic.

**The rule.** A task stays in `v1` while its expert solves at least 70% of surveyed seeds.
place_object_basket, at 45%, needs about 2.2 expert runs (~90 s) per scored episode and has the
longest pick-and-place horizon (one arm places the object, the other lifts the basket); it is out.
`v1` is the other nine tasks, across Pick and Place, Stacking and Press / Push. The task stays in
the table, and remains the harness's smoke-test task.

**Reading the table.**

- *one-arm* (in `robotwin-icil survey`'s own table; the V1 survey above predates it): of the
  successful demonstrations, how many moved exactly one arm. Each qpos row is split at half its
  width into the left arm and the right — 7 + 7 on aloha-agilex, 8 + 8 on two Frankas — and an
  arm moved if any of its joints or its gripper left its first-frame value by more than 0.05 at
  any frame of the demonstration
  — 0.05 rad for a joint, 0.05 of full travel for the gripper, whose value is RoboTwin's
  normalised [0, 1] opening. The JSON carries the verdict per seed
  (`tasks[].seeds_detail[].arms_moved`, `["left"]`, `["right"]`, both, or `[]`) next to the
  number it was read from (`tasks[].seeds_detail[].displacement`, each arm's largest departure,
  e.g. `{"left": 0.012, "right": 1.43}`), so a borderline seed can be told from an idle one
  without re-running the expert; and
  as counts: `one_arm_demonstrations`, `two_arm_demonstrations` and `no_arm_demonstrations`,
  which partition the successes. A task belongs in a one-arm suite only when every one of its
  successful demonstrations moved exactly one arm.
- *missed*: the expert ran to completion and RoboTwin's success check said no.
- *plan failed*: motion planning reported failure (`plan_success` false).
- *error*: `play_once()` raised — typically "target_pose cannot be None", no feasible grasp.
- *s / seed* includes scene construction and RoboTwin's 2500-step stability check.
  stack_blocks_two is the slowest: its expert never fails, but it is long.
- Expected expert runs per scored episode are about 1 / success rate; none of these is a model
  failure, and none enters a score's denominator.
