# Expert survey

A task can only be scored if RoboTwin's own expert can produce its demonstration. Every seed the
expert fails is rejected and costs a full expert run, so a task whose expert rarely succeeds
dominates a run's wall-clock and its rejection statistics without telling us anything about the
model. `robotwin-icil survey` runs the expert alone — through the same `attempt` an episode's
generator uses — over a fixed seed stream per task, and reports how often it succeeds and why not.

The tables below are historical measurements on a subset of tasks. The CLI surveys all cataloged
tasks by default, or a single task with `--task NAME`:

```bash
robotwin-icil survey --seeds 20 --seed 0 --json runs/survey-all.json
```

**No camera renders.** Everything the survey keeps is read from the robot's joints, so by default
it records its demonstrations without images: each frame reads the joint vector and the endpose
straight from the robot, from the accessors `get_obs` uses, and `get_obs` is never called.
RoboTwin ray-traces every camera at 32 samples per pixel for each frame, which we expect to be
most of a seed's cost (not yet timed), and SAPIEN's camera read is where a run hangs on a shared
GPU. What an expert reads does not
change: at the pinned commit no expert (`play_once`, `check_success` or a helper they call) reads
an observation, an image or a camera, and the one side effect of `get_obs` an expert could see —
the light colours `crazy_random_light` draws from numpy's RNG — is kept (`robotwin.robot_state`
says how this was checked). `tests/sim/test_capture_without_images.py` runs the check on two
Frankas. RoboTwin's expert does not repeat itself exactly between runs of one seed, so each seed
runs twice with images and once without, and the run without images must stay within what the
two rendered runs span: the same outcome unless they already disagree, a frame count within the
larger of their spread and two frames, the same arms moved and, when all three have as many
frames, joints and endposes row by row no further from a rendered run than the two are from each
other, plus 0.05.
One thing can still differ, on a GPU short of memory. Rendering holds GPU memory, and the batch
planner RoboTwin picks a grasp with (`CuroboPlanner.plan_batch`, `envs/robot/planner.py`) catches
every exception, a CUDA out-of-memory error included, and reports a failed plan. There a rendered
run can reject, as *plan failed* or *error*, a seed a run without images keeps, so a survey meant
to predict `eval`'s rejections on a shared GPU should pass `--images`. How much memory rendering
takes has not been measured. `--images` renders every frame as `eval` does; `eval` always
renders, since the demonstration it hands a policy needs its images. The JSON says which: it is
`{"images": false, "tasks": [...]}`, and every task entry carries `images` too.

The expert's rate is a property of the task *and* the robot: `--embodiment franka-panda` surveys
the same seeds on two Franka arms, and the table's `robot` column and the JSON's `embodiment`
field name the robot each row was measured on. `--arms 1` keeps only the tasks whose expert uses
one arm, and the two combine — every one-arm task, on two Frankas:

```bash
robotwin-icil survey --embodiment franka-panda --arms 1 --seeds 20 --seed 0 \
  --json runs/survey-franka.json
```

The Franka survey recorded below was smaller: see *Franka Panda survey*.

**The Franka survey ran without images.** It was gated on the sim test above passing, and the
test, in its earlier form, compared one rendered run with one run without images exactly, seed by
seed. That could not tell rendering apart from the expert's own variation: two rendered runs of
one seed had already differed by up to 0.13 rad in their qpos rows, and identical runs in #81
recorded 77 and 78 frames. The test's one completed run, on a GPU another simulator process
shared, failed only on click_bell seed 0, with 47 frames rendered against 48 without — within that
variation. Two independent audits of the pinned source had found no expert path that reads a
camera or an observation (`robotwin.robot_state` says how), so the gate was skipped rather than
held on a comparison that could not decide, and the test was reworked to compare against two
rendered runs; that form has not run on the simulator yet. The Franka survey's rejections carry
the GPU-memory caveat above: it does not predict every seed `eval` would reject on a shared GPU.

## Aloha-AgileX survey

20 seeds per task from global seed 0, `demo_clean` config, aloha-agilex, on the reference machine
in [install.md](install.md) (RTX A6000, CuRobo 0.7.8). Raw results:
[`results/survey-aloha-seed0.json`](results/survey-aloha-seed0.json), a plain list of task entries
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
holding about 40 GiB, so the *s / seed* column is inflated by contention. Contention alone does
not change success rates: each seed's scene and expert are deterministic. Running short of GPU
memory could, and would read as *plan failed* or *error* (see *No camera renders* above).

**Cost.** place_object_basket, at 45%, needs about 2.2 expert runs (~90 s) per scored episode and
has the longest pick-and-place horizon (one arm places the object, the other lifts the basket).
Low expert success raises generation cost and rejection counts, never a model's denominator.

**Reading the table.**

- *one-arm* (in `robotwin-icil survey`'s own table; the Aloha-AgileX survey above predates it): of the
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
  which partition the successes.
- *missed*: the expert ran to completion and RoboTwin's success check said no.
- *plan failed*: motion planning reported failure (`plan_success` false).
- *error*: `play_once()` raised — typically "target_pose cannot be None", no feasible grasp.
- *s / seed* includes scene construction and RoboTwin's 2500-step stability check.
  stack_blocks_two is the slowest: its expert never fails, but it is long.
- Expected expert runs per scored episode are about 1 / success rate; none of these is a model
  failure, and none enters a score's denominator.

## Franka Panda survey

This survey ran on 2026-09-14 at bfbc595: RoboTwin's expert alone on `--embodiment franka-panda`
(two Franka arms 0.8 m apart), without images (see *The Franka survey ran without images* above),
3 seeds per task from global seed 0 (scene seeds 1826701614, 1367864806 and 1097657231), on six
tasks, two each from Pick and Place, Stacking and Press / Push. Raw results:
[`results/survey-franka-seed0.json`](results/survey-franka-seed0.json).

**Reduced scope.** #83 asked for every one-arm task at 20 seeds: 26 × 20 expert runs, 3 to 13 GPU
hours. The survey was cut to 6 tasks × 3 seeds by decision. Every rate below is therefore a
three-seed estimate, and no task outside these six was measured.

| task | category | arms | successes | one-arm demonstrations | rejections | s / seed |
| --- | --- | --- | ---: | ---: | --- | ---: |
| place_a2b_left | Pick and Place | 1 | **1/3** | 1 of 1 | missed 1, plan failed 1 | 13.3 |
| place_empty_cup | Pick and Place | 1 | 3/3 | 3 of 3 | — | 11.9 |
| stack_blocks_two | Stacking | switching | 3/3 | **1 of 3** | — | 14.9 |
| stack_bowls_two | Stacking | switching | 2/3 | 2 of 2 | missed 1 | 13.7 |
| click_bell | Press / Push | 1 | 3/3 | 3 of 3 | — | 11.0 |
| press_stapler | Press / Push | 1 | 3/3 | 3 of 3 | — | 11.1 |

*arms* is the task's entry in `tasks.yml`, a static read of its expert; *one-arm demonstrations*
is how many of the successes moved exactly one arm, measured from the joints (see *Reading the
table* above). stack_blocks_two's expert moved both arms on seeds 1826701614 and 1367864806. The
first seed of every task took 31 to 35 s and the other two 1 to 5 s, so *s / seed* is mostly the
first seed's; it does not compare with the Aloha-AgileX survey's, which rendered every camera.

How the competition plugin uses these measurements is documented in
[`competition/README.md`](../competition/README.md).
