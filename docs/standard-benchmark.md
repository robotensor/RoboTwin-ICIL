# robotwin-icil-standard

The standard profile is RoboTwin 2.0's own policy evaluation with one change: the policy is given
a demonstration. Everything RoboTwin fixes stays fixed — every task, the scene seeds, the Easy and
Hard settings, the per-task step limits and `check_success()` — and the expert trajectory RoboTwin
already runs on each scored scene is kept and handed to the frozen policy as its single
demonstration, instead of being thrown away.

## Status

Implemented and covered by simulator-free tests. Not yet run in simulation: there is no
replay-oracle ceiling, expert survey or learned-policy result for this profile yet.

The training dataset is published and verified at revision `f0332a22c7e3a7b537f9eb4eba1ba70432aa1785`; see
[the release receipt](results/dataset-release-preparation.json).

## Three workflows

| Workflow | Commands | Selection | Result |
| --- | --- | --- | --- |
| ICIL duel | `materialize`, `run-unit`, `competition/` plugin | derived units | same prompt and exact initial scene for both models |
| Exploratory evaluation | `eval`, `report`, `survey` | all tasks or `--task`, optional `--arms`, robot, `--seed-stream` | per-task/skill/overall rates; no standard-profile claim |
| Robotensor standard | `standard-eval`, `standard-report` | all 50 tasks, Aloha, clean and randomized | mean task success per setting, with denominators and diagnostics |

All three use one expert demonstration, a frozen policy, a fresh exact-scene reset, fingerprint
verification and RoboTwin's unchanged binary success check.

## The protocol, beside RoboTwin's

RoboTwin's evaluation (`vendor/RoboTwin/scripts/eval_policy_xpolicylab.py`, `eval_remote_policy`)
and this profile, side by side:

| | RoboTwin 2.0 | robotwin-icil-standard |
| --- | --- | --- |
| Tasks | all 50 | all 50 |
| Settings | `demo_clean` (Easy), `demo_randomized` (Hard) | the same, scored separately as `clean` and `randomized` |
| Robot | aloha-agilex | aloha-agilex, `save_freq` 15 |
| Episodes | 100 per task | 100 per task |
| Scene seeds | from `100000 * (1 + seed)`, +1 per seed tried, seed 0 | the same stream, seed group 0 |
| Expert check | runs `play_once`; a seed it fails, or that is unstable, is skipped | the same; its successful trajectory is recorded |
| Rollout scene | `setup_demo` again with the same seed | the same, and the initial-state fingerprint must match the expert's |
| Policy input | observations and a language instruction | observations and the expert's demonstration; language is the neutral `Follow the demonstrated behavior.` |
| Episode end | `eval_success` or the task's `step_lim` | the same |
| Score | successes / 100 per task, averaged over tasks | the same, per setting, plus by skill category |

Two differences are deliberate. A scene whose rebuilt fingerprint differs from the expert's is
recorded as `invalid` and not scored, where RoboTwin would score it; and each episode may try at
most 50 seeds (`standard.MAX_EXPERT_ATTEMPTS`), where RoboTwin has no per-episode cap. The seed
stream continues after a capped episode exactly as it would after a successful one.

The scenes are RoboTwin's evaluation scenes, so a model's score here and its RoboTwin score are
measured on the same scenes when the expert behaves identically. RoboTwin's expert is not bitwise
repeatable across machines, so a seed it solves on one GPU can be skipped on another; the run
records every seed tried and why each was rejected.

## Scoring

`standard.json` holds the plan: profile, policy description, benchmark source digest, settings,
seed groups and episodes per task. `<setting>/seed_<seed>/` is an ordinary run directory: its
manifest and `episodes.jsonl`. `standard-report.json` is rewritten after each completed setting and
seed group, and `standard-report` aggregates a partial run without a simulator.

Per setting:

- `success_rate`: the arithmetic mean of the 50 tasks' success fractions (valid episodes only),
  available once every task has a valid episode;
- `by_category`: the same mean over each skill category's tasks;
- `by_task`: planned, recorded, scored and successful episodes per task;
- `diagnostics`: expert rejections by reason, invalid episodes and scene errors;
- `official_score`: `success_rate`, only when every planned episode is scored at seed group 0 with
  100 episodes per task; otherwise null.

A rejected or invalid episode never counts as a model failure, and it prevents an official score:
fix the harness and rerun in a fresh directory. `standard-report` refuses mixed policies, configs,
source digests, simulator revisions, seed streams, duplicate episodes and wrong task assignments.
Additional seed groups (`--seed 0 --seed 1 --seed 2`) are recorded and reported, but only seed
group 0 is the official budget, as in RoboTwin.

## Commands

```bash
# Full evaluation: 50 tasks x 100 episodes x {clean, randomized}; needs the simulator installation.
robotwin-icil standard-eval --policy your_adapter:Policy --policy-arg checkpoint=/absolute/model \
  --run-dir runs/standard/model
robotwin-icil standard-report runs/standard/model
robotwin-icil standard-report runs/standard/model --json

# One setting only.
robotwin-icil standard-eval --policy your_adapter:Policy --setting clean --run-dir runs/standard/clean

# Interface smoke test, explicitly provisional.
robotwin-icil standard-eval --policy replay --setting clean --episodes-per-task 1 \
  --run-dir runs/standard/smoke

# The same seed stream outside the profile, for a single task.
robotwin-icil eval --policy replay --task click_bell --seed-stream robotwin --episodes 5 \
  --run-dir runs/click-bell-robotwin
```

A run resumes when the same command is repeated. With RoboTwin's seed stream, a task's episodes
run in order and each continues the stream after the seeds its predecessors tried, so a resumed run
tries exactly the seeds an uninterrupted one would.

## Training data

[robotensor/robotwin-icil-aloha-clean](https://huggingface.co/datasets/robotensor/robotwin-icil-aloha-clean)
republishes RoboTwin's 50 × 50 clean Aloha trajectories with three aligned camera videos. All
2,500 trajectories are training data: as in RoboTwin, the benchmark scores in simulation, not on a
stored split. The trajectories were collected on RoboTwin's collection seeds; evaluation uses its
evaluation seeds, from 100000. Training on other data is allowed and must be disclosed.

Format: `robotwin-icil-hdf5-multiview-v1`. Source: `TianxingChen/RoboTwin2.0` at
`981c92aa34d8f94d4cff47e0d5bc2f7d4e0af042`. Only its 50 `dataset/<task>/demo_clean.zip` archives
are copied. No simulation replay is needed to recover videos: source HDF5 JPEG streams already
contain rendered simulation frames.

```text
README.md  LICENSE  manifest.json  schema.json  provenance.json  checksums.json
episodes.jsonl                one row per trajectory; the Hub viewer's rows
episodes/<task>/episode_0000000/
  trajectory.hdf5             original source bytes, unchanged
  head_camera.mp4             cam_head
  left_camera.mp4             cam_left_wrist
  right_camera.mp4            cam_right_wrist
  instruction.json            original metadata, not policy language
source_metadata/<task>/{scene_info.json,seed.txt}
```

Each video contains exactly N RGB frames aligned to the N original transition rows. It is H.264
yuv420p, 320x240, CRF 18, 30 presentation fps. Original HDF5 JPEGs remain authoritative training
images; MP4 is a lossy demonstration/inspection convenience. HDF5's extra third view and calibration
are retained as source data, not included in public policy camera inputs.

| Model field | Shape/type | Meaning |
| --- | --- | --- |
| `rgb[head_camera/left_camera/right_camera]` | `(N,240,320,3)` uint8 | standard RGB JPEG decoding |
| `qpos` | `(N,14)` float32 | left six joints+gripper, right six joints+gripper; drive targets, not encoders |
| `actions` | `(N,14)` float32 | absolute next-row joint/gripper commands, already shifted upstream |
| `endpose`, `action_endpose` | `(N,16)` float32 | per arm xyz, quaternion wxyz, gripper |
| `simulated_frame_times` | `None` | source physical timestamps are unavailable |
| `source_frequency_value` | integer | retained source value, not verified physical Hz |

Do not reverse JPEG RGB channels. Do not shift actions again, infer exact timestamps from video,
or manufacture terminal wrist images. Archived trajectories lack a verified initial fingerprint
and are not valid evaluation `prompt.npz` files. Live evaluation prompts have T frames, T-1
actions, float64 state/actions and actual simulated times; adapters explicitly cast as needed.

```python
from pathlib import Path
from robotwin_icil.dataset import ReleasedDataset

data = ReleasedDataset(Path("/absolute/path/to/dataset"))
sample = data.training_sample(index=0, row=10)
inputs = sample["inputs"]
target = sample["target_actions"]  # (1,14), supervised label only
context = inputs["demonstration"]  # complete source trajectory
query = inputs["observation"]  # rows 0..10 only, no target/future actions
assert query["qpos"].shape == (11, 14)
assert "actions" not in query
assert "task" not in context and "scene_seed" not in context
```

`training_sample` is a same-trajectory training proxy: future demonstration observations/actions
are intentionally available in context, as in Same Scene evaluation. `training_pair(context_index,
query_index, row)` uses two distinct episodes of the same task. Episode metadata stays available in
`data.episodes` for sampling only; do not feed it to the model.

The package supplies a loader, not a trainer or a trained checkpoint. Train in your own project
and implement the `ICILPolicy` adapter. Freeze parameters before evaluation; KV/history updates are
allowed, gradient steps are not. `robotwin_icil.policies.nearest:NearestPolicy` consumes the three
public cameras and qpos, matches demonstration transitions and emits absolute qpos actions. It is
untrained: an interface baseline, not evidence of learned ICIL performance.

```bash
pip install -e ".[dataset]"
robotwin-icil dataset download --revision f0332a22c7e3a7b537f9eb4eba1ba70432aa1785 --out /absolute/dataset
robotwin-icil dataset verify /absolute/dataset --videos
robotwin-icil dataset sample /absolute/dataset --episode-index 0 --row 10

# Rebuild from the pinned source, and publish (needs write access to the robotensor namespace).
robotwin-icil dataset build --out /absolute/dataset --cache-dir /absolute/source-cache --workers 4
robotwin-icil dataset upload /absolute/dataset --workers 4
```

Build resumes completed task conversion only after checking archive/output hashes. Every root
metadata file and every trajectory, video, instruction and source metadata file is SHA256
inventoried. Verification checks every file and all HDF5 rows; `--videos` also recounts every clip.
Upload verifies locally first, uses Hub resumable large-folder transfer (excluding `.build`,
`.cache` and partial files), removes Hub files the release no longer inventories, and returns the
immutable commit SHA. Source ZIPs and credentials are never uploaded.

MIT source attribution and the original RoboTwin copyright notice are retained. Simulator/object
assets are not copied; their licenses and installation remain upstream. See
[the dataset format audit](dataset-plan.md) for source-specific evidence and limitations.

## Reporting a result

Publish per setting `official_score`, `by_category` and `by_task`, with the model/checkpoint
digest, architecture/adapter, training data (this release or others) and training recipe, and the
benchmark source digest and RoboTwin commit from the run manifests.
