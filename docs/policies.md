# Writing a policy adapter

A policy is anything that can be reset, handed **one** demonstration, and then asked for actions
one observation at a time. The benchmark core knows nothing about any particular model; an adapter
converts between the benchmark's model-independent types and whatever the model consumes.

```bash
robotwin-icil eval --policy mypkg.adapters:MyPolicy --suite v1 --episodes 500 --seed 42 --run-dir runs/mine
```

`--policy` takes a built-in name (`replay`, `dummy`) or any importable `module:Class` that
subclasses `ICILPolicy`. Adapters live outside `robotwin_icil`.

## The lifecycle

For every episode, in this order and no other:

```text
policy.reset()                            # forget everything from the previous episode
policy.set_demonstration(demo)            # exactly once
loop:
    actions = policy.act(obs, action_dims)  # (k, action_dim); all k run before the next observation
```

`action_dims` is the live robot's action width per action type, read off its arms every rollout —
`{"qpos": 14, "ee": 16}` on aloha-agilex, `{"qpos": 16, "ee": 16}` on two Franka arms. The base
class checks `_act`'s result against it; an adapter never has to pass it anywhere.

`ICILPolicy` enforces this. Acting before a demonstration, or receiving a second one without a
reset, raises `PolicyError` and stops the run — an adapter breaking the protocol is a bug, not a
stream of zero scores.

## A minimal adapter

```python
from pathlib import Path

import torch

from robotwin_icil.policy import ICILPolicy


class MyPolicy(ICILPolicy):
    name = "my_icil_model"
    action_type = "qpos"  # or "ee"

    def __init__(self, checkpoint: str = "checkpoints/model.pt"):
        super().__init__()
        # Absolute now: the harness then runs from vendor/RoboTwin, where a relative path opened
        # later (in reset, say) would be looked up.
        self.checkpoint = str(Path(checkpoint).resolve())
        self.model = torch.load(self.checkpoint).eval().requires_grad_(False)

    def _reset(self):
        self.context = None  # clear KV cache, history, recurrent state

    def _set_demonstration(self, demo):
        with torch.inference_mode():
            self.context = self.model.encode(
                images=demo.images("head_camera"),  # (T, h, w, 3) uint8
                states=demo.qpos(),  # (T, qpos_dim)
                actions=demo.actions(),  # (T-1, qpos_dim)
            )

    def _act(self, obs):
        with torch.inference_mode():  # returns (k, qpos_dim)
            return self.model.act(self.context, obs.images["head_camera"], obs.qpos)

    def describe(self):
        return {**super().describe(), "model": self.name, "checkpoint": self.checkpoint}
```

`describe()` goes into the run manifest; report your model and checkpoint there.

## What the policy receives

**The demonstration** (`robotwin_icil.demo.Demonstration`) — one successful RoboTwin expert
trajectory from the very scene the rollout will start in:

| | |
| --- | --- |
| `frames` | per frame: `images` (camera name -> `(h, w, 3)` uint8 rgb), `qpos` `(qpos_dim,)`, `endpose`, `time_s` |
| `frequency` | nominal frames per second of the recording: the spacing of frames within one motion primitive |
| `cameras` | the camera names present in every frame |
| `qpos_dim` | the width of `qpos`, the same in every frame: the robot's |
| `qpos()` | `(T, qpos_dim)` robot state over the demonstration |
| `actions()` | `(T-1, qpos_dim)` the position target of each transition — the next frame's `qpos` |
| `images(camera)` | `(T, h, w, 3)` from one camera |
| `times()` | `(T,)` simulated seconds since the expert started, per frame — real, and uneven: each RoboTwin motion primitive records a frame before its first physics step, one after that step and after every `save_freq`-th step from it, and one after its last |

`endpose` is RoboTwin's dict per frame: `left_endpose` and `right_endpose` (`[x, y, z, qw, qx, qy,
qz]`, each arm's end-effector pose as RoboTwin's `get_arm_pose` reports it) and `left_gripper`,
`right_gripper`. On disk (`prompt.npz`, written by `robotwin-icil
materialize` and read by `run-unit`) it is one 16-wide row per frame, left arm then right, pose then
gripper — `robotwin_icil.prompt.flatten_endpose` — which is also the layout of an `ee` action.

**Each observation** (`robotwin_icil.policy.Observation`) has the same modalities as a frame —
`images`, `qpos`, `endpose` — plus `step` and `instruction`.

`qpos` is RoboTwin's joint vector: the left arm's joints then its gripper, then the right's. Its
width is the robot's, chosen per run with `--embodiment`: aloha-agilex (what every shipped task
config names) has six joints per arm, 14 in all; franka-panda is two seven-joint arms, 16.
Grippers run from 0 (closed) to 1
(open). Every episode record and the run manifest name the robot.

## What the policy never receives

The scene seed; the task name or RoboTwin's task instruction; the success condition; target
object or destination identities; the `info` dict from the expert's `play_once()`; ground-truth
object state; RoboTwin actor handles; planner internals. None of these are fields of
`Demonstration` or `Observation`, and a test pins that.

A model that requires language gets `obs.instruction == "Follow the demonstrated behavior."`, so
that what is measured is the demonstration and not the prompt.

## Actions

Return a `(k, action_dim)` array (a single `(action_dim,)` action is accepted). All `k` actions are
executed through RoboTwin's `take_action` before the next observation, so chunking policies work
unchanged; the episode ends as soon as RoboTwin latches success or the task's step limit is hit.

| `action_type` | `action_dim` | layout |
| --- | --- | --- |
| `qpos` | `qpos_dim` (14 on aloha-agilex, 16 on franka-panda) | the same joint vector as `qpos` above: absolute position targets |
| `ee` | 16 on any robot | per arm: end-effector pose (7) then gripper, as RoboTwin's `take_action(action_type='ee')` reads it |

The widths are read off the live robot before every rollout, not assumed. Actions of the wrong
width or containing non-finite values raise `PolicyError`.

## The policy is frozen

No `backward()`, no optimizer, no parameter writes, anywhere, during a benchmark run. The model
adapts only through the demonstration it is handed. Inference-time state — a KV cache, observation
or action history, a recurrent state — is fine, and `_reset()` must clear all of it.

## Validating an adapter

Run the replay oracle first, on the same suite and seed:

```bash
robotwin-icil eval --policy replay --suite v1 --episodes 20 --seed 42 --run-dir runs/replay --video
```

`ReplayPolicy` plays the demonstration's actions back verbatim. Under Same Scene it is the
harness's own upper bound, so it tells you what a perfect imitator scores on your machine. Then run
your adapter with `--video` and compare `demonstration.mp4` with `evaluation_same_scene.mp4` in a
few episode directories before trusting any number.

To iterate on one scene without regenerating its demonstration every time, save it once and
evaluate from the file — the competition runs adapters this way:

```bash
robotwin-icil materialize --task click_bell --scene-seed 42 --out runs/unit/prompt
robotwin-icil run-unit --prompt runs/unit/prompt/prompt.npz \
    --policy mypkg.adapters:MyPolicy --policy-arg checkpoint="$PWD/checkpoints/model.pt" --out runs/unit/run
```

`--policy-arg key=value` reaches the adapter's constructor as a string. The constructor runs in
the directory you ran the command from, but the simulator then moves the process into
`vendor/RoboTwin`: pass absolute paths, or resolve them in `__init__` as the adapter above does,
never in `reset` or later. `run-unit` writes
`result.json` (`success`, `void`, `steps`, `error`, ...) and `evaluation.mp4` into a directory of
its own; the adapter is handed the `Demonstration` read from the file and nothing of the prompt's
`meta`. An adapter at fault — raising from `reset` or `set_demonstration`, or returning an action
of the wrong width or a non-finite one — fails the unit, with the reason in `detail`; it is never
void, which is kept for what the harness could not give the policy (an unreadable prompt, a scene
that drifted, a GPU that failed, a fault of the harness's own). `eval` raises instead, so the bug
surfaces.
