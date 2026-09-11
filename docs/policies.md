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
policy.reset()                      # forget everything from the previous episode
policy.set_demonstration(demo)      # exactly once
loop:
    actions = policy.act(obs)       # (k, action_dim); all k run before the next observation
```

`ICILPolicy` enforces this. Acting before a demonstration, or receiving a second one without a
reset, raises `PolicyError` and stops the run — an adapter breaking the protocol is a bug, not a
stream of zero scores.

## A minimal adapter

```python
import torch

from robotwin_icil.policy import ICILPolicy


class MyPolicy(ICILPolicy):
    name = "my_icil_model"
    action_type = "qpos"  # or "ee"

    def __init__(self, checkpoint: str = "checkpoints/model.pt"):
        super().__init__()
        self.checkpoint = checkpoint
        self.model = torch.load(checkpoint).eval().requires_grad_(False)

    def _reset(self):
        self.context = None  # clear KV cache, history, recurrent state

    def _set_demonstration(self, demo):
        with torch.inference_mode():
            self.context = self.model.encode(
                images=demo.images("head_camera"),  # (T, h, w, 3) uint8
                states=demo.qpos(),  # (T, 14)
                actions=demo.actions(),  # (T-1, 14)
            )

    def _act(self, obs):
        with torch.inference_mode():
            return self.model.act(self.context, obs.images["head_camera"], obs.qpos)  # (k, 14)

    def describe(self):
        return {**super().describe(), "model": self.name, "checkpoint": self.checkpoint}
```

`describe()` goes into the run manifest; report your model and checkpoint there.

## What the policy receives

**The demonstration** (`robotwin_icil.demo.Demonstration`) — one successful RoboTwin expert
trajectory from the very scene the rollout will start in:

| | |
| --- | --- |
| `frames` | per frame: `images` (camera name -> `(h, w, 3)` uint8 rgb), `qpos` (14,), `endpose` |
| `frequency` | frames per second of the recording |
| `cameras` | the camera names present in every frame |
| `qpos()` | `(T, 14)` robot state over the demonstration |
| `actions()` | `(T-1, 14)` the position target of each transition — the next frame's `qpos` |
| `images(camera)` | `(T, h, w, 3)` from one camera |

**Each observation** (`robotwin_icil.policy.Observation`) has the same modalities as a frame —
`images`, `qpos`, `endpose` — plus `step` and `instruction`.

The 14-dim `qpos` is RoboTwin's bimanual joint vector: left arm joints (6), left gripper, right arm
joints (6), right gripper. Grippers run from 0 (closed) to 1 (open).

## Cameras

Under the default camera profile, `stock`, every frame and every observation carries four rgb
cameras, keyed by name — `demo.cameras` lists them:

| camera | where | image |
| --- | --- | --- |
| `head_camera` | over the robot's shoulders, 53° down; the arms enter from the bottom | 320x240, 37° |
| `front_camera` | 0.11 m above the table, looking at the wall | 320x240, 37° |
| `left_camera` | on the left gripper, looking along it | 320x240, 37° |
| `right_camera` | on the right gripper, looking along it | 320x240, 37° |

A camera profile, `--camera-profile` on `eval` and `survey`, changes which static cameras RoboTwin
renders. Profiles are data in
[`src/robotwin_icil/cameras.yml`](../src/robotwin_icil/cameras.yml):

| profile | static cameras | what changes |
| --- | --- | --- |
| `stock` | `head_camera`, `front_camera` | nothing: RoboTwin's own cameras |
| `far_side` | `head_camera`, `far_side_camera` | `far_side_camera` takes `front_camera`'s slot: across the table at (0, 0.36, 1.20), 38.9° down, 320x180 with a 45° vertical field; the arms enter from the top, and image left is the robot's right |

Profiles only **replace** a camera, in its slot. RoboTwin builds every static camera with
`create_camera`, which draws from numpy's global RNG after the scene seed is set and before any
object is placed, so a camera more or fewer would move every object of every seed — and the Same
Scene check could not notice, since both builds of an episode would carry it. A profile that adds
or removes a camera, touches `head_camera`, toggles `camera.collect_head_camera` or leaves two
cameras with one name is refused, with no override. Every profile therefore builds the same
scene from the same seed; only the images differ. The run manifest records the profile and the
static cameras, and a run directory does not resume under another profile.

A new or changed profile is checked by eye first — one PNG per camera of one scene:

```bash
robotwin-icil cameras --profile far_side --task click_bell --seed 0 --out runs/cameras
```

With `--video`, the clips show the profile's `video_camera` (`head_camera` unless the profile
says otherwise).

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
| `qpos` | 14 | the same joint vector as `qpos` above: absolute position targets |
| `ee` | 16 | per arm: end-effector pose (7) then gripper, as RoboTwin's `take_action(action_type='ee')` reads it |

Actions of the wrong width or containing non-finite values raise `PolicyError`.

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
