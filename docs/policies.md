# Writing a policy adapter

A policy is anything that can be reset, handed **one** demonstration, and then asked for actions
one observation at a time. The benchmark core knows nothing about any particular model; an adapter
converts between the benchmark's model-independent types and whatever the model consumes.

```bash
robotwin-icil eval --policy mypkg.adapters:MyPolicy --suite v1 --episodes 500 --seed 42 --run-dir runs/mine
```

`--policy` takes a built-in name (`replay`, `replay_ee`, `dummy`) or any importable
`module:Class` that subclasses `ICILPolicy`. Adapters live outside `robotwin_icil`.

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
| `frames` | per frame: `images` (camera name -> `(h, w, 3)` uint8 rgb), `qpos` (14,), `endpose`, `time_s`, `gripper_joints` |
| `frequency` | nominal frames per second: the spacing of frames within one motion primitive |
| `cameras` | the camera names present in every frame |
| `times()` | `(T,)` simulated seconds of each frame since the expert started; see [Time](#time) |
| `qpos()` | `(T, 14)` robot state over the demonstration |
| `actions()` | `(T-1, 14)` the position target of each transition — the next frame's `qpos` |
| `endposes()` | `(T, 16)` end-effector state in `take_action('ee')` layout; see [The `ee` path](#the-ee-path) |
| `ee_actions()` | `(T-1, 16)` the end-effector target of each transition — the next frame's `endposes()` row |
| `arms_moved(threshold_m=0.02)` | the arms, of `"left"` and `"right"`, whose flange path is longer than the threshold (provisional) |
| `images(camera)` | `(T, h, w, 3)` from one camera |

**Each observation** (`robotwin_icil.policy.Observation`) has the same modalities as a frame —
`images`, `qpos`, `endpose`, `time_s`, `gripper_joints` — plus `step` and `instruction`.

The 14-dim `qpos` is RoboTwin's bimanual joint vector: left arm joints (6), left gripper, right arm
joints (6), right gripper. Grippers run from 0 (closed) to 1 (open).

`endpose` is RoboTwin's own dict: `left_endpose` and `right_endpose`, each `[x, y, z, qw, qx, qy,
qz]` in the world frame, and `left_gripper` and `right_gripper`. The pose is the flange's
(`fl_link6` / `fr_link6` on aloha-agilex), not the tool centre's: the tool centre point is 0.12 m
further along the pose's own +x axis. Quaternions are scalar-first (wxyz).

The gripper values in `qpos` and `endpose` are the **command**, not a measurement: a gripper
closed on an object reads 0 however wide the object holds its fingers. `gripper_joints` is the
measurement — `"left"` and `"right"` map to the positions, in metres, of that arm's gripper
joints (on aloha-agilex the finger joint and its mimic, from -0.01 closed to 0.045 open). A real
robot's fingers report the same, so it is proprioception, not privileged state. Frames and
observations recorded before it was read carry `None`.

## Time

Demonstration frames are **not evenly spaced**, so a frame's index over `frequency` is not its
time. RoboTwin's expert acts in motion primitives — an arm move, a gripper open or close — and
records a frame before a primitive's first physics step, after every `save_freq`-th step from
its first, and after its last. Each primitive therefore adds a frame one step after its first, a
shorter remainder, and an exact duplicate where the next primitive starts; a gripper closing is
many frames of a still arm.

`Frame.time_s` and `Observation.time_s` are simulated seconds, counted in physics steps of
1/250 s: the demonstration's from the expert's first frame, the rollout's from its first
observation, both in the scene the fingerprint checked. `demo.times()` returns the frames'
times; they never decrease, and a tie is two frames with no physics step between them. A policy
that resamples a demonstration to its own control rate should interpolate over `times()`, not
over the frame index, and decide what to do with ties. For frames recorded without `time_s`,
`times()` estimates: an exact duplicate of the frame before shares its time and every other
frame follows by `1 / frequency`.

`obs.step` counts `take_action` calls, and one call runs as many physics steps as RoboTwin's
planner needs to reach its target, so `step` is not time either; `obs.time_s` is. The episode
record's `physics_steps` is the rollout's total. A clock is something a real robot has: neither
time is privileged.

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
| `far_side` | `head_camera`, `far_side_camera` | `far_side_camera` takes `front_camera`'s slot: across the table at (0, 0.90, 1.45), 38.9° down, 1.13 m from where it meets the table, 320x180 with a 45° vertical field; the arms enter from the top, and image left is the robot's right |

Profiles only **replace** a camera, in its slot. RoboTwin builds every static camera with
`create_camera`, which draws from numpy's global RNG after the scene seed is set and before any
object is placed, so a camera more or fewer would move every object of every seed — and the Same
Scene check could not notice, since both builds of an episode would carry it. A profile that adds
or removes a camera, touches `head_camera`, toggles `camera.collect_head_camera`, leaves two
cameras with one name or names a static camera `left_camera` or `right_camera` (the wrist
cameras' names) is refused, with no override: `SceneConfig.overrides` are held to the same
guard. Every profile therefore builds the same scene from the same seed; only the images differ.
A profile may still change a camera's type through `camera.head_camera_type` or
`camera.wrist_camera_type`: `head_camera` then keeps its name, slot and pose, but not its image
size or field of view.
The run manifest records the profile and the static cameras, and a run directory does not resume
under another profile.

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
| `ee` | 16 | per arm, left then right: flange pose (7) then gripper, as RoboTwin's `take_action(action_type='ee')` reads it |

Actions of the wrong width or containing non-finite values raise `PolicyError`.

### The `ee` path

An `ee` action is `[left pose (7), left gripper, right pose (7), right gripper]`, the layout of
`demo.endposes()`: each pose `[x, y, z, qw, qx, qy, qz]` in the world frame, wxyz, **the flange
pose** as `endpose` reports it — the tool centre point is 0.12 m further along the pose's own +x
axis, so an adapter that thinks at the tool centre converts back to the flange — and each
gripper a commanded value from 0 (closed) to 1 (open). Feeding an arm's own `endpose` back is an
exact round trip on aloha-agilex; RoboTwin's expert returns an arm to its start the same way.

Each call plans **both** arms with CuRobo from their measured joints, then runs max(left, right)
physics steps of 1/250 s: at least 31 (0.124 s) when a plan succeeds, even for an unchanged
target, and 50 with that arm not commanded when its plan fails. An arm the policy means to keep
still must still be given a pose; the fixed hold — its endpose from the episode's first
observation and its gripper value — is what the simulator tests check. The task's step limit
counts `take_action` calls, not physics steps: an `ee` call spends one step of the budget however
long it runs, and `obs.time_s` and the record's `physics_steps` show the time it took.

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
harness's own upper bound, so it tells you what a perfect imitator scores on your machine.

An `ee` adapter should also run `--policy replay_ee`, which plays `demo.ee_actions()` back through
`take_action('ee')` in order and holds the last. It is the ceiling of the `ee` path for any
model: where it scores below `replay`, the loss is in the path — planning, CuRobo's goal
tolerance, 31 physics steps a call — and no `ee` adapter will do better. Then run
your adapter with `--video` and compare `demonstration.mp4` with `evaluation_same_scene.mp4` in a
few episode directories before trusting any number.
