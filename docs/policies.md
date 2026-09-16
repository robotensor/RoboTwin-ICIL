# Writing a policy adapter

A policy is anything that can be reset, handed **one** demonstration, and then asked for actions
one observation at a time. The benchmark core knows nothing about any particular model; an adapter
converts between the benchmark's model-independent types and whatever the model consumes.

```bash
robotwin-icil eval --policy mypkg.adapters:MyPolicy --episodes 500 --seed 42 --run-dir runs/mine
```

`--policy` takes a built-in name (`replay`, `dummy`) or any importable `module:Class` that
subclasses `ICILPolicy`. Adapters live outside `robotwin_icil`.

## The lifecycle

For every episode, in this order and no other:

```text
policy.reset(episode)                     # forget everything from the previous episode
policy.set_demonstration(demo)            # exactly once
loop:
    actions = policy.act(obs, action_dims)  # (k, action_dim); all k run before the next observation
```

`action_dims` is the live robot's action width per action type, read off its arms every rollout —
`{"qpos": 14, "ee": 16}` on aloha-agilex, `{"qpos": 16, "ee": 16}` on two Franka arms. The base
class checks `_act`'s result against it; an adapter never has to pass it anywhere.

`episode` is an `EpisodeInfo` of public facts — `embodiment`, the robot's name; `action_dims`, the
same widths; `seed`, which in `run-unit` is drawn from the prompt file's bytes and is never the
scene seed — kept on `self.episode` for `_reset` and every later call. `close()` is called once a
`run-unit` unit is over, however it ended; the base class holds nothing to release.

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

`T` is not fixed for a scene. RoboTwin's expert plans with CuRobo on the GPU, whose trajectories
for one scene are not the same length every run, and a primitive one physics step longer can record
one more frame. A saved `prompt.npz` fixes the demonstration; `eval`, which generates one per
episode, can hand a policy a frame more or fewer for the same seed in another run.

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

Run the replay oracle first, with the same task selection and seed:

```bash
robotwin-icil eval --policy replay --episodes 20 --seed 42 --run-dir runs/replay --video
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

## A policy served at an address

In a competition the policy is not imported at all. It runs in a process (a container) of its
own, served by `python -m icil_policy.serve` from the orchestrator's `icil-policy` distribution,
and `run-unit` drives it over an authenticated socket:

```bash
export ICIL_POLICY_AUTHKEY=$(python -c "import secrets; print(secrets.token_hex(32))")
python -m icil_policy.serve --manifest my_policy/icil.yaml --address /tmp/policy.sock \
    --authkey-env ICIL_POLICY_AUTHKEY --log-file /tmp/policy.log &
robotwin-icil run-unit --prompt runs/unit/prompt/prompt.npz --policy-address /tmp/policy.sock \
    --authkey-env ICIL_POLICY_AUTHKEY --policy-log /tmp/policy.log --out runs/unit/served
```

Such a policy implements `icil_policy.Policy`, not `ICILPolicy` (see icil-policy's README), and is
handed named arrays, never the benchmark's types:

| call | what it is handed |
| --- | --- |
| `reset(seed)` | an integer drawn from the prompt file's sha256: the same for every policy handed that prompt, never the scene seed |
| `set_demonstration(arrays, info)` | `prompt.npz`'s own arrays — `frames_<camera>` `(T, h, w, 3)` uint8, `qpos` `(T, D)`, `endpose` `(T, 16)`, `actions` `(T-1, D)`, `times` `(T,)`, `frequency` `()` — and `info = {"frequency", "cameras", "embodiment", "action_dims"}`; `meta` never |
| `act(observation)` | `frames_<camera>` `(h, w, 3)`, `qpos` `(D,)`, `endpose` `(16,)`; it answers `{"action": ...}`, `(A,)` or `(H, A)` |

The policy declares `action_type` (`qpos` or `ee`) when it answers `hello`, and every action's
width and dtype are checked against `info["action_dims"][action_type]` before it is copied, as for
a local adapter; rows of a chunk past 4096, more than any task's step limit, are dropped. The key
is only ever the *name* of a variable on a command line.

How long a served policy may take is bounded three ways. Each call has its own timeout:
`--act-timeout-s` for one `act` (60 s by default), 60 s to connect, 300 s each for `hello`, `reset`
and `prompt`, since building a policy and encoding a demonstration are work, not a hang, and 5 s
for `close`. All the calls of a unit together get `--policy-budget-s` (300 s by default): the call
in flight when it runs out is cut short, and the unit is void on the policy, so a policy that
answers every call just in time cannot run its unit into its caller's kill. And
`--unit-timeout-s`, the seconds whoever started `run-unit` gives it, stops every call 30 s short of
that: a call cut short there while the policy was within its budget means the harness took the
time, and the unit is void on the harness, with how long each took in `error`. `result.json`
records `policy_wall_s` and `policy_budget_s`.

What a remote failure costs the unit, in `result.json`:

| what happened | `success` | `void` | `void_cause` |
| --- | --- | --- | --- |
| the policy answered `reset`, `prompt` or `act` with an error (it raised), or returned an action of the wrong width or a non-finite one | false | false | null |
| nothing listened, the key or `hello` was refused (an error reply to `hello` is a policy the server could not build), the declared `action_type` is neither `qpos` nor `ee`, a call ran past its timeout, the unit's calls ran past `--policy-budget-s`, the connection dropped, a reply was malformed | null | true | `"policy"` |
| what the harness could not give the policy: a prompt, a scene, a GPU, the time (`--unit-timeout-s` ran out while the policy was within its budget), a fault of its own | null | true | `"harness"` |

A void unit's `error` names the call and the reason, and ends with the tail of `--policy-log` when
it is given. The log is opened at the path given, never through a link the policy swapped in, but
only its last path component is guarded: give a copy in a directory the policy cannot write. What
the server says is bounded before it reaches `result.json`: its name to 200 characters, a
failure's text to its first 4000 and last 8192. `result.json` names `"policy": "remote"` and, once
the policy has answered `hello`, the policy the server reported as `served_policy`. `run-unit`
closes the connection however the unit ended, within 5 s, and the server exits with it; it exits 1
before any result when `--authkey-env` holds no usable key (missing, not hex, under 16 bytes) or
icil-policy is not installed.

A GPU that runs out of memory or is lost mid-rollout voids the unit on the harness, but a served
policy sharing that GPU could have filled it. The result then lists every process `nvidia-smi` saw
holding GPU memory (`gpu_processes`, pid and MiB) and `run_unit_pid`, so whoever started the
policy can tell whether its processes held the memory; better, give the policy another GPU or cap
its memory.
