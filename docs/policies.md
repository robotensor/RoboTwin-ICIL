# Writing a policy adapter

A policy is anything that can be reset, handed **one** demonstration, and then asked for actions
one observation at a time. The benchmark core knows nothing about any particular model; an adapter
converts between the benchmark's model-independent types and whatever the model consumes.

```bash
robotwin-icil eval --policy mypkg.adapters:MyPolicy --suite v1 --episodes 500 --seed 42 --run-dir runs/mine
```

`--policy` takes a built-in name (`replay`, `replay_ee`, `dummy`) or any importable
`module:Class` that subclasses `ICILPolicy`. Adapters live outside `robotwin_icil`. A model
whose libraries conflict with RoboTwin's runs in its own environment behind the built-in
`remote`; see [Running a model in its own environment](#running-a-model-in-its-own-environment).

## The lifecycle

For every episode, in this order and no other:

```text
policy.seed(s)                      # s from the policy's own stream, never the scene seed
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

`describe()` goes into the run manifest; report your model and checkpoint there, with the keys
of [the `describe()` convention](#what-describe-says).

## Configuration and hooks

### Arguments: `--policy-arg`

```bash
robotwin-icil eval --policy mypkg.adapters:MyPolicy --policy-arg config=configs/mine.yml \
    --policy-arg temperature=0.5 --suite v1 --episodes 500 --seed 42 --run-dir runs/mine
```

Each `--policy-arg KEY=VALUE`, repeatable, becomes one keyword argument of the policy's
constructor. The value is read with `yaml.safe_load`: numbers, booleans (`true`, `false`, and
YAML 1.1's `yes`, `no`, `on`, `off`) and `null` or an empty value arrive typed; a value in quotes
is the string inside them (`"revision='0123'"`, where `revision=0123` would be octal 83);
anything else, a path, a list or a date, stays the string given. Scientific notation needs a
dot and a signed exponent, as YAML 1.1 has it: `lr=1.0e-4` is the float 0.0001, but `lr=1e-4`
and `lr=1.0e4` stay the strings `1e-4` and `1.0e4`. `KEY` must be a Python identifier. A
malformed item, a non-finite number or a `KEY` given twice is a usage error before the simulator
loads; an argument the constructor does not take stops the run with a `PolicyError`.

An adapter with many settings takes one argument, `config=PATH` to a YAML file of its own, and
reads the rest from there; keep the command line for what changes between runs. The constructor
runs before the harness moves into `vendor/RoboTwin`, so resolve relative paths there
(`Path(config).resolve()`), not on first use.

Every policy still constructs with **no** arguments: give each one a default. The tests build
every built-in that way; do the same for yours. The one exception is `remote`, which has to be
told what to serve.

The manifest records the arguments as `policy_config`, paths as strings. They are part of the
run's identity: a run directory does not resume under other arguments.

### Hooks

Four optional methods on `ICILPolicy`, each with a default that does nothing, so a policy
overrides only what it needs:

| hook | called | default |
| --- | --- | --- |
| `seed(seed)` | before every `reset()` | nothing |
| `episode_info() -> dict` | after every rollout | `{}` |
| `close()` | once, when the run ends, however it ends | nothing |
| `environment() -> dict[str, str]` | once, when the run starts | `{}` |

**`seed(seed)`** gets an integer drawn from `np.random.default_rng([global_seed, episode,
POLICY_STREAM])`: the same for an episode whether or not the run was resumed, and never the scene
seed, which comes from `default_rng([global_seed, episode])` and is privileged. Only episodes
that reach a rollout seed the policy. Sampling policies, diffusion heads above all, become
reproducible per episode.

RoboTwin seeds torch's global RNG whenever it builds a scene: `_init_task_env_` calls
`torch.manual_seed` with the scene seed, at least twice an episode. Noise drawn from torch's
global RNG in the same process is therefore a function of the scene seed. An in-process adapter
must draw its sampling noise from a `torch.Generator` of its own, seeded from `seed()`, and never
from the global RNG:

```python
def seed(self, seed):
    self.generator = torch.Generator(device=self.device).manual_seed(seed)


def _act(self, obs):
    noise = torch.randn(self.noise_shape, generator=self.generator, device=self.device)
    ...
```

**`episode_info()`** reports what the adapter knows about the rollout it just ran: the arm it
drove, prompt chunks, how much proprioception fell outside its training range, clipped actions.
It is stored in the episode record as `policy_info`. Values must be plain JSON: `str`, `int`,
`float`, `bool`, `None`, lists and dicts of them. Convert numpy values (`.item()`, `.tolist()`).
NaN and infinity are refused too, and a value that is not JSON stops the run with a
`PolicyError` naming its key. Episodes without a rollout (rejected, invalid) record `{}`.

**`close()`** releases what the policy holds: a model on the GPU, a model server. `runner.run`
calls it exactly once, also when an episode raises or the run is refused.

**`environment()`** is the policy's own software environment, all strings: python, torch, CUDA,
the GPU, the commits of the model's repositories. The manifest records it as
`policy_environment`. Like the benchmark's own `environment`, it is not part of the run's
identity, so a run resumes on another machine.

### What `describe()` says

The base class returns `policy` (the name) and `action_type`; extend it with
`{**super().describe(), ...}`. These optional keys are the convention:

| key | type | what |
| --- | --- | --- |
| `adapter` | str | the adapter: its import path or distribution |
| `adapter_version` | str | the adapter's version |
| `checkpoint` | str or None | the checkpoint it loaded |
| `checkpoint_sha256` | 64 hex digits or None | that file's sha256 |
| `training_tasks` | list of task names, or `"unknown"` | what the model was trained on |
| `camera_profile_required` | a camera profile's name, or None | the only profile the policy runs under |
| `parameter_checksum` | str or None | a checksum of every parameter, for [the audit](#the-frozen-policy-audit) |

Any other key is the adapter's own; everything must be JSON. The runner holds the description to
the convention when a run starts (`check_description`) and refuses the run otherwise. A policy
whose `camera_profile_required` is not the run's `--camera-profile` is refused before one scene
is generated. The manifest records the description, and a run directory does not resume under a
different one, so another `adapter_version` or checkpoint is another run.

The runner describes the policy twice per run, at its start and at its end, not per episode.

The report's header names `adapter` and `adapter_version`, and counts the evaluated tasks that
`training_tasks` names: `evaluation tasks seen in training: k/N` over the run's N tasks, labelled
held out when k is 0 and `unknown` when the key is absent or "unknown". It labels the run; the
score is the same Same Scene 1-Demo Success Rate either way.

### The frozen-policy audit

When `describe()` gives a `parameter_checksum`, the runner describes the policy again at the end
of the run and compares. A different checksum raises `PolicyError: ... parameters changed during
the run: the policy must be frozen`. The audit also runs when the run raised, so an interrupted
run is audited too. The episodes already written stay on disk, and so does the failure: the run
directory gains `audit.json`, with both checksums, and from then on neither resumes nor reports.
Rerun under a new `--run-dir` once the policy is frozen. The audit is only as strong as the
checksum, so hash every parameter and buffer:

```python
def _checksum(self):
    digest = hashlib.sha256()
    for name, tensor in sorted(self.model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().flatten().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
```

A policy that gives no checksum is not audited.

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
or action history, a recurrent state — is fine, and `_reset()` must clear all of it. Give
`describe()` a `parameter_checksum` and [the audit](#the-frozen-policy-audit) holds you to it.

## Running a model in its own environment

BPP, UniSkill and RoboTwin pin libraries that conflict — transformers, diffusers,
huggingface_hub, and two different packages named `robomimic` — so no one Python environment
holds a model and the simulator. An adapter stays an ordinary `ICILPolicy`: it runs in-process
while you develop it, or behind a policy server in the model's own environment, which the
simulator's process reaches through the built-in `remote` (`robotwin_icil.remote.RemotePolicy`).
The simulator environment never imports model code.

```bash
robotwin-icil eval --policy remote \
    --policy-arg policy=mypkg.adapters:MyPolicy \
    --policy-arg python=/opt/envs/mymodel/bin/python \
    --policy-arg config=configs/mine.yml \
    --suite v1 --episodes 500 --seed 42 --run-dir runs/mine
```

`remote` starts `python -m robotwin_icil.serve --policy mypkg.adapters:MyPolicy --config ...`
under that interpreter, waits for it and forwards every call. The model environment needs the
adapter and its model, plus numpy and PyYAML, the benchmark's own dependencies; it needs neither
the simulator nor the benchmark installed. The server gets the `robotwin_icil` package on its
`PYTHONPATH`, the package alone through a link in a private directory, so a benchmark installed
in site-packages does not bring the simulator environment's libraries with it.

| argument | default | what |
| --- | --- | --- |
| `policy` | required | what the server serves: a built-in name or `module:Class`, importable in the model environment |
| `python` | the simulator's interpreter | the model environment's interpreter |
| `config` | none | the policy's own config file, resolved before the server starts and handed to the policy as `config=PATH` |
| `log` | `policy_server.log` in the run directory under `eval`, else a temporary file | where the server's stdout and stderr go, appended; kept after the run |
| `address` | none | `host:port` of a server already running (or its socket's path), instead of starting one; then give `authkey_file`, and none of `policy`, `python`, `config`, `log` or the policy's own arguments |
| `authkey_file` | none | the running server's key |
| `startup_timeout` | 900 | seconds for a started server to listen, and again for `hello`, whose reply comes once the model has loaded (BPP's checkpoint is 6.9 GB) |
| `timeout` | 120 | seconds for every other operation |
| any other | | the served policy's own keyword argument, passed on the server's command line as a `--policy-arg`: None, a boolean, a number, a string or a path, each read back as the value given; anything else is refused |

Each timeout bounds sending the request and, separately, waiting for its reply: a server that
stops reading, stopped or cut off by the network, would otherwise block a large message forever
once the socket's buffer is full.

Every one of them is a `--policy-arg`, so the manifest records it in `policy_config` and a run
does not resume under others; the log path `eval` picks is not recorded. The manifest's policy
description is the served policy's own, with `remote: {python, address, protocol_version}`
beside it: `python` is the interpreter the server ran under, and `address` is null for a server
`remote` started, whose socket changes every run.

### What happens in the server

The server builds the policy when the client says `hello`, then answers one operation at a time
by calling the policy's public methods: `describe`, `environment`, `seed`, `reset`, the
demonstration streamed as `demo_begin`, `demo_frames` (chunks of about 32 MB) and `demo_end`,
`act`, `info` (`episode_info()`), `ping`, and `shutdown`, which closes the policy before the
server exits. So the lifecycle checks of `ICILPolicy` run where the model lives, and, since
`RemotePolicy` is an `ICILPolicy` too, in the simulator's process as well, with the action shape
and finiteness checks. `describe()` asks the server each time, so the frozen-policy audit sees
the checksum the model ends the run with. Demonstrations and observations arrive bit for bit,
every field included.

A policy that raises in the server, a request or reply that does not go through within its
timeout, a server that hangs up or dies: each stops the server (SIGTERM to its process group,
SIGKILL 5 s later) and raises `PolicyError` naming the server's log and quoting its last lines;
every later call raises the same, the frozen-policy audit's included. The run stops as for any
`PolicyError`, and the same command resumes it from `episodes.jsonl`. `close()` asks the server
to shut down and stops it whatever it answers, so no server outlives a run, however the run
ends.

### A model on another machine

Start the server in the model's environment on the GPU machine, with the benchmark's `src` on
its `PYTHONPATH`, and point `remote` at it:

```bash
# on the GPU machine
PYTHONPATH=/path/to/robotwin-icil-benchmark/src python -m robotwin_icil.serve \
    --policy mypkg.adapters:MyPolicy --config configs/mine.yml \
    --address 0.0.0.0:41000 --authkey-file keys/policy.key
# on the simulator machine
robotwin-icil eval --policy remote --policy-arg address=gpu-box:41000 \
    --policy-arg authkey_file=keys/policy.key --suite v1 --episodes 500 --seed 42 --run-dir runs/mine
```

`serve` takes `--policy`, `--config` and `--policy-arg KEY=VALUE` as `eval` reads it, and the key
from `--authkey-file FILE` or `--authkey-env NAME`; it logs to stderr. It serves one client and
exits on `shutdown` or when that client hangs up, so start it again for the next run. The same
`--address` takes a Unix socket's path.

### Security

- Both ends prove they hold the key before anything else is said (`multiprocessing.connection`'s
  HMAC challenge). The key authenticates; it does not encrypt.
- Nothing is pickled. Only `send_bytes` and `recv_bytes` are used, never `Connection.send` or
  `recv`: a message is a JSON header and raw arrays of bool, integer and float dtypes, with object
  arrays refused at both ends, so neither peer can make the other run code.
- A server `remote` starts listens on a Unix socket in a private temporary directory, with a
  fresh 32-byte key handed over in an environment variable, never on the command line where
  `ps` would show it; the server removes it from its environment before the model loads.
- TCP only on a trusted network: the traffic is not encrypted, and a peer that connects and says
  nothing holds the server's handshake. A client with the wrong key is refused and the server
  listens on. Elsewhere, serve on `127.0.0.1` and reach it through an SSH tunnel.

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

Report your run with the oracles beside it:

```bash
robotwin-icil report runs/mine --reference runs/replay --reference runs/replay_ee
```

Each reference prints in a column of its own, overall, by category and by task, named by its
`adapter` or policy; `--json` lists them under `references`. A reference must have run the same
global seed, suite, expert budget, RoboTwin commit, configuration and camera profile, or the report
refuses it and names the field; the benchmark's own commit may differ. Episode i's scene depends
only on those and i, so each reference is rated over the episodes both runs recorded, its
diagnostics too: a finished oracle beside an interrupted run covers only the run's episodes, and a
reference that recorded only some of them says `only n of this run's m episodes`. One that shares
none is refused. The run's own score still covers all of its episodes. References are context for
the score, never a second one.

## The `policies/` distribution

Adapters live in `policies/`, the separate distribution `robotwin-icil-policies` (package
`icil_policies`). It depends on `robotwin-icil`; the core never imports it and nothing in
`src/robotwin_icil` names a model, which a test checks.

```bash
uv pip install -e . -e "policies[pure]"
pytest policies/tests -m "not sim"
```

| extra | what | installed in |
| --- | --- | --- |
| `pure` | numpy and pytest | CI, the simulator env, any adapter's pure tests |
| `bpp` | BPP's pins: torch 2.8.0, hydra-core 1.2.0, timm 1.0.20, diffusers 0.35.1, transformers 4.57.1, ... | `icil-bpp` |
| `uniskill` | UniSkill's pins: torch 2.8.0, transformers 4.57.1, diffusers 0.35.1, ... | `icil-uniskill` |

### The shared toolkit

`icil_policies.common` is numpy only, so it runs in the simulator env as well as in a model env,
and CI tests it on Python 3.10 and 3.12. Each constant in it cites the vendor file or plan
section it comes from; thresholds not yet measured (the arm-choice tie, the re-anchoring bounds,
the stall window) are marked provisional.

| module | what |
| --- | --- |
| `rotations` | wxyz quaternions (returned with w >= 0), matrices, axis-angle, rot6d as the first two rows (BPP's convention), slerp |
| `frames` | aloha-agilex: `LIBERO_TO_WORLD` (LIBERO's axes in the world, which is also the robot root's rotation), `arm_base(arm)`, `tcp_from_flange` and `flange_from_tcp` (0.12 m along the flange's +x), each arm's slots in qpos and `ee` actions |
| `resample` | `resample(demo, rate_hz=20)`: the demonstration on a fixed grid over `times()`; positions linear, orientations slerped, grippers held, images from the nearest frame; the final frame always sampled |
| `arms` | `choose_arm(demo)`: the arm whose tool centre travels further, a tie to the arm that moves first, then the left; `IdleArmHold`: the idle arm's first-observation joints or flange pose, and its gripper, for the whole episode |
| `images` | `arm_centred_view`: a square crop around a column, kept inside the frame, area resize to 128, bilinear to 224 with `align_corners=False` |
| `chunking` | `ChunkExecutor`: a history of `To` observations padded by repetition at the start, a queue of `Ta` actions, one per `act()` |
| `virtual_target` | `VirtualTarget`: an end-effector target that keeps sub-tolerance deltas and re-anchors on the measured pose past a bound or after a failed plan; `StallDetector` |
| `kinematics` | forward and inverse kinematics from a URDF, and `AlohaArm`: RoboTwin's endpose of an arm's six joints and back, for a `qpos_ik` execution mode only |

### `ADAPTER_VERSION`

Every adapter module declares `ADAPTER_VERSION`, a dotted string of integers such as `"1"` or
`"1.2"`, which names its conversion math: how a demonstration and each observation become model
inputs, and how model outputs become actions. Changing that math bumps it, and `describe()`
records it (#37), so runs converted differently are not mistaken for comparable ones.
`icil_policies.testing.assert_conversion_pinned` turns a forgotten bump into a failing test: an
adapter's test pins the digest of its conversion's outputs on fixed inputs, one entry per
version.

### Model environments

Each model runs in an environment of its own, built by
`bash scripts/install_policy_env.sh bpp|uniskill` from the lockfiles in `policies/envs/<name>/`.
The simulator env never imports model code; it reaches an adapter in a model env through a remote
policy (#39). See [Model environments](install.md#model-environments) in the install guide.
