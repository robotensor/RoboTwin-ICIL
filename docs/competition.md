# This benchmark as a competition plugin

The RoboTensor ICIL competition scores submissions on benchmarks, and a benchmark is a separate
repository. `competition/` is the distribution that makes this one of them:
**`robotwin-icil-competition`**, package `icil_benchmark_robotwin`, published through the
`icilval.benchmarks` entry point group.

It is a third distribution beside `robotwin-icil` and `robotwin-icil-policies`, for the same
reason `policies/` is separate: the benchmark core stays a benchmark and learns nothing about
duels, kings or stores, and the competition ABI moves at the competition's pace rather than this
repository's.

**It never imports `icilval`.** A benchmark that stands on its own cannot depend on the competition
that scores it, so the contract between them is duck-typed and carries `api_version`. The
orchestrator validates the shape structurally.

## The split that makes it work

| half | what | where it runs |
| --- | --- | --- |
| **pure** | `info`, `catalogue`, `derive_units`, `verify_prompt`, `read_result` | anywhere. No simulator, no assets, no GPU. |
| **commands** | `materialize_command`, `run_command` | nowhere — they return an **argv** |

The orchestrator host reads a catalogue, derives a unit list and verifies a published prompt; none
of those can build a RoboTwin scene, so none of them may need one. Everything that does need one
happens in a subprocess the orchestrator launches, which is why the orchestrator never imports
SAPIEN and why the simulator side can run in another image, or on another host, with nothing
changing here.

A test asserts the argv the plugin builds is the argv the CLI parses. That seam would otherwise rot
silently: the orchestrator only ever *runs* the simulator half, so a renamed flag would be found by
a failing duel rather than by a failing test.

## What a unit is here

A task and a **scene seed**. RoboTwin builds a scene deterministically from `(task, seed, config)`,
so the seed is the whole initial state and a published unit list is enough for a third party to
rebuild every scene it names.

Units derive from a sha256 counter rather than numpy's RNG, because a stream that depends on a
library version is not reproducible by someone holding the published record.

## The prompt

One demonstration, written as an `npz` of named arrays plus a JSON `meta` string — the convention
the orchestrator's loader already reads. Deliberately **not** a new file format, and no change to
`Demonstration` or `Frame`.

Arrays are grouped by channel so the orchestrator's demonstration view can allow or drop them as a
unit:

| channel | arrays |
| --- | --- |
| `video` | `frames_<camera>` |
| `proprio` | `qpos`, `endpose`, `gripper_joints` |
| `actions` | `actions` |

`times`, `frequency` and `meta` are metadata, not a channel.

`gripper_joints` is the **measured** position of each arm's fingers, `(T, 2, J)` in metres, left
arm then right. The gripper value inside `qpos` and `endpose` is RoboTwin's *command*, which reads
closed while the fingers rest on an object; a model trained on where the fingers actually are —
LIBERO's `gripper_states`, which is what the BPP adapter converts from — cannot run on a prompt
that carries only the command. A prompt written before this array, or from a demonstration
recorded before the benchmark read the measurement, simply does not have it and rebuilds without
one; the schema does not move for an array a reader may find absent.

**Redaction is the orchestrator's job, not this benchmark's.** It withholds the channels its field
withholds, over arrays like these. Keeping that decision there is what makes it hold for every
benchmark and survive this one changing underneath it — and it is why nothing in `competition/` has
a list of withheld channels.

The scene seed and a digest of the initial scene travel in `meta`, so Same Scene can be re-verified
against a published artifact. They are privileged, and a policy never sees `meta`.

## Only one view

`views` is `("video_only",)`. Same Scene makes the sensorimotor view **degenerate**: the rollout
starts in the very scene the demonstration was recorded in, so replaying the demonstration's own
actions solves the episode — the benchmark's own replay oracle does exactly that and scores 18/18
on the V1 suite. A field asking this benchmark for the sensorimotor view is refused rather than
quietly scored on something that measures playback.

## Running it

```bash
pip install -e . -e ./competition
robotwin-icil-competition info
robotwin-icil-competition catalogue
robotwin-icil-competition units --seed-material <duel id> --count 9
robotwin-icil-competition verify --prompt <dir> --task click_bell --scene-seed 7
```

Those need no simulator. `materialize` and `run-unit` do, and are run by the orchestrator rather
than by hand.

## Running a model that cannot share the simulator's process

`run-unit --policy-address` is how a competition keeps the entrant's weights out of the simulator:
the model runs in the competition's container behind a policy server, the simulator runs outside
it, and the two meet over an address the orchestrator hands both sides.

It is not a convenience. A model stack usually pins libraries RoboTwin also pins, and the BPP
adapter refuses outright to be constructed where `sapien` is importable — seeding torch's global
RNG in the simulator's process would make its diffusion noise a function of the scene seed, which
is privileged. Out of process is the only way such a model runs at all.

The orchestrator serves the policy:

```bash
/opt/envs/model/bin/python -m robotwin_icil.serve \
    --policy icil_policies.bpp:BPPPolicy --config configs/bpp.yml \
    --address 127.0.0.1:9001 --authkey-file /run/duel/authkey
```

and hands this side the same address and key:

```bash
robotwin-icil-competition run-unit --task click_bell --scene-seed 7 \
    --prompt /prompt/u0/prompt.npz --out /work/u0 \
    --policy-address 127.0.0.1:9001 --authkey-file /run/duel/authkey
```

`plugin.run_command` builds that argv; `authkey_file` travels through the same `**extra`
pass-through as any other option. The address is a `host:port` or the path of a Unix socket on a
shared mount. On this side it becomes the core's `RemotePolicy`, which is an `ICILPolicy` itself,
so the lifecycle and action checks run here as well as in the server. Every remote failure — an
error reply, a timeout, a server that hung up or died — is a `PolicyError` carrying the tail of
the server's log, and it stops the server: no server outlives its client. A `PolicyError` here
makes the unit `void`, not a loss, because a model that could not be reached is a harness fault.

`--policy module:Class` is the in-process path, for a policy whose dependencies do not conflict
with the simulator's. The two are alternatives, and giving both is refused: a unit runs one
policy, and the record has to say which.
