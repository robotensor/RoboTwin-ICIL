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
| `video` | `frames_*` — a prefix, one array per camera |
| `proprio` | `qpos`, `endpose` |
| `actions` | `actions` |
| `metadata` | `times` |

Both odd spellings are the orchestrator's, and both fail silently if we get them wrong — its view
is an allow-list, so an entry matching no array hands the policy nothing rather than raising. An
entry ending in `*` is a **prefix**, which is how a family the orchestrator cannot enumerate (it
does not know our camera list) stays inside an allow-list. `metadata` is the channel **every view
keeps**: frame times say when a frame was taken, never what the robot did, so no field's modality
list would ever claim them and a view that dropped them would leave the demonstration untimed.

`meta` is in no channel: it is a JSON string rather than an array, and it is privileged.

**Redaction is the orchestrator's job, not this benchmark's.** It withholds the channels its field
withholds, over arrays like these. Keeping that decision there is what makes it hold for every
benchmark and survive this one changing underneath it — and it is why nothing in `competition/` has
a list of withheld channels.

The scene seed and a digest of the initial scene travel in `meta`, so Same Scene can be re-verified
against a published artifact. They are privileged, and a policy never sees `meta`.

`verify_prompt` checks the file against every channel the map promises, not only the one the
default view is scored on: a channel with no array, an array in no channel, and rows that do not
line up with the frame count are all refusals. A prompt is one file read by both fields, and it
does not say which one will read it.

## Both views, and what each one measures here

`views` is `("video_only", "sensorimotor")`. The competition has a field for each and this is the
benchmark plugged into both, so serving one of them would leave the other field with nothing to
run on. `video_only` is the default when a field names no view.

Be accurate about the second one. Under Same Scene the sensorimotor view is **degenerate by
construction**: the rollout starts in the very scene the demonstration was recorded in, so
replaying the demonstration's own actions solves the episode — the benchmark's own replay oracle
does exactly that and scores 18/18 on the V1 suite. Serving the view is not a claim that a score
under it measures in-context imitation. It is the competition's field, and the **competition, not
this benchmark, owns the decision to score it**; what the benchmark owes is to serve both views
the same way and to say plainly what the difference between them is.

A view outside those two is refused — by the CLI's `choices`, and by `run_unit` itself, because
`result["view"]` is a claim about what the policy was shown and a claim nobody checked is worse
than no view at all.

Nothing else follows from the view. It is a passthrough: `run-unit` records it and never applies
it, `_demonstration` rebuilds the whole demonstration either way, and the prompt file is the same
bytes for both fields. **Redaction is the orchestrator's job**, over the channel map above.

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

## What is not wired yet

`run-unit --policy-address` is how a competition keeps the entrant's weights out of the simulator:
the model runs in the competition's container behind a policy server, the simulator runs outside
it, and the two meet over an address the orchestrator hands both sides. That needs
`robotwin_icil.serve` and `RemotePolicy`, which are built on another branch and not on `main`.
Rather than duplicate them here, `--policy-address` says exactly what it needs; `--policy
module:Class` is the in-process path and works today.
