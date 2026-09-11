# UniSkill on robotwin-icil

UniSkill ([arXiv 2505.08787](https://arxiv.org/abs/2505.08787)) imitates a demonstration video
through skills: a frozen inverse skill dynamics model (the ISD) turns each pair of frames k steps
apart into a 64-dim skill, and a diffusion policy conditioned on the current skill predicts the
robot's actions. The adapter is `icil_policies.uniskill` in the `policies/` distribution; it runs
in its own environment, `icil-uniskill`.

## Status

- **The released policy checkpoint is gone.** UniSkill-Policy's README links it on Google Drive
  (`1bBiOaJNK21x6ePoWQ3Kg3N4GDnMa6Duf`), which returns 404. Only the skill encoder is reusable,
  so this adapter pairs the frozen ISD with the fork's `DiffusionPolicyUNet` trained on RoboTwin
  exports outside the benchmark (plan B3, not yet done). Constructing `UniSkillPolicy` without a
  checkpoint raises `PolicyError` saying so.
- Everything else is built and tested without that checkpoint: the skill extractor on the GPU
  with the released weights, and the policy on a random network built from the same config code.
- The conversion oracle, `icil_policies.common.oracles:ResampledQposReplay`, replays the 20 Hz
  resampled demonstration through `take_action('qpos')`: the ceiling of any 20 Hz qpos model.
  It must come out close to the native replay's 18/18 on V1; that is measured in the simulator,
  not yet recorded here.

## Running it

```bash
bash scripts/install_policy_env.sh uniskill          # $ICIL_HOME/envs/icil-uniskill
PY=$HOME/.cache/robotwin-icil/envs/icil-uniskill/bin/python
$PY -m pytest policies/tests/test_uniskill_env.py      # the environment's contract (GPU tests need the lock)
```

The weights are expected under `$ICIL_HOME/models` (default `~/.cache/robotwin-icil/models`):
`uniskill/idm.pth` from `HanjungKim/UniSkill` (revision `b8d6cba`, `UniSkill_final_weight/idm.pth`)
and `depth-anything-v2-small/` from `depth-anything/Depth-Anything-V2-Small-hf` (revision `5426e4f`:
`config.json`, `preprocessor_config.json`, `model.safetensors`). Their sha256 are pinned in
`uniskill.yml` and checked before anything loads.

The policy is `icil_policies.uniskill.policy:UniSkillPolicy`, with `checkpoint=PATH` (and
optionally `checkpoint_sha256=`, `config=PATH` for a YAML overriding `uniskill.yml`, `device=`).
It runs with the `far_side` camera profile, in the `icil-uniskill` environment behind a remote
policy server (#39); keyword arguments reach it through `--policy-arg` (#37). The oracle needs
neither and runs in the simulator env:

```bash
robotwin-icil eval --policy icil_policies.common.oracles:ResampledQposReplay \
  --camera-profile far_side --suite v1 --episodes 18 --seed 42 --run-dir runs/qpos-oracle
```

## The skill extractor

`SkillExtractor.extract(demo)` returns one skill row per 20 Hz step of the demonstration:

1. The demonstration is resampled to 20 Hz on its frame times (`icil_policies.common.resample`);
   each step shows its nearest frame.
2. Each step's `far_side_camera` frame is cut to its centred 180x180 square (the ISD's view).
3. As in the ISD's training code (`diffusion/dataset/base_dataset.py`), the view is resized to
   224x224 by PIL bilinear interpolation and scaled to [0, 1]; Depth-Anything's slow
   `DPTImageProcessor` takes that image with `do_rescale=False` (518x518, ImageNet
   normalisation), and each predicted depth map is scaled to [0, 1] by its own extremes and
   resized to 224x224 (`diffusion/train_uniskill.py`).
4. Row i is `idm(depth[i, f(i)], image[i, f(i)], return_skill=True)` with f(i) = min(i + k, N - 1).

`extract(demo, variant=v, key=j)` gives the augmented variant `aug_v` of demonstration `j`, and
`extract_variants` all of `base`, `aug_0`..`aug_4`, named as the fork's skill directory names
them, for training data. Rows come back as (N, 64); the ISD itself returns (B, 1, 64), keeping
its last frame's time axis, and the fork's skill files keep it too (`extract_skill.py` saves
(T, 1, 64), and the fork's dataset hands `skill[index]` to a network that reads `[:, 0, :]`),
so a training export writes each row as (1, 64).

## The policy

At each episode the demonstration becomes N skill rows, once. The policy then returns one 14-dim
absolute qpos target per `act()` from a queue of `Ta` = 8; when the queue is empty it samples a
new chunk of `Tp` = 16 from the last `To` = 2 observations (padded by repetition at the start)
and the skill row of the number of actions executed so far, held at row N - 1 past the end, and
queues rows `To - 1` to `To + Ta - 2` of it, as the fork's `_get_action_trajectory` does.
`episode_info()` records the row of every re-plan against the demonstration's progress through
its rows (`replan_steps`, `replan_rows`, `skill_progress`, `held_past_end`); nothing re-anchors
the index in V1, so a policy that leads or lags the demonstration drifts from it (plan 6).

Each observation's `far_side_camera`, `left_camera` and `right_camera` frames are
area-resized to the checkpoint's input size (128x128 by default) and rounded to whole levels
(`conversion.policy_image`, which a training export must use too), then processed as robomimic
processes its dataset: [0, 1], channels first, the fork's 116-pixel centre crop at evaluation.

`describe()` records the adapter and `ADAPTER_VERSION`, the checkpoint and its sha256, the
checkpoint's `training_tasks`, `training_regime` and `augmentation` from its model card
(`"unknown"` when absent), `camera_profile_required: far_side`, a `parameter_checksum` of
every weight it runs (the policy network, the ISD and the depth model, computed at each call so
a run can compare its start and end), k, the horizons and the DDIM steps.

### The exported checkpoint

`model.save_checkpoint` writes, and `model.load_checkpoint` reads, a `torch.save` file whose
`format` is `icil-uniskill-policy/1`: the fork's config as JSON, `shape_metadata`
(`all_shapes`, `ac_dim`), the network's state dict **with the EMA average baked in**
(`model.bake_ema`, at export time), robomimic's action normaliser (`normalised = (action -
offset) / scale`) and a model card, `metadata`: `training_tasks`, `training_regime`,
`augmentation` and the `skills` settings it was trained on (camera, crop, rate_hz, k,
isd_sha256), which the policy holds against its own encoder. It loads with
`weights_only=True`; a fork training checkpoint is refused until exported.

## Constants and their sources

| Constant | Value | Source |
| --- | --- | --- |
| skill rate | 20 Hz | the ISD's LIBERO frames and the fork's control rate (`configs/uniskill_policy.json` rollouts at 20 Hz); plan A3 |
| k | 20 steps (1.0 s) | the paper's setting; plan A3 |
| skill rows | N per N-step demonstration, future clamped | the released `skills.zip` (recomputing it matched this, not `extract_skill.py`'s N - k rows) |
| upright frames | no flip | recomputed released skills matched upright LIBERO frames (cosine 0.964) far better than flipped (0.84); RoboTwin renders upright |
| ISD camera | `far_side_camera` | plan A3: the closest view to the ISD's LIBERO pretraining; M4's separation check may prefer `head_camera` |
| ISD crop | centred 180x180 | `far_side_camera` is 320x180 at 45° vertical, so the square is LIBERO's 45° field (plan 3.1) |
| ISD resolution | 224 | `dynamics/idm.py` `idm_resolution`; `extract_skill.py` default |
| ISD shape | 8 layers, 4 heads, 256 hidden, 64 skill, 768 out | `extract_skill.py` defaults, which `idm.pth` loads into strictly |
| depth | Depth-Anything-V2-Small, slow `DPTImageProcessor`, 518x518 | `diffusion/train_uniskill.py`, `base_dataset.py` |
| policy cameras | `far_side_camera`, `left_camera`, `right_camera` | plan B3 (the fork read `agentview` and the wrist) |
| policy images | 128x128, centre crop 116 | the fork's LIBERO renders and `CropRandomizer`; ASSUMPTION for RoboTwin: every camera is resized to a square, whatever its aspect |
| observation, action horizon | To 2, Ta 8, Tp 16 | `configs/uniskill_policy.json` |
| actions | 14-dim absolute next-step qpos, min-max normalised | plan B3 |
| scheduler | DDIM, 100 train steps, 10 at inference, `squaredcos_cap_v2`, epsilon | `configs/uniskill_policy.json` |
| EMA | power 0.75, baked in at export | `configs/uniskill_policy.json`; plan A3 |
| encoder | ResNet18Conv (not pretrained), SpatialSoftmax 32 keypoints, 64 features | `configs/uniskill_policy.json` |
| augmentation | 5 variants; brightness, contrast, saturation ±0.2; square crop 0.9-1.0 of the view | **ASSUMPTION**: the paper and code do not say how `aug_0..4` were made |

## Assumptions and deviations

- **Skill augmentation** is an assumption: colour jitter and a small crop, drawn once per
  demonstration and variant from `default_rng([seed, key, variant])` and applied alike to every
  frame, so a variant is one consistent video. Its parameters are `augmentation` in
  `uniskill.yml`; a trained checkpoint records its own in its model card.
- **EMA baked in.** The fork's rollout samples with the live weights: its `EMAModel` is built
  with `model_cls=nets`, the live network, which `_get_action_trajectory` then calls. Here the
  exported checkpoint holds the EMA average and the evaluator builds the network with no EMA
  model and no optimizer, so nothing calls `copy_to`, a parameter write.
- **Preprocessing follows the ISD's training code**, not `extract_skill.py`, which hands the
  depth processor raw 0-255 frames with `do_rescale=False`.
- **No warm-up.** The fork's `evaluate.py` sends 5 zero actions first; here the robot starts at
  its home pose and every call counts against the task's step limit.
- **Noise from a dedicated generator.** The fork draws diffusion noise from torch's global RNG;
  RoboTwin reseeds that RNG whenever it builds a scene, so the adapter draws from its own
  `torch.Generator`, which `seed()` reseeds.
- **The fork ignores `algo.unet`'s sizes**: its `ConditionalUnet1D` is always built with
  256/512/1024 channels, so the network is about 101M parameters whatever the config says, and
  the tests' small network is small only in its inputs.
- **`weights_only`.** `idm.pth` is a plain state dict and loads with `weights_only=True`, as does
  the exported checkpoint; the fork's own training checkpoints (loaded with
  `weights_only=False` by the fork, which torch 2.6 and later refuse by default) are never read
  by the evaluator.
- **The ISD's ResNet-18** is built without its ImageNet download (`dynamics/idm.py` asks for
  `pretrained=True`); `idm.pth` then sets every one of its weights.

## The environment

`icil-uniskill` (policies/envs/uniskill), built and verified on 2026-09-11 on an RTX 5090
(capability 12.0): python 3.10.21, torch 2.8.0+cu128, torchvision 0.23.0, transformers 4.57.1,
diffusers 0.35.1, numpy 1.26.4; the ISD checkout (`KimHanjung/UniSkill` at `eca49f0`) on the
path; the fork (`kang-jaehyun/UniSkill-Policy` at `2803ad6`) installed `--no-deps`, without its
LIBERO, robosuite and robocasa submodules, which only its LIBERO training and evaluation scripts
import (and robocasa pins numpy 1.23.3). xformers and flash-attn are not installed: nothing on
the adapter's path imports them. `policies/tests/test_uniskill_env.py` is its contract: the
fork builds `DiffusionPolicyUNet` from the adapter's config with diffusers' current `EMAModel`
accepting its `model_cls`, and the skill extractor runs on the GPU.

## Upstream issue (drafted, not posted)

To `kang-jaehyun/UniSkill-Policy`:

> **The pretrained policy checkpoint link returns 404**
>
> The README's "Pretrained Checkpoint" section links
> `https://drive.google.com/file/d/1bBiOaJNK21x6ePoWQ3Kg3N4GDnMa6Duf/view?usp=drive_link`,
> which returns 404 (checked 2026-09-11), so the released skill-conditioned policy cannot be
> evaluated. The skill encoder on Hugging Face (`HanjungKim/UniSkill`) and the skills archive
> still download. Could the checkpoint be re-uploaded, ideally with its training config and the
> skill directory it was trained on? It would also help to know how the augmented skills
> `aug_0`..`aug_4` in the skills archive were produced (which image augmentations, and with
> what parameters), since neither the paper nor the code says.
