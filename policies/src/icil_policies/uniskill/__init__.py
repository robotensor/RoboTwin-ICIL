"""The UniSkill adapter (plan A3): a frozen skill encoder and a skill-conditioned diffusion policy.

UniSkill's released policy checkpoint is gone, so this adapter pairs the released skill encoder
(the ISD, `idm.pth`, with Depth-Anything-V2-Small) with a robomimic-fork `DiffusionPolicyUNet`
trained on RoboTwin exports outside this package. Everything here works, and is tested, without
that checkpoint (docs/models/uniskill.md).

- `conversion`: numpy only, importable and tested anywhere the core is: the constants, skill rows
  and their indexing, the demonstration's 20 Hz steps, the ISD's view, skill augmentation and
  the policy's image path.
- `config`: the adapter's configuration (`uniskill.yml`), model paths and their sha256 checks.
- `skills`: `SkillExtractor`, the ISD on a demonstration (torch, the `uniskill` extra).
- `model`: the fork's network, its exported checkpoint and sampling (torch and the fork).
- `policy`: `UniSkillPolicy`, importable without torch; the model loads when it is constructed.
"""
