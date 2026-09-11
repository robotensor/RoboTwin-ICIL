"""The BPP LIBERO-Gen Combination transfer adapter (plan A2).

Behavior Prompting (BPP, `real-stanford/behavior_prompting` at `ec29e62`) is prompted with one
demonstration and predicts chunks of 20 Hz robosuite OSC deltas for a single Panda arm in LIBERO.
This adapter runs the released LIBERO-Gen Combination checkpoint on RoboTwin's bimanual
aloha-agilex, unchanged and frozen: it converts one RoboTwin `Demonstration` into BPP's prompt,
each `Observation` into BPP's observation, and BPP's actions back into RoboTwin actions.

The modules, so that only the last two need torch:

- `settings` and `conversion` — numpy only, importable in the simulator environment: the
  adapter's constants, frames, proprioception, images, prompt actions and execution. Built on
  `icil_policies.common`.
- `oracle` — `BPPConversionReplay`, the conversion oracle: the prompt's own converted actions
  replayed through the whole conversion and execution chain. Numpy only; its V1 fraction is the
  ceiling for any model behind this adapter.
- `model` and `policy` — the `bpp` extra: BPP's own Hydra composition, the slimmed checkpoint,
  and `BPPPolicy`, which runs in the `icil-bpp` environment behind the core's `remote`.

`ADAPTER_VERSION` names the conversion math: how a demonstration and each observation become
BPP inputs, and how BPP outputs become RoboTwin actions. Changing any of that bumps it, and
`policies/tests/test_bpp_conversion.py` pins the digest of each version's outputs.
"""

from __future__ import annotations

from typing import Any

ADAPTER_VERSION = "1"

__all__ = ["ADAPTER_VERSION", "BPPConversionReplay", "BPPPolicy"]


def __getattr__(name: str) -> Any:
    """Import the policies on demand, so `import icil_policies.bpp` needs no torch.

    `--policy-arg policy=icil_policies.bpp:BPPPolicy` resolves through this; the oracle stays
    importable in the simulator environment, where torch and BPP are absent.
    """
    if name == "BPPPolicy":
        from .policy import BPPPolicy

        return BPPPolicy
    if name == "BPPConversionReplay":
        from .oracle import BPPConversionReplay

        return BPPConversionReplay
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
