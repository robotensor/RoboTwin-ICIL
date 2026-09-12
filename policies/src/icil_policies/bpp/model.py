"""Building the BPP model, loading the LIBERO-Gen Combination checkpoint, and slimming it.

The `bpp` extra: torch, hydra and BPP itself, so this module is imported only inside the
`icil-bpp` environment (`scripts/install_policy_env.sh bpp`). Everything numpy is in
`conversion.py`, which the simulator's environment imports on its own.

The recipe, verified on an RTX 5090 (518.8M parameters, every key matched strictly):

1. `OmegaConf.register_new_resolver("eval", eval)`, as BPP's `train.py` does before composing.
2. `initialize_config_dir(<behavior_prompting>/train_network/config, version_base="1.2")` and
   `compose("libero_policy_dunetp", overrides=["task=liberogen_spatial_combination",
   "+modifiers=libero/liberogen_spatial_combination"])` — the repo's own composition, not the
   config stored in the checkpoint, which lacks `use_pool_modality_pos_embed` and would build a
   model whose keys do not match.
3. `pretrained` stays true. With `pretrained=false` BPP's `init_weights` raises "Unaccounted
   module Conv2d(3, 768, 16x16, bias=False)" on the CLIP ViT's bias-free patch layer; with it
   true, timm downloads `vit_base_patch16_clip_224.openai` once (the installer prefetches it, so
   servers run with `HF_HUB_OFFLINE=1`) and the checkpoint then overwrites those weights.
4. `hydra.utils.instantiate(cfg.model)`, then `BasePolicy.load_state_dict(..., strict=True)`,
   which pops `_extra_training_split_info` first.

`slim` does step 4's reading once: it verifies the 6.9 GB release file's sha256, loads it on the
CPU with dill, and writes the inference state dict, the composed config, the normalizer as plain
JSON (so the numpy conversion oracle can report out-of-range proprioception) and `SOURCE.json`.
The release payload's `state_dicts` holds `model` and `optimizer` only — there is no EMA copy,
so the model weights are the inference weights; the plan's "EMA state dict" is recorded as a
deviation in docs/models/bpp.md.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CONFIG_NAME = "libero_policy_dunetp"
CONFIG_OVERRIDES = (
    "task=liberogen_spatial_combination",
    "+modifiers=libero/liberogen_spatial_combination",
)
# What `slim` writes, and what `load_model` reads back.
STATE_DICT_FILE = "model.pt"
CONFIG_FILE = "config.yaml"
NORMALIZER_FILE = "normalizer.json"
SOURCE_FILE = "SOURCE.json"
# The keys of the checkpoint's normalizer the adapter reports against (`ee_ori` is normalized by
# the identity, and images by an image identity, so neither has a range to fall outside).
NORMALIZED_KEYS = ("ee_pos", "gripper_states", "action")
CHUNK_BYTES = 8 << 20


def sha256_of(path: str | Path) -> str:
    """The sha256 of a file, read in chunks: the release checkpoint is 6.9 GB."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def config_dir() -> Path:
    """BPP's own Hydra config directory, inside the installed `behavior_prompting` package."""
    import behavior_prompting

    return Path(behavior_prompting.__file__).resolve().parent / "train_network" / "config"


def compose_config() -> Any:
    """The composed BPP config for the LIBERO-Gen Combination checkpoint."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    # BPP's configs use `${eval:...}`; its train.py registers the resolver before composing.
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with initialize_config_dir(str(config_dir()), version_base="1.2"):
        return compose(CONFIG_NAME, overrides=list(CONFIG_OVERRIDES))


def build_model(cfg: Any = None) -> Any:
    """An untrained `DiffusionUnetPolicy` from the repo composition (24 s, 518.8M parameters)."""
    import hydra

    return hydra.utils.instantiate((cfg if cfg is not None else compose_config()).model)


def parameter_checksum(model: Any) -> str:
    """sha256 over every parameter and buffer, in sorted key order: the frozen-policy audit.

    The runner compares it at the start and the end of a run (#37), so it must see every tensor
    the model holds, the normalizer's included.
    """
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if not isinstance(tensor, torch.Tensor):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(
            tensor.detach().cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def normalizer_params(state_dict: dict[str, Any]) -> dict[str, dict[str, list[float]]]:
    """The normalizer's scale and offset per key, as plain lists (`normalizer.params_dict.*`)."""
    params: dict[str, dict[str, list[float]]] = {}
    for key in NORMALIZED_KEYS:
        scale = state_dict.get(f"normalizer.params_dict.{key}.scale")
        offset = state_dict.get(f"normalizer.params_dict.{key}.offset")
        if scale is None or offset is None:
            continue
        params[key] = {
            "scale": [float(v) for v in scale.detach().cpu().flatten()],
            "offset": [float(v) for v in offset.detach().cpu().flatten()],
        }
    return params


def slim(source: str | Path, out_dir: str | Path, expected_sha256: str | None = None) -> dict:
    """Write the inference checkpoint, the composed config, the normalizer and SOURCE.json.

    The release file is a dill payload of about 6.9 GB; a server that loaded it at every start
    would spend minutes and gigabytes on the optimizer state it never uses. Everything here runs
    on the CPU (`map_location="cpu"`).
    """
    import torch

    try:
        import dill
    except ImportError as exc:  # pragma: no cover - the bpp env always has it
        raise RuntimeError("slimming the release checkpoint needs dill") from exc

    source, out_dir = Path(source).resolve(), Path(out_dir).resolve()
    digest = sha256_of(source)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"{source}: sha256 is {digest}, expected {expected_sha256}")
    payload = torch.load(source, pickle_module=dill, map_location="cpu", weights_only=False)
    states = payload["state_dicts"]
    if "ema_model" in states or "ema" in states:  # pragma: no cover - not in this release
        raise RuntimeError(f"this checkpoint holds {sorted(states)}; pick the inference weights")
    state_dict = {
        key: value.detach().cpu().clone()
        for key, value in states["model"].items()
        if hasattr(value, "detach")
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, out_dir / STATE_DICT_FILE)
    (out_dir / NORMALIZER_FILE).write_text(
        json.dumps(normalizer_params(state_dict), indent=2, sort_keys=True), encoding="utf-8"
    )
    from omegaconf import OmegaConf

    (out_dir / CONFIG_FILE).write_text(OmegaConf.to_yaml(compose_config()), encoding="utf-8")
    written = sha256_of(out_dir / STATE_DICT_FILE)
    source_info = {
        "source": str(source),
        "source_sha256": digest,
        "state_dict_sha256": written,
        # Keys are tensors; `state_dict_elements` counts the numbers inside them, buffers and
        # the normalizer included, which is larger than the model's parameter count.
        "state_dict_keys": len(state_dict),
        "state_dict_elements": int(sum(v.numel() for v in state_dict.values())),
        "state_dicts_in_payload": sorted(states),
        "ema": False,
        "config_name": CONFIG_NAME,
        "config_overrides": list(CONFIG_OVERRIDES),
        "behavior_prompting": _repo_revision(),
    }
    (out_dir / SOURCE_FILE).write_text(
        json.dumps(source_info, indent=2, sort_keys=True), encoding="utf-8"
    )
    return source_info


def load_model(checkpoint: str | Path, device: str = "cuda", sha256: str | None = None) -> Any:
    """The frozen model, built from the repo composition and loaded strictly from `slim`'s output.

    `sha256` is checked against the state dict file before it is read, so a run records what it
    actually loaded. The model comes back in eval mode with gradients off: the benchmark's
    policy is frozen, and nothing here ever writes a parameter.
    """
    import torch

    directory = Path(checkpoint).resolve()
    path = directory / STATE_DICT_FILE if directory.is_dir() else directory
    if sha256:
        digest = sha256_of(path)
        if digest != sha256:
            raise ValueError(f"{path}: sha256 is {digest}, expected {sha256}")
    model = build_model()
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval().requires_grad_(False)
    return model


def _repo_revision() -> str | None:
    """The `behavior_prompting` checkout's commit, for SOURCE.json; None if it is not a git tree."""
    import subprocess

    try:
        import behavior_prompting
    except ImportError:  # pragma: no cover - only outside the bpp env
        return None
    repo = Path(behavior_prompting.__file__).resolve().parent.parent
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        return None
    return result.stdout.strip() or None
