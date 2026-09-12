"""`icil-bpp`: the BPP adapter's own tool, run inside the `icil-bpp` environment.

    icil-bpp slim --source CKPT --out DIR       # 6.9 GB of dill -> an inference checkpoint
    icil-bpp calibrate --data DIR --out YAML    # fit alpha_p and alpha_r on LIBERO-Gen data
    icil-bpp preflight --config YAML            # the two pre-flight gates
    icil-bpp libero-sanity --config YAML        # BPP's own LIBERO runner, as a sanity rollout

Every result is printed and, with `--json PATH`, written as JSON, so docs/models/bpp.md can
quote numbers that were actually produced here. Nothing in this module runs during a benchmark
episode.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..common.rotations import axis_angle_to_matrix, matrix_to_axis_angle
from .settings import (
    OSC_POSITION_SCALE_M,
    OSC_ROTATION_SCALE_RAD,
    dump,
    load,
)

AXES = ("x", "y", "z")


# ------------------------------------------------------------------------------------ slim


def _slim(args: argparse.Namespace) -> dict[str, Any]:
    from .model import slim

    settings = load(args.config) if args.config else load()
    info = slim(args.source, args.out, args.sha256 or settings.source_sha256 or None)
    print(f"wrote {args.out}")
    print(
        f"  state dict: {info['state_dict_keys']} tensors, {info['state_dict_elements']} elements"
    )
    print(f"  sha256:     {info['state_dict_sha256']}")
    print(f"  payload held state_dicts {info['state_dicts_in_payload']} (EMA: {info['ema']})")
    return info


# ------------------------------------------------------------------------------- calibrate


def _fit(achieved: np.ndarray, commanded: np.ndarray) -> dict[str, Any]:
    """Least squares of achieved on commanded motion, pooled and per axis, with its quality.

    An OSC delta is a goal offset: with robosuite's kp the arm covers `alpha` of it in one 20 Hz
    step (plan 3.4). `alpha` is the slope through the origin, so prompt deltas divide by it and
    executed deltas multiply by it.
    """
    pooled = float((commanded * achieved).sum() / (commanded * commanded).sum())
    per_axis = [
        float((commanded[:, i] * achieved[:, i]).sum() / (commanded[:, i] ** 2).sum())
        for i in range(achieved.shape[1])
    ]
    residual = achieved - pooled * commanded
    total = achieved - achieved.mean(axis=0)
    return {
        "alpha": round(pooled, 6),
        "per_axis": {axis: round(value, 6) for axis, value in zip(AXES, per_axis, strict=True)},
        "r2": round(float(1 - (residual**2).sum() / (total**2).sum()), 6),
        "steps": int(len(achieved)),
    }


def _calibrate(args: argparse.Namespace) -> dict[str, Any]:
    import h5py

    files = sorted(glob.glob(os.path.join(args.data, "*.hdf5")))[: args.files]
    if not files:
        raise SystemExit(f"no hdf5 files under {args.data}")
    moves, commanded_moves, turns, commanded_turns, actions = [], [], [], [], []
    for path in files:
        with h5py.File(path, "r") as handle:
            for name in sorted(handle["data"], key=lambda k: int(k.split("_")[1]))[: args.demos]:
                demo = handle["data"][name]
                raw = demo["actions"][:]
                position = demo["obs"]["ee_pos"][:]
                orientation = demo["obs"]["ee_ori"][:]  # axis-angle, as BPP's dataset reads it
                rotations = axis_angle_to_matrix(orientation)
                moves.append(np.diff(position, axis=0))
                commanded_moves.append(raw[:-1, :3] * OSC_POSITION_SCALE_M)
                turns.append(
                    matrix_to_axis_angle(np.einsum("tij,tkj->tik", rotations[1:], rotations[:-1]))
                )
                commanded_turns.append(raw[:-1, 3:6] * OSC_ROTATION_SCALE_RAD)
                actions.append(raw)
    moves, commanded_moves = np.concatenate(moves), np.concatenate(commanded_moves)
    turns, commanded_turns = np.concatenate(turns), np.concatenate(commanded_turns)
    every_action = np.concatenate(actions)

    position_fit, rotation_fit = _fit(moves, commanded_moves), _fit(turns, commanded_turns)
    report = {
        "files": [os.path.basename(f) for f in files],
        "demonstrations_per_file": args.demos,
        "position": position_fit,
        "rotation": rotation_fit,
        # What the checkpoint's own training data commanded: if RoboTwin's expert moves faster
        # than this, prompt actions clip and the config's `stretch` must rise (plan 3.4).
        "libero_step_motion_m": {
            "median": round(float(np.median(np.linalg.norm(moves, axis=1))), 6),
            "p99": round(float(np.percentile(np.linalg.norm(moves, axis=1), 99)), 6),
        },
        "libero_commanded_clipped_fraction": round(
            float(np.mean(np.abs(every_action[:, :6]) > 0.999)), 6
        ),
        # The time stretch stays 1.0 until RoboTwin's own speed is measured: the RoboTwin-side
        # numbers come from BPPConversionReplay.episode_info() on the calibration seed, not from
        # this command, which never touches a RoboTwin demonstration.
        "stretch": 1.0,
    }
    if args.out:
        settings = load(args.config) if args.config else load()
        fitted = load(
            args.config,
            alpha_p=position_fit["alpha"],
            alpha_r=rotation_fit["alpha"],
            stretch=report["stretch"],
            notes={**settings.notes, "calibrate": report["files"]},
        )
        Path(args.out).write_text(dump(fitted), encoding="utf-8")
        print(f"wrote {args.out}")
    print(
        f"alpha_p = {position_fit['alpha']} (per axis {position_fit['per_axis']}, "
        f"R2 {position_fit['r2']}, {position_fit['steps']} steps)"
    )
    print(
        f"alpha_r = {rotation_fit['alpha']} (per axis {rotation_fit['per_axis']}, "
        f"R2 {rotation_fit['r2']})"
    )
    print(f"LIBERO step motion: {report['libero_step_motion_m']}")
    return report


# ------------------------------------------------------------------------------- preflight


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Gate 1: prompt parity with BPP's own dataset. Gate 2: the adapter's actions are the model's."""
    report: dict[str, Any] = {"gate_1_prompt_parity": None, "gate_2_action_parity": None}
    if args.libero_data:
        report["gate_1_prompt_parity"] = _prompt_parity(args)
    else:
        report["gate_1_prompt_parity"] = {
            "ran": False,
            "why": "no --libero-data: gate 1 needs a LIBERO hdf5 and LiberoReplayImageDataset",
        }
    report["gate_2_action_parity"] = _action_parity(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def _prompt_parity(args: argparse.Namespace) -> dict[str, Any]:
    """The adapter's prompt tensors against `LiberoReplayImageDataset(only_prompt=True)`.

    Both sides read the same LIBERO demonstration: BPP's dataset through its zarr cache, and
    this adapter through `build_prompt` fed the hdf5's own arrays, with the loader's row flip
    and its channel move applied, so what is compared is the conversion and the chunking.
    """
    import h5py
    import torch
    from behavior_prompting.train_network.dataset.libero_replay_image_dataset import (
        LiberoReplayImageDataset,
        _receding_rgb_numpy_thwc_to_float_chw,
    )
    from behavior_prompting.train_network.utils.libero_util import (
        discover_and_register_benchmarks,
    )

    from .model import compose_config
    from .prompt_parity import adapter_prompt_from_hdf5

    # The dataset resolves a demonstration's task through `hdf5_to_task`, which looks its split
    # up in LIBERO's benchmark dict (`train_network/utils/libero_util.py:128`). LIBERO-Gen's
    # splits are not in it until this runs, one benchmark class per directory under LIBERO's
    # datasets root; BPP's own `utils/load_env.py:20` calls it at import, its dataset never does.
    discover_and_register_benchmarks()
    shape_meta = compose_config().shape_meta
    name = Path(args.libero_data).stem
    dataset = LiberoReplayImageDataset(
        shape_meta=shape_meta,
        dataset_splits=[Path(args.libero_data).parent.name],
        dataset_name=Path(args.libero_data).parent.name,
        use_cache=True,
        cache_dir=args.cache_dir,
        seed=0,
        val_ratio=0.0,
        sample_type="task",
        only_prompt=True,
        name_suffix=name,
        include_file_filters=[name],
    )
    theirs = dataset[0]["obs"]["prompt"] if "prompt" in dataset[0]["obs"] else dataset[0]["prompt"]
    with h5py.File(args.libero_data, "r") as handle:
        demo = handle["data"]["demo_0"]
        ours = adapter_prompt_from_hdf5(demo, shape_meta, _receding_rgb_numpy_thwc_to_float_chw)
    differences = {}
    for key in ("ee_pos", "ee_ori", "gripper_states", "agentview_rgb", "eye_in_hand_rgb"):
        mine, theirs_value = ours["obs"][key], np.asarray(theirs["obs"][key])
        size = min(len(mine), len(theirs_value))
        differences[key] = float(np.abs(mine[:size] - theirs_value[:size]).max())
    differences["action"] = float(
        np.abs(
            ours["action"][: len(theirs["action"])] - np.asarray(torch.as_tensor(theirs["action"]))
        ).max()
    )
    worst = max(differences.values())
    return {"ran": True, "max_abs_difference": differences, "passed": worst < 1e-5}


def _action_parity(args: argparse.Namespace) -> dict[str, Any]:
    """The adapter's first chunk against a direct `predict_action` on the same tensors.

    The adapter is driven through its public lifecycle — `seed`, `reset`, `set_demonstration`,
    `act` — on a synthetic RoboTwin demonstration, and the model is then asked directly, from the
    same seed, for the same observation history. Equal actions mean the adapter adds nothing to
    what the model itself predicts.
    """
    import torch

    from .policy import BPPPolicy
    from .synthetic import demonstration, observation

    policy = BPPPolicy(config=args.config, device=args.device)
    demo = demonstration()
    policy.seed(args.seed)
    policy.reset()
    policy.set_demonstration(demo)
    first = policy.act(observation(demo, 0, 0))

    # The same history the executor had: the first observation repeated to fill the horizon.
    from .conversion import observation_state

    arm = policy._choice.arm
    states = [observation_state(observation(demo, 0, 0), policy.settings, arm)] * 2
    batch = {
        key: torch.from_numpy(np.stack([s[key] for s in states]).astype(np.float32)[None]).to(
            policy.device
        )
        for key in states[0]
    }
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # The model keeps the prompt it was given; only the sampler's RNG is rewound.
    with torch.inference_mode():
        direct = policy.model.predict_action(batch)["action"].cpu().numpy()[0]
    from .conversion import Execution

    execution = Execution(policy.settings, arm, policy._arm_model)
    expected = execution.act(direct[0], observation(demo, 0, 0))
    difference = float(np.abs(np.asarray(first) - expected).max())
    policy.close()
    return {"ran": True, "max_abs_difference": difference, "passed": difference < 1e-5}


# --------------------------------------------------------------------------- libero sanity


def _libero_sanity(args: argparse.Namespace) -> dict[str, Any]:
    """BPP's own LIBERO runner on an unseen LIBERO-Gen combination task (the paper: about 71%)."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        from behavior_prompting.train_network.env_runner.libero_image_runner import (
            LiberoImageRunner,
        )
    except Exception as exc:  # pragma: no cover - depends on the LIBERO stack
        return {"ran": False, "why": f"{type(exc).__name__}: {exc}"}

    from .model import compose_config, load_model

    settings = load(args.config)
    model = load_model(settings.checkpoint, args.device, settings.checkpoint_sha256 or None)
    runner = LiberoImageRunner(
        dataset_path=args.dataset,
        output_dir=args.out,
        shape_meta=compose_config().shape_meta,
        cache_dir=args.cache_dir,
        n_envs=args.episodes,
        n_train=0,
        n_train_vis=0,
        n_test=args.episodes,
        n_test_vis=0,
        is_seen=False,
    )
    result = runner.run(model)
    return {"ran": True, "result": {k: float(v) for k, v in result.items() if _is_number(v)}}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# -------------------------------------------------------------------------------- the tool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="icil-bpp", description=__doc__.splitlines()[0])
    parser.add_argument("--json", help="also write this command's report as JSON")
    commands = parser.add_subparsers(dest="command", required=True)

    slim = commands.add_parser("slim", help="write an inference checkpoint from the release file")
    slim.add_argument("--source", required=True, help="the 6.9 GB release checkpoint")
    slim.add_argument("--out", required=True, help="the directory to write")
    slim.add_argument("--sha256", help="the release file's sha256 (default: the config's)")
    slim.add_argument("--config", help="a config whose source_sha256 to verify against")
    slim.set_defaults(run=_slim)

    calibrate = commands.add_parser("calibrate", help="fit the tracking gains on LIBERO-Gen data")
    calibrate.add_argument("--data", required=True, help="a directory of LIBERO-Gen hdf5 files")
    calibrate.add_argument("--files", type=int, default=10)
    calibrate.add_argument("--demos", type=int, default=10, help="demonstrations per file")
    calibrate.add_argument("--config", help="the config to start from")
    calibrate.add_argument("--out", help="write the fitted config here")
    calibrate.set_defaults(run=_calibrate)

    preflight = commands.add_parser("preflight", help="the pre-flight gates")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--device", default="cuda")
    preflight.add_argument("--seed", type=int, default=0)
    preflight.add_argument("--libero-data", help="one LIBERO hdf5, for the prompt parity gate")
    preflight.add_argument("--cache-dir", default="cache/libero/prompt_cache")
    preflight.set_defaults(run=_preflight)

    sanity = commands.add_parser("libero-sanity", help="BPP's own LIBERO rollout")
    sanity.add_argument("--config", required=True)
    sanity.add_argument("--dataset", required=True, help="the unseen LIBERO-Gen hdf5")
    sanity.add_argument("--episodes", type=int, default=10)
    sanity.add_argument("--device", default="cuda")
    sanity.add_argument("--out", default="runs/libero-sanity")
    sanity.add_argument("--cache-dir", default="cache/libero/rollout_cache")
    sanity.set_defaults(run=_libero_sanity)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = args.run(args)
    if args.json and report is not None:
        Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
