"""The BPP model side, skipped unless the `icil-bpp` environment is the one running the tests.

These need torch, hydra and BPP itself, so they never run in CI (`policies[pure]`) or in the
simulator environment. In the model environment they are the contract `BPPPolicy` relies on:

    $ICIL_HOME/envs/icil-bpp/bin/python -m pytest policies/tests/test_bpp_model_env.py

Building the model takes about half a minute and loading the checkpoint a few more, so the ones
that touch the checkpoint are marked `slow` and skip when it is not slimmed yet.
"""

import numpy as np
import pytest

from icil_policies.bpp import settings as bpp_settings
from icil_policies.bpp.synthetic import demonstration

bpp = pytest.importorskip("behavior_prompting", reason="not the icil-bpp environment")
torch = pytest.importorskip("torch")

from icil_policies.bpp import model as bpp_model  # noqa: E402
from icil_policies.bpp.prompt_parity import encode_actions  # noqa: E402

SETTINGS = bpp_settings.load(
    bpp_settings.Path(__file__).resolve().parents[1] / "configs" / "bpp_liberogen_combination.yaml"
)
CHECKPOINT = bpp_settings.Path(SETTINGS.checkpoint)
needs_checkpoint = pytest.mark.skipif(
    not (CHECKPOINT / bpp_model.STATE_DICT_FILE).is_file(),
    reason="run `icil-bpp slim` first",
)


def test_the_composition_is_the_one_the_checkpoint_was_trained_under():
    cfg = bpp_model.compose_config()
    shape_meta = cfg.shape_meta
    assert shape_meta["prompt_chunk_n_actions"] == bpp_settings.PROMPT_CHUNK_N_ACTIONS
    assert tuple(shape_meta["action"]["shape"]) == (bpp_settings.ACTION_DIM,)
    assert shape_meta["action"]["rep"] == "delta"
    assert shape_meta["action"]["horizon"] == bpp_settings.ACTION_HORIZON
    assert shape_meta["image_resolution"] == bpp_settings.IMAGE_SIZE
    assert cfg.task.env_runner.exec_action_horizon == bpp_settings.EXEC_ACTION_HORIZON
    for key in ("agentview_rgb", "eye_in_hand_rgb", "ee_pos", "ee_ori", "gripper_states"):
        assert key in shape_meta["obs"]
    # The stored config lacks this key, which is why the repo composition is used, not it.
    assert cfg.model.obs_encoder.obs_encoder.use_pool_modality_pos_embed is False


@pytest.mark.slow
@needs_checkpoint
def test_the_checkpoint_loads_strictly_and_comes_back_frozen():
    model = bpp_model.load_model(CHECKPOINT, device="cpu", sha256=SETTINGS.checkpoint_sha256)
    assert not model.training
    assert not any(p.requires_grad for p in model.parameters())
    assert sum(p.numel() for p in model.parameters()) > 500_000_000
    # The audit's checksum is stable across two reads of the same weights.
    assert bpp_model.parameter_checksum(model) == bpp_model.parameter_checksum(model)


@pytest.mark.slow
@needs_checkpoint
def test_describe_recomputes_the_checksum_the_audit_compares():
    """The frozen-policy audit is only real if `describe()` reads the weights again (#37)."""
    from icil_policies.bpp.policy import BPPPolicy

    policy = BPPPolicy(
        config=str(
            bpp_settings.Path(__file__).resolve().parents[1]
            / "configs"
            / "bpp_liberogen_combination.yaml"
        ),
        device="cpu",
    )
    before = policy.describe()["parameter_checksum"]
    assert before == policy.describe()["parameter_checksum"]
    with torch.no_grad():
        parameter = next(iter(policy.model.parameters()))
        parameter.add_(torch.ones_like(parameter))
    assert policy.describe()["parameter_checksum"] != before
    policy.close()


@needs_checkpoint
def test_slim_wrote_the_normalizer_the_numpy_oracle_reads():
    import json

    params = json.loads((CHECKPOINT / bpp_model.NORMALIZER_FILE).read_text())
    assert {"ee_pos", "gripper_states", "action"} <= set(params)
    for key in ("ee_pos", "gripper_states"):
        assert len(params[key]["scale"]) == len(params[key]["offset"])


def test_the_adapter_encodes_actions_exactly_as_bpps_dataset_does():
    """`encode_actions` against BPP's own rotation transformer, on the same random actions.

    `_convert_actions` is that transformer applied to an action's three rotation numbers, with
    the position and the gripper carried through; the test below compares against it directly,
    when its module imports.
    """
    from behavior_prompting.train_network.model.common.rotation_transformer import (
        RotationTransformer,
    )

    transformer = RotationTransformer(from_rep="axis_angle", to_rep="rotation_6d")
    raw = np.random.default_rng(0).uniform(-1, 1, size=(37, 7)).astype(np.float32)
    theirs = np.concatenate([raw[:, :3], transformer.forward(raw[:, 3:6]), raw[:, 6:]], axis=-1)
    np.testing.assert_allclose(encode_actions(raw), theirs, atol=1e-6)

    libero = pytest.importorskip("libero", reason="BPP's dataset module imports LIBERO")
    assert libero is not None
    from behavior_prompting.train_network.dataset.libero_replay_image_dataset import (
        _convert_actions,
    )

    np.testing.assert_allclose(encode_actions(raw), _convert_actions(raw, transformer), atol=1e-6)


def test_the_prompt_is_chunked_by_bpps_own_chunker():
    """The prompt an adapter builds has the shapes BPP's `prompt()` reads (the facts file)."""
    from behavior_prompting.common.np_util import add_batch_dim, remove_batch_dim
    from behavior_prompting.train_network.utils.prompt_util import (
        PromptActionChunker,
        collate_prompts,
    )

    from icil_policies.bpp.conversion import build_prompt, prompt_state

    chunker = PromptActionChunker(bpp_model.compose_config().shape_meta)
    demo = demonstration(steps=45)
    prompt = build_prompt(demo, SETTINGS, "left")
    state = prompt_state(prompt, SETTINGS)
    sample = {
        "obs": {key: value.astype(np.float32) for key, value in state.items()},
        "action": prompt.actions.astype(np.float32),
        "metadata": {},
    }
    chunked = remove_batch_dim(chunker.chunk_prompt(add_batch_dim(sample), obs_predownsampled=True))
    chunked["obs"] = {key: torch.from_numpy(value) for key, value in chunked["obs"].items()}
    chunked["action"] = torch.from_numpy(chunked["action"])
    batch = collate_prompts([{"obs": {"prompt": chunked}}])["obs"]["prompt"]

    chunks = prompt.chunks
    assert batch["action"].shape == (1, chunks, bpp_settings.PROMPT_CHUNK_N_ACTIONS, 10)
    assert batch["obs"]["agentview_rgb"].shape == (1, chunks, 3, 224, 224)
    assert batch["obs"]["ee_pos"].shape == (1, chunks, 3)
    assert batch["obs"]["ee_ori"].shape == (1, chunks, 6)
    assert batch["obs"]["gripper_states"].shape == (1, chunks, 2)
    # `pad_end_prompt_actions: zeros`: the last partial chunk is zero-padded, never dropped.
    assert bool(batch["metadata"]["mask"].any()) is False
    padded = chunks * bpp_settings.PROMPT_CHUNK_N_ACTIONS - len(prompt.actions)
    if padded:
        np.testing.assert_allclose(batch["action"][0, -1, -padded:].numpy(), 0.0)


def test_the_adapter_encodes_actions_exactly_as_convert_actions_does():
    """The same, against `_convert_actions` itself — which needs the LIBERO package it imports."""
    pytest.importorskip("libero", reason="BPP's dataset module imports LIBERO")
    from behavior_prompting.train_network.dataset.libero_replay_image_dataset import (
        _convert_actions,
    )
    from behavior_prompting.train_network.model.common.rotation_transformer import (
        RotationTransformer,
    )

    raw = np.random.default_rng(0).uniform(-1, 1, size=(37, 7)).astype(np.float32)
    transformer = RotationTransformer(from_rep="axis_angle", to_rep="rotation_6d")
    np.testing.assert_allclose(encode_actions(raw), _convert_actions(raw, transformer), atol=1e-6)
