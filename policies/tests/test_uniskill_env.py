"""The icil-uniskill environment's contract, and UniSkillPolicy on a small random network.

Skipped unless the `uniskill` extra and the fork import (`scripts/install_policy_env.sh
uniskill`). The CPU tests build the fork's network from the adapter's config code with random
weights; the GPU tests also need the skill encoder's weights under $ICIL_HOME/models and, on a
shared machine, the GPU lock. Run from the repository root:

    $ICIL_HOME/envs/icil-uniskill/bin/python -m pytest policies/tests/test_uniskill_env.py
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("diffusers")
pytest.importorskip("robomimic.algo")

from diffusers.schedulers.scheduling_ddim import DDIMScheduler  # noqa: E402
from diffusers.training_utils import EMAModel  # noqa: E402

from icil_policies.uniskill import model  # noqa: E402
from icil_policies.uniskill.config import load_config  # noqa: E402
from icil_policies.uniskill.conversion import skill_steps  # noqa: E402
from icil_policies.uniskill.policy import UniSkillPolicy  # noqa: E402
from robotwin_icil.demo import Demonstration, Frame  # noqa: E402
from robotwin_icil.policy import Observation, PolicyError  # noqa: E402
from synthetic import endpose_dict, endpose_row  # noqa: E402

# Small inputs and encoders. The fork's DiffusionPolicyUNet ignores `algo.unet`'s sizes (its
# ConditionalUnet1D always has 256/512/1024 channels), so the network stays ~100M parameters.
IMAGE = 32
SMALL = {
    "observation": {
        "encoder": {
            "rgb": {
                "core_kwargs": {"feature_dimension": 16, "pool_kwargs": {"num_kp": 8}},
                "obs_randomizer_kwargs": {"crop_height": 28, "crop_width": 28},
            }
        }
    }
}
SKILLS = {"camera": "far_side_camera", "crop": 180, "rate_hz": 20.0, "k": 20}
CAMERAS = {"far_side_camera": (180, 320), "left_camera": (240, 320), "right_camera": (240, 320)}
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


class FakeExtractor:
    """Skill rows without the ISD: row i is filled with i / (N - 1)."""

    skill_dim = 64

    def __init__(self, **describe):
        self._describe = {**SKILLS, "isd_sha256": "f" * 64, **describe}

    def extract(self, demonstration):
        rows = len(skill_steps(demonstration))
        return np.repeat(np.linspace(0, 1, rows)[:, None], self.skill_dim, axis=1)

    def modules(self):
        return ()

    def describe(self):
        return dict(self._describe)


def _images(rng, cameras=CAMERAS):
    return {
        name: rng.integers(0, 256, (*shape, 3), dtype=np.uint8) for name, shape in cameras.items()
    }


def demonstration(seconds=2.0, cameras=CAMERAS, seed=0):
    rng = np.random.default_rng(seed)
    times = np.arange(0, seconds + 1e-9, 15 / 250)
    frames = tuple(
        Frame(
            index=i,
            images=_images(rng, cameras),
            qpos=np.full(14, 0.01 * i),
            endpose=endpose_dict(endpose_row()),
            time_s=float(t),
        )
        for i, t in enumerate(times)
    )
    return Demonstration(frames=frames, frequency=250 / 15)


def observation(step, seed=1):
    rng = np.random.default_rng([seed, step])
    return Observation(step=step, images=_images(rng), qpos=np.zeros(14))


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    """A small random network with a made-up EMA, exported as a checkpoint."""
    config = model.robotwin_config(SMALL)
    shapes = model.default_shapes(IMAGE)
    network = model.build_network(config, shapes, 14, "cpu")
    ema = network.ema.state_dict()
    ema["shadow_params"] = [p.detach() + 0.01 for p in ema["shadow_params"]]
    baked = model.bake_ema(network, ema)
    directory = tmp_path_factory.mktemp("uniskill")
    path = directory / "policy.pth"
    metadata = {
        "training_tasks": ["beat_block_hammer"],
        "training_regime": "same-episode skills",
        "augmentation": {"variants": 5},
        "skills": SKILLS,
    }
    scale, offset = np.full(14, 0.5), np.linspace(-0.1, 0.1, 14)
    sha = model.save_checkpoint(path, config, shapes, 14, baked, scale, offset, metadata)
    settings = directory / "cpu.yml"
    settings.write_text("device: cpu\n")
    return {"path": path, "sha": sha, "config": str(settings), "network": network, "ema": ema}


def make_policy(small, **extractor):
    return UniSkillPolicy(
        config=small["config"],
        checkpoint=str(small["path"]),
        checkpoint_sha256=small["sha"],
        skill_extractor=FakeExtractor(**extractor),
    )


def rollout(policy, steps, seed=3):
    policy.seed(seed)
    policy.reset()
    policy.set_demonstration(demonstration())
    return np.stack([policy.act(observation(step))[0] for step in range(steps)])


# --- the environment's contract ------------------------------------------------------------


def test_the_fork_builds_the_robotwin_policy_with_a_current_diffusers_ema():
    config = model.robotwin_config()
    network = model.build_network(config, model.default_shapes(), 14, "cpu")
    # diffusers' EMAModel still takes the fork's model_cls, the live network.
    assert isinstance(network.ema, EMAModel) and network.ema.model_cls is network.nets
    assert network.optimizers == {}
    assert isinstance(network.noise_scheduler, DDIMScheduler)
    horizon = config.algo.horizon
    assert (horizon.observation_horizon, horizon.action_horizon, horizon.prediction_horizon) == (
        2,
        8,
        16,
    )
    assert config.algo.ddim.num_inference_timesteps == 10


def test_baking_takes_every_parameter_from_the_ema_and_every_buffer_from_the_network(small):
    network, ema = small["network"], small["ema"]
    baked = model.bake_ema(network, ema)
    parameters = dict(network.nets.named_parameters())
    for (name, _), shadow in zip(parameters.items(), ema["shadow_params"], strict=True):
        torch.testing.assert_close(baked[name], shadow)
    for name, buffer in network.nets.named_buffers():
        torch.testing.assert_close(baked[name], buffer)


def test_a_loaded_checkpoint_holds_its_baked_weights_and_no_optimizer_or_ema(small):
    loaded = model.load_checkpoint(small["path"], "cpu")
    assert loaded.network.ema is None and loaded.network.optimizers == {}
    assert not loaded.network.nets.training
    assert not any(p.requires_grad for p in loaded.network.nets.parameters())
    shadows = small["ema"]["shadow_params"]
    for parameter, shadow in zip(loaded.network.nets.parameters(), shadows, strict=True):
        torch.testing.assert_close(parameter, shadow)
    assert loaded.metadata["training_tasks"] == ["beat_block_hammer"]


def test_a_checkpoint_that_was_not_exported_is_refused(tmp_path):
    path = tmp_path / "model_epoch_2000.pth"
    torch.save({"model": {"nets": {}, "ema": {}}, "config": "{}"}, path)
    with pytest.raises(PolicyError, match="exported"):
        model.load_checkpoint(path, "cpu")


# --- UniSkillPolicy on the small network -----------------------------------------------------


def test_one_qpos_action_per_call_and_replans_on_the_executed_count(small):
    policy = make_policy(small)
    actions = rollout(policy, 60)
    assert actions.shape == (60, 14) and np.all(np.isfinite(actions))
    info = policy.episode_info()
    rows = info["skill_rows"]
    # Frames 15 physics steps apart up to 1.98 s: 40 steps at 20 Hz, and one more holding the end.
    assert rows == 41
    assert info["replan_steps"] == list(range(0, 60, 8))
    assert info["replan_rows"] == [0, 8, 16, 24, 32, 40, 40, 40]  # held past the end
    assert info["held_past_end"] and info["skill_progress"] == 1.0


def test_a_seed_fixes_the_actions_and_nothing_touches_the_weights_or_global_rng(small):
    policy = make_policy(small)
    start = policy.parameter_checksum()
    torch.manual_seed(1234)
    first = rollout(policy, 10, seed=5)
    drawn = torch.rand(1)
    torch.manual_seed(1234)
    assert torch.equal(drawn, torch.rand(1))  # the policy drew nothing from torch's global RNG
    np.testing.assert_array_equal(rollout(policy, 10, seed=5), first)
    assert not np.array_equal(rollout(policy, 10, seed=6), first)
    assert policy.parameter_checksum() == start


def test_describe_records_what_ran(small):
    description = make_policy(small).describe()
    assert description["adapter"] == "uniskill" and description["action_type"] == "qpos"
    assert description["checkpoint_sha256"] == small["sha"]
    assert description["training_tasks"] == ["beat_block_hammer"]
    assert description["camera_profile_required"] == "far_side"
    assert description["k"] == 20 and description["augmentation"] == {"variants": 5}
    assert description["horizons"] == {"observation": 2, "action": 8, "prediction": 16}
    assert description["inference_steps"] == 10
    assert len(description["parameter_checksum"]) == 64


def test_a_demonstration_without_the_far_side_camera_is_refused(small):
    policy = make_policy(small)
    policy.reset()
    with pytest.raises(PolicyError, match="far_side"):
        policy.set_demonstration(demonstration(cameras={"head_camera": (240, 320)}))


def test_skills_from_another_encoder_setting_are_refused(small):
    with pytest.raises(PolicyError, match="k 20"):
        make_policy(small, k=10)


# --- on the GPU, with the released skill encoder ------------------------------------------


def _weights_present():
    skills = load_config().skills
    return skills.isd_path.is_file() and (skills.depth_dir / "model.safetensors").is_file()


@gpu
@pytest.mark.skipif(not _weights_present(), reason="needs idm.pth and Depth-Anything-V2-Small")
def test_the_skill_extractor_gives_one_row_per_step_on_the_gpu():
    from icil_policies.uniskill.skills import SkillExtractor

    extractor = SkillExtractor(device="cuda")
    assert type(extractor.depth_processor).__name__ == "DPTImageProcessor"
    demo = demonstration(seconds=3.0, cameras={"far_side_camera": (180, 320)})
    base = extractor.extract(demo)
    assert base.shape == (len(skill_steps(demo)), 64) == (61, 64)  # 3 s at 20 Hz
    assert base.dtype == np.float32 and np.all(np.isfinite(base))
    np.testing.assert_allclose(extractor.extract(demo), base, rtol=1e-4, atol=1e-4)
    variant = extractor.extract(demo, variant=0, key=7)
    assert variant.shape == base.shape and not np.allclose(variant, base)


@gpu
def test_the_full_size_policy_samples_a_chunk_on_the_gpu():
    config = model.robotwin_config()
    network = model.build_network(config, model.default_shapes(), 14, "cuda")
    network.set_eval()
    loaded = model.LoadedPolicy(
        network=network,
        config=config,
        shapes=model.default_shapes(),
        ac_dim=14,
        action_scale=torch.ones(14),
        action_offset=torch.zeros(14),
        metadata={},
    )
    obs = {key: torch.rand(1, 2, 3, 128, 128, device="cuda") for key in model.RGB_KEYS}
    obs["qpos"] = torch.zeros(1, 2, 14, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(0)
    chunk = model.sample(loaded, obs, torch.zeros(1, 64, device="cuda"), generator)
    assert chunk.shape == (1, 15, 14) and torch.isfinite(chunk).all()
