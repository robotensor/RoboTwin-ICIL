"""What UniSkillPolicy does before any model code: pure, runs in CI without torch."""

import hashlib
import sys

import pytest

from robotwin_icil.policy import PolicyError


def test_the_policy_module_imports_without_torch():
    import icil_policies.uniskill.policy as policy

    assert policy.ADAPTER_VERSION
    if "torch" not in sys.modules:  # the host env has no torch: nothing pulled it in
        assert "robomimic" not in sys.modules


def test_construction_without_a_checkpoint_says_why(monkeypatch, tmp_path):
    from icil_policies.uniskill.policy import UniSkillPolicy

    monkeypatch.setenv("ICIL_HOME", str(tmp_path))
    with pytest.raises(PolicyError, match="404"):
        UniSkillPolicy()


def test_a_missing_or_mismatched_checkpoint_is_refused(tmp_path):
    from icil_policies.uniskill.policy import UniSkillPolicy

    with pytest.raises(PolicyError, match="not found"):
        UniSkillPolicy(checkpoint=str(tmp_path / "missing.pth"))
    path = tmp_path / "policy.pth"
    path.write_bytes(b"not a checkpoint")
    wrong = hashlib.sha256(b"something else").hexdigest()
    with pytest.raises(PolicyError, match="sha256"):
        UniSkillPolicy(checkpoint=str(path), checkpoint_sha256=wrong)
