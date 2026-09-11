from types import ModuleType

import numpy as np
import pytest

from icil_policies.testing import adapter_version, assert_conversion_pinned, conversion_digest


def _module(**attributes) -> ModuleType:
    module = ModuleType("fake_adapter")
    for name, value in attributes.items():
        setattr(module, name, value)
    return module


@pytest.mark.parametrize("version", ["1", "1.2", "10.0.3"])
def test_a_dotted_string_of_integers_is_a_version(version):
    assert adapter_version(_module(ADAPTER_VERSION=version)) == version


def test_a_module_without_a_version_fails():
    with pytest.raises(AssertionError, match="declares no ADAPTER_VERSION"):
        adapter_version(_module())


@pytest.mark.parametrize("version", [1, 1.2, "", "v1", "1.", "1.x", " 1"])
def test_anything_else_is_not_a_version(version):
    with pytest.raises(AssertionError, match="dotted string of integers"):
        adapter_version(_module(ADAPTER_VERSION=version))


def test_the_digest_follows_values_and_shapes():
    a = np.arange(6, dtype=np.float64)
    assert conversion_digest(a) == conversion_digest(a.copy())
    assert conversion_digest(a) != conversion_digest(a.reshape(2, 3))
    assert conversion_digest(a) != conversion_digest(a + 1e-3)
    assert conversion_digest(a) != conversion_digest(a, a)


def test_the_digest_ignores_float_noise_below_its_rounding():
    a = np.linspace(-1, 1, 7)
    assert conversion_digest(a) == conversion_digest(a + 1e-10)
    assert conversion_digest(np.array([0.0])) == conversion_digest(np.array([-0.0]))
    assert conversion_digest(a.astype(np.float32)) == conversion_digest(a.astype(np.float32))


def test_the_digest_takes_integers_and_booleans_but_not_objects():
    assert conversion_digest(np.array([1, 2], dtype=np.int32)) == conversion_digest(
        np.array([1, 2], dtype=np.int64)
    )
    conversion_digest(np.array([True, False]))
    with pytest.raises(TypeError):
        conversion_digest(np.array(["a"]))


def test_a_pinned_digest_passes():
    digest = conversion_digest(np.ones(3))
    assert_conversion_pinned(_module(ADAPTER_VERSION="2"), digest, {"1": "old", "2": digest})


def test_an_unpinned_version_asks_for_its_digest():
    digest = conversion_digest(np.ones(3))
    with pytest.raises(AssertionError, match=f"add '3': '{digest}'"):
        assert_conversion_pinned(_module(ADAPTER_VERSION="3"), digest, {"2": "other"})


def test_changed_outputs_under_an_old_version_ask_for_a_bump():
    old, new = conversion_digest(np.ones(3)), conversion_digest(np.zeros(3))
    with pytest.raises(AssertionError, match="bump ADAPTER_VERSION"):
        assert_conversion_pinned(_module(ADAPTER_VERSION="2"), new, {"2": old})
