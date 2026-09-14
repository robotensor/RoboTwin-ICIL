"""derive_units: a pure function of its arguments, drawn from a sha256 counter."""

import ast
import collections
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from icil_benchmark_robotwin import BENCHMARK, catalogue, units
from robotwin_icil import tasks

MATERIAL = "duel-1|franka_press_push"


def derive(**kwargs):
    arguments = {
        "seed_material": MATERIAL,
        "count": 12,
        "suite": "franka_1arm",
        "category": "press_push",
    }
    return BENCHMARK.derive_units(**{**arguments, **kwargs})


def test_the_same_arguments_derive_the_same_units_and_other_material_others():
    assert derive() == derive()
    assert derive(seed_material="duel-2|franka_press_push") != derive()


def test_units_are_drawn_only_from_the_requested_suite_and_category():
    table = tasks.table()
    suite = catalogue.suites(table)["franka_1arm"]
    for category in catalogue.FRANKA_1ARM_CATEGORIES:
        derived = derive(category=category, count=30)
        allowed = {name for name in suite if table[name].category == category}
        assert {u["task"] for u in derived} == allowed
        assert all(u["category"] == category and u["suite"] == "franka_1arm" for u in derived)
    # Without a category, the whole suite and nothing else; v1 is nine tasks.
    everything = derive(suite="v1", category=None, count=27)
    assert collections.Counter(u["task"] for u in everything) == dict.fromkeys(
        table.suites["v1"], 3
    )


def test_tasks_are_spread_evenly_over_the_suite():
    derived = derive(category="pick_and_place", count=20)  # 13 tasks
    counts = collections.Counter(u["task"] for u in derived)
    assert len(counts) == 13 and max(counts.values()) - min(counts.values()) <= 1


def test_each_unit_carries_its_candidates_and_its_robot():
    for unit in derive(count=8):
        assert set(unit) == {"task", "task_label", "suite", "category", "instance_params"}
        params = unit["instance_params"]
        seeds = params["scene_seeds"]
        assert len(seeds) == units.SCENE_SEED_CANDIDATES == len(set(seeds)) == 4
        assert all(type(s) is int and 0 <= s < units.SEED_BOUND for s in seeds)
        # The ABI names scene_seed; which candidate it is, materialize decides.
        assert params["scene_seed"] is None and params["embodiment"] == "franka-panda"
        assert unit["task_label"] == unit["task"].replace("_", " ").capitalize()
    assert {u["instance_params"]["embodiment"] for u in derive(suite="v1", category=None)} == {
        "aloha-agilex"
    }


def test_a_smaller_duel_is_the_start_of_a_larger_one():
    assert derive(count=3) == derive(count=12)[:3]
    assert derive(count=0) == []


def test_derivation_is_pinned_across_python_versions():
    # CI derives this on Python 3.10 and 3.12; a different digest is a different unit list.
    derived = BENCHMARK.derive_units(seed_material="pinned", count=9, suite="v1")
    text = json.dumps(derived, sort_keys=True, separators=(",", ":"))
    assert [u["task"] for u in derived[:3]] == PINNED_TASKS
    assert derived[0]["instance_params"]["scene_seeds"] == PINNED_SEEDS
    assert hashlib.sha256(text.encode()).hexdigest() == PINNED_SHA256


#: What `derive_units(seed_material="pinned", count=9, suite="v1")` returned when the derivation
#: was written, on Python 3.10 and 3.12: a change here changes every duel's units.
PINNED_TASKS = ["stack_bowls_two", "place_empty_cup", "click_alarmclock"]
PINNED_SEEDS = [1911900973, 111348362, 833122392, 158406989]
PINNED_SHA256 = "252476905dd8bc77004693c44457673dbf45ef7ecb861fbe866595864f37d8c7"


@pytest.mark.parametrize(
    ("kwargs", "error", "reason"),
    [
        ({"suite": "nope"}, ValueError, "unknown suite 'nope'"),
        ({"category": "nope"}, ValueError, "unknown category 'nope'"),
        ({"suite": "v1", "category": "bimanual"}, ValueError, "no task in category 'bimanual'"),
        ({"count": -1}, ValueError, "non-negative integer"),
        ({"count": True}, ValueError, "non-negative integer"),
        ({"seed_material": 7}, TypeError, "seed_material must be a string"),
    ],
)
def test_what_cannot_be_derived_is_refused(kwargs, error, reason):
    with pytest.raises(error, match=reason):
        derive(**kwargs)


def test_derivation_does_not_depend_on_the_hash_seed():
    code = (
        "import json; from icil_benchmark_robotwin import BENCHMARK; "
        "print(json.dumps(BENCHMARK.derive_units(seed_material='duel-1|franka_stacking', "
        "count=9, suite='franka_1arm', category='stacking'), sort_keys=True))"
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for hash_seed in ("0", "4242", "random")
    }
    assert len(outputs) == 1 and json.loads(outputs.pop())


def test_no_library_rng_is_consulted():
    # What the module imports and calls, not what its comments mention.
    tree = ast.parse(Path(units.__file__).read_text())
    imported = {
        alias.name.split(".")[0]
        for n in ast.walk(tree)
        if isinstance(n, ast.Import)
        for alias in n.names
    }
    imported |= {
        n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module
    }
    assert not imported & {"random", "numpy", "secrets"}
    called = {
        n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "hash" not in called
