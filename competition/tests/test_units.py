"""derive_units: a pure function of its arguments, drawn from a sha256 counter."""

import ast
import collections
import dataclasses
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
        assert {u["task"] for u in derived} == allowed == set(PINNED_FRANKA_1ARM[category])
        assert all(u["category"] == category and u["suite"] == "franka_1arm" for u in derived)
    # Without a category, the whole suite and nothing else: franka_1arm is four tasks.
    franka = derive(category=None, count=8)
    assert collections.Counter(u["task"] for u in franka) == dict.fromkeys(suite, 2)


def test_tasks_are_spread_evenly_over_the_suite():
    derived = derive(category=None, count=10)  # four tasks
    counts = collections.Counter(u["task"] for u in derived)
    assert len(counts) == 4 and max(counts.values()) - min(counts.values()) <= 1
    press_push = collections.Counter(u["task"] for u in derive(count=7))  # two tasks
    assert sorted(press_push.values()) == [3, 4]


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


def test_a_smaller_duel_is_the_start_of_a_larger_one():
    assert derive(count=3) == derive(count=12)[:3]
    assert derive(count=0) == []


def _sha256(derived):
    return hashlib.sha256(
        json.dumps(derived, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_derivation_is_pinned_across_python_versions():
    # CI derives these on Python 3.10 and 3.12; a different digest is a different unit list.
    # The one-arm Franka track's skills, one category of franka_1arm each.
    franka = {
        category: BENCHMARK.derive_units(
            seed_material="pinned", count=4, suite="franka_1arm", category=category
        )
        for category in catalogue.FRANKA_1ARM_CATEGORIES
    }
    assert {c: [u["task"] for u in derived] for c, derived in franka.items()} == (
        PINNED_FRANKA_TASKS
    )
    assert _sha256(franka) == PINNED_FRANKA_SHA256


#: What `derive_units(seed_material="pinned", ...)` returned under DERIVATION units/2, on Python
#: 3.10 and 3.12: a change here changes every duel's units.
PINNED_FRANKA_TASKS = {
    "pick_and_place": ["place_empty_cup"] * 4,
    "stacking": ["stack_bowls_two"] * 4,
    "press_push": ["click_bell", "press_stapler", "click_bell", "press_stapler"],
}
PINNED_FRANKA_SHA256 = "e7f7c2e9e39a3580da7e52581f8c7e4c382de384be1c7a1eb8229b4423fd37b0"


def test_the_catalogue_units_are_drawn_from_is_the_one_they_were_pinned_on():
    # The plugin's wheel is pinned; robotwin-icil's table is not in it. A table that changed
    # under the same pin would derive other units, so the digest is pinned with the derivation.
    assert units.catalogue_sha256(tasks.table()) == units.CATALOGUE_SHA256


#: `franka_1arm` per category, from the Franka survey (docs/results/survey-franka-seed0.json), when `CATALOGUE_SHA256`
#: was pinned. Stacking has no one-arm task:
#: stack_bowls_two, an arm-switching task, stands in for it.
PINNED_FRANKA_1ARM = {
    "pick_and_place": ["place_empty_cup"],
    "stacking": ["stack_bowls_two"],
    "press_push": ["click_bell", "press_stapler"],
}


def test_the_franka_suite_the_digest_covers_is_the_pinned_one():
    # The digest covers franka_1arm's members; this names them, so a table that moves one of them
    # to another category says which one before the digest refuses to derive.
    table = tasks.table()
    members = catalogue.suites(table)["franka_1arm"]
    by_category = {
        category: [name for name in members if table[name].category == category]
        for category in catalogue.FRANKA_1ARM_CATEGORIES
    }
    assert by_category == PINNED_FRANKA_1ARM
    assert list(members) == [name for names in PINNED_FRANKA_1ARM.values() for name in names]


def _changed_tables():
    table = tasks.table()
    moved = dict(table.tasks)
    moved["click_bell"] = dataclasses.replace(moved["click_bell"], category="pick_and_place")
    rearmed = dict(table.tasks)
    rearmed["lift_pot"] = dataclasses.replace(rearmed["lift_pot"], arms="1")
    return {
        "a task moved to another category": dataclasses.replace(table, tasks=moved),
        "a task whose expert uses other arms": dataclasses.replace(table, tasks=rearmed),
    }


@pytest.mark.parametrize("change", sorted(_changed_tables()))
def test_units_are_not_derived_from_a_catalogue_they_were_not_pinned_on(change):
    table = _changed_tables()[change]
    assert units.catalogue_sha256(table) != units.CATALOGUE_SHA256
    with pytest.raises(ValueError, match="units were pinned on"):
        units.derive_units(seed_material=MATERIAL, count=1, suite="franka_1arm", table=table)


@pytest.mark.parametrize(
    ("kwargs", "error", "reason"),
    [
        ({"suite": "nope"}, ValueError, "unknown suite 'nope'"),
        ({"category": "nope"}, ValueError, "unknown category 'nope'"),
        ({"category": "bimanual"}, ValueError, "no task in category 'bimanual'"),
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
