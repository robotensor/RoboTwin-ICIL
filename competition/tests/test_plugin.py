"""The plugin as the orchestrator sees it, checked structurally without importing the orchestrator.

The rules below mirror `icil_orchestrator.benchmarks.api.validate_plugin` and `plugins.py`: an id
and ABI version, every method callable with the keywords the orchestrator passes, an entry point in
the `icil.benchmarks` group, and a pure half that imports no simulator.
"""

import dataclasses
import inspect
import json
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest

import icil_benchmark_robotwin
import robotwin_icil
from icil_benchmark_robotwin import BENCHMARK, catalogue, plugin
from robotwin_icil import remote, tasks

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
PURE_METHODS = ("info", "catalogue", "derive_units", "verify_prompt", "read_result")
COMMAND_METHODS = ("materialize_command", "run_command")
REQUIRED_KEYWORDS = {
    "derive_units": ("seed_material", "count", "suite", "category"),
    "verify_prompt": ("path", "unit"),
    "read_result": ("out_dir",),
    "materialize_command": ("unit", "out_dir"),
    "run_command": ("unit", "prompt", "out_dir", "policy_address", "authkey_env"),
}
#: What `plugins.SIMULATOR_MODULES` refuses a plugin's import and pure calls to bring in.
SIMULATOR_MODULES = {"sapien", "mujoco", "robosuite", "torch", "curobo", "isaacgym"}
#: The spec's franka_1arm track: its suite and the categories its skills name.
SPEC_SUITE, SPEC_CATEGORIES = "franka_1arm", ("pick_and_place", "stacking", "press_push")


def test_the_plugin_has_the_id_and_abi_version_the_orchestrator_speaks():
    assert BENCHMARK.id == "robotwin"
    assert type(BENCHMARK.api_version) is int and BENCHMARK.api_version == 1


@pytest.mark.parametrize("name", PURE_METHODS + COMMAND_METHODS)
def test_every_method_takes_the_keywords_the_orchestrator_passes(name):
    method = getattr(BENCHMARK, name)
    assert callable(method)
    parameters = inspect.signature(method).parameters
    accepted = {
        n
        for n, p in parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    assert set(REQUIRED_KEYWORDS.get(name, ())) <= accepted
    if name == "run_command":
        assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())


def test_the_entry_point_names_the_plugin_object():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["name"] == "robotwin-icil-competition"
    assert project["entry-points"]["icil.benchmarks"] == {
        "robotwin": "icil_benchmark_robotwin:BENCHMARK"
    }
    # The benchmark it wraps is pinned to the version released with it; icil-policy is not
    # imported by the plugin at all.
    assert set(project["dependencies"]) == {
        f"robotwin-icil=={robotwin_icil.__version__}",
        "icil-policy",
    }
    installed = [
        ep for ep in metadata.entry_points(group="icil.benchmarks") if ep.name == "robotwin"
    ]
    if not installed:
        pytest.skip("robotwin-icil-competition is not installed; its metadata cannot be read")
    [entry_point] = installed
    assert entry_point.value == "icil_benchmark_robotwin:BENCHMARK"
    assert entry_point.load() is BENCHMARK


def test_the_plugin_never_imports_the_orchestrator():
    package = Path(icil_benchmark_robotwin.__file__).parent
    for source in package.glob("*.py"):
        text = source.read_text()
        assert "import icil_orchestrator" not in text and "from icil_orchestrator" not in text


def test_the_pure_half_and_the_builders_import_no_simulator_and_no_orchestrator(tmp_path):
    code = f"""
import json, sys
from icil_benchmark_robotwin import BENCHMARK
unit = BENCHMARK.derive_units(seed_material="m", count=1, suite="franka_1arm", category="stacking")[0]
BENCHMARK.info(); BENCHMARK.catalogue()
BENCHMARK.verify_prompt(path={str(tmp_path / "missing.npz")!r}, unit=unit)
BENCHMARK.read_result(out_dir={str(tmp_path)!r})
BENCHMARK.materialize_command(unit=unit, out_dir="/o")
BENCHMARK.run_command(unit=unit, prompt="/p.npz", out_dir="/o", policy_address="/s", authkey_env="K")
print(json.dumps(sorted({{m.split(".")[0] for m in sys.modules}})))
"""
    loaded = set(
        json.loads(
            subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, check=True
            ).stdout
        )
    )
    assert not loaded & SIMULATOR_MODULES
    assert "icil_orchestrator" not in loaded and "icil_policy" not in loaded


def test_the_catalogue_has_the_shape_the_orchestrator_reads_and_the_spec_needs():
    shown = BENCHMARK.catalogue()
    assert isinstance(shown["suites"], dict) and all(
        isinstance(v, list) for v in shown["suites"].values()
    )
    assert SPEC_SUITE in shown["suites"] and "v1" in shown["suites"]
    assert set(SPEC_CATEGORIES) <= set(shown["categories"])
    table = tasks.table()
    assert set(shown["tasks"]) == set(table.tasks)
    for name, task in shown["tasks"].items():
        assert task["category"] == table[name].category and task["arms"] in ("1", "switching", "2")
    # Every skill of the track can draw a unit.
    for category in SPEC_CATEGORIES:
        assert BENCHMARK.derive_units(
            seed_material="m", count=1, suite=SPEC_SUITE, category=category
        )
    assert json.loads(json.dumps(shown)) == shown


def test_the_franka_suite_is_the_task_tables_and_nothing_is_provisional():
    # franka_1arm is the suite the Franka survey named in robotwin_icil's table: the plugin
    # derives no suite of its own and serves nothing as provisional.
    table = tasks.table()
    assert catalogue.suites(table) == table.suites
    shown, info = BENCHMARK.catalogue(), BENCHMARK.info()
    assert shown["suites"][SPEC_SUITE] == list(table.suites[SPEC_SUITE])
    assert "provisional" not in shown and "provisional" not in info
    package = Path(icil_benchmark_robotwin.__file__).parent
    assert not [p.name for p in package.glob("*.py") if "PROVISIONAL" in p.read_text()]
    # info() says where the suite was chosen and what each category of the track holds.
    basis = info["franka_1arm"]
    assert basis["survey"] == catalogue.SURVEY == "docs/survey.md#franka-one-arm-survey"
    assert "\n## Franka one-arm survey\n" in (ROOT.parent / "docs" / "survey.md").read_text()
    assert basis["categories"] == {
        "pick_and_place": {"tasks": ["place_empty_cup"], "arms": ["1"]},
        "stacking": {"tasks": ["stack_bowls_two"], "arms": ["switching"]},
        "press_push": {"tasks": ["click_bell", "press_stapler"], "arms": ["1"]},
    }


def test_info_names_stack_bowls_two_as_the_arm_switching_stand_in_for_stacking():
    [(name, note)] = BENCHMARK.info()["franka_1arm"]["stand_ins"].items()
    assert name == "stack_bowls_two"
    assert note.startswith(
        "stack_bowls_two is an arm-switching stand-in for stacking, which has no one-arm task: "
    )
    assert "a scene the survey did not see can still make its expert move both arms" in note
    # Read off the table: a stand-in for a category that has one-arm tasks claims no such thing.
    table = tasks.table()
    switching = dataclasses.replace(table["press_stapler"], arms="switching")
    changed = dataclasses.replace(table, tasks={**table.tasks, "press_stapler": switching})
    notes = catalogue.stand_ins(changed)
    assert list(notes) == ["stack_bowls_two", "press_stapler"]
    assert notes["press_stapler"].startswith(
        "press_stapler is an arm-switching stand-in for press_push: every demonstration"
    )


def test_the_catalogue_shows_the_arms_robotwin_icils_table_records():
    table = tasks.table()
    shown = BENCHMARK.catalogue()["tasks"]
    assert {name: task["arms"] for name, task in shown.items()} == {
        name: task.arms for name, task in table.tasks.items()
    }


def test_info_names_the_robots_cameras_protocol_commits_and_command_line():
    info = BENCHMARK.info()
    assert (info["id"], info["api_version"]) == ("robotwin", 1)
    assert info["protocol"] == "same_initial_state" and info["views"] == ["sensorimotor"]
    assert info["embodiments"]["franka-panda"]["action_dims"] == {"qpos": 16, "ee": 16}
    assert info["embodiments"]["aloha-agilex"]["action_dims"] == {"qpos": 14, "ee": 16}
    assert info["embodiment_of_suite"]["franka_1arm"] == "franka-panda"
    assert info["action_types"] == ["qpos", "ee"] and info["cameras"]
    assert set(info["commits"]) == {"benchmark", "robotwin"}
    assert info["benchmark"]["source_sha256"] == robotwin_icil.source_sha256()
    assert info["catalogue_sha256"] == info["pinned_catalogue_sha256"]
    assert info["derivation"] == "robotwin-icil-competition/units/2"
    assert info["cli"]["module"] == "robotwin_icil.cli"
    assert info["cli"]["python_env"] == "ROBOTWIN_ICIL_PYTHON"
    assert json.loads(json.dumps(info)) == info


def test_info_names_the_time_limits_a_served_policy_gets_and_the_extra_that_sets_them():
    info = BENCHMARK.info()
    assert info["limits"]["policy_budget_s"] == remote.POLICY_BUDGET_S
    assert info["limits"]["act_timeout_s"] == remote.ACT_TIMEOUT_S
    assert info["limits"]["result_reserve_s"] == remote.RESULT_RESERVE_S
    assert info["run_extra"] == ["act_timeout_s", "policy_budget_s", "unit_timeout_s", "policy_log"]
    for flag in ("--act-timeout-s", "--policy-budget-s", "--unit-timeout-s", "--policy-log"):
        assert flag in info["cli"]["run"]


def test_the_robotwin_commit_is_read_from_the_benchmarks_checkout_never_one_around_it(tmp_path):
    # Installed from a wheel, REPO_ROOT is <venv>/lib/python3.X, which may sit in any repository,
    # even one with a gitlink at vendor/RoboTwin; `HEAD:path` reads from its root whatever -C says.
    repo = tmp_path / "unrelated"
    site = repo / ".venv" / "lib" / "python3.10"
    site.mkdir(parents=True)
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    gitlink = "0123456789abcdef0123456789abcdef01234567"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        [*git, "update-index", "--add", "--cacheinfo", f"160000,{gitlink},vendor/RoboTwin"],
        check=True,
    )
    subprocess.run([*git, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "c"], check=True)
    assert plugin.robotwin_commit(repo) == gitlink
    assert plugin.robotwin_commit(site) is None
