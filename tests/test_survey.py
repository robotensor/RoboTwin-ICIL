import json

from fake_robotwin import FakeConfig, FakeTaskEnv, FakeUnstable
from robotwin_icil import cli, robotwin, survey, tasks
from robotwin_icil.generate import scene_seeds


def test_a_survey_counts_successes_and_rejections_by_reason(monkeypatch):
    monkeypatch.setattr(robotwin, "unstable_error", lambda: FakeUnstable)
    env = FakeTaskEnv(unstable_seeds={1}, plan_fails_on={2})
    result = survey.survey_task(
        env, tasks.table()["place_object_basket"], [1, 2, 3, 4], FakeConfig()
    )
    assert (result.seeds, result.successes, result.success_rate) == (4, 2, 0.5)
    assert dict(result.rejections) == {"unstable": 1, "plan_failed": 1}
    assert result.frames == [env.expert_steps + 1] * 2
    assert result.to_json()["rejections"] == {"plan_failed": 1, "unstable": 1}
    # The expert's rate is measured on one robot; the result says which.
    assert result.embodiment == "fake-arms" and result.to_json()["embodiment"] == "fake-arms"


def test_a_survey_names_its_robot_even_with_nothing_measured():
    # The robot is the config's, not a seed's: a row with no seeds still says which one.
    result = survey.survey_task(FakeTaskEnv(), tasks.table()["click_bell"], [], FakeConfig())
    assert result.seeds == 0 and result.embodiment == "fake-arms"


def test_a_survey_keeps_one_record_per_seed_in_order(monkeypatch):
    monkeypatch.setattr(robotwin, "unstable_error", lambda: FakeUnstable)
    env = FakeTaskEnv(unstable_seeds={1}, plan_fails_on={2}, expert_misses_on={4})
    result = survey.survey_task(
        env, tasks.table()["place_object_basket"], [1, 2, 3, 4], FakeConfig()
    )
    detail = result.to_json()["seeds_detail"]
    assert [(r["seed"], r["outcome"], r["frames"]) for r in detail] == [
        (1, "unstable", None),
        (2, "plan_failed", None),
        (3, "ok", env.expert_steps + 1),
        (4, "expert_failed", None),
    ]
    assert all(r["seconds"] >= 0 for r in detail)
    assert sum(r["seconds"] for r in detail) == result.seconds


def test_render_lists_every_task_with_its_rate(monkeypatch):
    monkeypatch.setattr(robotwin, "unstable_error", lambda: FakeUnstable)
    table = tasks.table()
    results = [
        survey.survey_task(FakeTaskEnv(), table[name], [5, 6], FakeConfig())
        for name in ("place_object_basket", "click_bell")
    ]
    text = survey.render(results)
    assert "place_object_basket" in text and "click_bell" in text and "100% (2/2)" in text
    # A rate is the task's on one robot; the printed table says which, not only the JSON.
    header, first, second = text.splitlines()
    assert "robot" in header and "fake-arms" in first and "fake-arms" in second


def test_survey_rejects_zero_seeds(capsys):
    assert cli.main(["survey", "--suite", "v1", "--seeds", "0"]) == 2


def test_survey_json_is_rewritten_after_every_task(tmp_path, monkeypatch, capsys):
    # A survey runs for hours; one that dies mid-way keeps the tasks it has measured.
    first, second = (task.name for task in tasks.table().suite("v1")[:2])
    seeds = scene_seeds(0, 0, 3)

    def load_task(name):
        if name == first:
            return FakeTaskEnv(plan_fails_on={seeds[1]})
        raise robotwin.RoboTwinError(f"the simulator died before {name}")

    monkeypatch.setattr(robotwin, "unstable_error", lambda: FakeUnstable)
    monkeypatch.setattr(robotwin, "SceneConfig", FakeConfig)
    monkeypatch.setattr(robotwin, "load_task", load_task)
    out = tmp_path / "survey.json"

    assert cli.main(["survey", "--suite", "v1", "--seeds", "3", "--json", str(out)]) == 1
    assert f"the simulator died before {second}" in capsys.readouterr().err
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert [entry["task"] for entry in payload] == [first]
    assert payload[0]["seeds"] == 3
    detail = payload[0]["seeds_detail"]
    assert [r["seed"] for r in detail] == seeds
    assert [r["outcome"] for r in detail] == ["ok", "plan_failed", "ok"]
