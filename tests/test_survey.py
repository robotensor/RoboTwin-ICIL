from fake_robotwin import FakeConfig, FakeTaskEnv, FakeUnstable
from robotwin_icil import cli, robotwin, survey, tasks


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
