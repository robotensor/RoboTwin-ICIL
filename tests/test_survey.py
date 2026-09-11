from fake_robotwin import FakeConfig, FakeTaskEnv, FakeUnstable
from robotwin_icil import camera_profiles, cli, robotwin, survey, tasks


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


def test_a_survey_records_its_camera_profile(monkeypatch):
    monkeypatch.setattr(robotwin, "unstable_error", lambda: FakeUnstable)
    task = tasks.table()["click_bell"]
    # The survey written before camera profiles has no such key; stock is what it ran.
    assert survey.TaskSurvey(task=task).to_json()["camera_profile"]["name"] == "stock"
    config = FakeConfig()
    config.camera_profile = "far_side"
    result = survey.survey_task(FakeTaskEnv(), task, [5], config)
    assert result.to_json()["camera_profile"] == camera_profiles.get("far_side").identity()


def test_render_lists_every_task_with_its_rate(monkeypatch):
    monkeypatch.setattr(robotwin, "unstable_error", lambda: FakeUnstable)
    table = tasks.table()
    results = [
        survey.survey_task(FakeTaskEnv(), table[name], [5, 6], FakeConfig())
        for name in ("place_object_basket", "click_bell")
    ]
    text = survey.render(results)
    assert "place_object_basket" in text and "click_bell" in text and "100% (2/2)" in text


def test_survey_rejects_zero_seeds(capsys):
    assert cli.main(["survey", "--suite", "v1", "--seeds", "0"]) == 2
