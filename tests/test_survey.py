import json

from fake_robotwin import FakeConfig, FakeTaskEnv, FakeUnstable
from robotwin_icil import cli, robotwin, survey, tasks
from robotwin_icil.demo import DemonstrationError
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
    assert [(r["seed"], r["outcome"], r["frames"], r["arms_moved"]) for r in detail] == [
        (1, "unstable", None, None),
        (2, "plan_failed", None, None),
        (3, "ok", env.expert_steps + 1, ["left", "right"]),
        (4, "expert_failed", None, None),
    ]
    assert all(r["seconds"] >= 0 for r in detail)
    assert sum(r["seconds"] for r in detail) == result.seconds


def test_a_survey_records_which_arms_each_demonstration_moved(monkeypatch):
    monkeypatch.setattr(robotwin, "unstable_error", lambda: FakeUnstable)
    table = tasks.table()
    one_arm = survey.survey_task(
        FakeTaskEnv(moves=("right",), plan_fails_on={2}),
        table["click_bell"],
        [1, 2, 3],
        FakeConfig(),
    )
    payload = one_arm.to_json()
    assert [r["arms_moved"] for r in payload["seeds_detail"]] == [["right"], None, ["right"]]
    assert (
        payload["one_arm_demonstrations"],
        payload["two_arm_demonstrations"],
        payload["no_arm_demonstrations"],
    ) == (2, 0, 0)

    both = survey.survey_task(FakeTaskEnv(), table["lift_pot"], [1, 2], FakeConfig())
    assert [r.arms_moved for r in both.records] == [("left", "right")] * 2
    assert both.to_json()["two_arm_demonstrations"] == 2
    assert both.to_json()["one_arm_demonstrations"] == 0

    # A success that moved nothing is evidence for neither count; it is reported on its own.
    still = survey.survey_task(FakeTaskEnv(moves=()), table["click_bell"], [1], FakeConfig())
    counts = still.to_json()
    assert still.records[0].arms_moved == ()
    assert counts["no_arm_demonstrations"] == 1
    assert (counts["one_arm_demonstrations"], counts["two_arm_demonstrations"]) == (0, 0)


def test_render_lists_every_task_with_its_rate_and_one_arm_count(monkeypatch):
    monkeypatch.setattr(robotwin, "unstable_error", lambda: FakeUnstable)
    table = tasks.table()
    results = [
        survey.survey_task(FakeTaskEnv(moves=moves), table[name], [5, 6], FakeConfig())
        for name, moves in (("place_object_basket", ("left", "right")), ("click_bell", ("left",)))
    ]
    text = survey.render(results)
    header, basket, bell = text.splitlines()
    assert "place_object_basket" in basket and "click_bell" in bell and "100% (2/2)" in bell
    # A rate is the task's on one robot; the printed table says which, not only the JSON.
    assert "robot" in header and "fake-arms" in basket and "fake-arms" in bell
    assert "one-arm" in header
    column = header.index("one-arm")
    assert basket[column : column + 7].strip() == "0"
    assert bell[column : column + 7].strip() == "2"


def test_render_shows_no_one_arm_count_without_a_success(monkeypatch):
    monkeypatch.setattr(robotwin, "unstable_error", lambda: FakeUnstable)
    env = FakeTaskEnv(plan_fails_on={1, 2})
    result = survey.survey_task(env, tasks.table()["click_bell"], [1, 2], FakeConfig())
    header, row = survey.render([result]).splitlines()
    column = header.index("one-arm")
    assert row[column : column + 7].strip() == "—"


def test_render_keeps_its_columns_in_place_at_a_hundred_seeds():
    # "100% (100/100)" is one character wider than any 20-seed cell; the columns after it must
    # not shift on that row alone.
    table = tasks.table()
    hundred = survey.TaskSurvey(
        table["click_bell"],
        [survey.SeedRecord(seed, survey.OK, 78, ("right",), 1.0) for seed in range(100)],
    )
    twenty = survey.TaskSurvey(
        table["lift_pot"],
        [survey.SeedRecord(seed, survey.OK, 300, ("left", "right"), 1.0) for seed in range(20)],
    )
    header, bell, pot = survey.render([hundred, twenty]).splitlines()
    column = header.index("one-arm")
    assert "100% (100/100)" in bell and "100% (20/20)" in pot
    frames_column = header.index("frames")
    for row, count, frames in ((bell, "100", "78"), (pot, "0", "300")):
        assert row[column - 2 : column] == "  "
        assert row[column : column + 7].strip() == count
        assert row[frames_column : frames_column + 6].strip() == frames


def test_survey_rejects_zero_seeds(capsys):
    assert cli.main(["survey", "--suite", "v1", "--seeds", "0"]) == 2


def test_survey_stops_with_one_line_when_a_qpos_does_not_split_into_arms(
    tmp_path, monkeypatch, capsys
):
    # An embodiment whose two arms differ in width cannot be measured; the survey says so and
    # stops, rather than dying with a traceback at its first success. Frame pins the width on
    # this branch, so the refusal is raised in arms_moved's place.
    first = tasks.table().suite("v1")[0].name

    def refuse(demonstration):
        raise DemonstrationError("qpos width 15 does not split into two equal arms")

    monkeypatch.setattr(robotwin, "unstable_error", lambda: FakeUnstable)
    monkeypatch.setattr(robotwin, "SceneConfig", FakeConfig)
    monkeypatch.setattr(robotwin, "load_task", lambda name: FakeTaskEnv())
    monkeypatch.setattr(survey, "arms_moved", refuse)
    out = tmp_path / "survey.json"

    assert cli.main(["survey", "--task", first, "--seeds", "2", "--json", str(out)]) == 1
    captured = capsys.readouterr()
    assert captured.err == "robotwin-icil: qpos width 15 does not split into two equal arms\n"
    assert f"{first}:" not in captured.out
    assert not out.exists()


def test_survey_json_is_rewritten_after_every_task(tmp_path, monkeypatch, capsys):
    # A survey runs for hours; one that dies mid-way keeps the tasks it has measured.
    first, second = (task.name for task in tasks.table().suite("v1")[:2])
    seeds = scene_seeds(0, 0, 3)

    def load_task(name):
        if name == first:
            return FakeTaskEnv(plan_fails_on={seeds[1]}, moves=("left",))
        raise robotwin.RoboTwinError(f"the simulator died before {name}")

    monkeypatch.setattr(robotwin, "unstable_error", lambda: FakeUnstable)
    monkeypatch.setattr(robotwin, "SceneConfig", FakeConfig)
    monkeypatch.setattr(robotwin, "load_task", load_task)
    out = tmp_path / "survey.json"

    assert cli.main(["survey", "--suite", "v1", "--seeds", "3", "--json", str(out)]) == 1
    captured = capsys.readouterr()
    assert f"{first}: expert solved 2/3, 2 with one arm" in captured.out
    assert f"the simulator died before {second}" in captured.err
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert not out.with_suffix(".json.tmp").exists()  # written whole, then renamed into place
    assert [entry["task"] for entry in payload] == [first]
    assert payload[0]["seeds"] == 3
    assert (payload[0]["one_arm_demonstrations"], payload[0]["two_arm_demonstrations"]) == (2, 0)
    detail = payload[0]["seeds_detail"]
    assert [r["seed"] for r in detail] == seeds
    assert [r["outcome"] for r in detail] == ["ok", "plan_failed", "ok"]
    assert [r["arms_moved"] for r in detail] == [["left"], None, ["left"]]
