import json

import pytest

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
    # The displacement behind each verdict travels with it, so the threshold can be re-judged.
    moved, rejected, _ = payload["seeds_detail"]
    assert rejected["displacement"] is None
    assert moved["displacement"]["left"] == 0.0 and moved["displacement"]["right"] > 0.05
    assert list(moved["displacement"]) == ["left", "right"]
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


def test_a_survey_on_two_frankas_names_the_robot_and_the_arm_that_moved(monkeypatch):
    # 16 wide, split 8 + 8. A left-only expert must read as one arm: an aloha-style split at 7
    # would put the left gripper, index 7, in the right arm and count both.
    monkeypatch.setattr(robotwin, "unstable_error", lambda: FakeUnstable)
    env = FakeTaskEnv(qpos_dim=16, moves=("left",))
    config = FakeConfig(embodiment="franka-panda")
    payload = survey.survey_task(env, tasks.table()["click_bell"], [1, 2], config).to_json()
    assert payload["embodiment"] == "franka-panda"
    detail = payload["seeds_detail"]
    assert [r["arms_moved"] for r in detail] == [["left"], ["left"]]
    assert all(r["displacement"]["right"] == 0.0 for r in detail)
    assert (payload["one_arm_demonstrations"], payload["two_arm_demonstrations"]) == (2, 0)


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
    one = {"left": 0.0, "right": 1.0}
    two = {"left": 1.0, "right": 1.0}
    hundred = survey.TaskSurvey(
        table["click_bell"],
        [survey.SeedRecord(seed, survey.OK, 78, ("right",), one, 1.0) for seed in range(100)],
    )
    twenty = survey.TaskSurvey(
        table["lift_pot"],
        [
            survey.SeedRecord(seed, survey.OK, 300, ("left", "right"), two, 1.0)
            for seed in range(20)
        ],
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
    assert cli.main(["survey", "--seeds", "0"]) == 2


def test_survey_stops_with_one_line_when_a_qpos_does_not_split_into_arms(
    tmp_path, monkeypatch, capsys
):
    # An embodiment whose two arms differ in width cannot be measured; the survey says so and
    # stops, rather than dying with a traceback at its first success. A frame takes the robot's
    # own width, so a real 15-wide qpos reaches arms_moved, which refuses to guess the split.
    first = tasks.table().suite("v1")[0].name

    monkeypatch.setattr(robotwin, "unstable_error", lambda: FakeUnstable)
    monkeypatch.setattr(robotwin, "SceneConfig", FakeConfig)
    monkeypatch.setattr(robotwin, "load_task", lambda name: FakeTaskEnv(qpos_dim=15))
    out = tmp_path / "survey.json"

    assert cli.main(["survey", "--task", first, "--seeds", "2", "--json", str(out)]) == 1
    captured = capsys.readouterr()
    assert captured.err == "robotwin-icil: qpos width 15 does not split into two equal arms\n"
    assert f"{first}:" not in captured.out
    assert not out.exists()


def test_survey_json_is_rewritten_after_every_task(tmp_path, monkeypatch, capsys):
    # A survey runs for hours; one that dies mid-way keeps the tasks it has measured.
    first, second = (task.name for task in tasks.table().select()[:2])
    seeds = scene_seeds(0, 0, 3)

    def load_task(name):
        if name == first:
            return FakeTaskEnv(plan_fails_on={seeds[1]}, moves=("left",))
        raise robotwin.RoboTwinError(f"the simulator died before {name}")

    monkeypatch.setattr(robotwin, "unstable_error", lambda: FakeUnstable)
    monkeypatch.setattr(robotwin, "SceneConfig", FakeConfig)
    monkeypatch.setattr(robotwin, "load_task", load_task)
    out = tmp_path / "survey.json"

    assert cli.main(["survey", "--seeds", "3", "--json", str(out)]) == 1
    captured = capsys.readouterr()
    assert f"{first}: expert solved 2/3, 2 with one arm" in captured.out
    assert f"the simulator died before {second}" in captured.err
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert not out.with_suffix(".json.tmp").exists()  # written whole, then renamed into place
    assert payload["images"] is False
    [entry] = payload["tasks"]
    assert entry["task"] == first and entry["seeds"] == 3
    assert (entry["one_arm_demonstrations"], entry["two_arm_demonstrations"]) == (2, 0)
    detail = entry["seeds_detail"]
    assert [r["seed"] for r in detail] == seeds
    assert [r["outcome"] for r in detail] == ["ok", "plan_failed", "ok"]
    assert [r["arms_moved"] for r in detail] == [["left"], None, ["left"]]
    assert detail[0]["displacement"]["left"] > 0.05 and detail[0]["displacement"]["right"] == 0.0
    assert detail[1]["displacement"] is None


def test_a_survey_renders_no_camera_unless_asked_and_measures_the_same(monkeypatch):
    # Every number the survey keeps comes from the joints; rendering changes none of them.
    monkeypatch.setattr(robotwin, "unstable_error", lambda: FakeUnstable)
    task = tasks.table()["click_bell"]
    plain_env = FakeTaskEnv(moves=("right",), plan_fails_on={2})
    rendered_env = FakeTaskEnv(moves=("right",), plan_fails_on={2})
    plain = survey.survey_task(plain_env, task, [1, 2, 3], FakeConfig())
    rendered = survey.survey_task(rendered_env, task, [1, 2, 3], FakeConfig(), images=True)

    assert plain_env.get_obs_calls == 0 and rendered_env.get_obs_calls == 2 * (
        rendered_env.expert_steps + 1
    )
    assert (plain.images, rendered.images) == (False, True)
    assert (plain.to_json()["images"], rendered.to_json()["images"]) == (False, True)
    measured = ("seed", "outcome", "frames", "arms_moved", "displacement")
    assert [{k: r[k] for k in measured} for r in plain.to_json()["seeds_detail"]] == [
        {k: r[k] for k in measured} for r in rendered.to_json()["seeds_detail"]
    ]


def _counting_envs(monkeypatch):
    """Every env the survey command builds, so a test can count what they rendered."""
    envs = []

    def load_task(name):
        envs.append(FakeTaskEnv(moves=("left",)))
        return envs[-1]

    monkeypatch.setattr(robotwin, "unstable_error", lambda: FakeUnstable)
    monkeypatch.setattr(robotwin, "SceneConfig", FakeConfig)
    monkeypatch.setattr(robotwin, "load_task", load_task)
    return envs


def test_the_survey_command_renders_nothing_by_default(tmp_path, monkeypatch, capsys):
    envs = _counting_envs(monkeypatch)
    out = tmp_path / "survey.json"
    assert cli.main(["survey", "--task", "click_bell", "--seeds", "2", "--json", str(out)]) == 0
    assert "click_bell: expert solved 2/2" in capsys.readouterr().out
    assert len(envs) == 1 and envs[0].get_obs_calls == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["images"] is False
    assert [(entry["task"], entry["images"]) for entry in payload["tasks"]] == [
        ("click_bell", False)
    ]


@pytest.mark.parametrize("arms", ["1", "2"])
def test_survey_without_a_task_measures_the_catalog(arms, tmp_path, monkeypatch, capsys):
    envs = _counting_envs(monkeypatch)
    out = tmp_path / "survey.json"
    assert cli.main(["survey", "--arms", arms, "--seeds", "1", "--json", str(out)]) == 0
    expected = [task.name for task in tasks.table().select(arms=arms)]
    payload = json.loads(out.read_text())
    assert [entry["task"] for entry in payload["tasks"]] == expected
    assert len(envs) == len(expected)
    assert all(env.get_obs_calls == 0 for env in envs)


def test_the_survey_command_renders_every_frame_with_images(tmp_path, monkeypatch, capsys):
    envs = _counting_envs(monkeypatch)
    out = tmp_path / "survey.json"
    argv = ["survey", "--task", "click_bell", "--seeds", "2", "--json", str(out), "--images"]
    assert cli.main(argv) == 0
    assert "click_bell: expert solved 2/2" in capsys.readouterr().out
    assert envs[0].get_obs_calls == 2 * (envs[0].expert_steps + 1)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["images"] is True
    assert [(entry["task"], entry["images"]) for entry in payload["tasks"]] == [
        ("click_bell", True)
    ]
