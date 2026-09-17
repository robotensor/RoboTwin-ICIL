import json
from dataclasses import replace

import pytest

from robotwin_icil import cli, runner, source_sha256, standard, tasks
from robotwin_icil.policy import ReplayPolicy
from robotwin_icil.records import SAME_SCENE, EpisodeRecord, RunDir, RunManifest, Status, write_json
from robotwin_icil.robotwin import SceneConfig

NAMES = tuple(tasks.table().tasks)


def test_profile_is_robotwins_evaluation_on_every_task():
    meta = standard.metadata()
    assert meta["tasks"] == list(NAMES) and len(NAMES) == 50
    assert meta["seed_stream"] == "robotwin"
    assert standard.SETTINGS == {"clean": "demo_clean", "randomized": "demo_randomized"}
    assert (standard.SEEDS, standard.EPISODES_PER_TASK) == ((0,), 100)
    assert "dataset_revision" not in meta and "held_out_tasks" not in meta


def standard_spec(tmp_path, **changes):
    params = dict(
        run_dir=tmp_path,
        tasks=tuple(tasks.table()[n] for n in NAMES),
        episodes=50,
        global_seed=0,
        max_expert_attempts=standard.MAX_EXPERT_ATTEMPTS,
        seed_stream="robotwin",
        standard=standard.metadata(),
    )
    params.update(changes)
    return runner.RunSpec(**params)


def test_standard_rejects_changed_tasks_allocation_seeds_robot_and_config(tmp_path):
    for task_config in ("demo_clean", "demo_randomized"):
        standard.validate_run(
            standard_spec(tmp_path), SceneConfig(task_config=task_config, embodiment="aloha-agilex")
        )
    config = SceneConfig(embodiment="aloha-agilex")
    for change in (
        dict(episodes=51),
        dict(tasks=(tasks.table()["click_bell"],)),
        dict(max_expert_attempts=20),
        dict(seed_stream="independent"),
        dict(standard={"profile": "another"}),
    ):
        with pytest.raises(standard.StandardError):
            standard.validate_run(standard_spec(tmp_path, **change), config)
    for changed in (
        SceneConfig(embodiment="franka-panda"),
        SceneConfig(),
        SceneConfig(task_config="demo_other", embodiment="aloha-agilex"),
        SceneConfig(embodiment="aloha-agilex", save_freq=10),
        SceneConfig(embodiment="aloha-agilex", overrides={"save_freq": 15}),
    ):
        with pytest.raises(standard.StandardError):
            standard.validate_run(standard_spec(tmp_path), changed)


def populate(root, budget=1, seeds=(0,), settings=("clean", "randomized"), fail_every=50):
    policy = ReplayPolicy().describe()
    p = standard.plan(policy, list(settings), list(seeds), budget)
    standard.start(root, p)
    for setting in settings:
        for seed in seeds:
            rd = RunDir(standard.run_dir(root, setting, seed))
            rd.start(
                RunManifest(
                    global_seed=seed,
                    evaluation_setting=SAME_SCENE,
                    tasks=NAMES,
                    episodes=budget * 50,
                    max_expert_attempts=standard.MAX_EXPERT_ATTEMPTS,
                    seed_stream="robotwin",
                    policy=policy,
                    benchmark_commit="benchmark",
                    robotwin_commit="simulator",
                    benchmark_config={
                        "task_config": standard.SETTINGS[setting],
                        "save_freq": 15,
                        "head_camera": None,
                        "overrides": {},
                        "embodiment": "aloha-agilex",
                        "standard": p["standard"],
                        "source_sha256": source_sha256(),
                    },
                    robotwin_config={"embodiment_name": "aloha-agilex"},
                )
            )
            for index in range(budget * 50):
                task = tasks.table()[NAMES[index % 50]]
                rd.append(
                    EpisodeRecord(
                        episode=index,
                        evaluation_setting=SAME_SCENE,
                        skill_category=task.category,
                        task=task.name,
                        scene_seed=100000 * (1 + seed) + index,
                        status=Status.SCORED,
                        success=(index % fail_every != 0),
                        steps=10,
                        step_limit=100,
                        demonstration_frames=3,
                        expert_generation_attempts=1,
                        rejections={},
                        scene_max_error=0,
                        model="replay",
                        embodiment="aloha-agilex",
                    )
                )
    return p


def test_mean_over_all_tasks_and_reduced_budget_is_provisional(tmp_path):
    populate(tmp_path)
    result = standard.score(tmp_path)
    assert result["complete"]
    for setting in ("clean", "randomized"):
        row = result["settings"][setting]
        assert row["complete"] and row["official_score"] is None
        assert row["success_rate"] == pytest.approx(49 / 50)
        assert all(entry["scored"] == entry["planned"] == 1 for entry in row["by_task"].values())
        assert set(row["by_category"]) == set(tasks.table().categories)


def test_full_protocol_scores_each_setting(tmp_path):
    populate(tmp_path, budget=100, fail_every=10)
    result = standard.score(tmp_path)
    for setting, task_config in standard.SETTINGS.items():
        row = result["settings"][setting]
        assert row["task_config"] == task_config and row["official_protocol"]
        assert row["official_score"] == pytest.approx(0.9)
        assert row["diagnostics"]["scored"] == 5000


def test_other_seed_groups_are_not_the_official_budget(tmp_path):
    populate(tmp_path, budget=100, seeds=(0, 1), settings=("clean",))
    row = standard.score(tmp_path)["settings"]["clean"]
    assert row["complete"] and row["official_score"] is None


def test_missing_setting_and_harness_invalids_are_not_hidden(tmp_path):
    populate(tmp_path)
    (tmp_path / "randomized" / "seed_0" / "manifest.json").unlink()
    (tmp_path / "randomized" / "seed_0" / "episodes.jsonl").unlink()
    rd = RunDir(tmp_path / "clean" / "seed_0")
    records = rd.records()
    records[0] = replace(records[0], status=Status.INVALID, success=None)
    rd.episodes_path.write_text("".join(json.dumps(r.to_json()) + "\n" for r in records))
    result = standard.score(tmp_path)
    assert not result["complete"]
    clean, randomized = result["settings"]["clean"], result["settings"]["randomized"]
    assert not clean["complete"] and clean["success_rate"] is None
    assert clean["by_task"][NAMES[0]]["scored"] == 0
    assert clean["diagnostics"]["invalid"] == 1
    assert randomized["by_seed"] == {"0": {"recorded": 0, "scored": 0}}


@pytest.mark.parametrize(
    "field,value",
    [
        ("tasks", ["click_bell"]),
        ("policy", {"policy": "dummy"}),
        ("seed_stream", "independent"),
        ("episodes", 51),
    ],
)
def test_a_different_run_cannot_be_aggregated(tmp_path, field, value):
    populate(tmp_path)
    path = tmp_path / "clean" / "seed_0" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    write_json(path, manifest)
    with pytest.raises(standard.StandardError):
        standard.score(tmp_path)


def test_a_setting_run_under_the_other_task_config_is_rejected(tmp_path):
    populate(tmp_path)
    path = tmp_path / "clean" / "seed_0" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["benchmark_config"]["task_config"] = "demo_randomized"
    write_json(path, manifest)
    with pytest.raises(standard.StandardError, match="nonstandard scene"):
        standard.score(tmp_path)


def test_out_of_profile_episode_is_rejected(tmp_path):
    populate(tmp_path)
    rd = RunDir(tmp_path / "clean" / "seed_0")
    records = rd.records()
    records[0] = replace(records[0], task="click_bell")
    rd.episodes_path.write_text("".join(json.dumps(r.to_json()) + "\n" for r in records))
    with pytest.raises(standard.StandardError, match="out-of-profile"):
        standard.score(tmp_path)


def test_cli_standard_defaults_and_no_task_or_robot_override(tmp_path):
    parser = cli.build_parser()
    parsed = parser.parse_args(["standard-eval", "--policy", "replay", "--run-dir", str(tmp_path)])
    assert parsed.episodes_per_task == 100 and parsed.seed is None and parsed.setting is None
    for flag, value in (
        ("--task", "click_bell"),
        ("--embodiment", "franka-panda"),
        ("--setting", "hard"),
        ("--dataset-revision", "a" * 40),
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["standard-eval", "--policy", "replay", "--run-dir", str(tmp_path), flag, value]
            )
    argv = ["standard-eval", "--policy", "replay", "--run-dir", str(tmp_path)]
    assert cli.main([*argv, "--episodes-per-task", "0"]) == 1
    assert cli.main([*argv, "--seed", "1", "--seed", "1"]) == 1
    assert not (tmp_path / "standard.json").exists()


def test_standard_plan_is_resumable_only_with_identical_identity(tmp_path):
    p = standard.plan({"policy": "replay"}, ["clean"], [0], 1)
    standard.start(tmp_path, p)
    standard.start(tmp_path, p)
    with pytest.raises(standard.StandardError):
        standard.start(tmp_path, standard.plan({"policy": "dummy"}, ["clean"], [0], 1))
    with pytest.raises(standard.StandardError):
        standard.start(tmp_path, standard.plan({"policy": "replay"}, ["randomized"], [0], 1))


def test_plan_validation():
    with pytest.raises(standard.StandardError):
        standard.plan({}, ["clean"], [0, 0], 1)
    with pytest.raises(standard.StandardError):
        standard.plan({}, ["hard"], [0], 1)
    with pytest.raises(standard.StandardError):
        standard.plan({}, [], [0], 1)


def test_standard_cli_runs_both_settings_on_robotwins_seed_stream(tmp_path, monkeypatch):
    calls = []

    def run(spec, policy, config):
        standard.validate_run(spec, config)
        assert spec.episodes == 50 and spec.seed_stream == "robotwin"
        calls.append((config.task_config, spec.global_seed, spec.run_dir.relative_to(tmp_path)))
        return []

    monkeypatch.setattr(runner, "run", run)
    argv = ["standard-eval", "--policy", "replay", "--episodes-per-task", "1"]
    assert cli.main([*argv, "--run-dir", str(tmp_path)]) == 0
    assert [(c, s, str(d)) for c, s, d in calls] == [
        ("demo_clean", 0, "clean/seed_0"),
        ("demo_randomized", 0, "randomized/seed_0"),
    ]
    result = json.loads((tmp_path / "standard-report.json").read_text())
    assert not result["complete"]
    assert result["settings"]["clean"]["by_task"][NAMES[0]]["planned"] == 1


def test_malformed_standard_plan_has_a_clear_cli_error(tmp_path):
    (tmp_path / "standard.json").write_text('{"standard":')
    assert cli.main(["standard-report", str(tmp_path)]) == 1


def test_standard_report_text_names_each_setting(tmp_path, capsys):
    populate(tmp_path)
    assert cli.main(["standard-report", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "clean (demo_clean)" in out and "randomized (demo_randomized)" in out
    assert "success rate (mean over 50 tasks): 98.0%" in out
