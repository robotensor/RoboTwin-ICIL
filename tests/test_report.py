import dataclasses
import re

import pytest

from robotwin_icil import camera_profiles, report, tasks
from robotwin_icil.records import SAME_SCENE, EpisodeRecord, RecordError, RunDir, Status
from test_records import manifest


def record(
    episode, task, status=Status.SCORED, success=True, attempts=1, rejections=None, steps=50
):
    scored = status is Status.SCORED
    return EpisodeRecord(
        episode=episode,
        evaluation_setting=SAME_SCENE,
        skill_category=tasks.table()[task].category,
        task=task,
        scene_seed=None if status is Status.REJECTED else 1000 + episode,
        status=status,
        success=success if scored else None,
        steps=steps if scored else 0,
        step_limit=400,
        demonstration_frames=30,
        expert_generation_attempts=attempts,
        rejections=rejections or {},
        scene_max_error=0.0,
        model="replay",
    )


RECORDS = [
    record(0, "place_object_basket", success=True, steps=40),
    record(1, "place_object_basket", success=False, steps=400),
    record(
        2,
        "place_a2b_left",
        success=True,
        steps=60,
        attempts=3,
        rejections={"unstable": 1, "plan_failed": 1},
    ),
    record(3, "stack_blocks_two", success=False, steps=400),
    record(
        4, "stack_blocks_two", status=Status.REJECTED, attempts=20, rejections={"expert_failed": 20}
    ),
    record(5, "click_bell", status=Status.INVALID),
]


@pytest.fixture
def built():
    return report.build(RECORDS, tasks.table())


def test_rejected_and_invalid_episodes_never_enter_the_denominator(built):
    assert (built.overall.successes, built.overall.episodes) == (2, 4)
    assert built.overall.value == 0.5


def test_rows_follow_table_order(built):
    assert list(built.by_category) == ["pick_and_place", "stacking", "press_push"]
    assert list(built.by_task["pick_and_place"]) == ["place_a2b_left", "place_object_basket"]
    assert built.by_task["pick_and_place"]["place_object_basket"].value == 0.5
    assert built.by_category["stacking"].value == 0.0


def test_a_category_with_nothing_scored_is_empty_not_failing(built):
    # click_bell's only episode was invalid: that is no evidence about the model either way.
    assert built.by_category["press_push"].value is None


def test_diagnostics_are_counted_separately(built):
    d = built.diagnostics
    assert (d.episodes, d.scored, d.rejected, d.invalid) == (6, 4, 1, 1)
    assert d.expert_attempts == 27
    assert d.expert_rejection_rate == pytest.approx(22 / 27)
    assert d.simulation_failure_rate == pytest.approx(1 / 27)
    assert d.mean_rollout_steps == pytest.approx(225.0)
    assert d.mean_successful_rollout_steps == pytest.approx(50.0)


def test_json_keeps_fractions(built):
    data = built.to_json()
    assert data["overall"]["success_rate"] == 0.5
    assert data["by_category"]["press_push"]["success_rate"] is None


def test_render_is_the_only_place_rates_become_percentages(built):
    text = report.render(built, None, tasks.table())
    assert "Overall Same Scene 1-Demo Success:  50.0%  (2/4)" in text
    assert "Pick and Place" in text and "place_object_basket" in text
    assert "not a model score" in text


def test_an_empty_run_reports_without_crashing():
    empty = report.build([], tasks.table())
    assert empty.overall.value is None and empty.by_category == {}
    assert "—" in report.render(empty, None, tasks.table())


def test_the_report_says_what_rejections_actually_were():
    # A run whose every seed failed the same way must name the failure, not just count it.
    broken = dataclasses.replace(
        record(
            0, "click_bell", status=Status.REJECTED, attempts=20, rejections={"expert_error": 20}
        ),
        rejection_details={"expert_error": "RuntimeError: CUDA error: out of memory"},
    )
    built = report.build([broken], tasks.table())
    assert built.to_json()["diagnostics"]["rejection_examples"] == {
        "expert_error": "RuntimeError: CUDA error: out of memory"
    }
    assert "e.g. expert_error: RuntimeError: CUDA error: out of memory" in report.render(
        built, None, tasks.table()
    )


V1 = tuple(task.name for task in tasks.table().suite("v1"))


def v1_run(profile="stock", **policy):
    return manifest(
        suite="v1",
        tasks=V1,
        policy={"policy": "icil", **policy},
        benchmark_config={"camera_profile": camera_profiles.get(profile).identity()},
    )


def header(run):
    return report.render(report.build([], tasks.table()), run, tasks.table())


def test_the_header_labels_the_adapter_and_its_version_and_the_camera_profile():
    text = header(v1_run("far_side", adapter="icil_policies.bpp", adapter_version="0.3.0"))
    assert "Adapter:                     icil_policies.bpp (adapter_version 0.3.0)" in text
    sha = camera_profiles.get("far_side").sha256
    assert f"Camera profile:              far_side (sha256 {sha[:12]})" in text
    assert "Adapter:                     — (adapter_version —)" in header(v1_run())


def test_a_manifest_from_before_camera_profiles_is_labelled_stock():
    legacy = manifest(benchmark_config={"task_config": "demo_clean"})
    sha = camera_profiles.get("stock").sha256
    assert f"Camera profile:              stock (sha256 {sha[:12]})" in header(legacy)


@pytest.mark.parametrize(
    "training_tasks, line",
    [
        (
            ["place_object_basket", "open_laptop"],
            "held out (evaluation tasks seen in training: 0/9)",
        ),
        ([*V1[:2], "place_object_basket"], "seen tasks (evaluation tasks seen in training: 2/9)"),
        (list(V1), "seen tasks (evaluation tasks seen in training: 9/9)"),
        ("unknown", "unknown (evaluation tasks seen in training: unknown)"),
    ],
)
def test_the_header_counts_the_evaluation_tasks_seen_in_training(training_tasks, line):
    assert f"Training regime:             {line}" in header(v1_run(training_tasks=training_tasks))


def test_a_policy_that_does_not_say_what_it_trained_on_is_unknown():
    run = v1_run()
    assert "training_tasks" not in run.policy
    assert report.seen_in_training(run) is None
    assert "evaluation tasks seen in training: unknown" in header(run)
    assert report.seen_in_training(v1_run(training_tasks=["click_bell"])) == (1, 9)


def test_the_header_adds_no_score():
    # The training labels describe the run; the only score is still the overall success rate.
    text = header(v1_run(training_tasks=list(V1)))
    assert "%" not in text.split("Overall Same Scene 1-Demo Success:")[0]


def run_dir(path, run, outcomes):
    """A run directory holding `run` and one scored episode per (task, success)."""
    directory = RunDir(path)
    directory.start(run)
    for episode, (task, success) in enumerate(outcomes):
        directory.append(record(episode, task, success=success))
    return path


MODEL = [("click_bell", True), ("click_bell", False), ("stack_blocks_two", False)]
ORACLE = [("click_bell", True), ("click_bell", True), ("stack_blocks_two", True)]
BPP = v1_run(adapter="icil_policies.bpp", adapter_version="0.3.0")
# The episode ids the reported run recorded.
SHARED = range(len(MODEL))


def model_report():
    return report.build([record(i, t, success=s) for i, (t, s) in enumerate(MODEL)], tasks.table())


def heading(text):
    """The line naming each column: the one under "By manipulation skill:"."""
    lines = text.splitlines()
    return lines[lines.index("By manipulation skill:") + 1]


def line_of(text, start):
    [line] = [line for line in text.splitlines() if line.startswith(start)]
    return line


def test_a_reference_prints_in_a_column_beside_the_run(tmp_path):
    path = run_dir(tmp_path / "replay", v1_run(policy="replay"), ORACLE)
    reference = report.load_reference(path, BPP, tasks.table(), episodes=SHARED)
    assert reference.label == "replay" and reference.run_dir == str(path.resolve())
    text = report.render(model_report(), BPP, tasks.table(), [reference])

    assert "Overall Same Scene 1-Demo Success:  33.3%  (1/3)" in text
    assert f"  replay   100.0%  (3/3)  {path.resolve()}" in text
    columns = heading(text)
    assert columns.split() == ["icil_policies.bpp", "replay"]
    # Whole cells, the percentage right-aligned in seven characters, each under its column's label.
    for start, ours, theirs in [
        ("  Press / Push", "  50.0%  (1/2)", " 100.0%  (2/2)"),
        ("    click_bell", "  50.0%  (1/2)", " 100.0%  (2/2)"),
        ("  Stacking", "   0.0%  (0/1)", " 100.0%  (1/1)"),
        ("    stack_blocks_two", "   0.0%  (0/1)", " 100.0%  (1/1)"),
    ]:
        row = line_of(text, start)
        assert row.index(ours) == columns.index("icil_policies.bpp")
        assert row.index(theirs) == columns.index("replay")
    assert reference.to_json()["overall"]["success_rate"] == 1.0


def test_two_references_and_a_row_only_one_run_has(tmp_path):
    first = run_dir(tmp_path / "replay", v1_run(policy="replay"), ORACLE)
    second = run_dir(
        tmp_path / "replay-again",
        v1_run(policy="replay"),
        [("click_bell", False), ("place_a2b_left", True)],
    )
    references = [
        report.load_reference(p, BPP, tasks.table(), episodes=SHARED) for p in (first, second)
    ]
    text = report.render(model_report(), BPP, tasks.table(), references)
    # Two runs of one policy are two columns, numbered.
    assert heading(text).split() == ["icil_policies.bpp", "replay", "replay", "#2"]
    assert "  replay #2  " in line_of(text, "  replay #2")
    # A task the reported run never ran is a row, with a dash where it has no rate.
    row = line_of(text, "    place_a2b_left")
    assert row.split()[1] == "—" and row.endswith("100.0%  (1/1)")
    assert line_of(text, "    stack_blocks_two").rstrip().endswith("—")
    assert "0.0%  (0/1)" in line_of(text, "    click_bell")


def test_a_reference_is_rated_over_the_episodes_both_runs_recorded(tmp_path):
    # An interrupted model run beside a finished oracle: only episode 0's scene is in both.
    path = run_dir(tmp_path / "replay", v1_run(policy="replay"), ORACLE)
    reference = report.load_reference(path, BPP, tasks.table(), episodes=[0])
    assert (reference.report.overall.successes, reference.report.overall.episodes) == (1, 1)
    assert list(reference.report.by_task) == ["press_push"]
    assert reference.to_json()["diagnostics"]["episodes"] == 1
    partial = report.build([record(0, "click_bell", success=False)], tasks.table())
    text = report.render(partial, BPP, tasks.table(), [reference])
    assert line_of(text, "  replay").endswith(f"100.0%  (1/1)  {path.resolve()}")
    assert "rated over the episodes both runs recorded" in text


def test_a_reference_that_recorded_only_some_episodes_says_so(tmp_path):
    path = run_dir(tmp_path / "replay", v1_run(policy="replay"), ORACLE[:1])
    reference = report.load_reference(path, BPP, tasks.table(), episodes=SHARED)
    text = report.render(model_report(), BPP, tasks.table(), [reference])
    # The run's own score still covers all of its episodes.
    assert "Overall Same Scene 1-Demo Success:  33.3%  (1/3)" in text
    assert line_of(text, "  replay").endswith(
        f"100.0%  (1/1)  {path.resolve()}  (only 1 of this run's 3 episodes)"
    )


def test_a_reference_that_shares_no_episodes_is_refused(tmp_path):
    path = run_dir(tmp_path / "replay", v1_run(policy="replay"), ORACLE)
    with pytest.raises(report.ReportError, match="recorded none of the reported run's episodes"):
        report.load_reference(path, BPP, tasks.table(), episodes=[3, 4])


def test_without_references_the_report_reads_as_before(tmp_path):
    text = report.render(model_report(), BPP, tasks.table())
    assert "Reference runs" not in text
    assert "    click_bell                50.0%  (1/2)" in text.splitlines()
    assert heading(text) == "  Stacking                     0.0%  (0/1)"


STOCK = camera_profiles.get("stock").identity()


@pytest.mark.parametrize(
    "changes, field",
    [
        ({"global_seed": 7}, "global_seed (7, not 42)"),
        ({"suite": None}, "suite (None, not 'v1')"),
        ({"tasks": V1[:3]}, "tasks"),
        ({"evaluation_setting": "different_object_pose"}, "evaluation_setting"),
        ({"max_expert_attempts": 5}, "max_expert_attempts (5, not 20)"),
        (
            {"benchmark_config": {"camera_profile": STOCK, "save_freq": 5}},
            "benchmark_config.save_freq",
        ),
        ({"robotwin_config": {"task_config": "demo_randomized"}}, "robotwin_config.task_config"),
        # RoboTwin's task code builds the scenes, so another checkout may build others.
        ({"robotwin_commit": "1" * 40}, f"robotwin_commit ('{'1' * 40}', not 'def')"),
        ({"robotwin_commit": "def-dirty"}, "robotwin_commit ('def-dirty', not 'def')"),
        (
            {"benchmark_config": {"camera_profile": camera_profiles.get("far_side").identity()}},
            "camera profile (far_side (sha256 "
            f"{camera_profiles.get('far_side').identity()['sha256'][:12]}), not stock",
        ),
        # A profile edited under its old name is another set of cameras.
        (
            {"benchmark_config": {"camera_profile": {"name": "stock", "sha256": "0" * 64}}},
            "camera profile (stock (sha256 000000000000)",
        ),
    ],
)
def test_a_reference_of_other_scenes_is_refused_naming_the_field(tmp_path, changes, field):
    path = run_dir(tmp_path / "other", dataclasses.replace(v1_run(policy="replay"), **changes), [])
    with pytest.raises(report.ReportError, match=re.escape(field)):
        report.load_reference(path, BPP, tasks.table(), episodes=SHARED)


def test_a_reference_may_differ_in_policy_benchmark_commit_and_machine(tmp_path):
    other = dataclasses.replace(
        v1_run(policy="replay"),
        benchmark_commit="0" * 40,
        episodes=18,
        environment={"gpu": "NVIDIA RTX A6000"},
        policy_config={"temperature": 0.5},
    )
    assert report.differences(BPP, other) == []


def test_a_reference_from_before_camera_profiles_is_a_stock_run(tmp_path):
    legacy = dataclasses.replace(v1_run(policy="replay"), benchmark_config={})
    assert report.load_reference(
        run_dir(tmp_path / "legacy", legacy, ORACLE), BPP, tasks.table(), episodes=SHARED
    )
    far_side = v1_run("far_side")
    with pytest.raises(report.ReportError, match="camera profile"):
        report.load_reference(tmp_path / "legacy", far_side, tasks.table(), episodes=SHARED)


def test_a_reference_that_failed_the_frozen_policy_audit_is_refused(tmp_path):
    path = run_dir(tmp_path / "learning", v1_run(policy="learning"), ORACLE)
    RunDir(path).fail_audit({"policy": "learning"})
    with pytest.raises(RecordError, match="failed the frozen-policy audit"):
        report.load_reference(path, BPP, tasks.table(), episodes=SHARED)
