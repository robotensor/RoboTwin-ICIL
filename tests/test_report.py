import dataclasses

import pytest

from robotwin_icil import camera_profiles, report, tasks
from robotwin_icil.records import SAME_SCENE, EpisodeRecord, Status
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
