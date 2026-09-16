import dataclasses

import pytest

from robotwin_icil import report, tasks
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
        embodiment="aloha-agilex",
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


def test_render_says_when_a_run_was_one_arm_only(built):
    from test_records import manifest

    tasks_line = "Tasks:                       " + ", ".join(manifest().tasks)
    assert tasks_line + "\n" in report.render(built, manifest(), tasks.table())
    one_arm = report.render(built, manifest(arms="1"), tasks.table())
    assert tasks_line + " (one-arm tasks only)" in one_arm


def test_an_empty_run_reports_without_crashing():
    empty = report.build([], tasks.table())
    assert empty.overall.value is None and empty.by_category == {}
    assert "—" in report.render(empty, None, tasks.table())


def test_a_run_recorded_before_it_could_choose_its_robot_still_names_it(built):
    # Manifests from before #81 have no `embodiment_name`, but every one carries RoboTwin's
    # `embodiment` list, which names the robot on its own; only a manifest with neither says "?".
    def embodiment_line(robotwin_config):
        text = report.render(built, manifest(robotwin_config=robotwin_config), tasks.table())
        return next(line for line in text.splitlines() if line.startswith("Embodiment:"))

    assert embodiment_line({"embodiment": ["aloha-agilex"]}).endswith(" aloha-agilex")
    assert embodiment_line({"embodiment": ["franka-panda", "franka-panda", 0.8]}).endswith(
        " franka-panda"
    )
    assert embodiment_line({"task_config": "demo_clean"}).endswith(" ?")


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
