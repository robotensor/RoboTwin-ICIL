"""The static arms classifier, on small experts written in RoboTwin's idiom."""

from textwrap import dedent, indent

import pytest

from robotwin_icil import arms


def expert(play_once: str, helpers: str = "") -> str:
    """A task module with `play_once` (and helper methods) in RoboTwin's shape.

    `play_once` is the method body, `helpers` further methods; both are dedented first, so the
    fixtures read naturally. The first body line lands on line 9.
    """
    body = indent(dedent(play_once).strip("\n"), " " * 8)
    extra = indent(dedent(helpers).strip("\n"), " " * 4)
    return (
        "from ._base_task import Base_Task\n"
        "from .utils import ArmTag, Action\n"
        "\n"
        "class Fixture(Base_Task):\n"
        "    def load_actors(self):\n"
        '        self.arm_tag = ArmTag("left" if self.thing.get_pose().p[0] < 0 else "right")\n'
        "\n"
        "    def play_once(self):\n"
        f"{body}\n"
        "        return self.info\n"
        "\n"
        f"{extra}\n"
        "    def check_success(self):\n"
        "        return True\n"
    )


def classify(play_once: str, helpers: str = "") -> arms.Verdict:
    return arms.classify_source(expert(play_once, helpers), "fixture")


# --- one arm ----------------------------------------------------------------------------------


def test_an_arm_chosen_once_from_the_scene_is_one_arm():
    verdict = classify(
        """
        arm_tag = ArmTag("right" if self.thing.get_pose().p[0] > 0 else "left")
        self.move(self.grasp_actor(self.thing, arm_tag=arm_tag, pre_grasp_dis=0.1))
        for _ in range(3):
            self.move(self.move_by_displacement(arm_tag, z=0.05))
        self.move(self.place_actor(self.thing, arm_tag, target_pose=[0, 0, 0.8]))
        """
    )
    assert verdict.arms == arms.ONE
    assert "chosen(" in verdict.evidence


def test_a_fixed_single_arm_is_one_arm():
    verdict = classify(
        """
        arm_tag = ArmTag("left")
        self.move(self.grasp_actor(self.thing, arm_tag=arm_tag))
        self.move(self.open_gripper(arm_tag=arm_tag))
        """
    )
    assert verdict == arms.Verdict(arms.ONE, "one arm, left (line 10)")


def test_an_arm_chosen_in_load_actors_is_one_arm():
    verdict = classify(
        """
        self.move(self.grasp_actor(self.thing, arm_tag=self.arm_tag))
        self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=0.1))
        """
    )
    assert verdict.arms == arms.ONE
    assert "self.arm_tag" in verdict.evidence


def test_a_raw_action_on_the_chosen_arm_is_still_one_arm():
    # click_alarmclock builds its first motion by hand instead of through grasp_actor.
    verdict = classify(
        """
        arm_tag = ArmTag("right" if self.thing.get_pose().p[0] > 0 else "left")
        self.move((ArmTag(arm_tag), [Action(arm_tag, "move", [0, 0, 1, 0, 0, 0, 1])]))
        self.move(self.move_by_displacement(arm_tag, z=-0.05))
        """
    )
    assert verdict.arms == arms.ONE


# --- two arms ---------------------------------------------------------------------------------


def test_both_arms_moved_together_is_two_arms():
    verdict = classify(
        """
        left, right = ArmTag("left"), ArmTag("right")
        self.move(
            self.grasp_actor(self.pot, left, contact_point_id=0),
            self.grasp_actor(self.pot, right, contact_point_id=1),
        )
        """
    )
    assert verdict.arms == arms.TWO
    assert "together" in verdict.evidence


@pytest.mark.parametrize(
    "helper",
    [
        "self.together_close_gripper()",
        "self.together_open_gripper(save_freq=None)",
        "self.together_move_to_pose(self.left_pose, self.right_pose)",
    ],
)
def test_a_base_helper_that_drives_both_arms_is_two_arms(helper):
    # Base_Task's together_* helpers move both arms without going through self.move.
    verdict = classify(
        f"""
        {helper}
        self.move(self.grasp_actor(self.thing, arm_tag=ArmTag("left")))
        """
    )
    assert verdict == arms.Verdict(arms.TWO, "both arms move together at line 9")
    # On their own they are still a motion, so the expert is not read as moving nothing.
    assert classify(helper).arms == arms.TWO


def test_a_fixed_left_and_a_fixed_right_acting_in_turn_is_two_arms():
    verdict = classify(
        """
        grasp_arm, hang_arm = ArmTag("left"), ArmTag("right")
        self.move(self.grasp_actor(self.mug, arm_tag=grasp_arm))
        self.move(self.back_to_origin(grasp_arm), self.grasp_actor(self.mug, arm_tag=hang_arm))
        self.move(self.place_actor(self.mug, arm_tag=hang_arm, target_pose=[0, 0, 1]))
        """
    )
    assert verdict.arms == arms.TWO
    assert "left" in verdict.evidence and "right" in verdict.evidence


def test_a_chosen_arm_and_its_opposite_is_two_arms():
    verdict = classify(
        """
        arm_tag = ArmTag("right" if self.thing.get_pose().p[0] > 0 else "left")
        self.move(self.grasp_actor(self.thing, arm_tag=arm_tag))
        self.move(self.grasp_actor(self.drawer, arm_tag=arm_tag.opposite))
        """
    )
    assert verdict.arms == arms.TWO
    assert "opposite(" in verdict.evidence


def test_an_opposite_bound_to_a_name_is_two_arms():
    verdict = classify(
        """
        grasp_arm = ArmTag("left" if self.box.get_pose().p[0] < 0 else "right")
        place_arm = grasp_arm.opposite
        self.move(self.grasp_actor(self.box, arm_tag=grasp_arm))
        self.move(self.back_to_origin(grasp_arm), self.place_actor(self.box, arm_tag=place_arm))
        """
    )
    assert verdict.arms == arms.TWO


def test_a_chosen_arm_handing_over_to_a_fixed_arm_is_two_arms():
    verdict = classify(
        """
        grasp_arm = ArmTag("left" if self.bin.get_pose().p[0] < 0 else "right")
        pour_arm = ArmTag("left")
        if grasp_arm == "right":
            self.move(self.grasp_actor(self.bin, arm_tag=grasp_arm))
            self.move(self.back_to_origin(grasp_arm), self.grasp_actor(self.bin, arm_tag=pour_arm))
        else:
            self.move(self.grasp_actor(self.bin, arm_tag=pour_arm))
        self.move(self.move_by_displacement(arm_tag=pour_arm, z=0.1))
        """
    )
    assert verdict.arms == arms.TWO


def test_a_literal_arm_in_a_loop_alongside_a_chosen_one_is_two_arms():
    # put_bottles_dustbin: right-side bottles are handed to the left arm inside the loop.
    verdict = classify(
        """
        for bottle in self.bottles:
            arm_tag = ArmTag("left" if bottle.get_pose().p[0] < 0 else "right")
            if arm_tag == "left":
                self.move(self.grasp_actor(bottle, arm_tag=arm_tag))
            else:
                self.move(self.grasp_actor(bottle, arm_tag=arm_tag), self.back_to_origin("left"))
                self.move(self.grasp_actor(bottle, arm_tag="left"))
                self.move(self.open_gripper(ArmTag("right")))
            self.move(self.open_gripper("left"))
        """
    )
    assert verdict.arms == arms.TWO


def test_a_helper_written_as_a_nested_function_is_read():
    # place_bread_basket does its two-arm work in a def inside play_once.
    verdict = classify(
        """
        def remove():
            self.move(
                self.grasp_actor(self.bread[0], arm_tag="left"),
                self.grasp_actor(self.bread[1], arm_tag="right"),
            )
        remove()
        """
    )
    assert verdict.arms == arms.TWO


def test_a_nested_function_reads_the_arm_its_play_once_chose():
    verdict = classify(
        """
        arm_tag = ArmTag("left" if self.tray.get_pose().p[0] < 0 else "right")
        def hold():
            self.move(self.grasp_actor(self.tray, arm_tag=arm_tag.opposite))
        self.move(self.grasp_actor(self.thing, arm_tag=arm_tag))
        hold()
        """
    )
    assert verdict.arms == arms.TWO
    assert "opposite(chosen(" in verdict.evidence


def test_an_arm_handed_to_a_helper_is_read_at_each_call_site():
    # The helper moves whatever arm it is given, and the call sites give it both.
    verdict = classify(
        """
        arm_tag = ArmTag("left" if self.box.get_pose().p[0] < 0 else "right")
        self.pick(self.box, arm_tag)
        self.pick(self.tray, arm_tag.opposite)
        """,
        """
            def pick(self, thing, arm):
                self.move(self.grasp_actor(thing, arm_tag=arm, pre_grasp_dis=0.1))
                self.move(self.move_by_displacement(arm, z=0.1))
        """,
    )
    assert verdict.arms == arms.TWO
    assert "opposite(chosen(" in verdict.evidence


def test_a_helper_given_a_fixed_arm_at_each_call_site_is_two_arms():
    verdict = classify(
        """
        self.pick(self.box, ArmTag("left"))
        self.pick(self.tray, arm=ArmTag("right"))
        """,
        """
            def pick(self, thing, arm):
                self.move(self.grasp_actor(thing, arm_tag=arm))
        """,
    )
    assert verdict.arms == arms.TWO
    assert "left" in verdict.evidence and "right" in verdict.evidence


def test_a_helper_parameter_default_counts_where_a_call_site_omits_it():
    verdict = classify(
        """
        arm_tag = ArmTag("left" if self.box.get_pose().p[0] < 0 else "right")
        self.pick(self.box, arm_tag)
        self.pick(self.tray)
        """,
        """
            def pick(self, thing, arm=ArmTag("left")):
                self.move(self.grasp_actor(thing, arm_tag=arm))
        """,
    )
    assert verdict.arms == arms.TWO


def test_a_helper_given_the_same_arm_at_each_call_site_is_one_arm():
    verdict = classify(
        """
        arm_tag = ArmTag("left" if self.box.get_pose().p[0] < 0 else "right")
        self.pick(self.box, arm_tag)
        self.pick(self.tray, arm=arm_tag)
        """,
        """
            def pick(self, thing, arm=ArmTag("left")):
                self.move(self.grasp_actor(thing, arm_tag=arm))
                for _ in range(2):
                    self.move(self.move_by_displacement(arm, z=0.02))
        """,
    )
    assert verdict.arms == arms.ONE


# --- switching --------------------------------------------------------------------------------


def test_an_arm_chosen_per_object_in_a_repeated_helper_is_switching():
    verdict = classify(
        """
        self.last_gripper = None
        self.pick_and_place_block(self.block1)
        self.pick_and_place_block(self.block2)
        """,
        """
            def pick_and_place_block(self, block):
                arm_tag = ArmTag("left" if block.get_pose().p[0] < 0 else "right")
                if self.last_gripper is not None and self.last_gripper != arm_tag:
                    self.move(
                        self.grasp_actor(block, arm_tag=arm_tag),
                        self.back_to_origin(arm_tag=arm_tag.opposite),
                    )
                else:
                    self.move(self.grasp_actor(block, arm_tag=arm_tag))
                self.move(self.place_actor(block, arm_tag=arm_tag, target_pose=[0, 0, 1]))
                self.last_gripper = arm_tag
        """,
    )
    assert verdict.arms == arms.SWITCHING
    assert "pick_and_place_block, called 2 times" in verdict.evidence


def test_an_arm_chosen_per_object_in_a_loop_is_switching():
    verdict = classify(
        """
        for block in self.blocks:
            arm_tag = ArmTag("left" if block.get_pose().p[0] < 0 else "right")
            self.move(self.grasp_actor(block, arm_tag=arm_tag))
            self.move(self.place_actor(block, arm_tag=arm_tag, target_pose=[0, 0, 1]))
        """
    )
    assert verdict.arms == arms.SWITCHING
    assert "in a loop" in verdict.evidence


def test_an_arm_chosen_per_object_between_two_tags_is_switching():
    verdict = classify(
        """
        for block in self.blocks:
            arm_tag = ArmTag("left") if block.get_pose().p[0] < 0 else ArmTag("right")
            self.move(self.grasp_actor(block, arm_tag=arm_tag))
            self.move(self.place_actor(block, arm_tag=arm_tag, target_pose=[0, 0, 1]))
        """
    )
    assert verdict.arms == arms.SWITCHING


@pytest.mark.parametrize(
    "choice", ["self.arms[block.get_name()]", "self.arm_for(block)", "block.arm"]
)
def test_an_arm_looked_up_per_object_is_switching(choice):
    # However the per-object arm is computed — a lookup, a helper's answer, an attribute — it
    # is a choice made inside the loop, and that is what makes the expert switch.
    verdict = classify(
        f"""
        for block in self.blocks:
            arm_tag = {choice}
            self.move(self.grasp_actor(block, arm_tag=arm_tag))
        """
    )
    assert verdict.arms == arms.SWITCHING
    assert "at line 10 in a loop" in verdict.evidence


def test_a_side_chosen_per_object_that_is_not_an_arm_does_not_switch():
    verdict = classify(
        """
        arm_tag = ArmTag("left")
        for thing in self.things:
            side = "left" if thing.get_pose().p[0] < 0 else "right"
            self.move(self.grasp_actor(thing, arm_tag=arm_tag))
            self.move(self.place_actor(thing, arm_tag=arm_tag, target_pose=self.targets[side]))
        """
    )
    assert verdict == arms.Verdict(arms.ONE, "one arm, left (line 12)")


def test_a_helper_called_from_a_loop_is_repeated():
    verdict = classify(
        """
        for block in self.blocks:
            self.stack(block)
        """,
        """
            def stack(self, block):
                arm_tag = ArmTag("left" if block.get_pose().p[0] < 0 else "right")
                self.move(self.grasp_actor(block, arm_tag=arm_tag))
        """,
    )
    assert verdict.arms == arms.SWITCHING


def test_a_helper_called_once_that_chooses_once_is_one_arm():
    verdict = classify(
        """
        self.pick(self.thing)
        """,
        """
            def pick(self, thing):
                arm_tag = ArmTag("left" if thing.get_pose().p[0] < 0 else "right")
                self.move(self.grasp_actor(thing, arm_tag=arm_tag))
                for _ in range(2):
                    self.move(self.move_by_displacement(arm_tag, z=0.02))
        """,
    )
    assert verdict.arms == arms.ONE


def test_a_helper_that_moves_both_arms_wins_over_switching():
    verdict = classify(
        """
        self.stack(self.block1)
        self.stack(self.block2)
        """,
        """
            def stack(self, block):
                arm_tag = ArmTag("left" if block.get_pose().p[0] < 0 else "right")
                self.move(
                    self.grasp_actor(block, arm_tag=arm_tag),
                    self.grasp_actor(self.tray, arm_tag=arm_tag.opposite),
                )
        """,
    )
    assert verdict.arms == arms.TWO


# --- unreadable experts -----------------------------------------------------------------------


def test_a_module_without_play_once_is_an_error():
    with pytest.raises(arms.ArmsError, match="no class defines play_once"):
        arms.classify_source("class Nothing:\n    def load_actors(self):\n        pass\n", "x")


def test_an_expert_that_moves_no_arm_is_an_error():
    with pytest.raises(arms.ArmsError, match="fixture: play_once moves no arm"):
        classify(
            """
            self.delay(3)
            """
        )


# --- a directory of tasks ---------------------------------------------------------------------


def test_classify_arms_reads_every_task_file_and_skips_the_rest(tmp_path):
    (tmp_path / "one.py").write_text(
        expert(
            """
        self.move(self.grasp_actor(self.thing, arm_tag=ArmTag("left")))
        """
        )
    )
    (tmp_path / "two.py").write_text(
        expert(
            """
        self.move(
            self.grasp_actor(self.a, arm_tag=ArmTag("left")),
            self.grasp_actor(self.b, arm_tag=ArmTag("right")),
        )
        """
        )
    )
    for stem in arms.NON_TASK_STEMS:
        (tmp_path / f"{stem}.py").write_text("x = 1\n")
    assert arms.classify_arms(tmp_path) == {"one": arms.ONE, "two": arms.TWO}
    with pytest.raises(arms.ArmsError, match="init the submodule"):
        arms.classify_arms(tmp_path / "missing")
