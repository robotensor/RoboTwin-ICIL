"""Policies for the policy-server tests, imported by `robotwin_icil.serve` in a child process.

The tests put this directory on the child's PYTHONPATH; each class is served as
`served_policies:Name`.
"""

import os
import sys
import time

import numpy as np

from robotwin_icil.policy import ReplayPolicy


class Echo(ReplayPolicy):
    """Replays, and reports what it was handed: arguments, seeds, the last observation."""

    name = "echo"

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.seeds = []
        self.acts = 0
        self.last = None

    def seed(self, seed):
        self.seeds.append(seed)

    def _set_demonstration(self, demonstration):
        super()._set_demonstration(demonstration)
        self.frames = len(demonstration)

    def _act(self, observation):
        self.acts += 1
        self.last = observation
        return super()._act(observation)

    def describe(self):
        return {**super().describe(), "kwargs": self.kwargs}

    def environment(self):
        return {
            "python": sys.version.split()[0],
            "authkey": str("ROBOTWIN_ICIL_AUTHKEY" in os.environ),
        }

    def episode_info(self):
        return {
            "seeds": self.seeds,
            "acts": self.acts,
            "frames": self.frames,
            "time_s": self.last.time_s,
            "images": sorted(self.last.images),
            "gripper_joints": self.last.gripper_joints["left"].tolist(),
        }


class Crashing(ReplayPolicy):
    """Replays, then dies without a word on its `crash_at`-th action, as a segfault would."""

    name = "crashing"

    def __init__(self, crash_at=3):
        super().__init__()
        self.crash_at = crash_at
        self.acts = 0

    def _act(self, observation):
        self.acts += 1
        if self.acts == self.crash_at:
            print("crashing now", file=sys.stderr, flush=True)
            os._exit(3)
        return super()._act(observation)


class Hanging(ReplayPolicy):
    """Never answers an action."""

    name = "hanging"

    def _act(self, observation):
        print("hanging in act", file=sys.stderr, flush=True)
        time.sleep(3600)


class WrongShape(ReplayPolicy):
    """Returns five numbers where a qpos action has fourteen."""

    name = "wrong_shape"

    def _act(self, observation):
        return np.zeros(5)


class Unencodable(ReplayPolicy):
    """Reports what the protocol cannot carry."""

    name = "unencodable"

    def episode_info(self):
        return {"handle": object()}


class BrokenConstructor(ReplayPolicy):
    name = "broken"

    def __init__(self):
        raise RuntimeError("checkpoint not found: /models/missing.pt")
