"""Small observation-conditioned reference adapter, not a trained ICIL model.

Select a nearby demonstration transition using three-view RGB and joint command state.
This exercises the public camera/action interface for both duel and standard evaluation.
"""

from __future__ import annotations

import numpy as np

from ..dataset import CAMERAS
from ..demo import Demonstration
from ..policy import ICILPolicy, Observation, PolicyError


def features(images: dict[str, np.ndarray], qpos: np.ndarray, rgb_weight: float) -> np.ndarray:
    values = [np.asarray(qpos, np.float32) / np.sqrt(len(qpos))]
    for name in CAMERAS.values():
        if name not in images:
            raise PolicyError(f"nearest reference requires camera {name}")
        image = images[name]
        if image.ndim != 3 or image.shape[2] != 3:
            raise PolicyError(f"{name}: expected RGB image")
        y = np.linspace(0, image.shape[0] - 1, 12).astype(int)
        x = np.linspace(0, image.shape[1] - 1, 16).astype(int)
        thumbnail = image[y[:, None], x].astype(np.float32).ravel() / 255
        values.append(thumbnail * rgb_weight / np.sqrt(3 * len(thumbnail)))
    return np.concatenate(values)


class NearestPolicy(ICILPolicy):
    name = "nearest-reference"

    def __init__(self, window: str | int = 16, rgb_weight: str | float = 0.25):
        super().__init__()
        self.window, self.rgb_weight = int(window), float(rgb_weight)
        if self.window < 1 or not np.isfinite(self.rgb_weight) or self.rgb_weight < 0:
            raise PolicyError("window must be positive and RGB weight finite and nonnegative")

    def describe(self):
        return {
            **super().describe(),
            "window": self.window,
            "rgb_weight": self.rgb_weight,
            "training": "none; observation-conditioned reference only",
        }

    def _reset(self):
        self._features = None
        self._actions = None
        self._cursor = 0

    def _set_demonstration(self, demonstration: Demonstration):
        self._actions = demonstration.actions().astype(np.float32)
        self._features = np.stack(
            [
                features(frame.images, frame.qpos, self.rgb_weight)
                for frame in demonstration.frames[:-1]
            ]
        )

    def _act(self, observation: Observation):
        current = features(observation.images, observation.qpos, self.rgb_weight)
        start = min(self._cursor, len(self._actions) - 1)
        stop = min(start + self.window, len(self._actions))
        distances = np.sum((self._features[start:stop] - current) ** 2, axis=1)
        index = start + int(np.argmin(distances))
        self._cursor = index + 1
        return self._actions[index]
