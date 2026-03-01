"""Camera placeholder utilities.

Use these functions to return a fixed dummy image until a real camera is integrated.
"""

from __future__ import annotations

import numpy as np


def make_dummy_image(height: int = 480, width: int = 640, channels: int = 3) -> np.ndarray:
    """Return a black image placeholder (uint8)."""
    return np.zeros((height, width, channels), dtype=np.uint8)


def make_dummy_image_dict(camera_names: list[str]) -> dict[str, np.ndarray]:
    """Return a dict of dummy images keyed by camera name."""
    return {name: make_dummy_image() for name in camera_names}
