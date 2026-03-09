from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SceneState:
    box_pose: Optional[np.ndarray] = None

