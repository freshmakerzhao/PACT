from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class StepType(Enum):
    FIRST = 0
    MID = 1
    LAST = 2


@dataclass
class TimeStep:
    step_type: StepType
    reward: float
    discount: Optional[float]
    observation: Any
