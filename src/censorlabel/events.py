"""Events the pipeline emits so any front end (CLI, GUI, tests) can observe a run."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from censorlabel.domain import Segment

Stage = Literal["loading", "analyzing", "rendering"]


@dataclass(frozen=True)
class Progress:
    stage: Stage
    fraction: float


@dataclass(frozen=True)
class Log:
    message: str
    level: str = "info"


@dataclass(frozen=True)
class SegmentFound:
    segment: Segment


@dataclass(frozen=True)
class Preview:
    frame: np.ndarray


@dataclass(frozen=True)
class Finished:
    output: Path | None
    segments: tuple[Segment, ...]
    cancelled: bool = False


Event = Progress | Log | SegmentFound | Preview | Finished
Listener = Callable[[Event], None]
