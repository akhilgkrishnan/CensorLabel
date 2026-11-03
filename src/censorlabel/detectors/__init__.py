"""Object detectors. The pipeline depends only on this protocol, so tests can pass in fakes."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from censorlabel.domain import Detection


class ObjectDetector(Protocol):
    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]: ...


__all__ = ["ObjectDetector"]
