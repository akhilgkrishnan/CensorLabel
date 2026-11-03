"""Core domain types. No OpenCV or I/O in here, so everything is cheap to test."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class WarningKind(str, Enum):
    """Statutory warning to overlay. Values match the image file names in Images/statutory/<lang>/."""

    HELMET = "helmet"
    SMOKE = "smoke"
    ALCOHOL = "alcohol"
    SMOKE_ALCOHOL = "smokealcohol"
    SEATBELT = "seatbelt"

    @property
    def description(self) -> str:
        return _DESCRIPTIONS[self]


_DESCRIPTIONS = {
    WarningKind.HELMET: "Riding without helmet",
    WarningKind.SMOKE: "Smoking",
    WarningKind.ALCOHOL: "Alcohol consumption",
    WarningKind.SMOKE_ALCOHOL: "Alcohol and smoking",
    WarningKind.SEATBELT: "Driving without seatbelt",
}


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in pixel coordinates (top-left corner + size)."""

    x: int
    y: int
    w: int
    h: int

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> Box:
        return cls(int(x1), int(y1), int(x2 - x1), int(y2 - y1))

    @property
    def area(self) -> int:
        return max(self.w, 0) * max(self.h, 0)

    def top(self, fraction: float) -> Box:
        """The top `fraction` of this box, e.g. the head region of a person box."""
        return Box(self.x, self.y, self.w, max(int(self.h * fraction), 1))

    def intersection(self, other: Box) -> int:
        dx = min(self.x + self.w, other.x + other.w) - max(self.x, other.x)
        dy = min(self.y + self.h, other.y + other.h) - max(self.y, other.y)
        return dx * dy if dx > 0 and dy > 0 else 0

    def containment(self, other: Box) -> float:
        """Fraction of the smaller box that lies inside the other one (0..1)."""
        smaller = min(self.area, other.area)
        return self.intersection(other) / smaller if smaller else 0.0


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: Box


@dataclass(frozen=True)
class Segment:
    """A span of the video, in seconds, that gets one warning banner."""

    kind: WarningKind
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start
