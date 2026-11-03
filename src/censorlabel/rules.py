"""Pure per-frame rules: which violations a set of detections shows."""

from __future__ import annotations

from censorlabel.config import (
    ALCOHOL_OBJECTS,
    HELMET,
    MOTORCYCLE,
    NO_HELMET,
    NO_SEATBELT,
    PERSON,
    SEATBELT,
    SMOKING_OBJECTS,
    STEERING_WHEEL,
)
from censorlabel.domain import Box, Detection, WarningKind

HEAD_FRACTION = 0.25  # top quarter of a person box is treated as the head
RIDER_OVERLAP = 0.3
DRIVER_OVERLAP = 0.2
GEAR_OVERLAP = 0.3


def _boxes(detections: list[Detection], *labels: str) -> list[Box]:
    return [d.box for d in detections if d.label in labels]


def is_smoking(detections: list[Detection]) -> bool:
    """A smoking object that is not just the rim of a glass or bottle (a common zero-shot confusion)."""
    if any(d.label == "smoking" for d in detections):
        return True
    containers = _boxes(detections, *ALCOHOL_OBJECTS)
    return any(
        not any(c.containment(d.box) >= GEAR_OVERLAP for c in containers)
        for d in detections
        if d.label in SMOKING_OBJECTS
    )


def is_drinking(detections: list[Detection]) -> bool:
    return any(d.label in ALCOHOL_OBJECTS or d.label == "alcohol" for d in detections)


def is_riding_without_helmet(detections: list[Detection], min_person_height: int = 0) -> bool:
    """A person on a motorcycle whose head region has no helmet."""
    if any(d.label == NO_HELMET for d in detections):
        return True
    bikes = _boxes(detections, MOTORCYCLE)
    helmets = _boxes(detections, HELMET)
    riders = [
        p
        for p in _boxes(detections, PERSON)
        if p.h >= min_person_height and any(p.containment(b) >= RIDER_OVERLAP for b in bikes)
    ]
    return any(not any(r.top(HEAD_FRACTION).containment(h) >= GEAR_OVERLAP for h in helmets) for r in riders)


def is_driving_without_seatbelt(detections: list[Detection], min_person_height: int = 0) -> bool:
    """A person at a steering wheel with no seat belt detected across their body."""
    if any(d.label == NO_SEATBELT for d in detections):
        return True
    wheels = _boxes(detections, STEERING_WHEEL)
    belts = _boxes(detections, SEATBELT)
    drivers = [
        p
        for p in _boxes(detections, PERSON)
        if p.h >= min_person_height and any(p.containment(w) >= DRIVER_OVERLAP for w in wheels)
    ]
    return any(not any(d.containment(b) >= GEAR_OVERLAP for b in belts) for d in drivers)


def frame_violations(
    detections: list[Detection], enabled: frozenset[WarningKind], min_person_height: int = 0
) -> frozenset[WarningKind]:
    checks = {
        WarningKind.SMOKE: lambda: is_smoking(detections),
        WarningKind.ALCOHOL: lambda: is_drinking(detections),
        WarningKind.HELMET: lambda: is_riding_without_helmet(detections, min_person_height),
        WarningKind.SEATBELT: lambda: is_driving_without_seatbelt(detections, min_person_height),
    }
    return frozenset(kind for kind, check in checks.items() if kind in enabled and check())
