"""Turn per-sample violations into a clean, non-overlapping list of warning segments."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from censorlabel.config import BASE_KINDS
from censorlabel.domain import Segment, WarningKind

# When several warnings apply at once only one banner is shown, highest priority first.
PRIORITY = (WarningKind.SMOKE_ALCOHOL, WarningKind.HELMET, WarningKind.SEATBELT, WarningKind.SMOKE, WarningKind.ALCOHOL)


@dataclass(frozen=True)
class Sample:
    time: float
    kinds: frozenset[WarningKind]


@dataclass(frozen=True)
class TimelineRules:
    window: int = 3
    min_hits: int = 2
    hold_seconds: float = 3.0
    merge_gap: float = 2.0
    min_duration: float = 4.0


def _spans_for(kind: WarningKind, samples: list[Sample], rules: TimelineRules) -> list[tuple[float, float]]:
    """Debounced spans where `kind` is confirmed: min_hits within a sliding window of samples."""
    spans: list[tuple[float, float]] = []
    recent: deque[Sample] = deque(maxlen=rules.window)
    for sample in samples:
        recent.append(sample)
        hits = [s for s in recent if kind in s.kinds]
        if len(hits) < rules.min_hits or kind not in sample.kinds:
            continue
        start, end = hits[0].time, sample.time + rules.hold_seconds
        if spans and start - spans[-1][1] <= rules.merge_gap:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end))
        else:
            spans.append((start, end))
    return [(s, max(e, s + rules.min_duration)) for s, e in spans]


def _display_kind(active: set[WarningKind]) -> WarningKind:
    if {WarningKind.SMOKE, WarningKind.ALCOHOL} <= active:
        active = active | {WarningKind.SMOKE_ALCOHOL}
    return next(k for k in PRIORITY if k in active)


def build_timeline(samples: list[Sample], duration: float, rules: TimelineRules = TimelineRules()) -> list[Segment]:
    spans = {kind: _spans_for(kind, samples, rules) for kind in BASE_KINDS}
    cuts = sorted({0.0, duration} | {min(t, duration) for ss in spans.values() for s in ss for t in s})

    segments: list[Segment] = []
    for a, b in zip(cuts, cuts[1:]):
        if b <= a:
            continue
        mid = (a + b) / 2
        active = {k for k, ss in spans.items() if any(s <= mid < e for s, e in ss)}
        if not active:
            continue
        kind = _display_kind(active)
        if segments and segments[-1].kind == kind and abs(segments[-1].end - a) < 1e-6:
            segments[-1] = Segment(kind, segments[-1].start, b)
        else:
            segments.append(Segment(kind, a, b))
    return _absorb_slivers(segments)


def _absorb_slivers(segments: list[Segment], shortest: float = 1.0) -> list[Segment]:
    """Fold segments shorter than `shortest` into the preceding touching one, or drop them."""
    out: list[Segment] = []
    for seg in segments:
        if seg.duration >= shortest:
            if out and out[-1].kind == seg.kind and abs(out[-1].end - seg.start) < 1e-6:
                out[-1] = Segment(seg.kind, out[-1].start, seg.end)
            else:
                out.append(seg)
        elif out and abs(out[-1].end - seg.start) < 1e-6:
            out[-1] = Segment(out[-1].kind, out[-1].start, seg.end)
    return out
