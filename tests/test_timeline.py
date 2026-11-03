import pytest

from censorlabel.domain import Segment, WarningKind
from censorlabel.media import overlay_filter
from censorlabel.timeline import Sample, TimelineRules, build_timeline

SMOKE, ALCOHOL, HELMET = WarningKind.SMOKE, WarningKind.ALCOHOL, WarningKind.HELMET
RULES = TimelineRules(window=3, min_hits=2, hold_seconds=2.0, merge_gap=1.0, min_duration=3.0)


def samples(hits: dict[float, set[WarningKind]], until: float = 30.0, step: float = 0.5) -> list[Sample]:
    out, t = [], 0.0
    while t < until:
        out.append(Sample(t, frozenset(hits.get(t, set()))))
        t += step
    return out


def test_single_noisy_hit_is_ignored():
    assert build_timeline(samples({5.0: {SMOKE}}), 30, RULES) == []


def test_two_hits_in_window_start_at_first_hit_and_hold():
    timeline = build_timeline(samples({5.0: {SMOKE}, 6.0: {SMOKE}}), 30, RULES)
    assert timeline == [Segment(SMOKE, 5.0, 8.0)]


def test_short_detection_is_extended_to_min_duration():
    rules = TimelineRules(window=2, min_hits=2, hold_seconds=0.5, merge_gap=0, min_duration=4.0)
    assert build_timeline(samples({5.0: {SMOKE}, 5.5: {SMOKE}}), 30, rules) == [Segment(SMOKE, 5.0, 9.0)]


def test_nearby_runs_merge():
    hits = {t: {SMOKE} for t in (5.0, 5.5, 8.5, 9.0)}
    assert build_timeline(samples(hits), 30, RULES) == [Segment(SMOKE, 5.0, 11.0)]


def test_smoking_and_alcohol_together_use_combined_banner():
    hits = {t: {SMOKE, ALCOHOL} for t in (5.0, 5.5, 6.0)}
    assert [s.kind for s in build_timeline(samples(hits), 30, RULES)] == [WarningKind.SMOKE_ALCOHOL]


def test_overlapping_kinds_do_not_overlap_in_output():
    hits = {5.0: {ALCOHOL}, 5.5: {ALCOHOL}, 6.0: {ALCOHOL, HELMET}, 6.5: {HELMET}}
    timeline = build_timeline(samples(hits), 30, RULES)
    for a, b in zip(timeline, timeline[1:]):
        assert a.end <= b.start
    assert timeline[0].kind is ALCOHOL and timeline[1].kind is HELMET


def test_segments_clamped_to_video_end():
    timeline = build_timeline(samples({9.0: {SMOKE}, 9.5: {SMOKE}}, until=10), 10, RULES)
    assert timeline[-1].end == pytest.approx(10)


def test_overlay_filter_enables_each_banner_only_in_its_segments():
    graph = overlay_filter(
        [Segment(SMOKE, 1, 4), Segment(HELMET, 5, 9), Segment(SMOKE, 10, 12)],
        {SMOKE: 1, HELMET: 2},
        {SMOKE: (10, 600), HELMET: (10, 590)},
    )
    assert graph == (
        "[0:v][1:v]overlay=x=10:y=600:enable='between(t,1.000,4.000)+between(t,10.000,12.000)'[v0];"
        "[v0][2:v]overlay=x=10:y=590:enable='between(t,5.000,9.000)'[v1]"
    )
