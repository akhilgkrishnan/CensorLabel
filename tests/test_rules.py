from censorlabel import rules
from censorlabel.domain import Box, Detection, WarningKind

ALL = frozenset(WarningKind) - {WarningKind.SMOKE_ALCOHOL}


def det(label: str, x: int, y: int, w: int, h: int) -> Detection:
    return Detection(label, 0.9, Box(x, y, w, h))


class TestBox:
    def test_intersection_of_overlapping_boxes(self):
        assert Box(0, 0, 10, 10).intersection(Box(5, 5, 10, 10)) == 25

    def test_disjoint_boxes_do_not_intersect(self):
        assert Box(0, 0, 10, 10).intersection(Box(20, 20, 5, 5)) == 0

    def test_containment_is_relative_to_smaller_box(self):
        assert Box(0, 0, 100, 100).containment(Box(10, 10, 10, 10)) == 1.0

    def test_from_xyxy_and_top(self):
        box = Box.from_xyxy(10, 20, 50, 100)
        assert box == Box(10, 20, 40, 80)
        assert box.top(0.25) == Box(10, 20, 40, 20)


RIDER = det("person", 100, 50, 80, 200)
BIKE = det("motorcycle", 80, 150, 140, 150)
HELMET_ON_HEAD = det("helmet", 110, 45, 50, 45)


class TestHelmet:
    def test_rider_without_helmet(self):
        assert rules.is_riding_without_helmet([RIDER, BIKE])

    def test_rider_with_helmet(self):
        assert not rules.is_riding_without_helmet([RIDER, BIKE, HELMET_ON_HEAD])

    def test_helmet_elsewhere_does_not_count(self):
        assert rules.is_riding_without_helmet([RIDER, BIKE, det("helmet", 600, 400, 40, 40)])

    def test_pedestrian_without_motorcycle(self):
        assert not rules.is_riding_without_helmet([RIDER])

    def test_tiny_background_rider_ignored(self):
        assert not rules.is_riding_without_helmet([RIDER, BIKE], min_person_height=300)

    def test_fine_tuned_direct_label(self):
        assert rules.is_riding_without_helmet([det("no_helmet", 0, 0, 10, 10)])


DRIVER = det("person", 100, 50, 200, 250)
WHEEL = det("steering wheel", 150, 200, 100, 80)


class TestSeatbelt:
    def test_driver_without_seatbelt(self):
        assert rules.is_driving_without_seatbelt([DRIVER, WHEEL])

    def test_driver_with_seatbelt(self):
        assert not rules.is_driving_without_seatbelt([DRIVER, WHEEL, det("seat belt", 150, 100, 60, 150)])

    def test_person_not_at_wheel(self):
        assert not rules.is_driving_without_seatbelt([DRIVER, det("steering wheel", 900, 900, 50, 50)])


class TestSmokingAndAlcohol:
    def test_cigarette(self):
        assert rules.is_smoking([det("cigarette", 0, 0, 20, 5)])

    def test_glass_rim_mistaken_for_cigar_is_ignored(self):
        assert not rules.is_smoking([det("whisky glass", 0, 0, 100, 120), det("cigar", 10, 5, 40, 10)])

    def test_alcohol(self):
        assert rules.is_drinking([det("beer bottle", 0, 0, 20, 60)])
        assert not rules.is_drinking([det("person", 0, 0, 20, 60)])


def test_frame_violations_respects_enabled():
    dets = [det("cigarette", 0, 0, 20, 5), det("beer bottle", 50, 0, 20, 60)]
    assert rules.frame_violations(dets, ALL) == {WarningKind.SMOKE, WarningKind.ALCOHOL}
    assert rules.frame_violations(dets, frozenset({WarningKind.ALCOHOL})) == {WarningKind.ALCOHOL}
