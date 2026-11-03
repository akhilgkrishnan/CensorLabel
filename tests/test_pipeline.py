import subprocess
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

from censorlabel.config import Settings
from censorlabel.domain import Box, Detection, WarningKind
from censorlabel.events import Finished, Progress, SegmentFound
from censorlabel.media import ffmpeg_exe, probe
from censorlabel.overlay import Banners, load_banner
from censorlabel.pipeline import CancelToken, LabelingPipeline

REPO = Path(__file__).resolve().parents[1]
FPS, W, H, SECONDS = 25, 320, 240, 12


class CigaretteBetween:
    """Fake detector: reports a cigarette in frames whose brightness falls in a range."""

    def __init__(self, low: int, high: int):
        self.low, self.high = low, high
        self.frames_seen = 0

    def detect_batch(self, frames):
        self.frames_seen += len(frames)
        return [
            [Detection("cigarette", 0.9, Box(10, 10, 20, 5))] if self.low <= f.mean() <= self.high else []
            for f in frames
        ]


@pytest.fixture
def video(tmp_path: Path) -> Path:
    """12s clip with a sine soundtrack; brightness ramps 0→~240 over time."""
    silent = tmp_path / "silent.mp4"
    writer = cv.VideoWriter(str(silent), cv.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(FPS * SECONDS):
        writer.write(np.full((H, W, 3), i * 240 // (FPS * SECONDS), dtype=np.uint8))
    writer.release()
    path = tmp_path / "in.mp4"
    subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-y", "-i", str(silent), "-f", "lavfi", "-i", "sine=d=12",
         "-c:v", "copy", "-c:a", "aac", "-shortest", str(path)],
        check=True,
    )  # fmt: skip
    return path


def make_pipeline(detector) -> LabelingPipeline:
    settings = Settings(home=REPO, sample_fps=4, encoder="libx264")
    return LabelingPipeline(settings, detector, Banners(settings.warnings_dir / "English"))


def has_audio(path: Path) -> bool:
    out = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    return "Audio:" in out


def test_clean_video_is_copied_untouched(video, tmp_path):
    detector = CigaretteBetween(999, 999)
    result = make_pipeline(detector).run(video, tmp_path / "out.mp4")
    assert result.segments == ()
    assert (tmp_path / "out.mp4").read_bytes() == video.read_bytes()
    assert detector.frames_seen == pytest.approx(SECONDS * 4, abs=2)  # only 4 fps were analyzed


def test_smoking_scene_gets_banner_and_keeps_audio(video, tmp_path):
    out = tmp_path / "out.mp4"
    events = []
    # brightness 80..120 ↔ roughly seconds 4–6
    result = make_pipeline(CigaretteBetween(80, 120)).run(video, out, on_event=events.append)

    assert [s.kind for s in result.segments] == [WarningKind.SMOKE]
    segment = result.segments[0]
    assert 3.5 <= segment.start <= 4.5 and segment.duration >= 4.0
    assert any(isinstance(e, SegmentFound) for e in events)
    assert {e.stage for e in events if isinstance(e, Progress)} == {"analyzing", "rendering"}
    assert isinstance(events[-1], Finished)

    info = probe(out)
    assert info.frame_count == pytest.approx(FPS * SECONDS, abs=2)
    assert has_audio(out)

    cap = cv.VideoCapture(str(out))
    def frame_at(t):
        cap.set(cv.CAP_PROP_POS_MSEC, t * 1000)
        return cap.read()[1]
    banner_region = (slice(H - 60, H - 10), slice(10, 150))
    inside = frame_at(segment.start + 1)[banner_region]
    outside = frame_at(1.0)[banner_region]
    cap.release()
    assert inside.std() > 20  # banner text/icon present
    assert outside.std() < 5  # flat frame, no banner


def test_cancel_before_start(video, tmp_path):
    token = CancelToken()
    token.cancel()
    result = make_pipeline(CigaretteBetween(0, 255)).run(video, tmp_path / "o.mp4", cancel=token)
    assert result.cancelled and result.output is None
    assert not (tmp_path / "o.mp4").exists()


def test_black_background_banner_becomes_transparent():
    banner = load_banner(REPO / "Images/statutory/English/smoke.png")
    assert banner.shape[2] == 4
    assert banner[0, -1, 3] == 0  # black corner is transparent
    assert banner[:, :, 3].max() == 255  # text is opaque


def test_settings_languages_and_thresholds():
    s = Settings(home=REPO)
    assert "English" in s.available_languages()
    assert s.keeps("cigarette", 0.16) and not s.keeps("person", 0.16)
    assert s.detector_confidence == min(s.confidence, *s.class_confidence.values())
