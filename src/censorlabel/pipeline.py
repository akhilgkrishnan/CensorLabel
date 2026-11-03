"""Two-stage labelling run.

1. Analyze: decode a few frames per second, run one detector in batches, apply the rules and build
   a debounced timeline of warning segments.
2. Render: one ffmpeg pass burns the banners into those segments and copies the audio.
"""

from __future__ import annotations

import tempfile
import threading
import time
from dataclasses import dataclass, field
from itertools import batched
from pathlib import Path

import cv2 as cv
import numpy as np

from censorlabel import rules
from censorlabel.config import PROMPTS, Settings
from censorlabel.detectors import ObjectDetector
from censorlabel.domain import Detection, Segment
from censorlabel.events import Finished, Listener, Log, Preview, Progress, SegmentFound
from censorlabel.media import VideoInfo, encoder_args, probe, render, sample_frames
from censorlabel.overlay import Banners
from censorlabel.timeline import Sample, TimelineRules, build_timeline

MIN_PERSON_HEIGHT = 0.12  # ignore people smaller than this share of the frame height (background extras)


class CancelToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


@dataclass
class Analysis:
    info: VideoInfo
    samples: list[Sample]
    segments: list[Segment]
    seconds: float = 0.0


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}-labeled{input_path.suffix}")


def _draw(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = frame.copy()
    for d in detections:
        b = d.box
        cv.rectangle(out, (b.x, b.y), (b.x + b.w, b.y + b.h), (80, 220, 255), 2)
        cv.putText(out, f"{d.label} {d.confidence:.2f}", (b.x, max(b.y - 5, 12)), cv.FONT_HERSHEY_SIMPLEX, 0.45, (80, 220, 255), 1)
    return out


@dataclass
class LabelingPipeline:
    settings: Settings
    detector: ObjectDetector
    banners: Banners
    preview: bool = False
    _noop: Listener = field(default=lambda _e: None, repr=False)

    @classmethod
    def from_settings(cls, settings: Settings, *, preview: bool = False) -> LabelingPipeline:
        from censorlabel.detectors.yolo import YoloDetector

        detector = YoloDetector(
            settings.model,
            PROMPTS,
            models_dir=settings.home / "models",
            device=settings.device,
            image_size=settings.image_size,
            confidence=settings.detector_confidence,
        )
        banners = Banners(settings.warnings_dir / settings.language, width_ratio=settings.banner_width)
        return cls(settings, detector, banners, preview=preview)

    def analyze(self, source: Path, *, on_event: Listener | None = None, cancel: CancelToken | None = None) -> Analysis | None:
        emit = on_event or self._noop
        cancel = cancel or CancelToken()
        s = self.settings
        info = probe(source)
        started = time.monotonic()
        expected = max(int(info.duration * s.sample_fps), 1)
        emit(Log(f"Analyzing {source.name} ({info.width}×{info.height}, {info.duration:.0f}s) at {s.sample_fps:g} fps"))

        samples: list[Sample] = []
        for batch in batched(sample_frames(source, info, s.sample_fps, s.image_size), s.batch_size):
            if cancel.cancelled:
                return None
            times, frames = zip(*batch)
            min_height = int(frames[0].shape[0] * MIN_PERSON_HEIGHT)
            for t, raw in zip(times, self.detector.detect_batch(list(frames))):
                detections = [d for d in raw if s.keeps(d.label, d.confidence)]
                samples.append(Sample(t, rules.frame_violations(detections, s.enabled, min_height)))
            if self.preview:
                emit(Preview(_draw(frames[-1], detections)))
            emit(Progress("analyzing", min(len(samples) / expected, 1.0)))

        timeline = TimelineRules(s.window, s.min_hits, s.hold_seconds, s.merge_gap, s.min_duration)
        segments = build_timeline(samples, info.duration, timeline)
        elapsed = time.monotonic() - started
        for segment in segments:
            emit(SegmentFound(segment))
        speed = info.duration / elapsed if elapsed else 0
        emit(Log(f"Analysis done in {elapsed:.1f}s ({speed:.1f}× real time), {len(segments)} warning segment(s)"))
        return Analysis(info, samples, segments, elapsed)

    def render(
        self,
        source: Path,
        output: Path,
        analysis: Analysis,
        *,
        on_event: Listener | None = None,
        cancel: CancelToken | None = None,
    ) -> bool:
        emit = on_event or self._noop
        cancel = cancel or CancelToken()
        s, info = self.settings, analysis.info
        output.parent.mkdir(parents=True, exist_ok=True)
        emit(Log("Rendering with ffmpeg…" if analysis.segments else "No warnings needed, copying video…"))
        with tempfile.TemporaryDirectory(prefix="censorlabel-") as tmp:
            banners = {}
            for kind in dict.fromkeys(seg.kind for seg in analysis.segments):
                path = self.banners.export(kind, info.width, Path(tmp) / f"{kind.value}.png")
                height = self.banners.scaled(kind, info.width).shape[0]
                banners[kind] = (path, Banners.position(info.width, info.height, height))
            return render(
                source,
                output,
                analysis.segments,
                banners,
                duration=info.duration,
                video_args=encoder_args(s.encoder, s.encoder_preset, s.crf),
                on_progress=lambda f: emit(Progress("rendering", f)),
                should_stop=lambda: cancel.cancelled,
            )

    def run(
        self,
        source: Path,
        output: Path | None = None,
        *,
        on_event: Listener | None = None,
        cancel: CancelToken | None = None,
    ) -> Finished:
        emit = on_event or self._noop
        cancel = cancel or CancelToken()
        output = output or default_output_path(source)

        analysis = self.analyze(source, on_event=emit, cancel=cancel)
        if analysis is None or not self.render(source, output, analysis, on_event=emit, cancel=cancel):
            emit(Log("Cancelled", "warning"))
            finished = Finished(None, tuple(analysis.segments) if analysis else (), cancelled=True)
        else:
            emit(Log(f"Saved {output}", "positive"))
            finished = Finished(output, tuple(analysis.segments))
        emit(finished)
        return finished
