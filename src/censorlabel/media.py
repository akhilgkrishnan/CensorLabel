"""Video probing, fast frame sampling and single-pass rendering, all through ffmpeg."""

from __future__ import annotations

import functools
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np

from censorlabel.domain import Segment, WarningKind


@functools.cache
def ffmpeg_exe() -> str:
    """System ffmpeg if installed, otherwise the static build shipped with imageio-ffmpeg."""
    if found := shutil.which("ffmpeg"):
        return found
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


@functools.cache
def _encoders() -> str:
    return subprocess.run([ffmpeg_exe(), "-hide_banner", "-encoders"], capture_output=True, text=True).stdout


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


def probe(path: Path) -> VideoInfo:
    capture = cv.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video {path}")
        return VideoInfo(
            fps=capture.get(cv.CAP_PROP_FPS) or 25.0,
            width=int(capture.get(cv.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(capture.get(cv.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        capture.release()


def sample_frames(path: Path, info: VideoInfo, fps: float, width: int) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (timestamp, BGR frame) at `fps`, downscaled to `width`. ffmpeg decodes multithreaded."""
    width = min(width, info.width) // 2 * 2
    height = round(info.height * width / info.width / 2) * 2
    cmd = [
        ffmpeg_exe(), "-v", "error", "-nostdin", "-i", str(path),
        "-an", "-sn", "-vf", f"fps={fps},scale={width}:{height}",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]  # fmt: skip
    size = width * height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=size * 4)
    assert proc.stdout is not None
    try:
        index = 0
        while len(buf := proc.stdout.read(size)) == size:
            yield index / fps, np.frombuffer(buf, np.uint8).reshape(height, width, 3)
            index += 1
    finally:
        proc.kill()
        proc.wait()


def encoder_args(encoder: str, preset: str, crf: int) -> list[str]:
    if encoder == "auto":
        available = _encoders()
        if sys.platform == "darwin" and "h264_videotoolbox" in available:
            encoder = "h264_videotoolbox"
        elif "h264_nvenc" in available:
            encoder = "h264_nvenc"
        else:
            encoder = "libx264"
    match encoder:
        case "h264_videotoolbox":
            return ["-c:v", encoder, "-q:v", "65", "-allow_sw", "1"]
        case "h264_nvenc":
            return ["-c:v", encoder, "-preset", "p4", "-cq", str(crf + 1)]
        case _:
            return ["-c:v", encoder, "-preset", preset, "-crf", str(crf)]


def overlay_filter(segments: list[Segment], inputs: dict[WarningKind, int], positions: dict[WarningKind, tuple[int, int]]) -> str:
    """One overlay per banner kind, each enabled only during its segments."""
    chains = []
    current = "0:v"
    for n, (kind, index) in enumerate(inputs.items()):
        enable = "+".join(f"between(t,{s.start:.3f},{s.end:.3f})" for s in segments if s.kind == kind)
        x, y = positions[kind]
        label = f"v{n}"
        chains.append(f"[{current}][{index}:v]overlay=x={x}:y={y}:enable='{enable}'[{label}]")
        current = label
    return ";".join(chains)


def render(
    source: Path,
    output: Path,
    segments: list[Segment],
    banners: dict[WarningKind, tuple[Path, tuple[int, int]]],
    *,
    duration: float,
    video_args: list[str],
    on_progress: Callable[[float], None] = lambda _f: None,
    should_stop: Callable[[], bool] = lambda: False,
) -> bool:
    """Burn banners into `source`, copying audio. Returns False if stopped early."""
    if not segments:
        shutil.copyfile(source, output)
        on_progress(1.0)
        return True
    inputs = {kind: i + 1 for i, kind in enumerate(banners)}
    graph = overlay_filter(segments, inputs, {k: pos for k, (_p, pos) in banners.items()})
    last = f"[v{len(inputs) - 1}]"
    base = [ffmpeg_exe(), "-y", "-v", "error", "-nostdin", "-progress", "pipe:1", "-i", str(source)]
    for path, _pos in banners.values():
        base += ["-i", str(path)]
    base += ["-filter_complex", graph, "-map", last, "-map", "0:a?", *video_args, "-pix_fmt", "yuv420p"]

    for audio in (["-c:a", "copy"], ["-c:a", "aac", "-b:a", "192k"]):  # copy first; re-encode if the container refuses
        result = _run_with_progress([*base, *audio, str(output)], duration, on_progress, should_stop)
        if result is None:
            output.unlink(missing_ok=True)
            return False
        if result == 0:
            return True
    raise RuntimeError("ffmpeg failed to render the output video")


def _run_with_progress(
    cmd: list[str], duration: float, on_progress: Callable[[float], None], should_stop: Callable[[], bool]
) -> int | None:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        if should_stop():
            proc.kill()
            proc.wait()
            return None
        if (m := re.match(r"out_time_us=(\d+)", line)) and duration:
            on_progress(min(int(m.group(1)) / 1e6 / duration, 1.0))
    return proc.wait()
