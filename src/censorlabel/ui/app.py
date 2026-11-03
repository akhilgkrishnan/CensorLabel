"""NiceGUI front end: analyze → review the warning timeline → render.

Runs in a native window when pywebview is installed, otherwise in the browser.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2 as cv
import numpy as np
from nicegui import app, run, ui

from censorlabel.config import Settings
from censorlabel.domain import Segment, WarningKind
from censorlabel.events import Event, Log, Preview, Progress
from censorlabel.pipeline import Analysis, CancelToken, LabelingPipeline, default_output_path
from censorlabel.ui.file_picker import LocalFilePicker

VIDEO_TYPES = ("Video files (*.mp4;*.mkv;*.avi;*.mov;*.webm)", "All files (*.*)")
MODELS = {
    "yoloe-26n-seg.pt": "YOLOE-26 nano (fastest)",
    "yoloe-26s-seg.pt": "YOLOE-26 small (balanced)",
    "yoloe-26m-seg.pt": "YOLOE-26 medium (most accurate)",
}
KIND_COLORS = {
    WarningKind.SMOKE: "#f97316",
    WarningKind.ALCOHOL: "#a855f7",
    WarningKind.SMOKE_ALCOHOL: "#e11d48",
    WarningKind.HELMET: "#0ea5e9",
    WarningKind.SEATBELT: "#22c55e",
}
TOGGLEABLE = (WarningKind.SMOKE, WarningKind.ALCOHOL, WarningKind.HELMET, WarningKind.SEATBELT)
PREVIEW_WIDTH = 720
LOG_COLORS = {"warning": "text-amber-600", "negative": "text-red-600", "positive": "text-green-600"}
STAGE_TEXT = {"loading": "Loading model", "analyzing": "Analyzing", "rendering": "Rendering"}


@dataclass
class Job:
    """State shared between a worker thread (writes) and the UI timer (reads)."""

    cancel: CancelToken = field(default_factory=CancelToken)
    events: deque[Event] = field(default_factory=deque)
    preview: np.ndarray | None = None
    started: float = field(default_factory=time.monotonic)

    def on_event(self, event: Event) -> None:
        if isinstance(event, Preview):
            self.preview = event.frame  # only the latest frame matters
        else:
            self.events.append(event)


def _data_url(frame: np.ndarray) -> str:
    h, w = frame.shape[:2]
    if w > PREVIEW_WIDTH:
        frame = cv.resize(frame, (PREVIEW_WIDTH, int(h * PREVIEW_WIDTH / w)), interpolation=cv.INTER_AREA)
    ok, buf = cv.imencode(".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else ""


def _frame_at(path: Path, seconds: float) -> np.ndarray | None:
    capture = cv.VideoCapture(str(path))
    try:
        capture.set(cv.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, frame = capture.read()
        return frame if ok else None
    finally:
        capture.release()


def _reveal(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(path)])
    elif sys.platform == "win32":
        subprocess.Popen(["explorer", f"/select,{path}"])
    else:
        subprocess.Popen(["xdg-open", str(path.parent)])


def _clock(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes // 60:d}:{minutes % 60:02d}:{secs:02d}"


class PipelineCache:
    """Keeps the loaded model between runs; reloads only when model-affecting settings change."""

    def __init__(self) -> None:
        self._key: tuple | None = None
        self._pipeline: LabelingPipeline | None = None

    def get(self, settings: Settings) -> LabelingPipeline:
        key = (settings.model, settings.device, settings.image_size, settings.detector_confidence, settings.language)
        if self._pipeline is None or key != self._key:
            self._pipeline = LabelingPipeline.from_settings(settings, preview=True)
            self._key = key
        else:
            self._pipeline.settings = settings
        return self._pipeline


CACHE = PipelineCache()


def build_page() -> None:
    defaults = Settings()
    languages = defaults.available_languages()
    native = app.native.main_window is not None
    job: Job | None = None
    analysis: Analysis | None = None
    pipeline: LabelingPipeline | None = None  # the one that produced `analysis`
    source: Path | None = None
    output_file: Path | None = None

    dark = ui.dark_mode(None)
    ui.colors(primary="#4f46e5")

    with ui.header(elevated=True).classes("items-center justify-between px-6"):
        with ui.row().classes("items-center gap-3"):
            ui.icon("movie_filter", size="md")
            with ui.column().classes("gap-0"):
                ui.label("CensorLabel").classes("text-lg font-semibold leading-tight")
                ui.label("Automatic statutory warnings for movies").classes("text-xs opacity-80")
        ui.button(icon="dark_mode", on_click=dark.toggle).props("flat round color=white").tooltip("Toggle dark mode")

    with ui.row().classes("w-full max-w-7xl mx-auto p-4 gap-4 flex-col lg:flex-row items-start no-wrap"):
        # ---------- Settings column ----------
        with ui.card().classes("w-full lg:w-96 shrink-0 gap-3"):
            ui.label("Input").classes("text-base font-semibold")

            async def browse() -> None:
                current = Path(video.value).expanduser() if video.value else None
                start = current.parent if current and current.parent.is_dir() else None
                if native:
                    files = await app.native.main_window.create_file_dialog(
                        file_types=VIDEO_TYPES, directory=str(start or "")
                    )
                    chosen = files[0] if files else None
                else:
                    chosen = await LocalFilePicker(start)
                if chosen:
                    video.value = chosen

            with ui.input("Video file", placeholder="/path/to/movie.mp4").classes("w-full") as video:
                with video.add_slot("append"):
                    ui.button(icon="folder_open", on_click=browse).props("flat round dense").tooltip("Browse…")
            async def save_as() -> None:
                current = Path(output.value).expanduser() if output.value else None
                if current is None and video.value:
                    current = default_output_path(Path(video.value).expanduser())
                folder = current.parent if current and current.parent.is_dir() else None
                name = current.name if current else "labeled.mp4"
                if native:
                    import webview

                    save = webview.FileDialog.SAVE if hasattr(webview, "FileDialog") else webview.SAVE_DIALOG
                    result = await app.native.main_window.create_file_dialog(
                        dialog_type=save, directory=str(folder or ""), save_filename=name
                    )
                    chosen = (result[0] if isinstance(result, (list, tuple)) else result) if result else None
                else:
                    chosen = await LocalFilePicker(folder, save_name=name)
                if chosen:
                    output.value = chosen

            with ui.input("Output file").classes("w-full") as output:
                with output.add_slot("append"):
                    ui.button(icon="save_as", on_click=save_as).props("flat round dense").tooltip("Save as…")
            video.on_value_change(lambda e: output.set_value(str(default_output_path(Path(e.value)))) if e.value else None)

            language = ui.select(languages, value="English" if "English" in languages else None, label="Warning language").classes("w-full")
            ui.label("Check for").classes("text-sm opacity-70 mt-2")
            with ui.row().classes("gap-x-4 gap-y-0"):
                checks = {k: ui.checkbox(k.description, value=True) for k in TOGGLEABLE}

            with ui.expansion("Detection settings", icon="tune").classes("w-full"):
                model = ui.select(MODELS, value=defaults.model, label="Model", new_value_mode="add-unique").classes("w-full")
                ui.label("Pick a size or type a path to a fine-tuned .pt").classes("text-xs opacity-60")
                device = ui.select(["auto", "mps", "cuda", "cpu"], value="auto", label="Device").classes("w-full")
                ui.label("Frames analyzed per second").classes("text-sm opacity-70 mt-2")
                fps = ui.slider(min=1, max=10, step=1, value=defaults.sample_fps).props("label markers")
                ui.label("Detection confidence").classes("text-sm opacity-70")
                confidence = ui.slider(min=0.1, max=0.8, step=0.05, value=defaults.confidence).props("label")
                ui.label("Cigarette sensitivity (lower = catches more)").classes("text-sm opacity-70")
                cigarette = ui.slider(min=0.05, max=0.5, step=0.05, value=defaults.class_confidence["cigarette"]).props("label")
                live = ui.switch("Live preview", value=True)

            with ui.row().classes("w-full gap-2 mt-2 no-wrap"):
                analyze_btn = ui.button("Analyze", icon="search").classes("grow")
                render_btn = ui.button("Render", icon="movie").classes("grow").props("outline")
                cancel_btn = ui.button(icon="stop", color="negative").props("outline").tooltip("Cancel")
            render_btn.disable()
            cancel_btn.disable()

        # ---------- Results column ----------
        with ui.column().classes("grow gap-4 min-w-0 w-full"):
            with ui.card().classes("w-full p-0 overflow-hidden gap-0"):
                with ui.element("div").classes("w-full aspect-video bg-black flex items-center justify-center"):
                    placeholder = ui.icon("smart_display", size="4rem").classes("text-slate-600")
                    preview = ui.image().classes("w-full h-full").props("fit=contain no-spinner no-transition")
                    preview.set_visibility(False)
                timeline_bar = ui.element("div").classes("relative w-full h-3 bg-slate-200 dark:bg-slate-700")
                with ui.column().classes("w-full p-4 gap-2"):
                    status = ui.label("Choose a video, then Analyze").classes("text-sm opacity-80")
                    progress = ui.linear_progress(value=0, show_value=False).props("rounded size=8px")
                    with ui.row().classes("gap-2") as done_actions:
                        ui.button("Download", icon="download", on_click=lambda: output_file and ui.download.file(output_file)).props("flat dense")
                        ui.button("Show in folder", icon="folder", on_click=lambda: output_file and _reveal(output_file)).props("flat dense")
                    done_actions.set_visibility(False)

            with ui.card().classes("w-full"):
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label("Warning timeline").classes("text-base font-semibold")
                    ui.label("Click a row to preview, remove false detections before rendering").classes("text-xs opacity-60")
                table = ui.table(
                    columns=[
                        {"name": "warning", "label": "Warning", "field": "warning", "align": "left"},
                        {"name": "start", "label": "From", "field": "start"},
                        {"name": "end", "label": "To", "field": "end"},
                        {"name": "actions", "label": "", "field": "id"},
                    ],
                    rows=[],
                    row_key="id",
                ).props("flat dense").classes("w-full")
                table.add_slot(
                    "body-cell-actions",
                    '<q-td :props="props" auto-width><q-btn flat dense round size="sm" icon="delete" '
                    "@click.stop=\"() => $parent.$emit('remove', props.row)\" /></q-td>",
                )

            with ui.card().classes("w-full"):
                ui.label("Log").classes("text-base font-semibold")
                log = ui.log(max_lines=500).classes("w-full h-40 text-xs")

    def current_settings() -> Settings:
        return replace(
            defaults,
            language=language.value,
            model=model.value,
            device=device.value,
            sample_fps=float(fps.value),
            confidence=confidence.value,
            class_confidence={k: cigarette.value for k in defaults.class_confidence},
            enabled=frozenset(k for k, box in checks.items() if box.value),
            encoder="auto",
        )

    def show_segments() -> None:
        segments = analysis.segments if analysis else []
        table.rows = [
            {"id": i, "warning": s.kind.description, "start": _clock(s.start), "end": _clock(s.end)}
            for i, s in enumerate(segments)
        ]
        table.update()
        timeline_bar.clear()
        if analysis and analysis.info.duration:
            total = analysis.info.duration
            with timeline_bar:
                for s in segments:
                    ui.element("div").classes("absolute top-0 h-full").style(
                        f"left:{s.start / total:.4%};width:{s.duration / total:.4%};background:{KIND_COLORS[s.kind]}"
                    ).tooltip(f"{s.kind.description} {_clock(s.start)}–{_clock(s.end)}")
        render_btn.set_enabled(analysis is not None and job is None)

    def remove_segment(row: dict) -> None:
        if analysis is not None:
            del analysis.segments[row["id"]]
            show_segments()

    def preview_segment(row: dict) -> None:
        if analysis is None or source is None or pipeline is None:
            return
        segment: Segment = analysis.segments[row["id"]]
        frame = _frame_at(source, (segment.start + segment.end) / 2)
        if frame is not None:
            frame = pipeline.banners.apply(frame, segment.kind)
            preview.set_source(_data_url(frame))
            preview.set_visibility(True)
            placeholder.set_visibility(False)

    table.on("remove", lambda e: remove_segment(e.args))
    table.on("rowClick", lambda e: preview_segment(e.args[1]))

    def set_busy(busy: bool) -> None:
        analyze_btn.set_enabled(not busy)
        cancel_btn.set_enabled(busy)
        render_btn.set_enabled(not busy and analysis is not None)
        for el in (video, output, language, model, device, fps, confidence, cigarette, *checks.values()):
            el.set_enabled(not busy)

    def drain() -> None:
        if job is None:
            return
        if job.preview is not None and live.value:
            frame, job.preview = job.preview, None
            preview.set_source(_data_url(frame))
            preview.set_visibility(True)
            placeholder.set_visibility(False)
        while job.events:
            match event := job.events.popleft():
                case Progress():
                    progress.set_value(event.fraction)
                    elapsed = time.monotonic() - job.started
                    eta = elapsed / event.fraction - elapsed if event.fraction > 0.02 else None
                    status.set_text(
                        f"{STAGE_TEXT[event.stage]} · {event.fraction:.0%}"
                        + (f" · about {_clock(eta)} left" if eta is not None else "")
                    )
                case Log():
                    log.push(event.message, classes=LOG_COLORS.get(event.level, ""))

    ui.timer(0.2, drain)

    async def run_job(stage: str, fn, *args, **kwargs):
        nonlocal job
        job = Job()
        progress.set_value(0)
        status.set_text(f"{STAGE_TEXT[stage]}…")
        set_busy(True)
        try:
            return await run.io_bound(fn, *args, on_event=job.on_event, cancel=job.cancel, **kwargs)
        except Exception as exc:  # surfaced to the user rather than crashing the page
            job.on_event(Log(str(exc), "negative"))
            status.set_text("Failed, see the log for details")
            ui.notify(str(exc).splitlines()[0], type="negative", multi_line=True)
            return None
        finally:
            drain()
            job = None
            set_busy(False)

    async def analyze() -> None:
        nonlocal analysis, source, pipeline
        src = Path(video.value or "").expanduser()
        if not src.is_file():
            ui.notify("Select an existing video file", type="warning")
            return
        if not language.value:
            ui.notify("No warning images found in Images/statutory", type="negative")
            return
        settings = current_settings()
        log.clear()
        done_actions.set_visibility(False)
        analysis = None
        show_segments()
        status.set_text("Loading model… (first run downloads it)")
        set_busy(True)
        try:
            loaded = await run.io_bound(CACHE.get, settings)
        except Exception as exc:
            ui.notify(str(exc).splitlines()[0], type="negative", multi_line=True)
            status.set_text("Could not load the model")
            set_busy(False)
            return
        source = src
        result = await run_job("analyzing", loaded.analyze, src)
        if result is not None:
            analysis, pipeline = result, loaded
            progress.set_value(1)
            speed = result.info.duration / result.seconds if result.seconds else 0
            status.set_text(f"Found {len(result.segments)} warning segment(s) in {result.seconds:.0f}s ({speed:.0f}× real time). Review, then Render.")
        elif status.text.startswith("Analyzing"):
            status.set_text("Cancelled")
        show_segments()

    async def render() -> None:
        nonlocal output_file
        if analysis is None or source is None or pipeline is None:
            return
        dst = Path(output.value).expanduser() if output.value else default_output_path(source)
        ok = await run_job("rendering", pipeline.render, source, dst, analysis)
        if ok:
            output_file = dst
            progress.set_value(1)
            status.set_text(f"Saved {dst}")
            done_actions.set_visibility(True)
            ui.notify("Video rendered", type="positive")
        elif ok is False:
            status.set_text("Cancelled")

    def cancel() -> None:
        if job is not None:
            job.cancel.cancel()
            status.set_text("Cancelling…")

    analyze_btn.on_click(analyze)
    render_btn.on_click(render)
    cancel_btn.on_click(cancel)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="censorlabel-gui")
    parser.add_argument("--browser", action="store_true", help="open in the browser instead of a native window")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)

    native = not args.browser and importlib.util.find_spec("webview") is not None
    ui.run(
        build_page,
        title="CensorLabel",
        native=native,
        window_size=(1280, 860) if native else None,
        port=args.port,
        reload=False,
        show=not native and args.port is None,
        favicon="🎬",
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
