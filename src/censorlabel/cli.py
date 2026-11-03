"""Command-line interface: `censorlabel movie.mp4 -o labeled.mp4`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from censorlabel.config import Settings
from censorlabel.events import Event, Finished, Log, Progress, SegmentFound
from censorlabel.pipeline import CancelToken, LabelingPipeline, default_output_path


def _clock(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes // 60:d}:{minutes % 60:02d}:{secs:02d}"


def _print_event(event: Event, verbose: bool) -> None:
    match event:
        case Progress():
            print(f"\r  {event.stage:<10} {event.fraction:6.1%}", end="", flush=True)
        case SegmentFound(segment=seg):
            print(f"\n  ⚠ {seg.kind.description}: {_clock(seg.start)}–{_clock(seg.end)}", end="")
        case Log(level="debug") if not verbose:
            pass
        case Log():
            print(f"\n{event.message}")
        case Finished():
            print(f"Done. {len(event.segments)} warning segment(s).")


def main(argv: list[str] | None = None) -> int:
    defaults = Settings()
    parser = argparse.ArgumentParser(prog="censorlabel", description=__doc__)
    parser.add_argument("input", type=Path, help="input video")
    parser.add_argument("-o", "--output", type=Path, help="output video (default: <input>-labeled.<ext>)")
    parser.add_argument("-l", "--language", default=defaults.language, help="warning language (folder in Images/statutory)")
    parser.add_argument("-m", "--model", default=defaults.model, help="YOLOE model name or path to a fine-tuned .pt")
    parser.add_argument("--device", default=defaults.device, help="auto, cpu, cuda or mps")
    parser.add_argument("--fps", type=float, default=defaults.sample_fps, help="frames per second to analyze")
    parser.add_argument("--confidence", type=float, default=defaults.confidence)
    parser.add_argument("--encoder", default="auto", help="auto, libx264, h264_videotoolbox, h264_nvenc, …")
    parser.add_argument("--analyze-only", action="store_true", help="print the warning timeline as JSON, skip rendering")
    parser.add_argument("--home", type=Path, help="directory containing Images/ and models/ (default: repo root)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    settings = Settings(
        **({"home": args.home} if args.home else {}),
        language=args.language,
        model=args.model,
        device=args.device,
        sample_fps=args.fps,
        confidence=args.confidence,
        encoder=args.encoder,
    )
    if args.language not in settings.available_languages():
        parser.error(f"unknown language {args.language!r}; available: {', '.join(settings.available_languages())}")
    if not args.input.is_file():
        parser.error(f"no such file: {args.input}")

    pipeline = LabelingPipeline.from_settings(settings)
    cancel = CancelToken()
    on_event = (lambda e: None) if args.analyze_only else (lambda e: _print_event(e, args.verbose))
    try:
        if args.analyze_only:
            analysis = pipeline.analyze(args.input, cancel=cancel)
            assert analysis is not None
            json.dump(
                [{"warning": s.kind.value, "start": round(s.start, 2), "end": round(s.end, 2)} for s in analysis.segments],
                sys.stdout,
                indent=2,
            )
            print()
        else:
            pipeline.run(args.input, args.output or default_output_path(args.input), on_event=on_event, cancel=cancel)
    except KeyboardInterrupt:
        cancel.cancel()
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
