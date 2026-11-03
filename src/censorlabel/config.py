"""Settings and static configuration: detector vocabulary, sampling and timeline thresholds."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from censorlabel.domain import WarningKind


def _default_home() -> Path:
    """Directory holding the warning images: $CENSORLABEL_HOME, the repo checkout, or the CWD."""
    if env := os.environ.get("CENSORLABEL_HOME"):
        return Path(env).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "Images" / "statutory").is_dir():
        return repo_root
    return Path.cwd()


# Text prompts for open-vocabulary models (YOLOE / YOLO-World). A fine-tuned model can instead
# output the direct labels below (e.g. "no_helmet"), and the rules accept both.
SMOKING_OBJECTS = ("cigarette", "cigar", "hookah")
ALCOHOL_OBJECTS = ("beer bottle", "beer glass", "wine glass", "liquor bottle", "whisky glass")
PERSON, MOTORCYCLE, HELMET = "person", "motorcycle", "helmet"
STEERING_WHEEL, SEATBELT = "steering wheel", "seat belt"
NO_HELMET, NO_SEATBELT = "no_helmet", "no_seatbelt"

PROMPTS: tuple[str, ...] = (
    *SMOKING_OBJECTS,
    *ALCOHOL_OBJECTS,
    PERSON,
    MOTORCYCLE,
    HELMET,
    STEERING_WHEEL,
    SEATBELT,
)

BASE_KINDS = (WarningKind.SMOKE, WarningKind.ALCOHOL, WarningKind.HELMET, WarningKind.SEATBELT)


@dataclass
class Settings:
    home: Path = field(default_factory=_default_home)
    language: str = "English"

    # Detector. Open-vocabulary YOLOE works without training; pass a fine-tuned .pt for best accuracy.
    model: str = "yoloe-26s-seg.pt"
    device: str = "auto"  # auto | cpu | cuda | mps
    image_size: int = 640
    confidence: float = 0.25
    # Small objects score lower in zero-shot mode; the timeline's hit window filters the extra noise.
    class_confidence: dict[str, float] = field(default_factory=lambda: {"cigarette": 0.15, "cigar": 0.15, "hookah": 0.15})
    batch_size: int = 8

    # Analysis samples a few frames per second instead of every frame.
    sample_fps: float = 4.0
    # A warning starts once `min_hits` of `window` consecutive samples show the violation.
    window: int = 3
    min_hits: int = 2
    hold_seconds: float = 3.0  # keep the banner up this long after the last hit
    merge_gap: float = 2.0  # join segments of the same kind separated by less than this
    min_duration: float = 4.0  # never flash a banner for less than this

    enabled: frozenset[WarningKind] = frozenset(BASE_KINDS)

    # Rendering.
    banner_width: float = 0.45  # fraction of the video width
    encoder: str = "auto"  # picks VideoToolbox (macOS) or NVENC when available
    encoder_preset: str = "veryfast"
    crf: int = 20

    def __post_init__(self) -> None:
        self.home = Path(self.home)

    @property
    def detector_confidence(self) -> float:
        """Lowest threshold of any class; detections are filtered per class afterwards."""
        return min([self.confidence, *self.class_confidence.values()])

    def keeps(self, label: str, confidence: float) -> bool:
        return confidence >= self.class_confidence.get(label, self.confidence)

    @property
    def warnings_dir(self) -> Path:
        return self.home / "Images" / "statutory"

    def available_languages(self) -> list[str]:
        if not self.warnings_dir.is_dir():
            return []
        return sorted(p.name for p in self.warnings_dir.iterdir() if p.is_dir())
