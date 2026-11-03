"""Statutory-warning banners: load, make transparent, and composite or export for ffmpeg."""

from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np

from censorlabel.domain import WarningKind

BLACK_FLOOR = 16


def load_banner(path: Path) -> np.ndarray:
    """Load a banner as BGRA. Images without alpha use black as transparent (the bundled art style)."""
    image = cv.imread(str(path), cv.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Missing warning image {path}")
    if image.ndim == 2:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return image
    # Near-black (compression noise) is fully transparent; brighter pixels ramp up to opaque.
    peak = image.max(axis=2).astype(np.float32)
    alpha = np.clip((peak - BLACK_FLOOR) * 255 / (255 - BLACK_FLOOR), 0, 255).astype(np.uint8)
    # Un-premultiply so the colour survives compositing with the derived alpha.
    safe = np.maximum(alpha, 1)[..., None].astype(np.float32)
    color = np.clip(image.astype(np.float32) * 255 / safe, 0, 255).astype(np.uint8)
    return np.dstack([color, alpha])


class Banners:
    """Warning banners for one language, sized relative to the video frame."""

    def __init__(self, directory: Path, *, width_ratio: float = 0.45):
        if not directory.is_dir():
            raise FileNotFoundError(f"No warning images found in {directory}")
        self.directory = directory
        self.width_ratio = width_ratio
        self._cache: dict[tuple[WarningKind, int], np.ndarray] = {}

    def scaled(self, kind: WarningKind, frame_width: int) -> np.ndarray:
        key = (kind, frame_width)
        if key not in self._cache:
            banner = load_banner(self.directory / f"{kind.value}.png")
            width = max(int(frame_width * self.width_ratio), 1)
            height = max(int(banner.shape[0] * width / banner.shape[1]), 1)
            self._cache[key] = cv.resize(banner, (width, height), interpolation=cv.INTER_AREA)
        return self._cache[key]

    @staticmethod
    def position(frame_w: int, frame_h: int, banner_h: int) -> tuple[int, int]:
        """Bottom-left placement with margins proportional to the frame."""
        return int(frame_w * 0.02), max(frame_h - banner_h - int(frame_h * 0.05), 0)

    def export(self, kind: WarningKind, frame_width: int, path: Path) -> Path:
        cv.imwrite(str(path), self.scaled(kind, frame_width))
        return path

    def apply(self, frame: np.ndarray, kind: WarningKind) -> np.ndarray:
        fh, fw = frame.shape[:2]
        banner = self.scaled(kind, fw)
        x, y = self.position(fw, fh, banner.shape[0])
        h, w = min(banner.shape[0], fh - y), min(banner.shape[1], fw - x)
        out = frame.copy()
        alpha = banner[:h, :w, 3:].astype(np.float32) / 255
        region = out[y : y + h, x : x + w].astype(np.float32)
        out[y : y + h, x : x + w] = (banner[:h, :w, :3] * alpha + region * (1 - alpha)).astype(np.uint8)
        return out
