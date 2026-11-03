"""Ultralytics YOLO detector: open-vocabulary YOLOE out of the box, or any fine-tuned YOLO model."""

from __future__ import annotations

import contextlib
import hashlib
from pathlib import Path

import numpy as np

from censorlabel.domain import Box, Detection


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class YoloDetector:
    """Runs one YOLO model over batches of frames.

    For open-vocabulary models (file name starting with "yoloe" or containing "world") the
    `prompts` become the class list; their text embeddings are cached next to the model so the
    large text encoder is only needed once. Fine-tuned models keep their own class names.
    """

    def __init__(
        self,
        model: str | Path,
        prompts: tuple[str, ...],
        *,
        models_dir: Path,
        device: str = "auto",
        image_size: int = 640,
        confidence: float = 0.25,
    ):
        from ultralytics import YOLO, YOLOE

        models_dir.mkdir(parents=True, exist_ok=True)
        path = Path(model)
        if not path.is_absolute() and not path.exists():
            path = models_dir / path  # Ultralytics downloads known assets to this path
        name = path.name.lower()
        self.open_vocabulary = name.startswith("yoloe") or "world" in name

        with contextlib.chdir(models_dir):  # text encoder downloads land in models_dir too
            if self.open_vocabulary:
                self.model = YOLOE(str(path))
                self._set_prompts(prompts, models_dir / f"{path.stem}-{_digest(prompts)}.prompts.pt")
            else:
                self.model = YOLO(str(path))

        self.device = pick_device(device)
        self.image_size = image_size
        self.confidence = confidence

    def _set_prompts(self, prompts: tuple[str, ...], cache: Path) -> None:
        import torch

        if cache.exists():
            embeddings = torch.load(cache)
        else:
            embeddings = self.model.get_text_pe(list(prompts))
            torch.save(embeddings, cache)
        self.model.set_classes(list(prompts), embeddings)

    @property
    def labels(self) -> list[str]:
        return list(self.model.names.values())

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        results = self.model.predict(
            frames, imgsz=self.image_size, conf=self.confidence, device=self.device, verbose=False, retina_masks=False
        )
        out = []
        for result in results:
            names = result.names
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            scores = boxes.conf.cpu().numpy()
            out.append([Detection(names[c], float(s), Box.from_xyxy(*b)) for b, c, s in zip(xyxy, classes, scores)])
        return out


def _digest(prompts: tuple[str, ...]) -> str:
    return hashlib.sha1("\n".join(prompts).encode()).hexdigest()[:8]
