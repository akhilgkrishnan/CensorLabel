"""CensorLabel: automatic statutory-warning labelling for movies."""

from censorlabel.config import Settings
from censorlabel.domain import Segment, WarningKind
from censorlabel.pipeline import LabelingPipeline

__all__ = ["LabelingPipeline", "Segment", "Settings", "WarningKind"]
__version__ = "2.0.0"
