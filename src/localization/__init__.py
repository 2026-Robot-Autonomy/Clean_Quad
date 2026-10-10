"""Stage 4: RGB-D detections to a persistent list of litter positions."""

from src.localization.geometry import LocalizationConfig, LocalizedDetection, localize
from src.localization.pipeline import LocalizationPipeline
from src.localization.tracking import LitterTracker

__all__ = [
    "LocalizationConfig", "LocalizedDetection", "localize",
    "LocalizationPipeline", "LitterTracker",
]
