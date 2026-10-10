"""Runtime Stage 4 interface; no simulator ground truth is used here."""

from collections import Counter
from dataclasses import dataclass

import numpy as np

from src.localization.geometry import LocalizationConfig, localize
from src.localization.tracking import LitterTracker


@dataclass
class PipelineResult:
    observations: list
    rejected: dict
    assignments: dict


def _boxes_duplicate(a, b):
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    smaller = min(area_a, area_b)
    # A small fragment inside the object's box has low IoU even though nearly
    # its whole area overlaps. The caller also requires agreement in 3-D.
    return (
        (union > 0 and intersection / union >= 0.5)
        or (smaller > 0 and intersection / smaller >= 0.85)
    )


class LocalizationPipeline:
    """Feed Frame + Detection lists; read tracker.confirmed_tracks for Stage 5."""

    def __init__(self, method="depth", config=None, association_radius=0.20, min_observations=3, class_switch_radius=0.08):
        if method not in ("depth", "ground_plane"):
            raise ValueError(f"unknown localization method: {method}")
        self.method = method
        self.config = config or LocalizationConfig()
        self.tracker = LitterTracker(association_radius, min_observations, class_switch_radius=class_switch_radius)
        self.rejection_counts = Counter()

    def reset(self):
        self.tracker.reset()
        self.rejection_counts.clear()

    def update(self, frame, detections):
        detections = list(detections)
        observations, rejected = [], {}
        for index, detection in enumerate(detections):
            observation, reason = localize(frame, detection, self.config, self.method, index)
            if observation is None:
                rejected[index] = reason
                self.rejection_counts[reason] += 1
            else:
                observations.append(observation)
        # Class-specific detector NMS can leave two labels on one item. Require
        # agreement in image overlap/containment and world position before suppressing
        # the weaker box; nearby separate image boxes remain separate items.
        kept = []
        for observation in sorted(observations, key=lambda o: (-o.score, o.source_index)):
            duplicate = any(
                np.linalg.norm(observation.position - other.position) <= self.tracker.class_switch_radius
                and _boxes_duplicate(detections[observation.source_index].box, detections[other.source_index].box)
                for other in kept
            )
            if duplicate:
                rejected[observation.source_index] = "duplicate_box"
                self.rejection_counts["duplicate_box"] += 1
            else:
                kept.append(observation)
        observations = sorted(kept, key=lambda o: o.source_index)
        assignments = self.tracker.update(observations, frame.time)
        return PipelineResult(observations, rejected, assignments)
