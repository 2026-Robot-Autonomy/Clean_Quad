"""
Ground truth from the simulator: every pixel of a segmentation
image knows which geom it shows, so boxes need no hand labels.
"""

from dataclasses import dataclass

import numpy as np
import mujoco

from src.environment.litter import LITTER_CLASSES


# Pixel classes in the saved class masks.
BACKGROUND = 0
DISTRACTOR = len(LITTER_CLASSES) + 1      # litter classes are 1..N

MIN_PIXELS = 40
MIN_SIDE = 4


@dataclass
class GroundTruthBox:
    name: str
    category: str           # "litter" | "distractor"
    kind: str
    class_id: int           # litter class index, -1 for distractors
    box: tuple              # (x0, y0, x1, y1) pixels, x1/y1 exclusive
    pixels: int
    distance: float         # camera -> object centre [m]


def object_geom_ids(model, objects):
    """geom id -> GroundObject for litter and distractors."""

    return {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, o.geom_name): o
        for o in objects
    }


def class_id_of(obj):
    return LITTER_CLASSES.index(obj.kind) if obj.category == "litter" else -1


def ground_truth_boxes(frame, geom_map, min_pixels=MIN_PIXELS, min_side=MIN_SIDE):
    """
    One box per visible object. Objects that are tiny, far or
    mostly hidden (fewer than min_pixels) are skipped.
    """

    boxes = []
    ids, counts = np.unique(frame.geom_ids, return_counts=True)

    for gid, count in zip(ids, counts):

        obj = geom_map.get(int(gid))

        if obj is None or obj.collected or count < min_pixels:
            continue

        ys, xs = np.nonzero(frame.geom_ids == gid)
        box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

        if min(box[2] - box[0], box[3] - box[1]) < min_side:
            continue

        boxes.append(GroundTruthBox(
            name=obj.name,
            category=obj.category,
            kind=obj.kind,
            class_id=class_id_of(obj),
            box=box,
            pixels=int(count),
            distance=float(np.linalg.norm(obj.position - frame.cam_pos)),
        ))

    return boxes


def class_mask(frame, geom_map):
    """
    Per-pixel class image (uint8): 0 background, 1..N litter
    classes, DISTRACTOR for distractors.
    """

    mask = np.zeros(frame.geom_ids.shape, dtype=np.uint8)

    for gid, obj in geom_map.items():
        if obj.collected:
            continue
        value = class_id_of(obj) + 1 if obj.category == "litter" else DISTRACTOR
        mask[frame.geom_ids == gid] = value

    return mask
