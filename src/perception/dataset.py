"""
Automatically labelled litter dataset rendered from the campus.

For every scene seed: build a campus with extra litter and
distractors, teleport the robot to several random views (most
aimed at an object), render, and write

    images/<split>/<id>.jpg     RGB image
    labels/<split>/<id>.txt     YOLO labels: class cx cy w h (normalised)
    masks/<split>/<id>.png      per-pixel class (classical baseline fitting)
    meta/<split>.csv            every visible object, incl. distractors

Train, val and test use disjoint scene seeds, so test scenes
are never seen in training.
"""

from pathlib import Path
import csv
import math

import numpy as np
import mujoco
from PIL import Image

from src.config import PROJECT_ROOT
from src.environment import build_campus
from src.environment.litter import LITTER_CLASSES
from src.perception.camera import RobotCamera
from src.perception.labels import class_mask, ground_truth_boxes, object_geom_ids


DATASET_DIR = PROJECT_ROOT / "data" / "litter_dataset"

# split -> (first scene seed, number of scenes)
SPLITS = {
    "train": (1000, 600),
    "val": (2000, 100),
    "test": (3000, 100),
}

VIEWS_PER_SCENE = 5
SCENE_LITTER = 16
SCENE_DISTRACTORS = 16

VIEW_DISTANCE = (0.4, 3.0)          # robot -> aimed object [m]
VIEW_BEARING_DEG = 30.0             # aimed object up to this far off-centre
AIM_PROBABILITY = {"litter": 0.70, "distractor": 0.15}   # rest: random view

TRUNK_HEIGHT = (0.27, 0.32)         # standing .. walking
TRUNK_TILT_DEG = 3.0                # random roll / pitch (gait sway)
ROBOT_CLEARANCE = 0.30

META_FIELDS = [
    "image", "name", "category", "kind", "class_id",
    "x0", "y0", "x1", "y1", "pixels", "distance",
]


# =========================================================
# VIEWS
# =========================================================

def robot_fits(layout, x, y):

    if not layout.is_traversable(x, y):
        return False

    for k in range(8):
        a = k * math.pi / 4.0
        if not layout.is_traversable(x + ROBOT_CLEARANCE * math.cos(a), y + ROBOT_CLEARANCE * math.sin(a)):
            return False

    return True


def sample_view(campus, rng):
    """Random robot (x, y, yaw), usually looking at an object."""

    layout = campus.layout
    r = rng.random()

    if r < AIM_PROBABILITY["litter"]:
        targets = campus.litter
    elif r < AIM_PROBABILITY["litter"] + AIM_PROBABILITY["distractor"]:
        targets = campus.distractors
    else:
        targets = None

    for _ in range(1000):

        if targets:
            target = targets[rng.integers(len(targets))].position
            heading = rng.uniform(-math.pi, math.pi)
            d = rng.uniform(*VIEW_DISTANCE)
            x = target[0] - d * math.cos(heading)
            y = target[1] - d * math.sin(heading)
            yaw = heading + math.radians(rng.uniform(-VIEW_BEARING_DEG, VIEW_BEARING_DEG))
        else:
            x = rng.uniform(0.0, layout.width)
            y = rng.uniform(0.0, layout.depth)
            yaw = rng.uniform(-math.pi, math.pi)

        if robot_fits(layout, x, y):
            return x, y, yaw

    raise RuntimeError("no valid robot view found")


def set_view(campus, x, y, yaw, rng):
    """Teleport the robot; random trunk height and sway like walking."""

    data = campus.data
    tilt = math.radians(TRUNK_TILT_DEG)

    data.qpos[0] = x
    data.qpos[1] = y
    data.qpos[2] = campus.layout.ground_height_at(x, y) + rng.uniform(*TRUNK_HEIGHT)

    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, [rng.uniform(-tilt, tilt), rng.uniform(-tilt, tilt), yaw], "xyz")
    data.qpos[3:7] = quat

    mujoco.mj_forward(campus.model, data)


# =========================================================
# WRITING
# =========================================================

def yolo_lines(boxes, width, height):

    lines = []

    for b in boxes:
        if b.category != "litter":
            continue
        x0, y0, x1, y1 = b.box
        lines.append(
            f"{b.class_id} {(x0 + x1) / 2 / width:.6f} {(y0 + y1) / 2 / height:.6f} "
            f"{(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}"
        )

    return lines


def write_data_yaml(root=DATASET_DIR):

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(LITTER_CLASSES))
    (root / "data.yaml").write_text(
        f"path: {root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"names:\n{names}\n"
    )


def generate_split(split, root=DATASET_DIR, progress=print):

    first_seed, n_scenes = SPLITS[split]
    rng = np.random.default_rng(first_seed)

    for sub in ("images", "labels", "masks"):
        (root / sub / split).mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)

    counts = {"images": 0, "litter": 0, "distractor": 0}

    with open(root / "meta" / f"{split}.csv", "w", newline="") as f:

        writer = csv.DictWriter(f, fieldnames=META_FIELDS)
        writer.writeheader()

        for seed in range(first_seed, first_seed + n_scenes):

            campus = build_campus(
                seed=seed,
                litter_count=SCENE_LITTER,
                distractor_count=SCENE_DISTRACTORS,
            )
            geom_map = object_geom_ids(campus.model, campus.litter + campus.distractors)

            with RobotCamera(campus.model, campus.data) as camera:

                for view in range(VIEWS_PER_SCENE):

                    set_view(campus, *sample_view(campus, rng), rng)
                    frame = camera.capture(depth=False)

                    image_id = f"{seed}_{view}"
                    boxes = ground_truth_boxes(frame, geom_map)
                    h, w = frame.rgb.shape[:2]

                    Image.fromarray(frame.rgb).save(root / "images" / split / f"{image_id}.jpg", quality=92)
                    Image.fromarray(class_mask(frame, geom_map)).save(root / "masks" / split / f"{image_id}.png")
                    (root / "labels" / split / f"{image_id}.txt").write_text(
                        "\n".join(yolo_lines(boxes, w, h))
                    )

                    for b in boxes:
                        writer.writerow({
                            "image": image_id, "name": b.name, "category": b.category,
                            "kind": b.kind, "class_id": b.class_id,
                            "x0": b.box[0], "y0": b.box[1], "x1": b.box[2], "y1": b.box[3],
                            "pixels": b.pixels, "distance": f"{b.distance:.3f}",
                        })
                        counts[b.category] += 1

                    counts["images"] += 1

            if (seed - first_seed + 1) % 50 == 0:
                progress(f"  {split}: {seed - first_seed + 1}/{n_scenes} scenes")

    return counts


# =========================================================
# READING
# =========================================================

def load_split(split, root=DATASET_DIR):
    """
    List of samples:
        {"id", "image", "mask",
         "litter": [(class_id, box, distance)],
         "distractors": [box]}
    """

    samples = {
        p.stem: {
            "id": p.stem,
            "image": p,
            "mask": root / "masks" / split / f"{p.stem}.png",
            "litter": [],
            "distractors": [],
        }
        for p in sorted((root / "images" / split).glob("*.jpg"))
    }

    with open(root / "meta" / f"{split}.csv", newline="") as f:
        for row in csv.DictReader(f):
            box = tuple(int(row[k]) for k in ("x0", "y0", "x1", "y1"))
            sample = samples[row["image"]]
            if row["category"] == "litter":
                sample["litter"].append((int(row["class_id"]), box, float(row["distance"])))
            else:
                sample["distractors"].append(box)

    return list(samples.values())


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))
