"""
Stage 3: train both litter detectors on data/litter_dataset.

1. Colour baseline: fit P(class | colour) on the training set,
   tune threshold and minimum blob area on the validation set.
2. CNN: fine-tune YOLOv8n (GPU recommended, ~30-60 min on a
   GTX 1650).

    python experiments/23_train_detectors.py                 # both
    python experiments/23_train_detectors.py --skip-yolo     # baseline only
    python experiments/23_train_detectors.py --skip-color --epochs 50
"""

from pathlib import Path
import argparse
import itertools
import sys
import time

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from src.perception.color_detector import MODEL_PATH as COLOR_MODEL_PATH, ColorDetector
from src.perception.dataset import DATASET_DIR, load_rgb, load_split
from src.perception.detection import evaluate
from src.perception.yolo_detector import train_yolo


THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
MIN_AREAS = (15, 30, 60)


def load_mask(path):
    return np.asarray(Image.open(path))


def train_color_baseline():

    print("\n1. COLOUR BASELINE")

    start = time.perf_counter()
    detector = ColorDetector.fit(load_split("train"), load_rgb, load_mask)
    print(f"   fitted colour model on training set ({time.perf_counter() - start:.0f} s)")

    val = load_split("val")
    images = [load_rgb(s["image"]) for s in val]
    truths = [[(c, b) for c, b, _ in s["litter"]] for s in val]

    best = None

    for threshold, min_area in itertools.product(THRESHOLDS, MIN_AREAS):
        detector.threshold, detector.min_area = threshold, min_area
        score = evaluate([detector.detect(im) for im in images], truths)["mAP50"]
        print(f"   threshold {threshold:.1f}  min_area {min_area:3d}  ->  val mAP50 {score:.3f}")
        if best is None or score > best[0]:
            best = (score, threshold, min_area)

    _, detector.threshold, detector.min_area = best
    detector.save()

    print(
        f"   best: threshold {detector.threshold}, min_area {detector.min_area}, "
        f"val mAP50 {best[0]:.3f}\n   saved {COLOR_MODEL_PATH.relative_to(PROJECT_ROOT)}"
    )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-color", action="store_true")
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="'0' = first GPU, 'cpu' = CPU")
    args = parser.parse_args()

    print("=" * 70)
    print("CLEANQUAD STAGE 3: TRAIN DETECTORS")
    print("=" * 70)

    if not (DATASET_DIR / "data.yaml").exists():
        sys.exit("No dataset: run experiments/22_generate_dataset.py first")

    if not args.skip_color:
        train_color_baseline()

    if not args.skip_yolo:
        print("\n2. YOLOv8n")
        start = time.perf_counter()
        path = train_yolo(
            DATASET_DIR / "data.yaml",
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
        )
        print(
            f"   trained in {(time.perf_counter() - start) / 60:.1f} min, "
            f"saved {path.relative_to(PROJECT_ROOT)}"
        )


if __name__ == "__main__":
    main()
