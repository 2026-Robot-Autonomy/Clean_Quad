"""
Classical baseline: colour-histogram Bayes classifier + blobs.

Fitting (no learning beyond counting):
    From the training images and their class masks, count how
    often each quantised HSV colour belongs to each class
    (background, the litter classes, distractor). That gives a
    lookup table P(class | colour).

Detection:
    1. every pixel takes its most likely class
       (if that probability >= threshold)
    2. per litter class: morphological clean-up (opening,
       closing), connected components (scipy.ndimage)
    3. each component large enough -> one box; score = mean
       class probability over its pixels

Threshold and minimum blob area are tuned on the validation set.
"""

from pathlib import Path

import numpy as np
from matplotlib.colors import rgb_to_hsv
from scipy import ndimage

from src.config import PROJECT_ROOT
from src.environment.litter import LITTER_CLASSES
from src.perception.detection import Detection, Detector
from src.perception.labels import DISTRACTOR


MODEL_PATH = PROJECT_ROOT / "models" / "color_baseline.npz"

H_BINS, S_BINS, V_BINS = 32, 16, 16
N_CLASSES = DISTRACTOR + 1          # background, litter classes, distractor
SMOOTHING = 1.0                     # Laplace smoothing of the counts

OPEN_KERNEL = np.ones((3, 3), bool)
CLOSE_KERNEL = np.ones((7, 7), bool)
CONNECTIVITY = np.ones((3, 3), bool)


def colour_bins(rgb):
    """Quantised HSV bin index per pixel."""

    hsv = rgb_to_hsv(rgb.astype(np.float32) / 255.0)      # all channels 0..1

    def quantise(channel, n):
        return np.minimum((channel * n).astype(np.int32), n - 1)

    h = quantise(hsv[..., 0], H_BINS)
    s = quantise(hsv[..., 1], S_BINS)
    v = quantise(hsv[..., 2], V_BINS)

    return (h * S_BINS + s) * V_BINS + v


class ColorDetector(Detector):

    name = "colour baseline"

    def __init__(self, posterior, threshold=0.5, min_area=30):

        self.posterior = posterior.astype(np.float32)   # (bins, N_CLASSES)
        self.threshold = threshold
        self.min_area = min_area

    # -----------------------------------------------------
    # FITTING
    # -----------------------------------------------------

    @classmethod
    def fit(cls, samples, load_rgb, load_mask):
        """samples: dataset records with 'image' and 'mask' paths."""

        n_bins = H_BINS * S_BINS * V_BINS
        counts = np.zeros((n_bins, N_CLASSES), dtype=np.float64)

        for sample in samples:
            bins = colour_bins(load_rgb(sample["image"])).ravel()
            labels = load_mask(sample["mask"]).ravel().astype(np.int64)
            counts += np.bincount(
                bins * N_CLASSES + labels, minlength=n_bins * N_CLASSES
            ).reshape(n_bins, N_CLASSES)

        counts += SMOOTHING
        posterior = counts / counts.sum(axis=1, keepdims=True)

        return cls(posterior)

    def save(self, path=MODEL_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, posterior=self.posterior, threshold=self.threshold, min_area=self.min_area
        )

    @classmethod
    def load(cls, path=MODEL_PATH):
        f = np.load(path)
        return cls(f["posterior"], float(f["threshold"]), int(f["min_area"]))

    # -----------------------------------------------------
    # DETECTION
    # -----------------------------------------------------

    def detect(self, rgb):

        probs = self.posterior[colour_bins(rgb)]            # (H, W, N_CLASSES)
        label = probs.argmax(axis=2)
        confident = probs.max(axis=2) >= self.threshold

        detections = []

        for class_id in range(len(LITTER_CLASSES)):

            mask = (label == class_id + 1) & confident
            if not mask.any():
                continue

            mask = ndimage.binary_opening(mask, OPEN_KERNEL)
            mask = ndimage.binary_closing(mask, CLOSE_KERNEL)

            components, n = ndimage.label(mask, CONNECTIVITY)
            if n == 0:
                continue

            index = np.arange(1, n + 1)
            areas = ndimage.sum_labels(mask, components, index)
            scores = ndimage.mean(probs[..., class_id + 1], components, index)

            for k, region in enumerate(ndimage.find_objects(components)):
                if areas[k] < self.min_area:
                    continue
                ys, xs = region
                detections.append(Detection(
                    class_id, float(scores[k]), (xs.start, ys.start, xs.stop, ys.stop)
                ))

        return detections
