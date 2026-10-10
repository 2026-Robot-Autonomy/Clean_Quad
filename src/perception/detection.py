"""
Common detection types and metrics, shared by the classical
baseline and the CNN so both are scored identically.

Metrics follow the usual object-detection definitions:
a detection is a true positive if it overlaps an unmatched
ground-truth box of the same class with IoU >= threshold.
AP uses 101-point interpolation (COCO style).
"""

from dataclasses import dataclass

import numpy as np

from src.environment.litter import LITTER_CLASSES


@dataclass
class Detection:
    class_id: int
    score: float
    box: tuple              # (x0, y0, x1, y1) pixels

    @property
    def kind(self):
        return LITTER_CLASSES[self.class_id]


class Detector:
    """Interface: every detector maps an RGB image to detections."""

    name = "detector"

    def detect(self, rgb):
        raise NotImplementedError

    def detect_batch(self, images):
        return [self.detect(rgb) for rgb in images]


# =========================================================
# GEOMETRY
# =========================================================

def iou(a, b):

    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter

    return inter / union if union > 0 else 0.0


def match_image(detections, gt_boxes, iou_threshold=0.5):
    """
    Greedy matching in descending score order.

    gt_boxes: list of (class_id, box)
    Returns (is_true_positive per detection, matched gt index or -1).
    """

    order = sorted(range(len(detections)), key=lambda i: -detections[i].score)
    used = set()
    tp = [False] * len(detections)
    matched = [-1] * len(detections)

    for i in order:
        det = detections[i]
        best, best_iou = -1, iou_threshold

        for j, (cls, box) in enumerate(gt_boxes):
            if j in used or cls != det.class_id:
                continue
            overlap = iou(det.box, box)
            if overlap >= best_iou:
                best, best_iou = j, overlap

        if best >= 0:
            used.add(best)
            tp[i] = True
            matched[i] = best

    return tp, matched


# =========================================================
# AVERAGE PRECISION
# =========================================================

def average_precision(predictions, ground_truths, class_id, iou_threshold):
    """
    predictions:   list (per image) of Detection lists
    ground_truths: list (per image) of (class_id, box) lists
    """

    n_gt = sum(1 for gts in ground_truths for c, _ in gts if c == class_id)
    if n_gt == 0:
        return float("nan")

    scored = []

    for dets, gts in zip(predictions, ground_truths):
        dets = [d for d in dets if d.class_id == class_id]
        gts = [g for g in gts if g[0] == class_id]
        tp, _ = match_image(dets, gts, iou_threshold)
        scored.extend(zip([d.score for d in dets], tp))

    if not scored:
        return 0.0

    scored.sort(key=lambda s: -s[0])
    tp = np.cumsum([t for _, t in scored])
    fp = np.cumsum([not t for _, t in scored])

    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-9)

    # Precision envelope, then sample at 101 recall points.
    for k in range(len(precision) - 2, -1, -1):
        precision[k] = max(precision[k], precision[k + 1])

    points = np.linspace(0.0, 1.0, 101)
    idx = np.searchsorted(recall, points, side="left")

    return float(np.mean([precision[i] if i < len(precision) else 0.0 for i in idx]))


def evaluate(predictions, ground_truths, score_threshold=0.25):
    """
    Standard summary for a detector.

    Returns dict with mAP50, mAP50_95, per-class AP50, and
    precision / recall / F1 at `score_threshold`.
    """

    classes = range(len(LITTER_CLASSES))
    thresholds = np.arange(0.50, 0.96, 0.05)

    ap50 = {LITTER_CLASSES[c]: average_precision(predictions, ground_truths, c, 0.5) for c in classes}
    ap_range = [
        np.nanmean([average_precision(predictions, ground_truths, c, t) for c in classes])
        for t in thresholds
    ]

    tp = fp = 0
    n_gt = sum(len(g) for g in ground_truths)

    for dets, gts in zip(predictions, ground_truths):
        dets = [d for d in dets if d.score >= score_threshold]
        flags, _ = match_image(dets, gts, 0.5)
        tp += sum(flags)
        fp += len(flags) - sum(flags)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(n_gt, 1)

    return {
        "mAP50": float(np.nanmean(list(ap50.values()))),
        "mAP50_95": float(np.mean(ap_range)),
        "AP50": ap50,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "true_positives": tp,
        "false_positives": fp,
        "ground_truth": n_gt,
    }


def pr_curve(predictions, ground_truths, iou_threshold=0.5):
    """
    Precision and recall over all classes as the score
    threshold is lowered (for plotting).
    """

    n_gt = sum(len(g) for g in ground_truths)
    scored = []

    for dets, gts in zip(predictions, ground_truths):
        tp, _ = match_image(dets, gts, iou_threshold)
        scored.extend(zip([d.score for d in dets], tp))

    scored.sort(key=lambda s: -s[0])
    tp = np.cumsum([t for _, t in scored])
    fp = np.cumsum([not t for _, t in scored])

    return tp / max(n_gt, 1), tp / np.maximum(tp + fp, 1e-9)
