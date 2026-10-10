"""
Stage 3: compare the colour baseline and the YOLOv8n CNN on the
held-out test scenes (never seen in training or tuning).

Each detector's operating score threshold is the one that
maximises F1 on the VALIDATION set; test numbers use it as is.

    python experiments/24_evaluate_detectors.py

Outputs:
    results/logs/perception/detector_comparison.csv
    results/logs/perception/per_class_ap50.csv
    results/logs/perception/recall_by_distance.csv
    results/figures/detector_comparison.png
    results/figures/detection_examples.png
"""

from pathlib import Path
import csv
import math
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from src.environment.litter import LITTER_CLASSES
from src.perception.color_detector import ColorDetector
from src.perception.dataset import load_rgb, load_split
from src.perception.detection import evaluate, iou, match_image, pr_curve
from src.perception.yolo_detector import YoloDetector


LOG_DIR = PROJECT_ROOT / "results" / "logs" / "perception"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

DISTANCE_BINS = ((0.0, 1.0, "< 1 m"), (1.0, 2.0, "1-2 m"), (2.0, math.inf, "> 2 m"))
DISTRACTOR_IOU = 0.3
LATENCY_FRAMES = 50

# Fixed identity colours (validated categorical slots 1 and 2).
COLOURS = {"YOLOv8n (CNN)": "#2a78d6", "colour baseline": "#eb6834"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"


# ============================================================
# RUNNING
# ============================================================

def truths_of(samples):
    return [[(c, b) for c, b, _ in s["litter"]] for s in samples]


def run(detector, images):
    """Predictions for all images, and throughput [images/s]."""

    start = time.perf_counter()
    predictions = detector.detect_batch(images)
    return predictions, len(images) / (time.perf_counter() - start)


def latency_ms(detector, images):
    """Single-frame latency, as on the robot (one image at a time)."""

    detector.detect(images[0])                      # warm-up
    start = time.perf_counter()
    for rgb in images[:LATENCY_FRAMES]:
        detector.detect(rgb)
    return 1000.0 * (time.perf_counter() - start) / LATENCY_FRAMES


def best_f1_threshold(predictions, truths):

    best = (0.0, 0.5)

    for t in np.arange(0.05, 0.96, 0.01):
        f1 = evaluate(predictions, truths, score_threshold=t)["f1"]
        if f1 > best[0]:
            best = (f1, float(t))

    return best[1]


def above(predictions, threshold):
    return [[d for d in dets if d.score >= threshold] for dets in predictions]


def recall_by_distance(predictions, samples):

    found = {label: [0, 0] for _, _, label in DISTANCE_BINS}

    for dets, sample in zip(predictions, samples):
        gts = sample["litter"]
        _, matched = match_image(dets, [(c, b) for c, b, _ in gts])
        hit = set(matched)

        for j, (_, _, distance) in enumerate(gts):
            for lo, hi, label in DISTANCE_BINS:
                if lo <= distance < hi:
                    found[label][0] += j in hit
                    found[label][1] += 1

    return {label: (k / n if n else math.nan, n) for label, (k, n) in found.items()}


def false_alarms(predictions, samples):
    """(false positives per image, of which on a distractor)."""

    fp = on_distractor = 0

    for dets, sample in zip(predictions, samples):
        tp, _ = match_image(dets, [(c, b) for c, b, _ in sample["litter"]])
        for det, ok in zip(dets, tp):
            if ok:
                continue
            fp += 1
            if any(iou(det.box, box) >= DISTRACTOR_IOU for box in sample["distractors"]):
                on_distractor += 1

    n = len(samples)
    return fp / n, on_distractor / n


# ============================================================
# FIGURES
# ============================================================

def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)


def grouped_bars(ax, labels, series, title, ylabel):
    """series: {detector name: values}. Values printed above bars."""

    names = list(series)
    width = 0.36
    x = np.arange(len(labels))

    for k, name in enumerate(names):
        offset = (k - (len(names) - 1) / 2) * (width + 0.02)
        values = np.nan_to_num(series[name])
        bars = ax.bar(x + offset, values, width, color=COLOURS[name], label=name, zorder=2)
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.2f}",
                ha="center", va="bottom", fontsize=7.5, color=INK_SECONDARY,
            )

    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    style(ax, title)


def comparison_figure(results):

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), facecolor=SURFACE)
    names = list(results)

    metrics = ["mAP50", "mAP50_95", "precision", "recall", "f1"]
    grouped_bars(
        axes[0, 0], ["mAP@0.5", "mAP@0.5:0.95", "precision", "recall", "F1"],
        {n: [results[n]["metrics"][m] for m in metrics] for n in names},
        "(a) Overall, test scenes", "score",
    )

    grouped_bars(
        axes[0, 1], list(LITTER_CLASSES),
        {n: [results[n]["metrics"]["AP50"][c] for c in LITTER_CLASSES] for n in names},
        "(b) AP@0.5 per litter class", "AP@0.5",
    )

    bins = [label for _, _, label in DISTANCE_BINS]
    grouped_bars(
        axes[1, 0], bins,
        {n: [results[n]["distance"][b][0] for b in bins] for n in names},
        "(c) Recall by distance (operating threshold)", "recall",
    )

    ax = axes[1, 1]
    for n in names:
        recall, precision = results[n]["pr"]
        ax.plot(recall, precision, color=COLOURS[n], lw=2, label=n)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("recall", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("precision", color=INK_SECONDARY, fontsize=9)
    style(ax, "(d) Precision-recall, IoU 0.5, all classes")
    ax.grid(axis="x", color=GRID, lw=0.8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURE_DIR / "detector_comparison.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def examples_figure(samples, images, results, n=4):
    """Same test images, both detectors (top: CNN, bottom: baseline)."""

    picks = [i for i, s in enumerate(samples) if len(s["litter"]) >= 3 and s["distractors"]][:n]
    names = list(results)

    fig, axes = plt.subplots(
        len(names), n, figsize=(4.2 * n, 3.4 * len(names)), facecolor=SURFACE, squeeze=False
    )

    for row, name in enumerate(names):
        for col, i in enumerate(picks):
            ax = axes[row, col]
            ax.imshow(images[i])
            for x0, y0, x1, y1 in samples[i]["distractors"]:
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="white", ls="--", lw=0.8))
            for d in results[name]["operating"][i]:
                x0, y0, x1, y1 = d.box
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=COLOURS[name], lw=1.6))
                ax.text(x0, y0 - 3, f"{d.kind} {d.score:.2f}", fontsize=6.5, color="white",
                        bbox=dict(fc=COLOURS[name], ec="none", pad=1))
            ax.axis("off")
            if col == 0:
                ax.set_title(name, loc="left", fontsize=10, color=INK)

    fig.suptitle("Test-scene detections (dashed white = distractor, not litter)", color=INK)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "detection_examples.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 96)
    print("CLEANQUAD STAGE 3: DETECTOR EVALUATION (test scenes)")
    print("=" * 96)

    val, test = load_split("val"), load_split("test")
    val_images = [load_rgb(s["image"]) for s in val]
    test_images = [load_rgb(s["image"]) for s in test]
    val_truths, test_truths = truths_of(val), truths_of(test)

    print(
        f"test: {len(test)} images, {sum(len(t) for t in test_truths)} litter boxes, "
        f"{sum(len(s['distractors']) for s in test)} distractor boxes"
    )

    detectors = []

    try:
        detectors.append(YoloDetector())
    except (ImportError, FileNotFoundError) as error:
        print(f"\nWARNING: CNN skipped ({error}). Comparing the baseline alone.")

    detectors.append(ColorDetector.load())
    results = {}

    for detector in detectors:

        val_pred, _ = run(detector, val_images)
        threshold = best_f1_threshold(val_pred, val_truths)

        test_pred, fps = run(detector, test_images)
        operating = above(test_pred, threshold)

        results[detector.name] = {
            "threshold": threshold,
            "metrics": evaluate(test_pred, test_truths, score_threshold=threshold),
            "distance": recall_by_distance(operating, test),
            "false_alarms": false_alarms(operating, test),
            "pr": pr_curve(test_pred, test_truths),
            "fps": fps,
            "latency_ms": latency_ms(detector, test_images),
            "operating": operating,
        }

    # ---------------- table ----------------

    print(
        f"\n{'detector':<18}{'mAP50':>7}{'mAP50:95':>10}{'prec':>7}{'recall':>8}{'F1':>7}"
        f"{'thr':>6}{'FP/img':>8}{'on distr.':>10}{'img/s':>8}{'ms/img':>8}"
    )

    for name, r in results.items():
        m = r["metrics"]
        fp, fp_d = r["false_alarms"]
        print(
            f"{name:<18}{m['mAP50']:7.3f}{m['mAP50_95']:10.3f}{m['precision']:7.3f}{m['recall']:8.3f}"
            f"{m['f1']:7.3f}{r['threshold']:6.2f}{fp:8.2f}{fp_d:10.2f}{r['fps']:8.1f}{r['latency_ms']:8.1f}"
        )

    print("\nAP@0.5 per class")
    for name, r in results.items():
        print(f"  {name:<18}" + "  ".join(f"{c} {r['metrics']['AP50'][c]:.3f}" for c in LITTER_CLASSES))

    print("\nRecall by distance")
    for name, r in results.items():
        print(f"  {name:<18}" + "  ".join(f"{b} {v:.3f} (n={n})" for b, (v, n) in r["distance"].items()))

    # ---------------- files ----------------

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    with open(LOG_DIR / "detector_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["detector", "mAP50", "mAP50_95", "precision", "recall", "f1", "threshold",
                    "fp_per_image", "fp_on_distractor_per_image", "images_per_s", "latency_ms"])
        for name, r in results.items():
            m = r["metrics"]
            w.writerow([name, m["mAP50"], m["mAP50_95"], m["precision"], m["recall"], m["f1"],
                        r["threshold"], *r["false_alarms"], r["fps"], r["latency_ms"]])

    with open(LOG_DIR / "per_class_ap50.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["detector", *LITTER_CLASSES])
        for name, r in results.items():
            w.writerow([name, *(r["metrics"]["AP50"][c] for c in LITTER_CLASSES)])

    with open(LOG_DIR / "recall_by_distance.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["detector", "distance_bin", "recall", "n"])
        for name, r in results.items():
            for b, (v, n) in r["distance"].items():
                w.writerow([name, b, v, n])

    comparison_figure(results)
    examples_figure(test, test_images, results)

    print(f"\nSaved: {LOG_DIR.relative_to(PROJECT_ROOT)}/, results/figures/detector_comparison.png, detection_examples.png")


if __name__ == "__main__":
    main()
