"""
Stage 3: render the automatically labelled litter dataset.

    python experiments/22_generate_dataset.py            # train / val / test
    python experiments/22_generate_dataset.py --preview  # also save label previews

Output: data/litter_dataset/ (see src/perception/dataset.py).
Needs a desktop session (OpenGL through GLFW).
"""

from pathlib import Path
import argparse
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from src.environment.litter import LITTER_CLASSES
from src.perception.dataset import (
    DATASET_DIR,
    SPLITS,
    generate_split,
    load_rgb,
    load_split,
    write_data_yaml,
)


FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

BOX_COLOURS = ["#2ca02c", "#d62728", "#1f77b4", "#ff7f0e"]


def save_preview(split="train", n=8):
    """Ground-truth boxes drawn on sample images (litter + distractors)."""

    samples = [s for s in load_split(split) if s["litter"]][:n]

    fig, axes = plt.subplots(2, n // 2, figsize=(4 * n // 2, 6.2))

    for ax, sample in zip(axes.flat, samples):
        ax.imshow(load_rgb(sample["image"]))
        for cls, (x0, y0, x1, y1), dist in sample["litter"]:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color=BOX_COLOURS[cls], lw=1.5))
            ax.text(x0, y0 - 3, f"{LITTER_CLASSES[cls]} {dist:.1f}m", color=BOX_COLOURS[cls], fontsize=7)
        for x0, y0, x1, y1 in sample["distractors"]:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="white", lw=1, ls="--"))
        ax.set_title(sample["id"], fontsize=8)
        ax.axis("off")

    fig.suptitle("Auto-generated labels (dashed white = distractor, not litter)")
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "dataset_preview.png", dpi=120)
    plt.close(fig)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("CLEANQUAD STAGE 3: DATASET GENERATION")
    print("=" * 70)

    start = time.perf_counter()

    for split in SPLITS:
        counts = generate_split(split)
        print(
            f"{split:<6} {counts['images']:5d} images, "
            f"{counts['litter']:6d} litter boxes, {counts['distractor']:6d} distractor boxes"
        )

    write_data_yaml()

    if args.preview:
        save_preview()
        print(f"Preview: {(FIGURE_DIR / 'dataset_preview.png').relative_to(PROJECT_ROOT)}")

    print(f"\nSaved: {DATASET_DIR.relative_to(PROJECT_ROOT)}/  ({time.perf_counter() - start:.0f} s)")


if __name__ == "__main__":
    main()
