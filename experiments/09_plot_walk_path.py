import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOG_PATH = (
    PROJECT_ROOT
    / "results"
    / "logs"
    / "straight_walk.csv"
)


xs = []
ys = []


with open(LOG_PATH, "r") as f:

    reader = csv.DictReader(f)

    for row in reader:
        xs.append(float(row["x"]))
        ys.append(float(row["y"]))


plt.figure()

plt.plot(xs, ys)

plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")

plt.title("A1 Walking Path")

plt.axis("equal")
plt.grid()

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "walk_path.png"
)

plt.savefig(
    OUTPUT_PATH,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved plot to: {OUTPUT_PATH}"
)