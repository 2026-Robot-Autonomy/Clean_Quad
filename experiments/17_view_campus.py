"""
Stage 2: build the campus, check it, save the map, view it.

    python experiments/17_view_campus.py               # checks + map + viewer
    python experiments/17_view_campus.py --no-viewer   # checks + map only
    python experiments/17_view_campus.py --seed 3      # another litter layout
"""

from pathlib import Path
import argparse
import sys
import time

import numpy as np
import mujoco
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from src.controllers.locomotion_interface import LocomotionInterface
from src.environment import build_campus
from src.environment.plotting import plot_campus


FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

# (x, y) -> expected zone
ZONE_CHECKS = [
    ((0.8, 1.0), "road"),
    ((4.0, 1.8), "ramp"),
    ((2.0, 2.0), "curb"),
    ((4.0, 2.5), "pavement"),
    ((1.0, 4.0), "grass"),
    ((6.0, 4.0), "plaza"),
    ((2.7, 4.0), "planter"),
    ((6.5, 4.8), "stairs"),
    ((6.5, 5.5), "entrance"),
    ((1.0, 5.0), "obstacle"),
    ((7.4, 2.6), "obstacle"),
    ((-0.1, 1.0), "outside"),
]

STAND_CHECK_TIME = 5.0


# ============================================================
# CHECKS
# ============================================================

def describe(campus):

    layout = campus.layout

    print("\nZONES")
    for s in layout.surfaces:
        r = s.rect
        print(
            f"  {s.kind:<10} x {r.x0:.1f}-{r.x1:.1f}  y {r.y0:.1f}-{r.y1:.1f}  "
            f"height {s.height:.2f} m  friction {s.friction:.1f}  "
            f"walkable {layout.traversable[s.kind]}"
        )

    print("\nOBSTACLES")
    for o in layout.obstacles:
        x, y = o.shape.center
        print(f"  {o.name:<16} ({x:.2f}, {y:.2f})")

    print(f"\nLITTER (seed {campus.seed})")
    for item in campus.litter:
        x, y, _ = item.position
        print(f"  {item.name:<20} ({x:.2f}, {y:.2f})  on {layout.zone_at(x, y)}")


def check_zones(campus):

    ok = True

    for (x, y), expected in ZONE_CHECKS:
        got = campus.layout.zone_at(x, y)
        if got != expected:
            print(f"  zone_at({x}, {y}) = {got}, expected {expected}")
            ok = False

    return ok


def check_seed(seed):

    a = build_campus(seed=seed)
    b = build_campus(seed=seed)
    c = build_campus(seed=seed + 1)

    pos = lambda campus: np.array([it.position for it in campus.litter])

    return (
        np.array_equal(pos(a), pos(b))
        and not np.array_equal(pos(a), pos(c))
    )


def check_standing(campus):

    model, data = campus.model, campus.data
    loco = LocomotionInterface()

    while data.time < STAND_CHECK_TIME:
        data.ctrl[:] = loco.update(data, model.opt.timestep)
        mujoco.mj_step(model, data)

    x0, y0, _ = campus.layout.base_pose
    moved = np.hypot(data.qpos[0] - x0, data.qpos[1] - y0)
    ok = data.qpos[2] > 0.25 and moved < 0.05

    campus.reset()

    return ok


def save_map(campus):

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "campus_map.png"

    fig, ax = plt.subplots(figsize=(10, 6.5))
    plot_campus(ax, campus.layout, litter=campus.litter, distractors=campus.distractors)
    ax.set_title(f"CleanQuad campus (seed {campus.seed})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path


# ============================================================
# VIEWER
# ============================================================

def view(campus):

    import mujoco.viewer

    model, data = campus.model, campus.data
    loco = LocomotionInterface()

    print("\nOpening viewer (robot standing at base). Close the window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:

        viewer.cam.lookat[:] = [4.0, 3.0, 0.0]
        viewer.cam.distance = 9.0
        viewer.cam.elevation = -50.0
        viewer.cam.azimuth = 90.0

        while viewer.is_running():

            start = time.perf_counter()

            data.ctrl[:] = loco.update(data, model.opt.timestep)
            mujoco.mj_step(model, data)
            viewer.sync()

            sleep = model.opt.timestep - (time.perf_counter() - start)
            if sleep > 0:
                time.sleep(sleep)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print("CLEANQUAD STAGE 2: CAMPUS")
    print("=" * 80)

    campus = build_campus(seed=args.seed)

    print(
        f"model: {campus.model.ngeom} geoms, {campus.model.nbody} bodies, "
        f"{campus.model.nmocap} litter items"
    )

    describe(campus)

    print("\nCHECKS")
    print("  zone queries      :", "PASS" if check_zones(campus) else "FAIL")
    print("  seeded litter     :", "PASS" if check_seed(args.seed) else "FAIL")
    print("  robot stands 5 s  :", "PASS" if check_standing(campus) else "FAIL")

    path = save_map(campus)
    print(f"\nMap saved: {path.relative_to(PROJECT_ROOT)}")

    if not args.no_viewer:
        view(campus)


if __name__ == "__main__":
    main()
