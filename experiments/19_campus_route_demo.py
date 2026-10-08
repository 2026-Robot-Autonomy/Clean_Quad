"""
Stage 2 milestone M2: the robot walks a fixed route through
the campus: road -> up the ramp -> pavement -> grass (around
the planter) -> down the ramp -> base.

The waypoints are hand-picked and steering is a simple
heading follower using validated V2_FAST commands:

    vx = 0.3 m/s,  wz in {-0.5, 0, +0.5} rad/s

This is a demo of the world, not navigation: there is no
planning or obstacle avoidance (that is Stage 6).

    python experiments/19_campus_route_demo.py             # headless, ~10 s
    python experiments/19_campus_route_demo.py --viewer    # watch in real time, ~1 min
"""

from pathlib import Path
import argparse
import csv
import math
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


from src.controllers.locomotion_interface import LocomotionInterface, wrap_angle
from src.environment import build_campus
from src.environment.plotting import plot_campus
from src.utils.orientation import quaternion_to_euler


# ============================================================
# ROUTE
# ============================================================

WAYPOINTS = [
    (3.3, 1.0),     # east along the road
    (4.0, 2.6),     # up the ramp onto the pavement
    (4.4, 4.4),     # grass, east of the planter
    (2.0, 4.8),     # north of the planter
    (1.0, 3.6),     # grass, west
    (2.6, 2.7),     # pavement
    (3.4, 2.7),     # line up above the ramp
    (4.0, 1.1),     # down the ramp
    (0.8, 1.0),     # home
]

REACHED_DISTANCE = 0.25

VX = 0.30
WZ = 0.50

# Hysteresis on heading error [rad]
START_TURN = math.radians(8.0)
STOP_TURN = math.radians(3.0)

MAX_TIME_PER_METRE = 10.0       # generous: nominal is ~3.3 s per metre

LOG_PERIOD = 0.1

FALL_HEIGHT = 0.20
FALL_ANGLE_DEG = 30.0

LOG_DIR = PROJECT_ROOT / "results" / "logs" / "campus_route"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"


# ============================================================
# HEADING FOLLOWER
# ============================================================

class HeadingFollower:
    """Bang-bang steering towards a point, with hysteresis."""

    def __init__(self):
        self.wz = 0.0

    def command(self, x, y, yaw, target):

        error = wrap_angle(math.atan2(target[1] - y, target[0] - x) - yaw)

        if self.wz == 0.0 and abs(error) > START_TURN:
            self.wz = math.copysign(WZ, error)
        elif self.wz != 0.0 and (abs(error) < STOP_TURN or error * self.wz < 0.0):
            self.wz = 0.0

        return VX, self.wz


# ============================================================
# RUN
# ============================================================

def run_route(campus, viewer=None, speed=1.0):

    model, data = campus.model, campus.data
    layout = campus.layout

    loco = LocomotionInterface()
    follower = HeadingFollower()

    log = []
    next_log = 0.0
    fall = False
    reached = []

    for i, target in enumerate(WAYPOINTS):

        x, y, _ = campus.robot_pose()
        deadline = data.time + MAX_TIME_PER_METRE * max(1.0, math.dist((x, y), target))
        done = False

        while data.time < deadline:

            wall_start = time.perf_counter()

            x, y, yaw = campus.robot_pose()

            if math.dist((x, y), target) < REACHED_DISTANCE:
                done = True
                break

            loco.set_velocity(*follower.command(x, y, yaw, target))
            data.ctrl[:] = loco.update(data, model.opt.timestep)
            mujoco.mj_step(model, data)

            roll, pitch, _ = quaternion_to_euler(data.qpos[3:7])

            if data.time >= next_log:
                next_log += LOG_PERIOD
                log.append({
                    "time": data.time,
                    "x": x,
                    "y": y,
                    "yaw_deg": math.degrees(yaw),
                    "roll_deg": math.degrees(roll),
                    "pitch_deg": math.degrees(pitch),
                    "zone": layout.zone_at(x, y),
                    "waypoint": i,
                    "wz_command": follower.wz,
                })

            if (
                data.qpos[2] - layout.ground_height_at(x, y) < FALL_HEIGHT
                or abs(math.degrees(roll)) > FALL_ANGLE_DEG
                or abs(math.degrees(pitch)) > FALL_ANGLE_DEG
            ):
                fall = True
                break

            if viewer is not None:
                if not viewer.is_running():
                    return log, reached, fall
                viewer.sync()
                sleep = model.opt.timestep / speed - (time.perf_counter() - wall_start)
                if sleep > 0:
                    time.sleep(sleep)

        reached.append(done)

        print(
            f"  waypoint {i + 1}/{len(WAYPOINTS)} {target}: "
            f"{'reached' if done else 'NOT reached'} at t = {data.time:6.1f} s, "
            f"zone {layout.zone_at(*campus.robot_pose()[:2])}"
        )

        if fall or not done:
            break

    loco.stop()

    return log, reached, fall


# ============================================================
# OUTPUT
# ============================================================

def save(campus, log):

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    with open(LOG_DIR / "route.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log[0]))
        writer.writeheader()
        writer.writerows(log)

    path = np.array([[r["x"], r["y"]] for r in log])

    fig, ax = plt.subplots(figsize=(10, 6.5))
    plot_campus(ax, campus.layout, litter=campus.litter, path=path)
    ax.plot(*zip(*WAYPOINTS), "x", color="red", ms=9, mew=2, label="waypoints")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    ax.set_title("Stage 2 route demo")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "campus_route.png", dpi=150)
    plt.close(fig)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0, help="viewer speed-up")
    args = parser.parse_args()

    print("=" * 80)
    print("CLEANQUAD STAGE 2: CAMPUS ROUTE DEMO (M2)")
    print("=" * 80)

    campus = build_campus(seed=args.seed)

    if args.viewer:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(campus.model, campus.data) as viewer:
            viewer.cam.lookat[:] = [4.0, 3.0, 0.0]
            viewer.cam.distance = 9.0
            viewer.cam.elevation = -50.0
            viewer.cam.azimuth = 90.0
            log, reached, fall = run_route(campus, viewer, args.speed)
    else:
        log, reached, fall = run_route(campus)

    zones = [r["zone"] for r in log]
    visited = sorted(set(zones))
    blocked = sorted({z for z in visited if not campus.layout.traversable[z]})
    max_roll = max(abs(r["roll_deg"]) for r in log)
    max_pitch = max(abs(r["pitch_deg"]) for r in log)
    # 1 s samples: the trunk sways within each gait cycle.
    xy = np.array([[r["x"], r["y"]] for r in log])[:: int(round(1.0 / LOG_PERIOD))]
    distance = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))

    save(campus, log)

    passed = (
        len(reached) == len(WAYPOINTS)
        and all(reached)
        and not fall
        and not blocked
        and {"road", "ramp", "pavement", "grass"} <= set(visited)
    )

    print("\nRESULT")
    print(f"  waypoints reached : {sum(reached)}/{len(WAYPOINTS)}")
    print(f"  time / distance   : {log[-1]['time']:.1f} s / {distance:.2f} m")
    print(f"  zones visited     : {', '.join(visited)}")
    print(f"  no-go zones hit   : {', '.join(blocked) or 'none'}")
    print(f"  max roll / pitch  : {max_roll:.1f} / {max_pitch:.1f} deg")
    print(f"  fall              : {fall}")
    print(f"\n  M2 ROUTE DEMO     : {'PASS' if passed else 'FAIL'}")
    print("\nSaved: results/logs/campus_route/route.csv, results/figures/campus_route.png")


if __name__ == "__main__":
    main()
