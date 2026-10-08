"""
Stage 2: what terrain can the Stage-1 robot cross? (headless)

Each case builds a small straight test track with ONE
feature at y = 0 (friction change, curb, or ramp) using the
same terrain code as the campus. The robot walks along the
track at cruise speed through LocomotionInterface (default
gait profile):

    up   cases: start below the feature, face +y
    down cases: start above the feature, face -y

Outcome:
    PASS      whole robot crossed, no fall, stable
    UNSTABLE  crossed, but roll/pitch above limits
    BLOCKED   did not get across in time
    FALL      trunk too low or tipped over

These results decide the campus design in
src/environment/layout.py: whatever the robot cannot cross
safely is a no-go zone.
"""

from pathlib import Path
import csv
import math
import sys

import numpy as np
import mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from src.controllers.locomotion_interface import LocomotionInterface
from src.environment.layout import (
    CURB_HEIGHT,
    FRICTION_GRASS,
    FRICTION_PAVED,
    Ramp,
    Rect,
    Surface,
)
from src.environment.terrain import (
    add_ramp,
    add_sky_and_light,
    add_surface,
    add_underfloor,
)
from src.simulation.robot_setup import load_a1_spec, reset_robot
from src.utils.orientation import quaternion_to_euler


# ============================================================
# SETTINGS
# ============================================================

VX = 0.30
MAX_DURATION = 30.0

START_DISTANCE = 0.50       # trunk start distance from the feature
FINISH_DISTANCE = 0.40      # trunk distance past the feature = fully crossed

TRACK_HALF_WIDTH = 1.0
TRACK_LENGTH = 3.0

# Same rise as the campus ramp
RAMP_RISE = CURB_HEIGHT

CURB_HEIGHTS = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08)
RAMP_SLOPES = (3, 5, 8, 12, 15)

FALL_CLEARANCE = 0.20
FALL_ANGLE_DEG = 30.0
MAX_ROLL_DEG = 8.0
MAX_PITCH_DEG = 8.0         # on ramps: + slope

OUTPUT_DIR = PROJECT_ROOT / "results" / "logs" / "terrain_traversal"
OUTPUT_FILE = OUTPUT_DIR / "terrain_traversal.csv"


# (name, feature, value, direction)
#   feature: "friction" (value = friction after y=0)
#            "curb"     (value = height [m])
#            "ramp"     (value = slope [deg])

CASES = (
    [
        ("flat_road", "friction", FRICTION_PAVED, "up"),
        ("road_to_grass", "friction", FRICTION_GRASS, "up"),
    ]
    + [(f"curb_up_{round(h * 100)}cm", "curb", h, "up") for h in CURB_HEIGHTS]
    + [(f"curb_down_{round(h * 100)}cm", "curb", h, "down") for h in CURB_HEIGHTS]
    + [(f"ramp_up_{s}deg", "ramp", s, "up") for s in RAMP_SLOPES]
    + [(f"ramp_down_{s}deg", "ramp", s, "down") for s in RAMP_SLOPES]
)


# ============================================================
# TEST TRACK
# ============================================================

class Track:
    """Low surface for y < 0, feature, high surface after it."""

    def __init__(self, feature, value):

        w = TRACK_HALF_WIDTH
        L = TRACK_LENGTH

        self.ramp = None
        self.slope_deg = 0.0

        if feature == "friction":
            self.high = 0.0
            self.feature_end = 0.0
            self.surfaces = [
                Surface("road", Rect(-w, -L, w, 0.0), 0.0, FRICTION_PAVED),
                Surface("grass", Rect(-w, 0.0, w, L), 0.0, value),
            ]

        elif feature == "curb":
            self.high = value
            self.feature_end = 0.0
            self.surfaces = [
                Surface("road", Rect(-w, -L, w, 0.0), 0.0, FRICTION_PAVED),
                Surface("pavement", Rect(-w, 0.0, w, L), value, FRICTION_PAVED),
            ]

        elif feature == "ramp":
            self.high = RAMP_RISE
            self.slope_deg = value
            self.ramp = Ramp("ramp", -w, w, 0.0, 0.0, RAMP_RISE, value, FRICTION_PAVED)
            self.feature_end = self.ramp.y_top
            self.surfaces = [
                Surface("road", Rect(-w, -L, w, 0.0), 0.0, FRICTION_PAVED),
                Surface("pavement", Rect(-w, self.feature_end, w, self.feature_end + L),
                        RAMP_RISE, FRICTION_PAVED),
            ]

    def ground_height_at(self, y):

        if y < 0.0:
            return 0.0

        if self.ramp is not None and y <= self.feature_end:
            return self.ramp.height_at(y)

        return self.high

    def build(self):

        spec = load_a1_spec()

        add_sky_and_light(spec)
        add_underfloor(spec)

        for surface in self.surfaces:
            add_surface(spec, surface)

        if self.ramp is not None:
            add_ramp(spec, self.ramp)

        return spec.compile()


# ============================================================
# RUN ONE CASE
# ============================================================

def run_case(name, feature, value, direction):

    track = Track(feature, value)
    model = track.build()
    data = mujoco.MjData(model)

    if direction == "up":
        start_y = -START_DISTANCE
        yaw0 = math.pi / 2.0
        goal_reached = lambda y: y >= track.feature_end + FINISH_DISTANCE
    else:
        start_y = track.feature_end + START_DISTANCE
        yaw0 = -math.pi / 2.0
        goal_reached = lambda y: y <= -FINISH_DISTANCE

    reset_robot(
        model,
        data,
        x=0.0,
        y=start_y,
        yaw=yaw0,
        ground_z=track.ground_height_at(start_y),
    )

    loco = LocomotionInterface()
    loco.set_velocity(VX, 0.0)

    max_roll = 0.0
    max_pitch = 0.0
    min_clearance = np.inf
    outcome = "BLOCKED"

    n_steps = int(MAX_DURATION / model.opt.timestep)

    for _ in range(n_steps):

        data.ctrl[:] = loco.update(data, model.opt.timestep)
        mujoco.mj_step(model, data)

        x, y, z = data.qpos[0:3]
        roll, pitch, _ = quaternion_to_euler(data.qpos[3:7])

        roll_deg = abs(math.degrees(roll))
        pitch_deg = abs(math.degrees(pitch))
        clearance = z - track.ground_height_at(y)

        max_roll = max(max_roll, roll_deg)
        max_pitch = max(max_pitch, pitch_deg)
        min_clearance = min(min_clearance, clearance)

        if (
            clearance < FALL_CLEARANCE
            or roll_deg > FALL_ANGLE_DEG
            or pitch_deg > FALL_ANGLE_DEG
        ):
            outcome = "FALL"
            break

        if goal_reached(y):
            outcome = "PASS"
            break

    if outcome == "PASS" and (
        max_roll > MAX_ROLL_DEG
        or max_pitch > MAX_PITCH_DEG + track.slope_deg
    ):
        outcome = "UNSTABLE"

    _, _, yaw = quaternion_to_euler(data.qpos[3:7])

    return {
        "name": name,
        "feature": feature,
        "value": value,
        "direction": direction,
        "outcome": outcome,
        "time": data.time,
        "progress": abs(data.qpos[1] - start_y),
        "avg_speed": abs(data.qpos[1] - start_y) / max(data.time, 1e-9),
        "lateral_drift": abs(data.qpos[0]),
        "heading_error_deg": math.degrees(
            (yaw - yaw0 + math.pi) % (2 * math.pi) - math.pi
        ),
        "max_roll_deg": max_roll,
        "max_pitch_deg": max_pitch,
        "min_clearance": min_clearance,
    }


# ============================================================
# MAIN
# ============================================================

def summarise(results):

    def max_passed(feature, direction):
        passed = [
            r["value"] for r in results
            if r["feature"] == feature and r["direction"] == direction and r["outcome"] == "PASS"
        ]
        return max(passed) if passed else None

    print("\n" + "=" * 100)
    print("TRAVERSABILITY SUMMARY")
    print("=" * 100)

    for feature, unit, scale in (("curb", "cm", 100.0), ("ramp", "deg", 1.0)):
        for direction in ("up", "down"):
            best = max_passed(feature, direction)
            text = "none passed" if best is None else f"{best * scale:g} {unit}"
            print(f"max {feature} {direction:<5}: {text}")


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("CLEANQUAD STAGE 2: TERRAIN TRAVERSAL TEST")
    print("=" * 100)
    print(
        f"{'case':<18}{'outcome':>10}{'time s':>9}{'speed':>9}"
        f"{'drift m':>9}{'yaw err':>9}{'roll':>7}{'pitch':>7}{'min clr':>9}"
    )

    results = []

    for case in CASES:

        r = run_case(*case)
        results.append(r)

        print(
            f"{r['name']:<18}{r['outcome']:>10}{r['time']:>9.1f}{r['avg_speed']:>9.4f}"
            f"{r['lateral_drift']:>9.3f}{r['heading_error_deg']:>9.1f}"
            f"{r['max_roll_deg']:>7.1f}{r['max_pitch_deg']:>7.1f}{r['min_clearance']:>9.3f}"
        )

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    summarise(results)

    print(f"\nSaved: {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
