"""
Calibrate the V2_FAST gait (headless, flat scene.xml).

A. Stride -> speed table
   Fixed raw strides with heading hold; steady-state
   body-frame forward speed. Output is the calibration table
   for gait_profiles.V2_FAST.

B. Turn gains
   Open loop (no heading feedback) at the cruise speed: a
   differential stride +-d gives a yaw rate w. The gain is
   K = d / w, fitted separately for left and right turns.

    python experiments/20_fast_gait_calibration.py
"""

from dataclasses import replace
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


from src.config import A1_SCENE_PATH
from src.controllers.gait_profiles import V2_FAST
from src.controllers.locomotion_interface import wrap_angle
from src.controllers.trot_controller import TrotController
from src.simulation.robot_setup import reset_robot
from src.utils.orientation import quaternion_to_euler


# ============================================================
# SETTINGS
# ============================================================

DURATION = 15.0
STEADY_STATE_START = 5.0

STRIDES = np.round(np.arange(0.06, -0.141, -0.01), 3)

CRUISE_SPEED = 0.30
TURN_STRIDES = (0.0025, 0.005, 0.0075, 0.010)

OUTPUT_DIR = PROJECT_ROOT / "results" / "logs" / "fast_gait_calibration"


# ============================================================
# MEASUREMENT
# ============================================================

def measure(model, data, controller, heading_hold):
    """
    Run one experiment; return steady-state body-frame
    forward speed, yaw rate, max roll and pitch [deg].
    """

    reset_robot(model, data)
    _, _, yaw0 = quaternion_to_euler(data.qpos[3:7])

    t = 0.0
    forward = 0.0
    yaws = []
    max_roll = max_pitch = 0.0
    previous_xy = data.qpos[0:2].copy()

    while t < DURATION:

        roll, pitch, yaw = quaternion_to_euler(data.qpos[3:7])
        error = wrap_angle(yaw0 - yaw) if heading_hold else 0.0

        data.ctrl[:] = controller.get_command(t, yaw_error=error)
        mujoco.mj_step(model, data)
        t += model.opt.timestep

        roll, pitch, yaw = quaternion_to_euler(data.qpos[3:7])

        if t >= STEADY_STATE_START:
            delta = data.qpos[0:2] - previous_xy
            forward += delta[0] * math.cos(yaw) + delta[1] * math.sin(yaw)
            yaws.append(yaw)
            max_roll = max(max_roll, abs(math.degrees(roll)))
            max_pitch = max(max_pitch, abs(math.degrees(pitch)))

        previous_xy = data.qpos[0:2].copy()

        if data.qpos[2] < 0.20:
            return math.nan, math.nan, max_roll, max_pitch

    span = DURATION - STEADY_STATE_START
    yaws = np.unwrap(yaws)

    return forward / span, (yaws[-1] - yaws[0]) / span, max_roll, max_pitch


def raw_stride_controller(stride, **overrides):
    """V2 gait, but every vx command maps to `stride`."""

    kwargs = replace(V2_FAST, **overrides).controller_kwargs()
    kwargs["velocity_to_step_length"] = lambda v: stride

    return TrotController(**kwargs)


# ============================================================
# A. SPEED TABLE
# ============================================================

def calibrate_speed(model, data):

    print("\nA. STRIDE -> SPEED (heading hold)")
    print(f"{'stride':>8}{'vx m/s':>10}{'wz':>9}{'roll':>7}{'pitch':>7}")

    rows = []

    for stride in STRIDES:

        controller = raw_stride_controller(stride)
        controller.set_velocity(1.0, 0.0)       # any nonzero vx

        vx, wz, roll, pitch = measure(model, data, controller, heading_hold=True)
        rows.append((float(stride), vx, wz, roll, pitch))

        print(f"{stride:8.3f}{vx:10.4f}{wz:9.4f}{roll:7.1f}{pitch:7.1f}")

    velocities = np.array([r[1] for r in rows])
    monotonic = bool(np.all(np.diff(velocities) > 0))

    print(f"\nmonotonic: {monotonic}")

    # Table in increasing velocity order
    order = np.argsort(velocities)
    v_sorted = velocities[order]
    s_sorted = np.array([r[0] for r in rows])[order]
    in_place = float(np.interp(0.0, v_sorted, s_sorted))

    print("\nPaste into gait_profiles.V2_FAST:")
    print("    calibration_velocities=(")
    print("        " + ", ".join(f"{v:.4f}" for v in v_sorted) + ",")
    print("    ),")
    print("    calibration_steps=(")
    print("        " + ", ".join(f"{s:.2f}" for s in s_sorted) + ",")
    print("    ),")
    print(f"    in_place_step_length={in_place:.4f},")

    return rows


# ============================================================
# B. TURN GAINS
# ============================================================

def calibrate_turning(model, data):

    print(f"\nB. TURN GAINS at vx = {CRUISE_SPEED} m/s (open loop)")
    print(f"{'d stride':>10}{'wz rad/s':>11}{'vx m/s':>9}")

    base = V2_FAST.velocity_to_step_length(CRUISE_SPEED)

    def run(d):
        controller = raw_stride_controller(
            base,
            k_yaw=0.0,
            left_turn_gain=1.0,
            right_turn_gain=1.0,
            max_turn_correction=1.0,
        )
        # With gain 1, the turn correction equals d.
        controller.set_velocity(CRUISE_SPEED, d)
        vx, wz, _, _ = measure(model, data, controller, heading_hold=False)
        print(f"{d:10.4f}{wz:11.4f}{vx:9.4f}")
        return wz

    bias = run(0.0)
    rows = [(0.0, bias)]

    for d in TURN_STRIDES:
        rows.append((d, run(d)))
        rows.append((-d, run(-d)))

    def fit(sign):
        pts = [(d, w - bias) for d, w in rows if d * sign > 0]
        d = np.array([p[0] for p in pts])
        w = np.array([p[1] for p in pts])
        return float(np.sum(d * w) / np.sum(w * w))

    left, right = fit(+1), fit(-1)

    print(f"\nopen-loop yaw bias at d = 0: {bias:+.4f} rad/s")
    print("Paste into gait_profiles.V2_FAST:")
    print(f"    left_turn_gain={left:.4f},")
    print(f"    right_turn_gain={right:.4f},")

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("V2_FAST GAIT CALIBRATION")
    print(
        f"gait {V2_FAST.gait_frequency} Hz, "
        f"foot lift {V2_FAST.step_height * 100:.0f} cm"
    )
    print("=" * 70)

    model = mujoco.MjModel.from_xml_path(str(A1_SCENE_PATH))
    data = mujoco.MjData(model)

    speed_rows = calibrate_speed(model, data)
    turn_rows = calibrate_turning(model, data)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "speed_table.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stride", "vx", "wz", "max_roll_deg", "max_pitch_deg"])
        writer.writerows(speed_rows)

    with open(OUTPUT_DIR / "turn_gains.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["differential_stride", "wz"])
        writer.writerows(turn_rows)

    print(f"\nSaved: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
