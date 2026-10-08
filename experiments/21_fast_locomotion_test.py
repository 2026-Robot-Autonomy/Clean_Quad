"""
Certification of the V2_FAST gait (headless, flat scene.xml).

Same method and pass criteria as the Stage-1 certification
(15_final_locomotion_test.py), driven through
LocomotionInterface:

    straight vx error  <= 20 %      curve vx error <= 25 %
    wz error           <= 20 %      straight drift <= 15 %
    final straight yaw <= 5 deg     roll, pitch    <= 8 deg
    no fall

plus a stop-from-full-speed test.

    python experiments/21_fast_locomotion_test.py
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


from src.config import A1_SCENE_PATH
from src.controllers.gait_profiles import V2_FAST
from src.controllers.locomotion_interface import LocomotionInterface
from src.simulation.robot_setup import reset_robot
from src.utils.orientation import quaternion_to_euler


# ============================================================
# SETTINGS
# ============================================================

DURATION = 10.0
STEADY_STATE_START = 3.0

MAX_STRAIGHT_VX_ERROR = 20.0
MAX_CURVE_VX_ERROR = 25.0
MAX_WZ_ERROR = 20.0
MAX_STRAIGHT_DRIFT = 15.0
MAX_STRAIGHT_YAW_DEG = 5.0
MAX_ROLL_DEG = 8.0
MAX_PITCH_DEG = 8.0
MAX_STAND_SPEED = 0.005

FALL_HEIGHT = 0.20
FALL_ANGLE_DEG = 30.0

OUTPUT_FILE = PROJECT_ROOT / "results" / "logs" / "fast_locomotion" / "fast_locomotion_summary.csv"


# (name, vx, wz, required)
TESTS = [
    ("stand", 0.0, 0.0, True),

    ("forward_010", 0.10, 0.0, True),
    ("forward_020", 0.20, 0.0, True),
    ("forward_030", 0.30, 0.0, True),
    ("forward_040", 0.40, 0.0, True),
    ("forward_050", 0.50, 0.0, True),

    ("backward_010", -0.10, 0.0, True),
    ("backward_020", -0.20, 0.0, True),
    ("backward_030", -0.30, 0.0, True),

    ("left_030_030", 0.30, 0.30, True),
    ("right_030_030", 0.30, -0.30, True),
    ("left_030_050", 0.30, 0.50, True),
    ("right_030_050", 0.30, -0.50, True),
    ("left_010_030", 0.10, 0.30, True),
    ("right_010_030", 0.10, -0.30, True),

    ("back_left_020_030", -0.20, 0.30, False),
    ("back_right_020_030", -0.20, -0.30, False),
    ("rotate_left_030", 0.0, 0.30, False),
    ("rotate_right_030", 0.0, -0.30, False),
]


# ============================================================
# ONE TEST
# ============================================================

def classify(vx, wz):
    if vx == 0.0 and wz == 0.0:
        return "stand"
    if vx == 0.0:
        return "rotation"
    return "curve" if wz != 0.0 else "straight"


def run_test(model, data, name, vx_cmd, wz_cmd, required):

    reset_robot(model, data)
    _, _, yaw0 = quaternion_to_euler(data.qpos[3:7])
    start_xy = data.qpos[0:2].copy()

    loco = LocomotionInterface(V2_FAST)
    loco.set_velocity(vx_cmd, wz_cmd)

    forward = 0.0
    yaws = []
    max_roll = max_pitch = 0.0
    fall = False
    previous_xy = data.qpos[0:2].copy()

    while data.time < DURATION:

        data.ctrl[:] = loco.update(data, model.opt.timestep)
        mujoco.mj_step(model, data)

        roll, pitch, yaw = quaternion_to_euler(data.qpos[3:7])

        if data.time >= STEADY_STATE_START:
            delta = data.qpos[0:2] - previous_xy
            forward += delta[0] * math.cos(yaw) + delta[1] * math.sin(yaw)
            yaws.append(yaw)

        previous_xy = data.qpos[0:2].copy()

        max_roll = max(max_roll, abs(math.degrees(roll)))
        max_pitch = max(max_pitch, abs(math.degrees(pitch)))

        if (
            data.qpos[2] < FALL_HEIGHT
            or abs(math.degrees(roll)) > FALL_ANGLE_DEG
            or abs(math.degrees(pitch)) > FALL_ANGLE_DEG
        ):
            fall = True
            break

    span = DURATION - STEADY_STATE_START
    kind = classify(vx_cmd, wz_cmd)

    if fall or len(yaws) < 2:
        vx = wz = math.nan
    else:
        yaws = np.unwrap(yaws)
        vx = forward / span
        wz = (yaws[-1] - yaws[0]) / span

    # Straight-line drift: lateral offset relative to the start heading
    d = data.qpos[0:2] - start_xy
    along = d[0] * math.cos(yaw0) + d[1] * math.sin(yaw0)
    lateral = -d[0] * math.sin(yaw0) + d[1] * math.cos(yaw0)
    drift = 100.0 * abs(lateral) / max(abs(along), 1e-9)

    final_yaw = math.degrees(math.remainder(quaternion_to_euler(data.qpos[3:7])[2] - yaw0, 2 * math.pi))

    vx_err = 100.0 * abs(vx - vx_cmd) / abs(vx_cmd) if vx_cmd else math.nan
    wz_err = 100.0 * abs(wz - wz_cmd) / abs(wz_cmd) if wz_cmd else math.nan

    stable = not fall and max_roll <= MAX_ROLL_DEG and max_pitch <= MAX_PITCH_DEG

    if kind == "stand":
        tracking = abs(vx) < MAX_STAND_SPEED
    elif kind == "straight":
        tracking = (
            vx_err <= MAX_STRAIGHT_VX_ERROR
            and drift <= MAX_STRAIGHT_DRIFT
            and abs(final_yaw) <= MAX_STRAIGHT_YAW_DEG
        )
    elif kind == "curve":
        tracking = vx_err <= MAX_CURVE_VX_ERROR and wz_err <= MAX_WZ_ERROR
    else:
        tracking = wz_err <= MAX_WZ_ERROR and abs(vx) < 0.05

    return {
        "name": name,
        "type": kind,
        "required": required,
        "vx_target": vx_cmd,
        "vx_measured": vx,
        "vx_error_percent": vx_err,
        "wz_target": wz_cmd,
        "wz_measured": wz,
        "wz_error_percent": wz_err,
        "drift_percent": drift if kind == "straight" else math.nan,
        "final_yaw_deg": final_yaw,
        "max_roll_deg": max_roll,
        "max_pitch_deg": max_pitch,
        "fall": fall,
        "pass": bool(stable and tracking),
    }


def run_stop_test(model, data):
    """Full speed for 5 s, then stop: must settle without falling."""

    reset_robot(model, data)
    loco = LocomotionInterface(V2_FAST)
    loco.set_velocity(V2_FAST.max_forward_speed, 0.0)

    max_roll = max_pitch = 0.0
    fall = False
    stop_xy = None

    while data.time < 8.0:

        if data.time >= 5.0 and stop_xy is None:
            loco.stop()
            stop_xy = data.qpos[0:2].copy()

        data.ctrl[:] = loco.update(data, model.opt.timestep)
        mujoco.mj_step(model, data)

        roll, pitch, _ = quaternion_to_euler(data.qpos[3:7])

        if stop_xy is not None:
            max_roll = max(max_roll, abs(math.degrees(roll)))
            max_pitch = max(max_pitch, abs(math.degrees(pitch)))

        if data.qpos[2] < FALL_HEIGHT or max(max_roll, max_pitch) > FALL_ANGLE_DEG:
            fall = True
            break

    overshoot = float(np.linalg.norm(data.qpos[0:2] - stop_xy)) if stop_xy is not None else math.nan
    passed = not fall and max_roll <= MAX_ROLL_DEG and max_pitch <= MAX_PITCH_DEG

    return passed, overshoot, max_roll, max_pitch, fall


# ============================================================
# MAIN
# ============================================================

def fmt(x, spec):
    """Format a number, or a right-aligned '-' for NaN."""
    if isinstance(x, float) and math.isnan(x):
        return "-".rjust(int(spec.split(".")[0]))
    return format(x, spec)


def main():

    print("=" * 112)
    print(
        f"V2_FAST LOCOMOTION CERTIFICATION  "
        f"({V2_FAST.gait_frequency} Hz, foot lift {V2_FAST.step_height * 100:.0f} cm)"
    )
    print("=" * 112)
    print(
        f"{'test':<20}{'vx*':>7}{'vx':>8}{'err%':>7}{'wz*':>7}{'wz':>8}{'err%':>7}"
        f"{'drift%':>8}{'yaw':>7}{'roll':>6}{'pitch':>7}{'REQ':>5}{'PASS':>6}"
    )

    model = mujoco.MjModel.from_xml_path(str(A1_SCENE_PATH))
    data = mujoco.MjData(model)

    results = []

    for name, vx, wz, required in TESTS:

        r = run_test(model, data, name, vx, wz, required)
        results.append(r)

        print(
            f"{r['name']:<20}{vx:7.2f}{fmt(r['vx_measured'], '8.3f')}{fmt(r['vx_error_percent'], '7.1f')}"
            f"{wz:7.2f}{fmt(r['wz_measured'], '8.3f')}{fmt(r['wz_error_percent'], '7.1f')}"
            f"{fmt(r['drift_percent'], '8.1f')}{r['final_yaw_deg']:7.1f}"
            f"{r['max_roll_deg']:6.1f}{r['max_pitch_deg']:7.1f}"
            f"{'Y' if required else 'N':>5}{'Y' if r['pass'] else 'N':>6}"
        )

    stop_ok, overshoot, roll, pitch, fall = run_stop_test(model, data)

    print(
        f"\nstop from {V2_FAST.max_forward_speed} m/s: "
        f"{'PASS' if stop_ok else 'FAIL'}  (coasts {overshoot:.3f} m, "
        f"roll {roll:.1f}, pitch {pitch:.1f} deg, fall {fall})"
    )

    required = [r for r in results if r["required"]]
    optional = [r for r in results if not r["required"]]
    n_req = sum(r["pass"] for r in required) + stop_ok

    print(f"\nRequired tests passed : {n_req}/{len(required) + 1}")
    print(f"Optional tests passed : {sum(r['pass'] for r in optional)}/{len(optional)}")
    print(
        "\nV2_FAST LOCOMOTION:",
        "PASSED" if n_req == len(required) + 1 else "FAILED",
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved: {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
