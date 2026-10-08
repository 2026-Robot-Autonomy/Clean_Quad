"""
Stage-1 -> Stage-2 refactor check (headless), on the
original certified gait (V1_SLOW).

1. Equivalence: LocomotionInterface reproduces the certified
   loop of 15_final_locomotion_test.py exactly for all
   7 required tests (same scene.xml, same trajectory).

2. MjSpec scene: the Python-built flat scene gives the same
   trajectory as scene.xml.

3. Spawn: reset_robot at a non-zero pose walks along the
   spawned heading.

4. Stop-and-go: straight -> stop -> left curve -> right
   curve, with the blend restarting after the stop.
"""

from pathlib import Path
import sys

import numpy as np
import mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from src.config import A1_SCENE_PATH
from src.controllers.gait_profiles import V1_SLOW
from src.controllers.trot_controller import TrotController
from src.controllers.locomotion_interface import (
    LocomotionInterface,
    wrap_angle,
)
from src.simulation.robot_setup import (
    build_flat_scene,
    reset_robot,
)
from src.utils.orientation import quaternion_to_euler


# ============================================================
# SETTINGS
# ============================================================

EXPERIMENT_DURATION = 10.0

REQUIRED_TESTS = [
    ("stand",          0.00,  0.00),
    ("forward_002",    0.02,  0.00),
    ("forward_003",    0.03,  0.00),
    ("backward_002",  -0.02,  0.00),
    ("backward_003",  -0.03,  0.00),
    ("forward_left",   0.03,  0.05),
    ("forward_right",  0.03, -0.05),
]

# Same fall rule as 15_final_locomotion_test.py
FALL_HEIGHT = 0.20
FALL_ANGLE_DEG = 30.0

MAX_ROLL_DEG = 8.0
MAX_PITCH_DEG = 8.0


# ============================================================
# REFERENCE LOOP (copy of the certified Stage-1 loop)
# ============================================================

def run_reference(
    model,
    data,
    vx,
    wz,
):

    reset_robot(model, data)

    _, _, initial_yaw = quaternion_to_euler(
        data.qpos[3:7].copy()
    )

    controller = TrotController(
        gait_frequency=1.0,
        step_height=0.03,
        nominal_height=-0.30,
        ramp_time=2.0,
        k_yaw=0.03,
    )

    controller.set_velocity(vx=vx, wz=wz)

    trajectory = []
    simulation_time = 0.0

    while simulation_time < EXPERIMENT_DURATION:

        _, _, yaw = quaternion_to_euler(
            data.qpos[3:7]
        )

        desired_yaw = (
            initial_yaw
            + wz * simulation_time
        )

        yaw_error = wrap_angle(
            desired_yaw - yaw
        )

        data.ctrl[:] = controller.get_command(
            simulation_time,
            yaw_error=yaw_error,
        )

        mujoco.mj_step(model, data)

        simulation_time += model.opt.timestep

        trajectory.append(data.qpos.copy())

    return np.array(trajectory)


# ============================================================
# INTERFACE LOOP
# ============================================================

def run_interface(
    model,
    data,
    vx,
    wz,
):

    reset_robot(model, data)

    loco = LocomotionInterface(V1_SLOW)

    loco.set_velocity(vx, wz)

    trajectory = []
    simulation_time = 0.0

    while simulation_time < EXPERIMENT_DURATION:

        data.ctrl[:] = loco.update(
            data,
            model.opt.timestep,
        )

        mujoco.mj_step(model, data)

        simulation_time += model.opt.timestep

        trajectory.append(data.qpos.copy())

    return np.array(trajectory)


# ============================================================
# SCRIPTED RUN WITH STABILITY LOGGING
# ============================================================

def run_schedule(
    model,
    data,
    loco,
    schedule,
):
    """
    schedule: list of (duration, vx, wz)

    Returns per-segment logs and overall stability.
    """

    segments = []
    max_roll = 0.0
    max_pitch = 0.0
    fall = False

    for duration, vx, wz in schedule:

        loco.set_velocity(vx, wz)

        start_qpos = data.qpos.copy()
        _, _, start_yaw = quaternion_to_euler(start_qpos[3:7])

        yaws = [start_yaw]
        forward_distance = 0.0
        previous_xy = start_qpos[0:2].copy()

        n_steps = int(round(duration / model.opt.timestep))

        for _ in range(n_steps):

            data.ctrl[:] = loco.update(
                data,
                model.opt.timestep,
            )

            mujoco.mj_step(model, data)

            roll, pitch, yaw = quaternion_to_euler(
                data.qpos[3:7]
            )

            # Body-frame forward displacement
            delta = data.qpos[0:2] - previous_xy
            forward_distance += (
                delta[0] * np.cos(yaw)
                + delta[1] * np.sin(yaw)
            )
            previous_xy = data.qpos[0:2].copy()

            yaws.append(yaw)

            roll_deg = abs(np.degrees(roll))
            pitch_deg = abs(np.degrees(pitch))

            max_roll = max(max_roll, roll_deg)
            max_pitch = max(max_pitch, pitch_deg)

            if (
                data.qpos[2] < FALL_HEIGHT
                or roll_deg > FALL_ANGLE_DEG
                or pitch_deg > FALL_ANGLE_DEG
            ):
                fall = True
                break

        yaw_change = float(
            np.unwrap(yaws)[-1] - np.unwrap(yaws)[0]
        )

        segments.append({
            "vx": vx,
            "wz": wz,
            "duration": duration,
            "vx_avg": forward_distance / duration,
            "wz_avg": yaw_change / duration,
            "end_xy": data.qpos[0:2].copy(),
            "end_yaw_deg": float(np.degrees(
                quaternion_to_euler(data.qpos[3:7])[2]
            )),
        })

        if fall:
            break

    return segments, max_roll, max_pitch, fall


# ============================================================
# CHECKS
# ============================================================

def check_equivalence(xml_model, spec_model):

    print("\n" + "=" * 88)
    print("1 + 2. EQUIVALENCE WITH THE CERTIFIED STAGE-1 LOOP")
    print("=" * 88)
    print(
        f"{'test':<16}"
        f"{'interface vs ref':>20}"
        f"{'MjSpec vs ref':>20}"
        f"{'final x':>12}"
        f"{'final y':>12}"
    )

    xml_data = mujoco.MjData(xml_model)
    spec_data = mujoco.MjData(spec_model)

    all_identical = True
    spec_identical = True

    for name, vx, wz in REQUIRED_TESTS:

        reference = run_reference(xml_model, xml_data, vx, wz)
        interface = run_interface(xml_model, xml_data, vx, wz)
        spec_run = run_interface(spec_model, spec_data, vx, wz)

        diff_interface = float(np.max(np.abs(interface - reference)))
        diff_spec = float(np.max(np.abs(spec_run - reference)))

        all_identical &= diff_interface == 0.0
        spec_identical &= diff_spec == 0.0

        print(
            f"{name:<16}"
            f"{diff_interface:>20.3e}"
            f"{diff_spec:>20.3e}"
            f"{reference[-1, 0]:>12.4f}"
            f"{reference[-1, 1]:>12.4f}"
        )

    print("\n(max |qpos difference| over the whole 10 s run)")

    return all_identical, spec_identical


def check_spawn(model):

    print("\n" + "=" * 88)
    print("3. SPAWN AT x=1.0, y=2.0, yaw=90 deg, then vx=+0.03 for 10 s")
    print("=" * 88)

    data = mujoco.MjData(model)

    reset_robot(
        model,
        data,
        x=1.0,
        y=2.0,
        yaw=np.pi / 2.0,
    )

    print(
        f"spawn qpos[0:3] = {np.round(data.qpos[0:3], 4)}, "
        f"yaw = {np.degrees(quaternion_to_euler(data.qpos[3:7])[2]):.2f} deg"
    )

    segments, max_roll, max_pitch, fall = run_schedule(
        model,
        data,
        LocomotionInterface(V1_SLOW),
        [(10.0, 0.03, 0.0)],
    )

    end = segments[-1]["end_xy"]
    dx = end[0] - 1.0
    dy = end[1] - 2.0

    print(
        f"displacement: dx = {dx:+.4f} m, dy = {dy:+.4f} m, "
        f"end yaw = {segments[-1]['end_yaw_deg']:.2f} deg"
    )
    print(
        f"max roll = {max_roll:.2f} deg, "
        f"max pitch = {max_pitch:.2f} deg, fall = {fall}"
    )

    # Walking along +y: most motion in y, heading kept near 90 deg
    passed = (
        not fall
        and dy > 0.15
        and abs(dx) < 0.15 * dy
        and abs(segments[-1]["end_yaw_deg"] - 90.0) < 5.0
    )

    print("SPAWN:", "PASS" if passed else "FAIL")

    return passed


def check_stop_and_go(model):

    print("\n" + "=" * 88)
    print("4. STOP-AND-GO / COMMAND CHANGES")
    print("=" * 88)

    data = mujoco.MjData(model)
    reset_robot(model, data)

    schedule = [
        (8.0, 0.03, 0.00),
        (3.0, 0.00, 0.00),
        (10.0, 0.03, 0.05),
        (10.0, 0.03, -0.05),
        (3.0, 0.00, 0.00),
    ]

    segments, max_roll, max_pitch, fall = run_schedule(
        model,
        data,
        LocomotionInterface(V1_SLOW),
        schedule,
    )

    print(
        f"{'segment':<10}{'vx*':>8}{'wz*':>8}"
        f"{'vx avg':>10}{'wz avg':>10}"
        f"{'end x':>9}{'end y':>9}{'end yaw':>10}"
    )

    for i, seg in enumerate(segments, start=1):
        print(
            f"{i:<10}{seg['vx']:>8.3f}{seg['wz']:>8.3f}"
            f"{seg['vx_avg']:>10.4f}{seg['wz_avg']:>10.4f}"
            f"{seg['end_xy'][0]:>9.3f}{seg['end_xy'][1]:>9.3f}"
            f"{seg['end_yaw_deg']:>10.2f}"
        )

    print(
        f"\nmax roll = {max_roll:.2f} deg, "
        f"max pitch = {max_pitch:.2f} deg, fall = {fall}"
    )
    print(
        "(segment averages include the 2 s startup blend, "
        "so they read lower than steady state)"
    )

    passed = (
        not fall
        and max_roll <= MAX_ROLL_DEG
        and max_pitch <= MAX_PITCH_DEG
        and segments[0]["vx_avg"] > 0.0
        and segments[2]["wz_avg"] > 0.0
        and segments[3]["wz_avg"] < 0.0
    )

    print("STOP-AND-GO:", "PASS" if passed else "FAIL")

    return passed


# ============================================================
# MAIN
# ============================================================

def main():

    xml_model = mujoco.MjModel.from_xml_path(
        str(A1_SCENE_PATH)
    )

    spec_model, _ = build_flat_scene()

    identical, spec_identical = check_equivalence(
        xml_model,
        spec_model,
    )

    spawn_ok = check_spawn(spec_model)
    stop_go_ok = check_stop_and_go(spec_model)

    print("\n" + "=" * 88)
    print("SUMMARY")
    print("=" * 88)
    print(
        "Interface == certified loop (7 required tests):",
        "PASS" if identical else "FAIL",
    )
    print(
        "MjSpec flat scene == scene.xml:",
        "PASS" if spec_identical else "DIFFERS (see table)",
    )
    print("Spawn at arbitrary pose:", "PASS" if spawn_ok else "FAIL")
    print("Stop-and-go:", "PASS" if stop_go_ok else "FAIL")


if __name__ == "__main__":
    main()
