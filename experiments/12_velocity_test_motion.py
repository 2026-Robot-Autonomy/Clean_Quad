from pathlib import Path
import sys
import time
import csv

import numpy as np
import mujoco
import mujoco.viewer


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))


from src.controllers.trot_controller import TrotController
from src.utils.orientation import quaternion_to_euler


# =========================================================
# MODEL PATH
# =========================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "assets"
    / "mujoco_menagerie"
    / "unitree_a1"
    / "scene.xml"
)


# =========================================================
# OUTPUT DIRECTORY
# =========================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "logs"
    / "velocity_tests"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# TEST CONFIGURATION
# =========================================================

EXPERIMENT_DURATION = 10.0


TESTS = [
    {
        "name": "forward",
        "vx": 0.06,
        "wz": 0.00,
    },

    {
        "name": "backward",
        "vx": -0.04,
        "wz": 0.00,
    },

    {
        "name": "left_curve",
        "vx": 0.04,
        "wz": 0.10,
    },

    {
        "name": "right_curve",
        "vx": 0.04,
        "wz": -0.10,
    },
]


# =========================================================
# ANGLE WRAPPING
# =========================================================

def wrap_angle(angle):
    """
    Wrap angle to [-pi, pi].
    """

    return (
        angle + np.pi
    ) % (
        2 * np.pi
    ) - np.pi


# =========================================================
# RUN ONE EXPERIMENT
# =========================================================

def run_test(test_config):

    test_name = test_config["name"]
    vx_command = test_config["vx"]
    wz_command = test_config["wz"]


    print("\n")
    print("=" * 60)
    print(f"STARTING TEST: {test_name.upper()}")
    print("=" * 60)

    print(
        f"vx = {vx_command:.3f} m/s | "
        f"wz = {wz_command:.3f} rad/s"
    )


    # =====================================================
    # 1. LOAD MODEL
    # =====================================================

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )

    data = mujoco.MjData(model)


    # =====================================================
    # 2. RESET TO HOME POSE
    # =====================================================

    home_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home"
    )

    mujoco.mj_resetDataKeyframe(
        model,
        data,
        home_id
    )

    mujoco.mj_forward(
        model,
        data
    )


    # =====================================================
    # 3. INITIAL STATE
    # =====================================================

    initial_x = data.qpos[0]
    initial_y = data.qpos[1]
    initial_z = data.qpos[2]

    initial_quaternion = (
        data.qpos[3:7].copy()
    )

    (
        initial_roll,
        initial_pitch,
        initial_yaw
    ) = quaternion_to_euler(
        initial_quaternion
    )


    # =====================================================
    # 4. CONTROLLER
    # =====================================================

    controller = TrotController(
        gait_frequency=1.0,
        step_height=0.03,
        nominal_height=-0.30,
        ramp_time=2.0,
        k_yaw=0.03,
    )


    controller.set_velocity(
        vx=vx_command,
        wz=wz_command,
    )


    # =====================================================
    # 5. SIMULATION VARIABLES
    # =====================================================

    simulation_time = 0.0

    step_count = 0

    steps_per_second = int(
        1.0 / model.opt.timestep
    )


    # =====================================================
    # 6. LOG STORAGE
    # =====================================================

    times = []

    xs = []
    ys = []
    zs = []

    dxs = []
    dys = []

    rolls = []
    pitches = []

    actual_yaws = []
    desired_yaws = []
    yaw_errors = []

    avg_vxs = []
    drift_ratios = []


    fall_detected = False


    # =====================================================
    # 7. VIEWER + SIMULATION
    # =====================================================

    with mujoco.viewer.launch_passive(
        model,
        data
    ) as viewer:

        while (
            viewer.is_running()
            and simulation_time < EXPERIMENT_DURATION
        ):

            wall_start = time.time()


            # =================================================
            # A. CURRENT ORIENTATION
            # =================================================

            quat = data.qpos[3:7]

            roll, pitch, yaw = (
                quaternion_to_euler(
                    quat
                )
            )


            # =================================================
            # B. DESIRED YAW
            # =================================================
            #
            # Straight motion:
            #
            # wz = 0
            #
            # desired_yaw stays equal to initial_yaw.
            #
            #
            # Turning:
            #
            # desired_yaw =
            # initial_yaw + wz * time
            #
            # =================================================

            desired_yaw = (
                initial_yaw
                + wz_command
                * simulation_time
            )


            # =================================================
            # C. YAW ERROR
            # =================================================

            yaw_error = (
                desired_yaw - yaw
            )

            yaw_error = wrap_angle(
                yaw_error
            )


            # =================================================
            # D. CONTROLLER
            # =================================================

            command = controller.get_command(
                simulation_time,
                yaw_error=yaw_error,
            )

            data.ctrl[:] = command


            # =================================================
            # E. PHYSICS STEP
            # =================================================

            mujoco.mj_step(
                model,
                data
            )


            simulation_time += (
                model.opt.timestep
            )

            step_count += 1


            # =================================================
            # F. ROBOT POSITION
            # =================================================

            robot_x = data.qpos[0]
            robot_y = data.qpos[1]
            robot_z = data.qpos[2]


            dx = (
                robot_x - initial_x
            )

            dy = (
                robot_y - initial_y
            )


            # =================================================
            # G. AVERAGE X VELOCITY
            # =================================================

            if simulation_time > 0:

                avg_vx = (
                    dx
                    / simulation_time
                )

            else:

                avg_vx = 0.0


            # =================================================
            # H. DRIFT RATIO
            # =================================================
            #
            # Mainly meaningful for forward/backward tests.
            #
            # For curves dy is intentional, so this metric
            # should not be interpreted as "bad drift".
            # =================================================

            if abs(dx) > 1e-6:

                drift_ratio = (
                    abs(dy)
                    / abs(dx)
                )

            else:

                drift_ratio = 0.0


            # =================================================
            # I. READ ORIENTATION AFTER PHYSICS STEP
            # =================================================

            quat = data.qpos[3:7]

            roll, pitch, yaw = (
                quaternion_to_euler(
                    quat
                )
            )


            relative_yaw = wrap_angle(
                yaw - initial_yaw
            )


            desired_relative_yaw = (
                desired_yaw
                - initial_yaw
            )


            roll_deg = np.degrees(
                roll
            )

            pitch_deg = np.degrees(
                pitch
            )

            yaw_deg = np.degrees(
                relative_yaw
            )

            desired_yaw_deg = np.degrees(
                desired_relative_yaw
            )

            yaw_error_deg = np.degrees(
                yaw_error
            )


            # =================================================
            # J. FALL DETECTION
            # =================================================

            if (
                robot_z < 0.20
                or abs(roll_deg) > 30
                or abs(pitch_deg) > 30
            ):

                print("\n!!! FALL DETECTED !!!")

                fall_detected = True

                break


            # =================================================
            # K. SAVE LOG VALUES
            # =================================================

            times.append(
                simulation_time
            )

            xs.append(
                robot_x
            )

            ys.append(
                robot_y
            )

            zs.append(
                robot_z
            )

            dxs.append(
                dx
            )

            dys.append(
                dy
            )

            avg_vxs.append(
                avg_vx
            )

            drift_ratios.append(
                drift_ratio
            )

            rolls.append(
                roll_deg
            )

            pitches.append(
                pitch_deg
            )

            actual_yaws.append(
                yaw_deg
            )

            desired_yaws.append(
                desired_yaw_deg
            )

            yaw_errors.append(
                yaw_error_deg
            )


            # =================================================
            # L. PRINT ONCE PER SECOND
            # =================================================

            if (
                step_count
                % steps_per_second
                == 0
            ):

                print(
                    f"time={simulation_time:5.2f}s | "
                    f"dx={dx:7.3f}m | "
                    f"dy={dy:7.3f}m | "
                    f"z={robot_z:6.3f}m | "
                    f"vx_avg={avg_vx:6.3f}m/s | "
                    f"roll={roll_deg:6.2f}° | "
                    f"pitch={pitch_deg:6.2f}° | "
                    f"yaw={yaw_deg:7.2f}° | "
                    f"desired_yaw={desired_yaw_deg:7.2f}° | "
                    f"error={yaw_error_deg:7.2f}°"
                )


            # =================================================
            # M. VIEWER
            # =================================================

            viewer.sync()


            # =================================================
            # N. REAL-TIME SLEEP
            # =================================================

            elapsed = (
                time.time()
                - wall_start
            )

            remaining = (
                model.opt.timestep
                - elapsed
            )

            if remaining > 0:

                time.sleep(
                    remaining
                )


    # =====================================================
    # CHECK THAT DATA EXISTS
    # =====================================================

    if len(times) == 0:

        print(
            f"No data collected for {test_name}"
        )

        return None


    # =====================================================
    # 8. PATH LENGTH
    # =====================================================

    path_length = 0.0

    for i in range(
        1,
        len(xs)
    ):

        segment_dx = (
            xs[i]
            - xs[i - 1]
        )

        segment_dy = (
            ys[i]
            - ys[i - 1]
        )

        path_length += np.sqrt(
            segment_dx**2
            + segment_dy**2
        )


    # =====================================================
    # 9. STEADY-STATE VELOCITY
    # =====================================================
    #
    # Ignore first 3 seconds because of smooth startup.
    # =====================================================

    steady_start_index = None

    for i, t in enumerate(times):

        if t >= 3.0:

            steady_start_index = i

            break


    if (
        steady_start_index is not None
        and steady_start_index < len(times) - 1
    ):

        steady_dx = (
            xs[-1]
            - xs[steady_start_index]
        )

        steady_dt = (
            times[-1]
            - times[steady_start_index]
        )

        steady_vx = (
            steady_dx
            / steady_dt
        )

    else:

        steady_vx = (
            dxs[-1]
            / times[-1]
        )


    # =====================================================
    # 10. FINAL METRICS
    # =====================================================

    final_dx = dxs[-1]

    final_dy = dys[-1]

    final_yaw = actual_yaws[-1]

    final_desired_yaw = (
        desired_yaws[-1]
    )


    final_yaw_error = (
        final_desired_yaw
        - final_yaw
    )


    avg_height = np.mean(
        zs
    )


    height_std = np.std(
        zs
    )


    max_roll = np.max(
        np.abs(
            rolls
        )
    )


    max_pitch = np.max(
        np.abs(
            pitches
        )
    )


    max_yaw_error = np.max(
        np.abs(
            yaw_errors
        )
    )


    final_drift_ratio = (
        drift_ratios[-1]
    )


    # =====================================================
    # 11. TURN RADIUS
    # =====================================================

    final_yaw_rad = np.radians(
        final_yaw
    )


    if abs(final_yaw_rad) > 1e-6:

        turn_radius = (
            path_length
            / abs(final_yaw_rad)
        )

    else:

        turn_radius = np.inf


    # =====================================================
    # 12. SAVE CSV
    # =====================================================

    output_file = (
        OUTPUT_DIR
        / f"{test_name}.csv"
    )


    with open(
        output_file,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(
            file
        )


        writer.writerow([
            "time",
            "x",
            "y",
            "z",
            "dx",
            "dy",
            "avg_vx",
            "drift_ratio",
            "roll_deg",
            "pitch_deg",
            "actual_yaw_deg",
            "desired_yaw_deg",
            "yaw_error_deg",
        ])


        for row in zip(
            times,
            xs,
            ys,
            zs,
            dxs,
            dys,
            avg_vxs,
            drift_ratios,
            rolls,
            pitches,
            actual_yaws,
            desired_yaws,
            yaw_errors,
        ):

            writer.writerow(
                row
            )


    # =====================================================
    # 13. PRINT TEST SUMMARY
    # =====================================================

    print("\n")
    print("-" * 55)

    print(
        f"{test_name.upper()} TEST SUMMARY"
    )

    print("-" * 55)


    print(
        f"Command vx            : "
        f"{vx_command:.3f} m/s"
    )


    print(
        f"Command wz            : "
        f"{wz_command:.3f} rad/s"
    )


    print(
        f"X displacement        : "
        f"{final_dx:.3f} m"
    )


    print(
        f"Y displacement        : "
        f"{final_dy:.3f} m"
    )


    print(
        f"Path length           : "
        f"{path_length:.3f} m"
    )


    print(
        f"Steady-state vx       : "
        f"{steady_vx:.3f} m/s"
    )


    print(
        f"Average height        : "
        f"{avg_height:.3f} m"
    )


    print(
        f"Height std            : "
        f"{height_std:.4f} m"
    )


    print(
        f"Maximum roll          : "
        f"{max_roll:.2f}°"
    )


    print(
        f"Maximum pitch         : "
        f"{max_pitch:.2f}°"
    )


    print(
        f"Desired yaw change    : "
        f"{final_desired_yaw:.2f}°"
    )


    print(
        f"Actual yaw change     : "
        f"{final_yaw:.2f}°"
    )


    print(
        f"Final yaw error       : "
        f"{final_yaw_error:.2f}°"
    )


    print(
        f"Maximum yaw error     : "
        f"{max_yaw_error:.2f}°"
    )


    # Drift ratio only makes sense for straight tests
    if abs(wz_command) < 1e-6:

        print(
            f"Drift ratio           : "
            f"{final_drift_ratio * 100:.2f}%"
        )


    else:

        print(
            f"Approx. turn radius   : "
            f"{turn_radius:.3f} m"
        )


    print(
        f"Fall detected         : "
        f"{fall_detected}"
    )


    print(
        f"CSV saved             : "
        f"{output_file}"
    )


    # =====================================================
    # RETURN SUMMARY
    # =====================================================

    return {

        "name": test_name,

        "vx_command": vx_command,

        "wz_command": wz_command,

        "dx": final_dx,

        "dy": final_dy,

        "path_length": path_length,

        "steady_vx": steady_vx,

        "avg_height": avg_height,

        "height_std": height_std,

        "max_roll": max_roll,

        "max_pitch": max_pitch,

        "desired_yaw": final_desired_yaw,

        "actual_yaw": final_yaw,

        "yaw_error": final_yaw_error,

        "max_yaw_error": max_yaw_error,

        "drift_ratio": final_drift_ratio,

        "turn_radius": turn_radius,

        "fall": fall_detected,
    }


# =========================================================
# FINAL COMPARISON
# =========================================================

def print_final_comparison(
    results
):

    print("\n\n")

    print("=" * 100)

    print(
        "FINAL LOCOMOTION TEST COMPARISON"
    )

    print("=" * 100)


    header = (
        f"{'Test':<14}"
        f"{'vx':>8}"
        f"{'wz':>8}"
        f"{'dx':>9}"
        f"{'dy':>9}"
        f"{'vx_meas':>10}"
        f"{'roll':>9}"
        f"{'pitch':>9}"
        f"{'yaw_des':>11}"
        f"{'yaw_act':>11}"
        f"{'yaw_err':>11}"
        f"{'fall':>8}"
    )

    print(
        header
    )

    print(
        "-" * 100
    )


    for result in results:

        print(
            f"{result['name']:<14}"
            f"{result['vx_command']:>8.3f}"
            f"{result['wz_command']:>8.3f}"
            f"{result['dx']:>9.3f}"
            f"{result['dy']:>9.3f}"
            f"{result['steady_vx']:>10.3f}"
            f"{result['max_roll']:>9.2f}"
            f"{result['max_pitch']:>9.2f}"
            f"{result['desired_yaw']:>11.2f}"
            f"{result['actual_yaw']:>11.2f}"
            f"{result['yaw_error']:>11.2f}"
            f"{str(result['fall']):>8}"
        )


    print("=" * 100)


# =========================================================
# MAIN
# =========================================================

def main():

    results = []


    for test in TESTS:

        result = run_test(
            test
        )

        if result is not None:

            results.append(
                result
            )


    print_final_comparison(
        results
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    main()