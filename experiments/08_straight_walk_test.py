from pathlib import Path
import sys
import time
import csv

import numpy as np
import mujoco
import mujoco.viewer


# ---------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))


from src.controllers.trot_controller import TrotController
from src.utils.orientation import quaternion_to_euler


# ---------------------------------------------------------
# MODEL PATH
# ---------------------------------------------------------

MODEL_PATH = (
    PROJECT_ROOT
    / "assets"
    / "mujoco_menagerie"
    / "unitree_a1"
    / "scene.xml"
)


# ---------------------------------------------------------
# OUTPUT PATH
# ---------------------------------------------------------

OUTPUT_DIR = PROJECT_ROOT / "results" / "logs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "straight_walk_velocity_control.csv"


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

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
    # 3. INITIAL ROBOT STATE
    # =====================================================

    initial_x = data.qpos[0]
    initial_y = data.qpos[1]
    initial_z = data.qpos[2]

    initial_quat = data.qpos[3:7].copy()

    initial_roll, initial_pitch, initial_yaw = (
        quaternion_to_euler(initial_quat)
    )


    # =====================================================
    # 4. CREATE CONTROLLER
    # =====================================================

    controller = TrotController(
        gait_frequency=1.0,
        step_height=0.03,
        nominal_height=-0.30,
        ramp_time=2.0,
        k_yaw=0.03,
    )


    # -----------------------------------------------------
    # HIGH-LEVEL VELOCITY COMMAND
    # -----------------------------------------------------
    #
    # Positive vx = forward
    #
    # wz = 0 means maintain straight heading
    #
    # Internally the controller converts:
    #
    # vx = +0.06
    #
    # into approximately:
    #
    # step_length = -0.06
    #
    # because your current gait convention uses
    # negative stride for forward motion.
    # -----------------------------------------------------

    controller.set_velocity(
        vx=0.06,
        wz=0.0,
    )


    # =====================================================
    # 5. EXPERIMENT SETTINGS
    # =====================================================

    simulation_time = 0.0

    experiment_duration = 10.0

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

    avg_vxs = []

    drift_ratios = []

    rolls = []
    pitches = []
    yaws = []

    yaw_errors = []

    desired_yaws = []


    # =====================================================
    # 7. SIMULATION
    # =====================================================

    with mujoco.viewer.launch_passive(
        model,
        data
    ) as viewer:

        while (
            viewer.is_running()
            and simulation_time < experiment_duration
        ):

            wall_start = time.time()


            # =================================================
            # A. READ CURRENT ORIENTATION
            # =================================================

            quat = data.qpos[3:7]

            roll, pitch, yaw = (
                quaternion_to_euler(quat)
            )


            # =================================================
            # B. DESIRED YAW
            # =================================================
            #
            # General form:
            #
            # desired_yaw =
            # initial_yaw + wz * time
            #
            # For straight walking:
            #
            # wz = 0
            #
            # therefore:
            #
            # desired_yaw = initial_yaw
            # =================================================

            desired_yaw = (
                initial_yaw
                + controller.wz_command
                * simulation_time
            )


            # =================================================
            # C. YAW ERROR
            # =================================================

            yaw_error = (
                desired_yaw - yaw
            )


            # Wrap to [-pi, pi]

            yaw_error = (
                yaw_error + np.pi
            ) % (
                2 * np.pi
            ) - np.pi


            # =================================================
            # D. GET CONTROL COMMAND
            # =================================================

            command = controller.get_command(
                simulation_time,
                yaw_error=yaw_error,
            )


            data.ctrl[:] = command


            # =================================================
            # E. ADVANCE PHYSICS
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
            # F. READ STATE AFTER PHYSICS STEP
            # =================================================

            robot_x = data.qpos[0]
            robot_y = data.qpos[1]
            robot_z = data.qpos[2]


            # =================================================
            # G. DISPLACEMENT
            # =================================================

            dx = (
                robot_x - initial_x
            )

            dy = (
                robot_y - initial_y
            )


            # =================================================
            # H. AVERAGE FORWARD VELOCITY
            # =================================================

            if simulation_time > 0:

                avg_vx = (
                    dx / simulation_time
                )

            else:

                avg_vx = 0.0


            # =================================================
            # I. DRIFT RATIO
            # =================================================

            if abs(dx) > 1e-6:

                drift_ratio = (
                    abs(dy)
                    / abs(dx)
                )

            else:

                drift_ratio = 0.0


            # =================================================
            # J. ORIENTATION AFTER PHYSICS STEP
            # =================================================

            quat = data.qpos[3:7]

            roll, pitch, yaw = (
                quaternion_to_euler(quat)
            )


            relative_yaw = (
                yaw - initial_yaw
            )


            # Wrap relative yaw

            relative_yaw = (
                relative_yaw + np.pi
            ) % (
                2 * np.pi
            ) - np.pi


            roll_deg = np.degrees(
                roll
            )

            pitch_deg = np.degrees(
                pitch
            )

            yaw_deg = np.degrees(
                relative_yaw
            )

            yaw_error_deg = np.degrees(
                yaw_error
            )

            desired_yaw_deg = np.degrees(
                desired_yaw - initial_yaw
            )


            # =================================================
            # K. STORE LOGS
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

            yaws.append(
                yaw_deg
            )


            yaw_errors.append(
                yaw_error_deg
            )

            desired_yaws.append(
                desired_yaw_deg
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
                    f"vx_avg={avg_vx:6.3f}m/s | "
                    f"z={robot_z:6.3f}m | "
                    f"roll={roll_deg:6.2f}° | "
                    f"pitch={pitch_deg:6.2f}° | "
                    f"yaw={yaw_deg:6.2f}° | "
                    f"yaw_error={yaw_error_deg:6.2f}° | "
                    f"drift={drift_ratio * 100:6.2f}%"
                )


            # =================================================
            # M. UPDATE VIEWER
            # =================================================

            viewer.sync()


            # =================================================
            # N. KEEP SIMULATION CLOSE TO REAL TIME
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
    # 8. SAVE CSV
    # =====================================================

    with open(
        OUTPUT_FILE,
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
            "yaw_deg",
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
            yaws,
            desired_yaws,
            yaw_errors,
        ):

            writer.writerow(
                row
            )


    # =====================================================
    # 9. FINAL METRICS
    # =====================================================

    final_dx = dxs[-1]

    final_dy = dys[-1]

    final_vx = avg_vxs[-1]

    final_drift = (
        drift_ratios[-1]
    )


    avg_height = np.mean(
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


    final_yaw = yaws[-1]


    max_yaw_error = np.max(
        np.abs(
            yaw_errors
        )
    )


    # =====================================================
    # 10. FINAL SUMMARY
    # =====================================================

    print(
        "\n===================================="
    )

    print(
        "STRAIGHT WALK TEST SUMMARY"
    )

    print(
        "===================================="
    )


    print(
        f"Commanded vx     : "
        f"{controller.vx_command:.3f} m/s"
    )


    print(
        f"Commanded wz     : "
        f"{controller.wz_command:.3f} rad/s"
    )


    print(
        f"Forward distance : "
        f"{final_dx:.3f} m"
    )


    print(
        f"Lateral drift    : "
        f"{final_dy:.3f} m"
    )


    print(
        f"Average velocity : "
        f"{final_vx:.3f} m/s"
    )


    print(
        f"Drift ratio      : "
        f"{final_drift * 100:.2f}%"
    )


    print(
        f"Average height   : "
        f"{avg_height:.3f} m"
    )


    print(
        f"Maximum roll     : "
        f"{max_roll:.2f}°"
    )


    print(
        f"Maximum pitch    : "
        f"{max_pitch:.2f}°"
    )


    print(
        f"Final yaw change : "
        f"{final_yaw:.2f}°"
    )


    print(
        f"Maximum yaw error: "
        f"{max_yaw_error:.2f}°"
    )


    print(
        f"\nLog saved to:\n"
        f"{OUTPUT_FILE}"
    )


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    main()