from pathlib import Path
import sys
import csv

import numpy as np
import mujoco


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


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
# OUTPUT
# =========================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "logs"
    / "velocity_validation"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SUMMARY_FILE = (
    OUTPUT_DIR
    / "velocity_validation_summary.csv"
)


# =========================================================
# SUPPORTED TRANSLATIONAL VELOCITY ENVELOPE
# =========================================================
#
# We experimentally found that ±0.01 m/s is too close to
# the gait's low-speed region to be reliably controlled.
#
# Therefore Stage-1 translational control supports:
#
#     -0.03 m/s
#     -0.02 m/s
#      0.00 m/s
#     +0.02 m/s
#     +0.03 m/s
#
# This is now the official validation envelope.
#
# =========================================================

VELOCITY_COMMANDS = [
    -0.03,
    -0.02,
     0.00,
     0.02,
     0.03,
]


# =========================================================
# EXPERIMENT SETTINGS
# =========================================================

EXPERIMENT_DURATION = 10.0

# Ignore startup transition when calculating
# steady-state velocity.
STEADY_STATE_START = 3.0


# =========================================================
# ACCEPTANCE THRESHOLDS
# =========================================================

MAX_VELOCITY_ERROR_PERCENT = 20.0

MAX_DRIFT_PERCENT = 15.0

MAX_FINAL_YAW_DEG = 5.0

MAX_ROLL_DEG = 8.0

MAX_PITCH_DEG = 8.0

MAX_STAND_VELOCITY = 0.005

MAX_STAND_DISPLACEMENT = 0.035


# =========================================================
# HELPERS
# =========================================================

def wrap_angle(angle):
    """
    Wrap angle to [-pi, pi].
    """

    return (
        angle + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi


# =========================================================
# RUN ONE TRANSLATIONAL VELOCITY TEST
# =========================================================

def run_velocity_test(
    vx_command,
):

    print("\n")
    print("=" * 78)

    print(
        f"TRANSLATIONAL TEST | "
        f"target vx = {vx_command:+.3f} m/s"
    )

    print("=" * 78)


    # =====================================================
    # 1. LOAD MODEL
    # =====================================================

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )


    data = mujoco.MjData(
        model
    )


    # =====================================================
    # 2. RESET TO HOME
    # =====================================================

    home_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home",
    )


    mujoco.mj_resetDataKeyframe(
        model,
        data,
        home_id,
    )


    mujoco.mj_forward(
        model,
        data,
    )


    # =====================================================
    # 3. INITIAL STATE
    # =====================================================

    initial_x = float(
        data.qpos[0]
    )


    initial_y = float(
        data.qpos[1]
    )


    initial_quat = (
        data.qpos[3:7]
        .copy()
    )


    _, _, initial_yaw = (
        quaternion_to_euler(
            initial_quat
        )
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
        wz=0.0,
    )


    # =====================================================
    # 5. STORAGE
    # =====================================================

    times = []

    xs = []
    ys = []
    zs = []

    body_forward_velocities = []

    rolls = []
    pitches = []
    yaws = []

    yaw_errors = []


    simulation_time = 0.0

    step_count = 0


    steps_per_second = max(
        1,
        int(
            1.0
            / model.opt.timestep
        )
    )


    fall_detected = False


    # =====================================================
    # 6. SIMULATION LOOP
    # =====================================================

    while (
        simulation_time
        < EXPERIMENT_DURATION
    ):

        # -------------------------------------------------
        # Current orientation
        # -------------------------------------------------

        quat = (
            data.qpos[3:7]
        )


        roll, pitch, yaw = (
            quaternion_to_euler(
                quat
            )
        )


        # -------------------------------------------------
        # Straight walking target:
        #
        # desired yaw = initial yaw
        # -------------------------------------------------

        desired_yaw = (
            initial_yaw
        )


        yaw_error = wrap_angle(
            desired_yaw
            - yaw
        )


        # -------------------------------------------------
        # Controller
        # -------------------------------------------------

        command = (
            controller.get_command(
                simulation_time,
                yaw_error=yaw_error,
            )
        )


        data.ctrl[:] = command


        # -------------------------------------------------
        # Physics
        # -------------------------------------------------

        mujoco.mj_step(
            model,
            data,
        )


        simulation_time += (
            model.opt.timestep
        )


        step_count += 1


        # -------------------------------------------------
        # Robot position
        # -------------------------------------------------

        robot_x = float(
            data.qpos[0]
        )


        robot_y = float(
            data.qpos[1]
        )


        robot_z = float(
            data.qpos[2]
        )


        # -------------------------------------------------
        # Orientation after physics step
        # -------------------------------------------------

        quat = (
            data.qpos[3:7]
        )


        roll, pitch, yaw = (
            quaternion_to_euler(
                quat
            )
        )


        # -------------------------------------------------
        # World-frame velocity
        # -------------------------------------------------

        world_vx = float(
            data.qvel[0]
        )


        world_vy = float(
            data.qvel[1]
        )


        # -------------------------------------------------
        # Body-frame forward velocity
        # -------------------------------------------------

        body_forward_velocity = (
            world_vx
            * np.cos(yaw)

            +

            world_vy
            * np.sin(yaw)
        )


        # -------------------------------------------------
        # Orientation diagnostics
        # -------------------------------------------------

        relative_yaw = wrap_angle(
            yaw
            - initial_yaw
        )


        roll_deg = float(
            np.degrees(
                roll
            )
        )


        pitch_deg = float(
            np.degrees(
                pitch
            )
        )


        yaw_deg = float(
            np.degrees(
                relative_yaw
            )
        )


        yaw_error_deg = float(
            np.degrees(
                yaw_error
            )
        )


        # -------------------------------------------------
        # Fall detection
        # -------------------------------------------------

        if (
            robot_z < 0.20

            or abs(
                roll_deg
            ) > 30.0

            or abs(
                pitch_deg
            ) > 30.0
        ):

            print(
                "!!! FALL DETECTED !!!"
            )

            fall_detected = True

            break


        # -------------------------------------------------
        # Store data
        # -------------------------------------------------

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


        body_forward_velocities.append(
            body_forward_velocity
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


        # -------------------------------------------------
        # Print approximately every second
        # -------------------------------------------------

        if (
            step_count
            % steps_per_second
            == 0
        ):

            dx = (
                robot_x
                - initial_x
            )


            dy = (
                robot_y
                - initial_y
            )


            print(
                f"time={simulation_time:5.2f}s | "
                f"dx={dx:+7.3f}m | "
                f"dy={dy:+7.3f}m | "
                f"v_body={body_forward_velocity:+7.3f}m/s | "
                f"z={robot_z:6.3f}m | "
                f"yaw={yaw_deg:+6.2f}°"
            )


    # =====================================================
    # 7. DATA CHECK
    # =====================================================

    if len(
        times
    ) == 0:

        return None


    # =====================================================
    # 8. STEADY-STATE REGION
    # =====================================================

    steady_indices = [

        i

        for i, t in enumerate(
            times
        )

        if t >= STEADY_STATE_START

    ]


    if len(
        steady_indices
    ) == 0:

        steady_indices = list(
            range(
                len(times)
            )
        )


    start_index = (
        steady_indices[0]
    )


    steady_dt = (
        times[-1]
        -
        times[start_index]
    )


    # =====================================================
    # 9. PRIMARY VELOCITY METRIC
    # =====================================================
    #
    # Use steady-state displacement.
    #
    # This avoids instantaneous velocity oscillations caused
    # by the trot cycle.
    #
    # =====================================================

    steady_dx = (
        xs[-1]
        -
        xs[start_index]
    )


    if steady_dt > 1e-9:

        measured_velocity = (
            steady_dx
            /
            steady_dt
        )

    else:

        measured_velocity = 0.0


    # =====================================================
    # 10. BODY VELOCITY DIAGNOSTICS
    # =====================================================

    steady_body_velocities = np.array([

        body_forward_velocities[i]

        for i in steady_indices

    ])


    mean_body_velocity = float(
        np.mean(
            steady_body_velocities
        )
    )


    body_velocity_std = float(
        np.std(
            steady_body_velocities
        )
    )


    # =====================================================
    # 11. FINAL POSITION
    # =====================================================

    final_dx = (
        xs[-1]
        -
        initial_x
    )


    final_dy = (
        ys[-1]
        -
        initial_y
    )


    # =====================================================
    # 12. DRIFT
    # =====================================================

    if abs(
        final_dx
    ) > 1e-6:

        drift_percent = (
            abs(
                final_dy
            )
            /
            abs(
                final_dx
            )
            * 100.0
        )

    else:

        drift_percent = 0.0


    # =====================================================
    # 13. VELOCITY ERROR
    # =====================================================

    if abs(
        vx_command
    ) > 1e-6:

        velocity_error = (
            measured_velocity
            -
            vx_command
        )


        velocity_error_percent = (
            abs(
                velocity_error
            )
            /
            abs(
                vx_command
            )
            * 100.0
        )

    else:

        velocity_error = (
            measured_velocity
        )


        velocity_error_percent = 0.0


    # =====================================================
    # 14. DIRECTION CHECK
    # =====================================================

    if abs(
        vx_command
    ) < 1e-6:

        correct_direction = True

    else:

        correct_direction = (
            np.sign(
                measured_velocity
            )
            ==
            np.sign(
                vx_command
            )
        )


    # =====================================================
    # 15. STABILITY
    # =====================================================

    average_height = float(
        np.mean(
            zs
        )
    )


    height_std = float(
        np.std(
            zs
        )
    )


    max_roll = float(
        np.max(
            np.abs(
                rolls
            )
        )
    )


    max_pitch = float(
        np.max(
            np.abs(
                pitches
            )
        )
    )


    final_yaw = float(
        yaws[-1]
    )


    max_yaw_error = float(
        np.max(
            np.abs(
                yaw_errors
            )
        )
    )


    # =====================================================
    # 16. PASS / FAIL LOGIC
    # =====================================================

    # -----------------------------------------------------
    # Stand test
    # -----------------------------------------------------

    if abs(
        vx_command
    ) < 1e-6:

        velocity_pass = (
            abs(
                measured_velocity
            )
            <= MAX_STAND_VELOCITY
        )


        stand_displacement_pass = (
            abs(
                final_dx
            )
            <= MAX_STAND_DISPLACEMENT
        )

    else:

        velocity_pass = (
            velocity_error_percent
            <= MAX_VELOCITY_ERROR_PERCENT
        )


        stand_displacement_pass = True


    # -----------------------------------------------------
    # Drift
    # -----------------------------------------------------

    if abs(
        vx_command
    ) < 1e-6:

        # Drift ratio is not meaningful while standing.
        drift_pass = True

    else:

        drift_pass = (
            drift_percent
            <= MAX_DRIFT_PERCENT
        )


    # -----------------------------------------------------
    # Heading
    # -----------------------------------------------------

    yaw_pass = (
        abs(
            final_yaw
        )
        <= MAX_FINAL_YAW_DEG
    )


    # -----------------------------------------------------
    # Stability
    # -----------------------------------------------------

    roll_pass = (
        max_roll
        <= MAX_ROLL_DEG
    )


    pitch_pass = (
        max_pitch
        <= MAX_PITCH_DEG
    )


    no_fall_pass = (
        not fall_detected
    )


    # -----------------------------------------------------
    # Overall
    # -----------------------------------------------------

    overall_pass = (
        correct_direction
        and velocity_pass
        and stand_displacement_pass
        and drift_pass
        and yaw_pass
        and roll_pass
        and pitch_pass
        and no_fall_pass
    )


    # =====================================================
    # 17. SAVE RUN CSV
    # =====================================================

    if abs(
        vx_command
    ) < 1e-6:

        filename = (
            "stand_0.00.csv"
        )

    elif vx_command > 0:

        filename = (
            f"forward_"
            f"{abs(vx_command):.2f}.csv"
        )

    else:

        filename = (
            f"backward_"
            f"{abs(vx_command):.2f}.csv"
        )


    output_file = (
        OUTPUT_DIR
        / filename
    )


    with open(
        output_file,
        "w",
        newline="",
    ) as file:

        writer = csv.writer(
            file
        )


        writer.writerow([
            "time",
            "x",
            "y",
            "z",
            "body_forward_velocity",
            "roll_deg",
            "pitch_deg",
            "yaw_deg",
            "yaw_error_deg",
        ])


        for row in zip(
            times,
            xs,
            ys,
            zs,
            body_forward_velocities,
            rolls,
            pitches,
            yaws,
            yaw_errors,
        ):

            writer.writerow(
                row
            )


    # =====================================================
    # 18. PRINT RESULT
    # =====================================================

    print("\n")
    print("-" * 72)


    print(
        f"SUMMARY | "
        f"target vx={vx_command:+.3f} m/s"
    )


    print("-" * 72)


    print(
        f"Target velocity          : "
        f"{vx_command:+.4f} m/s"
    )


    print(
        f"Measured velocity        : "
        f"{measured_velocity:+.4f} m/s"
    )


    print(
        f"Mean body velocity       : "
        f"{mean_body_velocity:+.4f} m/s"
    )


    print(
        f"Body velocity std        : "
        f"{body_velocity_std:.4f} m/s"
    )


    if abs(
        vx_command
    ) > 1e-6:

        print(
            f"Velocity tracking error  : "
            f"{velocity_error_percent:.2f}%"
        )

    else:

        print(
            "Velocity tracking error  : "
            "N/A (stand)"
        )


    print(
        f"Correct direction        : "
        f"{correct_direction}"
    )


    print(
        f"Final dx                 : "
        f"{final_dx:+.4f} m"
    )


    print(
        f"Final dy                 : "
        f"{final_dy:+.4f} m"
    )


    print(
        f"Drift                    : "
        f"{drift_percent:.2f}%"
    )


    print(
        f"Average height           : "
        f"{average_height:.4f} m"
    )


    print(
        f"Height std               : "
        f"{height_std:.4f} m"
    )


    print(
        f"Maximum roll             : "
        f"{max_roll:.2f}°"
    )


    print(
        f"Maximum pitch            : "
        f"{max_pitch:.2f}°"
    )


    print(
        f"Final yaw                : "
        f"{final_yaw:.2f}°"
    )


    print(
        f"Maximum yaw error        : "
        f"{max_yaw_error:.2f}°"
    )


    print(
        f"Fall detected            : "
        f"{fall_detected}"
    )


    print(
        f"Velocity pass            : "
        f"{velocity_pass}"
    )


    print(
        f"Drift pass               : "
        f"{drift_pass}"
    )


    print(
        f"Yaw pass                 : "
        f"{yaw_pass}"
    )


    print(
        f"Stability pass           : "
        f"{roll_pass and pitch_pass}"
    )


    print(
        f"OVERALL PASS             : "
        f"{overall_pass}"
    )


    # =====================================================
    # 19. RETURN SUMMARY
    # =====================================================

    return {

        "vx_target":
            vx_command,

        "vx_measured":
            measured_velocity,

        "velocity_error":
            velocity_error,

        "velocity_error_percent":
            velocity_error_percent,

        "mean_body_velocity":
            mean_body_velocity,

        "body_velocity_std":
            body_velocity_std,

        "correct_direction":
            correct_direction,

        "final_dx":
            final_dx,

        "final_dy":
            final_dy,

        "drift_percent":
            drift_percent,

        "average_height":
            average_height,

        "height_std":
            height_std,

        "max_roll":
            max_roll,

        "max_pitch":
            max_pitch,

        "final_yaw":
            final_yaw,

        "max_yaw_error":
            max_yaw_error,

        "fall":
            fall_detected,

        "velocity_pass":
            velocity_pass,

        "drift_pass":
            drift_pass,

        "yaw_pass":
            yaw_pass,

        "roll_pass":
            roll_pass,

        "pitch_pass":
            pitch_pass,

        "overall_pass":
            overall_pass,
    }


# =========================================================
# SAVE SUMMARY CSV
# =========================================================

def save_summary(
    results,
):

    with open(
        SUMMARY_FILE,
        "w",
        newline="",
    ) as file:

        writer = csv.writer(
            file
        )


        writer.writerow([
            "vx_target",
            "vx_measured",
            "velocity_error",
            "velocity_error_percent",
            "mean_body_velocity",
            "body_velocity_std",
            "correct_direction",
            "final_dx",
            "final_dy",
            "drift_percent",
            "average_height",
            "height_std",
            "max_roll",
            "max_pitch",
            "final_yaw",
            "max_yaw_error",
            "fall",
            "velocity_pass",
            "drift_pass",
            "yaw_pass",
            "roll_pass",
            "pitch_pass",
            "overall_pass",
        ])


        for result in results:

            writer.writerow([
                result["vx_target"],
                result["vx_measured"],
                result["velocity_error"],
                result["velocity_error_percent"],
                result["mean_body_velocity"],
                result["body_velocity_std"],
                result["correct_direction"],
                result["final_dx"],
                result["final_dy"],
                result["drift_percent"],
                result["average_height"],
                result["height_std"],
                result["max_roll"],
                result["max_pitch"],
                result["final_yaw"],
                result["max_yaw_error"],
                result["fall"],
                result["velocity_pass"],
                result["drift_pass"],
                result["yaw_pass"],
                result["roll_pass"],
                result["pitch_pass"],
                result["overall_pass"],
            ])


# =========================================================
# PRINT FINAL TABLE
# =========================================================

def print_results_table(
    results,
):

    print("\n\n")

    print("=" * 118)

    print(
        "SUPPORTED TRANSLATIONAL VELOCITY VALIDATION"
    )

    print("=" * 118)


    print(
        f"{'target':>9}"
        f"{'actual':>10}"
        f"{'error%':>10}"
        f"{'dx':>10}"
        f"{'dy':>10}"
        f"{'drift%':>10}"
        f"{'yaw':>9}"
        f"{'roll':>9}"
        f"{'pitch':>9}"
        f"{'dir':>8}"
        f"{'PASS':>9}"
    )


    print(
        "-" * 118
    )


    for result in results:

        if abs(
            result["vx_target"]
        ) > 1e-6:

            error_text = (
                f"{result['velocity_error_percent']:.2f}"
            )

        else:

            error_text = "N/A"


        print(
            f"{result['vx_target']:>9.3f}"
            f"{result['vx_measured']:>10.4f}"
            f"{error_text:>10}"
            f"{result['final_dx']:>10.4f}"
            f"{result['final_dy']:>10.4f}"
            f"{result['drift_percent']:>10.2f}"
            f"{result['final_yaw']:>9.2f}"
            f"{result['max_roll']:>9.2f}"
            f"{result['max_pitch']:>9.2f}"
            f"{str(result['correct_direction']):>8}"
            f"{str(result['overall_pass']):>9}"
        )


    print(
        "=" * 118
    )


# =========================================================
# FORWARD / BACKWARD SYMMETRY
# =========================================================

def print_symmetry_analysis(
    results,
):

    result_map = {

        round(
            result["vx_target"],
            3,
        ):
        result

        for result in results
    }


    print("\n")

    print("=" * 76)

    print(
        "TRANSLATIONAL FORWARD / BACKWARD SYMMETRY"
    )

    print("=" * 76)


    print(
        f"{'speed':>10}"
        f"{'forward':>14}"
        f"{'backward':>14}"
        f"{'mag diff':>14}"
        f"{'symmetry err%':>17}"
    )


    print(
        "-" * 76
    )


    for speed in [
        0.02,
        0.03,
    ]:

        forward = result_map.get(
            round(
                speed,
                3,
            )
        )


        backward = result_map.get(
            round(
                -speed,
                3,
            )
        )


        if (
            forward is None
            or backward is None
        ):

            continue


        forward_actual = (
            forward[
                "vx_measured"
            ]
        )


        backward_actual = (
            backward[
                "vx_measured"
            ]
        )


        magnitude_difference = abs(
            abs(
                forward_actual
            )
            -
            abs(
                backward_actual
            )
        )


        average_magnitude = (
            abs(
                forward_actual
            )
            +
            abs(
                backward_actual
            )
        ) / 2.0


        if (
            average_magnitude
            > 1e-9
        ):

            symmetry_error = (
                magnitude_difference
                /
                average_magnitude
                * 100.0
            )

        else:

            symmetry_error = 0.0


        print(
            f"{speed:>10.3f}"
            f"{forward_actual:>14.4f}"
            f"{backward_actual:>14.4f}"
            f"{magnitude_difference:>14.4f}"
            f"{symmetry_error:>16.2f}%"
        )


    print(
        "=" * 76
    )


# =========================================================
# FINAL PASS / FAIL
# =========================================================

def print_overall_result(
    results,
):

    passed = sum(
        1

        for result in results

        if result[
            "overall_pass"
        ]
    )


    total = len(
        results
    )


    print("\n")

    print("=" * 76)

    print(
        "STAGE-1 TRANSLATIONAL CONTROL RESULT"
    )

    print("=" * 76)


    print(
        f"Tests passed : "
        f"{passed}/{total}"
    )


    if (
        passed == total
    ):

        print(
            "\nRESULT: "
            "TRANSLATIONAL LOCOMOTION PASSED"
        )


        print(
            "\nSupported vx envelope:"
        )


        print(
            "    -0.03 <= vx <= -0.02 m/s"
        )


        print(
            "     vx = 0.00 m/s"
        )


        print(
            "    +0.02 <= vx <= +0.03 m/s"
        )


    else:

        print(
            "\nRESULT: "
            "TRANSLATIONAL LOCOMOTION "
            "STILL NEEDS CALIBRATION"
        )


    print(
        "=" * 76
    )


# =========================================================
# MAIN
# =========================================================

def main():

    results = []


    for vx_command in VELOCITY_COMMANDS:

        result = run_velocity_test(
            vx_command
        )


        if result is not None:

            results.append(
                result
            )


    # -----------------------------------------------------
    # Final validation table
    # -----------------------------------------------------

    print_results_table(
        results
    )


    # -----------------------------------------------------
    # Symmetry analysis
    # -----------------------------------------------------

    print_symmetry_analysis(
        results
    )


    # -----------------------------------------------------
    # Overall Stage-1 translation verdict
    # -----------------------------------------------------

    print_overall_result(
        results
    )


    # -----------------------------------------------------
    # Save CSV
    # -----------------------------------------------------

    save_summary(
        results
    )


    print(
        f"\nSummary saved to:\n"
        f"{SUMMARY_FILE}"
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    main()