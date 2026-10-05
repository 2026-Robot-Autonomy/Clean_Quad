from pathlib import Path
import sys
import csv

import numpy as np
import mujoco


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from src.controllers.trot_controller import TrotController
from src.utils.orientation import quaternion_to_euler


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "assets"
    / "mujoco_menagerie"
    / "unitree_a1"
    / "scene.xml"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "logs"
    / "turning_validation"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SUMMARY_FILE = (
    OUTPUT_DIR
    / "turning_validation_summary.csv"
)


# ============================================================
# STAGE-1 TURNING SETTINGS
# ============================================================

VX_COMMAND = 0.02


# Required Stage-1 turning envelope
WZ_COMMANDS = [
    -0.05,
     0.00,
     0.05,
]


EXPERIMENT_DURATION = 10.0

STEADY_STATE_START = 3.0


# ============================================================
# ACCEPTANCE LIMITS
# ============================================================

MAX_VX_ERROR_PERCENT = 25.0

MAX_WZ_ERROR_PERCENT = 20.0

MAX_STRAIGHT_WZ = 0.02

MAX_ROLL_DEG = 8.0

MAX_PITCH_DEG = 8.0


# ============================================================
# HELPERS
# ============================================================

def wrap_angle(angle):

    return (
        angle + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi


def percent_error(
    actual,
    target,
):

    if abs(target) < 1e-9:

        return 0.0


    return (
        abs(
            actual
            - target
        )
        /
        abs(
            target
        )
        * 100.0
    )


# ============================================================
# RUN TEST
# ============================================================

def run_test(
    wz_command,
):

    print("\n")
    print("=" * 84)

    print(
        f"TURN TEST | "
        f"vx={VX_COMMAND:+.3f} m/s | "
        f"wz={wz_command:+.3f} rad/s"
    )

    print("=" * 84)


    # ========================================================
    # MODEL
    # ========================================================

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )


    data = mujoco.MjData(
        model
    )


    # ========================================================
    # RESET
    # ========================================================

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


    # ========================================================
    # INITIAL STATE
    # ========================================================

    initial_x = float(
        data.qpos[0]
    )


    initial_y = float(
        data.qpos[1]
    )


    initial_quaternion = (
        data.qpos[3:7]
        .copy()
    )


    _, _, initial_yaw = (
        quaternion_to_euler(
            initial_quaternion
        )
    )


    # ========================================================
    # CONTROLLER
    # ========================================================

    controller = TrotController(
        gait_frequency=1.0,
        step_height=0.03,
        nominal_height=-0.30,
        ramp_time=2.0,
        k_yaw=0.03,

        # Calibrated asymmetric gains
        left_turn_gain=0.042,
        right_turn_gain=0.080,
    )


    controller.set_velocity(
        vx=VX_COMMAND,
        wz=wz_command,
    )


    # ========================================================
    # STORAGE
    # ========================================================

    times = []

    xs = []
    ys = []
    zs = []

    raw_yaws = []

    rolls = []
    pitches = []

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


    # ========================================================
    # SIMULATION
    # ========================================================

    while (
        simulation_time
        < EXPERIMENT_DURATION
    ):

        # ----------------------------------------------------
        # Current orientation
        # ----------------------------------------------------

        quat = data.qpos[3:7]


        roll, pitch, yaw = (
            quaternion_to_euler(
                quat
            )
        )


        # ----------------------------------------------------
        # Desired heading
        # ----------------------------------------------------

        desired_yaw = (
            initial_yaw
            +
            wz_command
            * simulation_time
        )


        yaw_error = wrap_angle(
            desired_yaw
            - yaw
        )


        # ----------------------------------------------------
        # Controller
        # ----------------------------------------------------

        command = (
            controller.get_command(
                simulation_time,
                yaw_error=yaw_error,
            )
        )


        data.ctrl[:] = command


        # ----------------------------------------------------
        # Physics
        # ----------------------------------------------------

        mujoco.mj_step(
            model,
            data,
        )


        simulation_time += (
            model.opt.timestep
        )


        step_count += 1


        # ----------------------------------------------------
        # State after step
        # ----------------------------------------------------

        x = float(
            data.qpos[0]
        )


        y = float(
            data.qpos[1]
        )


        z = float(
            data.qpos[2]
        )


        quat = data.qpos[3:7]


        roll, pitch, yaw = (
            quaternion_to_euler(
                quat
            )
        )


        current_desired_yaw = (
            initial_yaw
            +
            wz_command
            * simulation_time
        )


        current_yaw_error = wrap_angle(
            current_desired_yaw
            - yaw
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


        relative_yaw_deg = float(
            np.degrees(
                wrap_angle(
                    yaw
                    - initial_yaw
                )
            )
        )


        yaw_error_deg = float(
            np.degrees(
                current_yaw_error
            )
        )


        # ----------------------------------------------------
        # Fall detection
        # ----------------------------------------------------

        if (
            z < 0.20

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


        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        times.append(
            simulation_time
        )

        xs.append(
            x
        )

        ys.append(
            y
        )

        zs.append(
            z
        )

        raw_yaws.append(
            yaw
        )

        rolls.append(
            roll_deg
        )

        pitches.append(
            pitch_deg
        )

        yaw_errors.append(
            yaw_error_deg
        )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            step_count
            % steps_per_second
            == 0
        ):

            dx = (
                x
                - initial_x
            )


            dy = (
                y
                - initial_y
            )


            desired_relative_yaw = (
                np.degrees(
                    wz_command
                    * simulation_time
                )
            )


            print(
                f"time={simulation_time:5.2f}s | "
                f"dx={dx:+7.3f}m | "
                f"dy={dy:+7.3f}m | "
                f"yaw={relative_yaw_deg:+7.2f}° | "
                f"yaw_des={desired_relative_yaw:+7.2f}°"
            )


    # ========================================================
    # CHECK
    # ========================================================

    if len(
        times
    ) < 2:

        return None


    times = np.array(
        times
    )

    xs = np.array(
        xs
    )

    ys = np.array(
        ys
    )

    yaws = np.unwrap(
        np.array(
            raw_yaws
        )
    )


    # ========================================================
    # STEADY STATE
    # ========================================================

    steady_indices = np.where(
        times
        >= STEADY_STATE_START
    )[0]


    if len(
        steady_indices
    ) < 2:

        steady_indices = np.arange(
            len(
                times
            )
        )


    start_index = int(
        steady_indices[0]
    )


    dt = (
        times[-1]
        -
        times[start_index]
    )


    # ========================================================
    # YAW RATE
    # ========================================================

    measured_wz = (
        yaws[-1]
        -
        yaws[start_index]
    ) / dt


    # ========================================================
    # CURVED-MOTION FORWARD VELOCITY
    # ========================================================

    forward_distance = 0.0

    path_length = 0.0


    for i in range(
        start_index + 1,
        len(
            times
        )
    ):

        dx = (
            xs[i]
            -
            xs[i - 1]
        )


        dy = (
            ys[i]
            -
            ys[i - 1]
        )


        yaw_mid = (
            yaws[i]
            +
            yaws[i - 1]
        ) / 2.0


        forward_distance += (
            dx
            * np.cos(
                yaw_mid
            )

            +

            dy
            * np.sin(
                yaw_mid
            )
        )


        path_length += np.sqrt(
            dx ** 2
            +
            dy ** 2
        )


    measured_vx = (
        forward_distance
        /
        dt
    )


    path_speed = (
        path_length
        /
        dt
    )


    # ========================================================
    # ERRORS
    # ========================================================

    vx_error_percent = percent_error(
        measured_vx,
        VX_COMMAND,
    )


    if abs(
        wz_command
    ) > 1e-9:

        wz_error_percent = percent_error(
            measured_wz,
            wz_command,
        )

    else:

        wz_error_percent = 0.0


    # ========================================================
    # TURN RADIUS
    # ========================================================

    if abs(
        measured_wz
    ) > 1e-6:

        measured_radius = (
            abs(
                measured_vx
            )
            /
            abs(
                measured_wz
            )
        )

    else:

        measured_radius = np.inf


    if abs(
        wz_command
    ) > 1e-6:

        desired_radius = (
            VX_COMMAND
            /
            abs(
                wz_command
            )
        )

    else:

        desired_radius = np.inf


    # ========================================================
    # STABILITY
    # ========================================================

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


    avg_height = float(
        np.mean(
            zs
        )
    )


    max_yaw_error = float(
        np.max(
            np.abs(
                yaw_errors
            )
        )
    )


    # ========================================================
    # PASS / FAIL
    # ========================================================

    vx_pass = (
        vx_error_percent
        <= MAX_VX_ERROR_PERCENT
    )


    if abs(
        wz_command
    ) < 1e-9:

        wz_pass = (
            abs(
                measured_wz
            )
            <= MAX_STRAIGHT_WZ
        )

    else:

        wz_pass = (
            wz_error_percent
            <= MAX_WZ_ERROR_PERCENT

            and

            np.sign(
                measured_wz
            )
            ==
            np.sign(
                wz_command
            )
        )


    stability_pass = (
        max_roll
        <= MAX_ROLL_DEG

        and

        max_pitch
        <= MAX_PITCH_DEG

        and

        not fall_detected
    )


    overall_pass = (
        vx_pass
        and wz_pass
        and stability_pass
    )


    # ========================================================
    # RESULT
    # ========================================================

    result = {

        "wz_target":
            wz_command,

        "wz_measured":
            measured_wz,

        "wz_error_percent":
            wz_error_percent,

        "vx_target":
            VX_COMMAND,

        "vx_measured":
            measured_vx,

        "vx_error_percent":
            vx_error_percent,

        "desired_radius":
            desired_radius,

        "measured_radius":
            measured_radius,

        "path_speed":
            path_speed,

        "max_yaw_error":
            max_yaw_error,

        "max_roll":
            max_roll,

        "max_pitch":
            max_pitch,

        "avg_height":
            avg_height,

        "fall":
            fall_detected,

        "overall_pass":
            overall_pass,
    }


    print("\n")
    print("-" * 74)


    print(
        f"RESULT | "
        f"wz={wz_command:+.3f}"
    )


    print("-" * 74)


    print(
        f"Target vx               : "
        f"{VX_COMMAND:+.4f} m/s"
    )


    print(
        f"Measured vx             : "
        f"{measured_vx:+.4f} m/s"
    )


    print(
        f"vx error                : "
        f"{vx_error_percent:.2f}%"
    )


    print(
        f"Target wz               : "
        f"{wz_command:+.4f} rad/s"
    )


    print(
        f"Measured wz             : "
        f"{measured_wz:+.4f} rad/s"
    )


    if abs(
        wz_command
    ) > 1e-9:

        print(
            f"wz error                : "
            f"{wz_error_percent:.2f}%"
        )


    print(
        f"Desired radius          : "
        f"{desired_radius:.3f} m"
    )


    print(
        f"Measured radius         : "
        f"{measured_radius:.3f} m"
    )


    print(
        f"Max roll                : "
        f"{max_roll:.2f}°"
    )


    print(
        f"Max pitch               : "
        f"{max_pitch:.2f}°"
    )


    print(
        f"PASS                    : "
        f"{overall_pass}"
    )


    return result


# ============================================================
# SAVE SUMMARY
# ============================================================

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
            "wz_target",
            "wz_measured",
            "wz_error_percent",
            "vx_target",
            "vx_measured",
            "vx_error_percent",
            "desired_radius",
            "measured_radius",
            "path_speed",
            "max_yaw_error",
            "max_roll",
            "max_pitch",
            "avg_height",
            "fall",
            "overall_pass",
        ])


        for r in results:

            writer.writerow([
                r["wz_target"],
                r["wz_measured"],
                r["wz_error_percent"],
                r["vx_target"],
                r["vx_measured"],
                r["vx_error_percent"],
                r["desired_radius"],
                r["measured_radius"],
                r["path_speed"],
                r["max_yaw_error"],
                r["max_roll"],
                r["max_pitch"],
                r["avg_height"],
                r["fall"],
                r["overall_pass"],
            ])


# ============================================================
# FINAL TABLE
# ============================================================

def print_summary(
    results,
):

    print("\n\n")

    print("=" * 104)

    print(
        "STAGE-1 SUPPORTED TURNING VALIDATION"
    )

    print("=" * 104)


    print(
        f"{'wz target':>11}"
        f"{'wz actual':>12}"
        f"{'wz err%':>11}"
        f"{'vx actual':>12}"
        f"{'vx err%':>11}"
        f"{'R actual':>12}"
        f"{'roll':>9}"
        f"{'pitch':>9}"
        f"{'PASS':>9}"
    )


    print(
        "-" * 104
    )


    for r in results:

        radius_text = (
            f"{r['measured_radius']:.3f}"

            if np.isfinite(
                r["measured_radius"]
            )

            else "inf"
        )


        print(
            f"{r['wz_target']:>11.3f}"
            f"{r['wz_measured']:>12.4f}"
            f"{r['wz_error_percent']:>11.2f}"
            f"{r['vx_measured']:>12.4f}"
            f"{r['vx_error_percent']:>11.2f}"
            f"{radius_text:>12}"
            f"{r['max_roll']:>9.2f}"
            f"{r['max_pitch']:>9.2f}"
            f"{str(r['overall_pass']):>9}"
        )


    print(
        "=" * 104
    )


    passed = sum(
        1
        for r in results
        if r[
            "overall_pass"
        ]
    )


    print(
        f"\nTurning tests passed: "
        f"{passed}/{len(results)}"
    )


    if passed == len(
        results
    ):

        print(
            "\nSTAGE-1 SUPPORTED TURNING: PASSED"
        )

        print(
            "\nNext:"
        )

        print(
            "python "
            "experiments/15_final_locomotion_test.py"
        )

    else:

        print(
            "\nSUPPORTED TURNING STILL NEEDS "
            "MINOR CALIBRATION."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    results = []


    for wz in WZ_COMMANDS:

        result = run_test(
            wz
        )


        if result is not None:

            results.append(
                result
            )


    print_summary(
        results
    )


    save_summary(
        results
    )


    print(
        f"\nSummary saved to:\n"
        f"{SUMMARY_FILE}"
    )


if __name__ == "__main__":

    main()