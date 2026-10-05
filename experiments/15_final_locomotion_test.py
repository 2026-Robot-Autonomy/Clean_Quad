from pathlib import Path
import sys
import csv
import time

import numpy as np
import mujoco
import mujoco.viewer


# ============================================================
# PROJECT SETUP
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
    / "final_locomotion"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SUMMARY_FILE = (
    OUTPUT_DIR
    / "final_locomotion_summary.csv"
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

EXPERIMENT_DURATION = 10.0

STEADY_STATE_START = 3.0


# ============================================================
# VIEWER SETTINGS
# ============================================================

# Run approximately in real time so the robot motion
# can actually be watched.

REAL_TIME_VIEWER = True


# Short pause between experiments.

PAUSE_BETWEEN_TESTS = 2.0


# ============================================================
# VALIDATED STAGE-1 LOCOMOTION ENVELOPE
# ============================================================
#
# REQUIRED:
#
#   Standing
#
#   Straight:
#
#       vx = +0.02
#       vx = +0.03
#       vx = -0.02
#       vx = -0.03
#
#   Forward turning:
#
#       vx = +0.03
#       wz = +0.05
#
#       vx = +0.03
#       wz = -0.05
#
#
# OPTIONAL:
#
#   Slow forward curves
#   Backward curves
#   Rotation in place
#
# ============================================================

TESTS = [

    # ========================================================
    # REQUIRED TESTS
    # ========================================================

    {
        "name": "stand",
        "vx": 0.00,
        "wz": 0.00,
        "required": True,
        "type": "stand",
    },

    {
        "name": "forward_002",
        "vx": 0.02,
        "wz": 0.00,
        "required": True,
        "type": "straight",
    },

    {
        "name": "forward_003",
        "vx": 0.03,
        "wz": 0.00,
        "required": True,
        "type": "straight",
    },

    {
        "name": "backward_002",
        "vx": -0.02,
        "wz": 0.00,
        "required": True,
        "type": "straight",
    },

    {
        "name": "backward_003",
        "vx": -0.03,
        "wz": 0.00,
        "required": True,
        "type": "straight",
    },

    {
        "name": "forward_left",
        "vx": 0.03,
        "wz": 0.05,
        "required": True,
        "type": "curve",
    },

    {
        "name": "forward_right",
        "vx": 0.03,
        "wz": -0.05,
        "required": True,
        "type": "curve",
    },


    # ========================================================
    # OPTIONAL TESTS
    # ========================================================

    {
        "name": "slow_forward_left",
        "vx": 0.02,
        "wz": 0.05,
        "required": False,
        "type": "curve",
    },

    {
        "name": "slow_forward_right",
        "vx": 0.02,
        "wz": -0.05,
        "required": False,
        "type": "curve",
    },

    {
        "name": "backward_left",
        "vx": -0.02,
        "wz": 0.05,
        "required": False,
        "type": "curve",
    },

    {
        "name": "backward_right",
        "vx": -0.02,
        "wz": -0.05,
        "required": False,
        "type": "curve",
    },

    {
        "name": "rotate_left",
        "vx": 0.00,
        "wz": 0.05,
        "required": False,
        "type": "rotation",
    },

    {
        "name": "rotate_right",
        "vx": 0.00,
        "wz": -0.05,
        "required": False,
        "type": "rotation",
    },
]


# ============================================================
# ACCEPTANCE THRESHOLDS
# ============================================================

MAX_STRAIGHT_VX_ERROR_PERCENT = 20.0

MAX_CURVE_VX_ERROR_PERCENT = 25.0

MAX_WZ_ERROR_PERCENT = 20.0

MAX_STRAIGHT_DRIFT_PERCENT = 15.0

MAX_STRAIGHT_FINAL_YAW_DEG = 5.0

MAX_ROLL_DEG = 8.0

MAX_PITCH_DEG = 8.0

MAX_STAND_SPEED = 0.005

MAX_STAND_DISPLACEMENT = 0.035

MAX_ROTATION_TRANSLATIONAL_SPEED = 0.01


# ============================================================
# HELPERS
# ============================================================

def wrap_angle(angle):
    """
    Wrap angle to [-pi, pi].
    """

    return (
        angle + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi


def percent_error(
    actual,
    target,
):
    """
    Relative percentage error.
    """

    if abs(target) < 1e-9:

        return 0.0


    return (
        abs(
            actual - target
        )
        /
        abs(
            target
        )
        * 100.0
    )


# ============================================================
# RESET ROBOT
# ============================================================

def reset_robot(
    model,
    data,
):

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


# ============================================================
# REAL-TIME PHYSICS / VIEWER SYNC
# ============================================================

def realtime_step(
    model,
    data,
    viewer,
):
    """
    Advance simulation by one physics step and synchronize
    the MuJoCo viewer.

    When REAL_TIME_VIEWER is True, sleep just enough so the
    simulation approximately matches wall-clock time.
    """

    wall_start = time.perf_counter()


    mujoco.mj_step(
        model,
        data,
    )


    if viewer is not None:

        viewer.sync()


    if REAL_TIME_VIEWER:

        computation_time = (
            time.perf_counter()
            -
            wall_start
        )


        sleep_time = (
            model.opt.timestep
            -
            computation_time
        )


        if sleep_time > 0:

            time.sleep(
                sleep_time
            )


# ============================================================
# RUN ONE TEST
# ============================================================

def run_test(
    model,
    data,
    viewer,
    test,
):

    name = test["name"]

    vx_command = test["vx"]

    wz_command = test["wz"]

    test_type = test["type"]

    required = test["required"]


    print("\n\n")

    print("=" * 96)

    print(
        f"RUNNING TEST: {name}"
    )

    print(
        f"vx = {vx_command:+.3f} m/s"
    )

    print(
        f"wz = {wz_command:+.3f} rad/s"
    )

    print(
        "TYPE = "
        + (
            "REQUIRED"
            if required
            else "OPTIONAL"
        )
    )

    print("=" * 96)


    # ========================================================
    # 1. RESET ROBOT
    # ========================================================

    reset_robot(
        model,
        data,
    )


    if viewer is not None:

        viewer.sync()


    # Give the viewer time to visibly show the reset.

    time.sleep(
        0.5
    )


    # ========================================================
    # 2. INITIAL STATE
    # ========================================================

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


    # ========================================================
    # 3. CONTROLLER
    # ========================================================

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


    # ========================================================
    # 4. LOGGING STORAGE
    # ========================================================

    times = []

    xs = []
    ys = []
    zs = []

    raw_yaws = []

    rolls = []
    pitches = []

    yaw_errors = []

    body_velocities = []


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
    # 5. SIMULATION LOOP
    # ========================================================

    while (
        simulation_time
        < EXPERIMENT_DURATION
    ):

        # ----------------------------------------------------
        # If viewer window was closed, stop everything.
        # ----------------------------------------------------

        if (
            viewer is not None
            and not viewer.is_running()
        ):

            print(
                "\nViewer closed by user."
            )

            return None


        # ----------------------------------------------------
        # Current orientation
        # ----------------------------------------------------

        quat = (
            data.qpos[3:7]
        )


        roll, pitch, yaw = (
            quaternion_to_euler(
                quat
            )
        )


        # ----------------------------------------------------
        # Desired yaw trajectory
        # ----------------------------------------------------
        #
        # ψ_des(t) =
        #
        # ψ_0 + wz * t
        #
        # ----------------------------------------------------

        desired_yaw = (
            initial_yaw
            +
            wz_command
            * simulation_time
        )


        # ----------------------------------------------------
        # Heading error
        # ----------------------------------------------------

        yaw_error = wrap_angle(
            desired_yaw
            - yaw
        )


        # ----------------------------------------------------
        # Controller
        # ----------------------------------------------------

        control_command = (
            controller.get_command(
                simulation_time,
                yaw_error=yaw_error,
            )
        )


        data.ctrl[:] = (
            control_command
        )


        # ----------------------------------------------------
        # Physics + viewer
        # ----------------------------------------------------

        realtime_step(
            model,
            data,
            viewer,
        )


        simulation_time += (
            model.opt.timestep
        )


        step_count += 1


        # ----------------------------------------------------
        # Position
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


        # ----------------------------------------------------
        # Updated orientation
        # ----------------------------------------------------

        quat = (
            data.qpos[3:7]
        )


        roll, pitch, yaw = (
            quaternion_to_euler(
                quat
            )
        )


        # ----------------------------------------------------
        # World velocity
        # ----------------------------------------------------

        world_vx = float(
            data.qvel[0]
        )

        world_vy = float(
            data.qvel[1]
        )


        # ----------------------------------------------------
        # Body-frame forward velocity
        # ----------------------------------------------------

        body_velocity = (
            world_vx
            * np.cos(
                yaw
            )

            +

            world_vy
            * np.sin(
                yaw
            )
        )


        # ----------------------------------------------------
        # Current desired yaw
        # ----------------------------------------------------

        current_desired_yaw = (
            initial_yaw
            +
            wz_command
            * simulation_time
        )


        current_yaw_error = (
            wrap_angle(
                current_desired_yaw
                - yaw
            )
        )


        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

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


        yaw_error_deg = float(
            np.degrees(
                current_yaw_error
            )
        )


        relative_yaw = wrap_angle(
            yaw
            - initial_yaw
        )


        relative_yaw_deg = float(
            np.degrees(
                relative_yaw
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
                "\n!!! FALL DETECTED !!!"
            )

            fall_detected = True

            break


        # ----------------------------------------------------
        # Store data
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

        body_velocities.append(
            body_velocity
        )


        # ----------------------------------------------------
        # Progress every approximately 1 second
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


            print(
                f"time={simulation_time:5.2f}s | "
                f"dx={dx:+7.3f}m | "
                f"dy={dy:+7.3f}m | "
                f"v_body={body_velocity:+7.3f}m/s | "
                f"yaw={relative_yaw_deg:+7.2f}°"
            )


    # ========================================================
    # 6. CHECK COLLECTED DATA
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

    raw_yaws = np.array(
        raw_yaws
    )


    # ========================================================
    # 7. UNWRAP YAW
    # ========================================================

    yaws = np.unwrap(
        raw_yaws
    )


    # ========================================================
    # 8. STEADY-STATE REGION
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


    steady_dt = (
        times[-1]
        -
        times[start_index]
    )


    # ========================================================
    # 9. MEASURED ANGULAR VELOCITY
    # ========================================================

    if steady_dt > 1e-9:

        measured_wz = (
            yaws[-1]
            -
            yaws[start_index]
        ) / steady_dt

    else:

        measured_wz = 0.0


    # ========================================================
    # 10. MEASURED BODY-FRAME TRANSLATIONAL VELOCITY
    # ========================================================
    #
    # For curved motion, world-frame dx/dt is incorrect.
    #
    # We integrate displacement projected along the
    # instantaneous robot heading:
    #
    # ds_forward =
    #
    # dx cos(psi) + dy sin(psi)
    #
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


        forward_component = (
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


        forward_distance += (
            forward_component
        )


        path_length += (
            np.sqrt(
                dx ** 2
                +
                dy ** 2
            )
        )


    if steady_dt > 1e-9:

        measured_vx = (
            forward_distance
            /
            steady_dt
        )


        path_speed = (
            path_length
            /
            steady_dt
        )

    else:

        measured_vx = 0.0

        path_speed = 0.0


    # ========================================================
    # 11. FINAL DISPLACEMENT
    # ========================================================

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


    total_displacement = (
        np.sqrt(
            final_dx ** 2
            +
            final_dy ** 2
        )
    )


    # ========================================================
    # 12. STRAIGHT-LINE DRIFT
    # ========================================================

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


    # ========================================================
    # 13. TRACKING ERRORS
    # ========================================================

    vx_error_percent = (
        percent_error(
            measured_vx,
            vx_command,
        )
    )


    wz_error_percent = (
        percent_error(
            measured_wz,
            wz_command,
        )
    )


    # ========================================================
    # 14. FINAL HEADING
    # ========================================================

    final_relative_yaw = (
        yaws[-1]
        -
        yaws[0]
    )


    final_relative_yaw_deg = float(
        np.degrees(
            final_relative_yaw
        )
    )


    desired_final_yaw_deg = float(
        np.degrees(
            wz_command
            * times[-1]
        )
    )


    # ========================================================
    # 15. STABILITY
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


    max_yaw_error = float(
        np.max(
            np.abs(
                yaw_errors
            )
        )
    )


    average_height = float(
        np.mean(
            zs
        )
    )


    # ========================================================
    # 16. PASS / FAIL CONDITIONS
    # ========================================================

    stability_pass = (
        max_roll
        <= MAX_ROLL_DEG

        and

        max_pitch
        <= MAX_PITCH_DEG

        and

        not fall_detected
    )


    # Defaults

    vx_pass = True

    wz_pass = True

    drift_pass = True

    heading_pass = True

    stand_pass = True

    rotation_translation_pass = True


    # ========================================================
    # STAND
    # ========================================================

    if test_type == "stand":

        stand_pass = (
            abs(
                measured_vx
            )
            <= MAX_STAND_SPEED

            and

            total_displacement
            <= MAX_STAND_DISPLACEMENT
        )


    # ========================================================
    # STRAIGHT MOTION
    # ========================================================

    elif test_type == "straight":

        vx_pass = (
            vx_error_percent
            <= MAX_STRAIGHT_VX_ERROR_PERCENT

            and

            np.sign(
                measured_vx
            )
            ==
            np.sign(
                vx_command
            )
        )


        drift_pass = (
            drift_percent
            <= MAX_STRAIGHT_DRIFT_PERCENT
        )


        heading_pass = (
            abs(
                final_relative_yaw_deg
            )
            <= MAX_STRAIGHT_FINAL_YAW_DEG
        )


    # ========================================================
    # CURVED MOTION
    # ========================================================

    elif test_type == "curve":

        vx_pass = (
            vx_error_percent
            <= MAX_CURVE_VX_ERROR_PERCENT

            and

            np.sign(
                measured_vx
            )
            ==
            np.sign(
                vx_command
            )
        )


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


    # ========================================================
    # ROTATION IN PLACE
    # ========================================================

    elif test_type == "rotation":

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


        rotation_translation_pass = (
            abs(
                measured_vx
            )
            <= MAX_ROTATION_TRANSLATIONAL_SPEED
        )


    # ========================================================
    # OVERALL RESULT
    # ========================================================

    overall_pass = (
        stability_pass
        and vx_pass
        and wz_pass
        and drift_pass
        and heading_pass
        and stand_pass
        and rotation_translation_pass
    )


    # ========================================================
    # 17. SAVE PER-TEST CSV
    # ========================================================

    run_file = (
        OUTPUT_DIR
        / f"{name}.csv"
    )


    with open(
        run_file,
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
            "yaw_rad",
            "roll_deg",
            "pitch_deg",
            "yaw_error_deg",
            "body_velocity",
        ])


        for row in zip(
            times,
            xs,
            ys,
            zs,
            raw_yaws,
            rolls,
            pitches,
            yaw_errors,
            body_velocities,
        ):

            writer.writerow(
                row
            )


    # ========================================================
    # 18. PRINT RESULT
    # ========================================================

    print("\n")

    print("-" * 76)

    print(
        f"RESULT: {name}"
    )

    print("-" * 76)


    print(
        f"Target vx       : "
        f"{vx_command:+.4f} m/s"
    )


    print(
        f"Measured vx     : "
        f"{measured_vx:+.4f} m/s"
    )


    if abs(
        vx_command
    ) > 1e-9:

        print(
            f"vx error        : "
            f"{vx_error_percent:.2f}%"
        )


    print(
        f"Target wz       : "
        f"{wz_command:+.4f} rad/s"
    )


    print(
        f"Measured wz     : "
        f"{measured_wz:+.4f} rad/s"
    )


    if abs(
        wz_command
    ) > 1e-9:

        print(
            f"wz error        : "
            f"{wz_error_percent:.2f}%"
        )


    print(
        f"Max roll        : "
        f"{max_roll:.2f}°"
    )


    print(
        f"Max pitch       : "
        f"{max_pitch:.2f}°"
    )


    print(
        f"Fall            : "
        f"{fall_detected}"
    )


    print(
        f"Required        : "
        f"{required}"
    )


    print(
        f"PASS            : "
        f"{overall_pass}"
    )


    # ========================================================
    # PAUSE SO USER CAN SEE END OF TEST
    # ========================================================

    if viewer is not None:

        viewer.sync()


    time.sleep(
        PAUSE_BETWEEN_TESTS
    )


    # ========================================================
    # RETURN
    # ========================================================

    return {

        "name":
            name,

        "type":
            test_type,

        "required":
            required,

        "vx_target":
            vx_command,

        "vx_measured":
            measured_vx,

        "vx_error_percent":
            vx_error_percent,

        "wz_target":
            wz_command,

        "wz_measured":
            measured_wz,

        "wz_error_percent":
            wz_error_percent,

        "path_speed":
            path_speed,

        "final_dx":
            final_dx,

        "final_dy":
            final_dy,

        "drift_percent":
            drift_percent,

        "desired_yaw":
            desired_final_yaw_deg,

        "actual_yaw":
            final_relative_yaw_deg,

        "max_yaw_error":
            max_yaw_error,

        "max_roll":
            max_roll,

        "max_pitch":
            max_pitch,

        "average_height":
            average_height,

        "fall":
            fall_detected,

        "vx_pass":
            vx_pass,

        "wz_pass":
            wz_pass,

        "stability_pass":
            stability_pass,

        "overall_pass":
            overall_pass,
    }


# ============================================================
# SUMMARY TABLE
# ============================================================

def print_summary_table(
    results,
):

    print("\n\n")

    print("=" * 142)

    print(
        "FINAL STAGE-1 LOCOMOTION VALIDATION"
    )

    print("=" * 142)


    print(
        f"{'test':<23}"
        f"{'vx*':>8}"
        f"{'vx':>9}"
        f"{'vxerr%':>10}"
        f"{'wz*':>9}"
        f"{'wz':>10}"
        f"{'wzerr%':>10}"
        f"{'roll':>9}"
        f"{'pitch':>9}"
        f"{'fall':>8}"
        f"{'REQ':>8}"
        f"{'PASS':>9}"
    )


    print(
        "-" * 142
    )


    for result in results:

        vx_error_text = (
            f"{result['vx_error_percent']:.1f}"
            if abs(
                result["vx_target"]
            ) > 1e-9
            else "-"
        )


        wz_error_text = (
            f"{result['wz_error_percent']:.1f}"
            if abs(
                result["wz_target"]
            ) > 1e-9
            else "-"
        )


        print(
            f"{result['name']:<23}"
            f"{result['vx_target']:>8.3f}"
            f"{result['vx_measured']:>9.4f}"
            f"{vx_error_text:>10}"
            f"{result['wz_target']:>9.3f}"
            f"{result['wz_measured']:>10.4f}"
            f"{wz_error_text:>10}"
            f"{result['max_roll']:>9.2f}"
            f"{result['max_pitch']:>9.2f}"
            f"{str(result['fall']):>8}"
            f"{str(result['required']):>8}"
            f"{str(result['overall_pass']):>9}"
        )


    print(
        "=" * 142
    )


# ============================================================
# FINAL VERDICT
# ============================================================

def print_final_verdict(
    results,
):

    required_results = [
        result
        for result in results
        if result["required"]
    ]


    optional_results = [
        result
        for result in results
        if not result["required"]
    ]


    required_passed = sum(
        1
        for result in required_results
        if result["overall_pass"]
    )


    optional_passed = sum(
        1
        for result in optional_results
        if result["overall_pass"]
    )


    print("\n")

    print("=" * 88)

    print(
        "STAGE-1 FINAL VERDICT"
    )

    print("=" * 88)


    print(
        f"Required tests passed : "
        f"{required_passed}/"
        f"{len(required_results)}"
    )


    print(
        f"Optional tests passed : "
        f"{optional_passed}/"
        f"{len(optional_results)}"
    )


    # ========================================================
    # PASS
    # ========================================================

    if (
        required_passed
        ==
        len(
            required_results
        )
    ):

        print("\n")

        print(
            "STAGE 1 LOCOMOTION: PASSED"
        )


        print("\n")

        print(
            "VALIDATED OPERATING ENVELOPE"
        )


        print("\nStraight translation:")

        print(
            "    vx = +0.02 m/s"
        )

        print(
            "    vx = +0.03 m/s"
        )

        print(
            "    vx = -0.02 m/s"
        )

        print(
            "    vx = -0.03 m/s"
        )


        print("\nForward turning:")

        print(
            "    vx = +0.03 m/s"
        )

        print(
            "    wz = ±0.05 rad/s"
        )


        print("\nReverse:")

        print(
            "    straight reverse supported"
        )

        print(
            "    reverse steering experimental"
        )


        print("\n")

        print(
            "Stage 1 locomotion can now be frozen."
        )


    # ========================================================
    # FAIL
    # ========================================================

    else:

        print("\n")

        print(
            "STAGE 1 LOCOMOTION: NOT YET PASSED"
        )


        print(
            "\nFailed REQUIRED tests:"
        )


        for result in required_results:

            if not result["overall_pass"]:

                print(
                    f"    - "
                    f"{result['name']}"
                )


    print(
        "=" * 88
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

def save_summary(
    results,
):

    if len(
        results
    ) == 0:

        return


    with open(
        SUMMARY_FILE,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                results[0].keys()
            ),
        )


        writer.writeheader()


        for result in results:

            writer.writerow(
                result
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 88)

    print(
        "CLEANQUAD"
    )

    print(
        "FINAL STAGE-1 LOCOMOTION CERTIFICATION"
    )

    print("=" * 88)


    # ========================================================
    # LOAD MODEL ONCE
    # ========================================================

    model = mujoco.MjModel.from_xml_path(
        str(
            MODEL_PATH
        )
    )


    data = mujoco.MjData(
        model
    )


    reset_robot(
        model,
        data,
    )


    results = []


    # ========================================================
    # OPEN MUJOCO VIEWER
    # ========================================================

    print("\n")

    print(
        "Opening MuJoCo viewer..."
    )


    print(
        "Close the viewer window at any time "
        "to terminate the test."
    )


    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

        viewer.sync()


        time.sleep(
            1.0
        )


        # ====================================================
        # RUN ALL TESTS
        # ====================================================

        for test_number, test in enumerate(
            TESTS,
            start=1,
        ):

            if not viewer.is_running():

                print(
                    "\nViewer closed. "
                    "Stopping certification."
                )

                break


            print("\n\n")

            print("#" * 96)

            print(
                f"TEST "
                f"{test_number}/"
                f"{len(TESTS)}"
            )

            print(
                f"NEXT MOTION: "
                f"{test['name']}"
            )

            print("#" * 96)


            result = run_test(
                model,
                data,
                viewer,
                test,
            )


            if result is None:

                break


            results.append(
                result
            )


        # ====================================================
        # FINAL VIEWER PAUSE
        # ====================================================

        if viewer.is_running():

            print("\n")

            print(
                "All locomotion tests completed."
            )


            print(
                "Closing viewer shortly..."
            )


            time.sleep(
                2.0
            )


    # ========================================================
    # RESULTS
    # ========================================================

    print_summary_table(
        results
    )


    print_final_verdict(
        results
    )


    save_summary(
        results
    )


    print("\n")

    print(
        f"Summary saved to:\n"
        f"{SUMMARY_FILE}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()