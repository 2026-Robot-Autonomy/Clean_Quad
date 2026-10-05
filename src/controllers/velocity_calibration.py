import numpy as np


# =========================================================
# EXPERIMENTALLY MEASURED A1 GAIT CALIBRATION
# =========================================================
#
# These values came from our straight-line,
# yaw-stabilized velocity calibration experiment.
#
# measured velocity = actual steady-state translation
#
# step_length = raw gait stride used internally
# =========================================================


MEASURED_VELOCITIES = np.array([
    -0.0572,
    -0.0530,
    -0.0468,
    -0.0414,
    -0.0233,
    -0.0149,
    -0.0056,
     0.0027,
     0.0156,
     0.0314,
])


STEP_LENGTHS = np.array([
     0.05,
     0.04,
     0.03,
     0.02,
    -0.02,
    -0.03,
    -0.04,
    -0.05,
    -0.06,
    -0.07,
])


MIN_CALIBRATED_VELOCITY = MEASURED_VELOCITIES[0]

MAX_CALIBRATED_VELOCITY = MEASURED_VELOCITIES[-1]


def velocity_to_step_length(
    target_velocity,
):
    """
    Convert desired physical forward velocity [m/s]
    into experimentally calibrated gait step length.

    Positive target_velocity  -> forward
    Negative target_velocity  -> backward
    """

    target_velocity = np.clip(
        target_velocity,
        MIN_CALIBRATED_VELOCITY,
        MAX_CALIBRATED_VELOCITY,
    )


    step_length = np.interp(
        target_velocity,
        MEASURED_VELOCITIES,
        STEP_LENGTHS,
    )


    return float(step_length)