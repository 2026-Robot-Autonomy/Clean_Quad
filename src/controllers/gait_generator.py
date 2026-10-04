import numpy as np


def foot_trajectory(
    phase,
    step_length=0.10,# foot moves b/w 0.05 to -0.05 relative to hip
    step_height=0.05,
    nominal_height=-0.30,
):
    """
    Generate desired x,z position for one foot.

    phase must be in [0, 1).
    """

    phase = phase % 1.0

    if phase < 0.5:
        # STANCE
        stance_phase = phase / 0.5

        x = (
            step_length / 2
            - step_length * stance_phase
        )

        z = nominal_height

    else:
        # SWING
        swing_phase = (phase - 0.5) / 0.5

        x = (
            -step_length / 2
            + step_length * swing_phase
        )
        # swing arc
        z = (
            nominal_height
            + step_height
            * np.sin(np.pi * swing_phase)
        )

    return x, z