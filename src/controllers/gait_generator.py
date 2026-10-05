import numpy as np


def foot_trajectory(
    phase,
    step_length=0.10,
    step_height=0.05,
    nominal_height=-0.30,
):
    phase = phase % 1.0

    if phase < 0.5:
        # STANCE: foot moves backward relative to body
        stance_phase = phase / 0.5

        x = (
            -step_length / 2
            + step_length * stance_phase
        )

        z = nominal_height

    else:
        # SWING: foot moves forward
        swing_phase = (phase - 0.5) / 0.5

        x = (
            step_length / 2
            - step_length * swing_phase
        )

        z = (
            nominal_height
            + step_height * np.sin(np.pi * swing_phase)
        )

    return x, z