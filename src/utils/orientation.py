import numpy as np


def quaternion_to_euler(q):
    """
    MuJoCo quaternion:
    q = [w, x, y, z]

    Returns:
    roll, pitch, yaw
    """

    w, x, y, z = q

    # Roll
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)

    roll = np.arctan2(
        sinr_cosp,
        cosr_cosp
    )

    # Pitch
    sinp = 2 * (w * y - z * x)

    sinp = np.clip(
        sinp,
        -1.0,
        1.0
    )

    pitch = np.arcsin(sinp)

    # Yaw
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)

    yaw = np.arctan2(
        siny_cosp,
        cosy_cosp
    )

    return roll, pitch, yaw