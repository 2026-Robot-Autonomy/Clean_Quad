import numpy as np


# ---------------------------------------------------------
# A1 LEG GEOMETRY
# ---------------------------------------------------------

HIP_LENGTH = 0.085
THIGH_LENGTH = 0.20
CALF_LENGTH = 0.20


# ---------------------------------------------------------
# PLANAR IK
# ---------------------------------------------------------

def solve_planar_ik(x, z):
    """
    Solve thigh/calf IK in the sagittal x-z plane.

    x:
        positive/negative forward position relative to hip

    z:
        vertical foot position.
        Negative means below the hip.
    """

    L1 = THIGH_LENGTH
    L2 = CALF_LENGTH

    D = (
        x**2
        + z**2
        - L1**2
        - L2**2
    ) / (
        2 * L1 * L2
    )

    # Prevent numerical errors such as
    # acos(1.00000001)
    D = np.clip(
        D,
        -1.0,
        1.0
    )

    q_calf = -np.arccos(D)

    q_thigh = (
        np.arctan2(
            -x,
            -z
        )
        -
        np.arctan2(
            L2 * np.sin(q_calf),
            L1
            + L2 * np.cos(q_calf)
        )
    )

    return q_thigh, q_calf


# ---------------------------------------------------------
# 3D LEG IK
# ---------------------------------------------------------

def solve_leg_ik(
    x,
    y,
    z,
    side_sign,
):
    """
    Solve 3D inverse kinematics for one A1 leg.

    Parameters
    ----------
    x : float
        Forward/backward foot position.

    y : float
        Lateral foot position relative to hip.

    z : float
        Vertical foot position.
        Negative means below robot.

    side_sign : int
        +1 for LEFT legs
        -1 for RIGHT legs

    Returns
    -------
    q_hip
    q_thigh
    q_calf
    """

    hip = HIP_LENGTH

    # -----------------------------------------------------
    # 1. Distance in lateral/vertical plane
    # -----------------------------------------------------

    yz_squared = (
        y**2 + z**2
    )

    # Need enough distance to accommodate hip link
    if yz_squared <= hip**2:
        raise ValueError(
            "Foot target too close to hip joint."
        )


    # -----------------------------------------------------
    # 2. Effective vertical distance after accounting
    #    for lateral hip link
    # -----------------------------------------------------

    z_planar = -np.sqrt(
        yz_squared - hip**2
    )


    # -----------------------------------------------------
    # 3. Hip abduction/adduction angle
    # -----------------------------------------------------

    q_hip = (
        np.arctan2(z, y)
        -
        np.arctan2(
            z_planar,
            side_sign * hip
        )
    )


    # Wrap angle to [-pi, pi]
    q_hip = (
        q_hip + np.pi
    ) % (
        2 * np.pi
    ) - np.pi


    # -----------------------------------------------------
    # 4. Remaining leg becomes planar problem
    # -----------------------------------------------------

    q_thigh, q_calf = solve_planar_ik(
        x,
        z_planar
    )


    return (
        q_hip,
        q_thigh,
        q_calf,
    )