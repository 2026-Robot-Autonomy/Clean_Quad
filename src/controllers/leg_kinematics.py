import numpy as np


THIGH_LENGTH = 0.20
CALF_LENGTH = 0.20


def solve_planar_ik(x, z):
    """
    Solve the 2-link inverse kinematics for one leg.

    Parameters
    ----------
    x : float
        Forward/backward foot position relative to hip.

    z : float
        Vertical foot position relative to hip.
        Negative means below the hip.

    Returns
    -------
    q_thigh : float
    q_calf : float
    """

    L1 = THIGH_LENGTH
    L2 = CALF_LENGTH

    D = (
        x**2 + z**2 - L1**2 - L2**2
    ) / (2 * L1 * L2)

    # Numerical safety
    D = np.clip(D, -1.0, 1.0)

    q_calf = -np.arccos(D)

    q_thigh = (
        np.arctan2(-x, -z)
        - np.arctan2(
            L2 * np.sin(q_calf),
            L1 + L2 * np.cos(q_calf)
        )
    )

    return q_thigh, q_calf

def forward_kinematics(q_thigh, q_calf):
    """
    Convert joint angles back to foot position.
    """

    L1 = THIGH_LENGTH
    L2 = CALF_LENGTH

    x = (
        -L1 * np.sin(q_thigh)
        -L2 * np.sin(q_thigh + q_calf)
    )

    z = (
        -L1 * np.cos(q_thigh)
        -L2 * np.cos(q_thigh + q_calf)
    )

    return x, z

'''
    foot position
      ↓
IK
      ↓
joint angles
      ↓
FK
      ↓
reconstructed foot position

'''

