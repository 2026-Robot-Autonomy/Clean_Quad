import sys
import os

# Adds the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.controllers.leg_kinematics import solve_leg_ik


NOMINAL_Z = -0.30
HIP_OFFSET = 0.085


print("\nFRONT RIGHT")

q_hip, q_thigh, q_calf = solve_leg_ik(
    x=0.0,
    y=-HIP_OFFSET,
    z=NOMINAL_Z,
    side_sign=-1,
)

print("hip   :", q_hip)
print("thigh :", q_thigh)
print("calf  :", q_calf)


print("\nFRONT LEFT")

q_hip, q_thigh, q_calf = solve_leg_ik(
    x=0.0,
    y=HIP_OFFSET,
    z=NOMINAL_Z,
    side_sign=1,
)

print("hip   :", q_hip)
print("thigh :", q_thigh)
print("calf  :", q_calf)