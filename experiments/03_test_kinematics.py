from src.controllers.leg_kinematics import (
    solve_planar_ik,
    forward_kinematics,
)


x_target = 0.0
z_target = -0.30


q_thigh, q_calf = solve_planar_ik(
    x_target,
    z_target
)


print("Target foot position:")
print("x =", x_target)
print("z =", z_target)

print("\nJoint angles:")
print("thigh =", q_thigh)
print("calf  =", q_calf)


x_reconstructed, z_reconstructed = forward_kinematics(
    q_thigh,
    q_calf
)

print("\nReconstructed foot position:")
print("x =", x_reconstructed)
print("z =", z_reconstructed)