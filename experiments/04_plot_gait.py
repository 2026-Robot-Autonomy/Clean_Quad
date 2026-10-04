import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

from src.controllers.gait_generator import foot_trajectory


phases = np.linspace(0, 1, 200)

xs = []
zs = []

for phase in phases:
    x, z = foot_trajectory(
        phase,
        step_length=0.10,
        step_height=0.05,
        nominal_height=-0.30,
    )

    xs.append(x)
    zs.append(z)


plt.figure()

plt.plot(xs, zs)

plt.xlabel("Forward position x (m)")
plt.ylabel("Vertical position z (m)")
plt.title("Foot Trajectory")
plt.grid()

plt.savefig(
    "results/foot_trajectory.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("Saved: results/foot_trajectory.png")