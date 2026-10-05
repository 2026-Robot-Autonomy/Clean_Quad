from pathlib import Path

import mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "assets"
    / "mujoco_menagerie"
    / "unitree_a1"
    / "scene.xml"
)


model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(model)

mujoco.mj_forward(
    model,
    data
)


print("\n=== BODIES ===")

for i in range(model.nbody):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        i
    )

    if name is None:
        continue

    if any(
        prefix in name
        for prefix in [
            "FR",
            "FL",
            "RR",
            "RL"
        ]
    ):

        pos = data.xpos[i]

        print(
            f"{name:20s} "
            f"x={pos[0]:.3f} "
            f"y={pos[1]:.3f} "
            f"z={pos[2]:.3f}"
        )