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


model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)


names = [
    "FR_hip",
    "FL_hip",
    "RR_hip",
    "RL_hip",
]


for name in names:

    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        name
    )

    if body_id == -1:
        print(f"{name}: not found")
        continue

    pos = data.xpos[body_id]

    print(
        f"{name}: "
        f"x={pos[0]:.3f}, "
        f"y={pos[1]:.3f}, "
        f"z={pos[2]:.3f}"
    )