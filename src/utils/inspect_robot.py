from pathlib import Path

import mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "assets"
    / "mujoco_menagerie"
    / "unitree_a1"
    / "scene.xml"
)


model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))


print("\n============================")
print("ROBOT INFORMATION")
print("============================")

print("nq =", model.nq)
print("nv =", model.nv)
print("nu =", model.nu)


print("\n============================")
print("JOINTS")
print("============================")

for i in range(model.njnt):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        i
    )

    print(i, name)


print("\n============================")
print("ACTUATORS")
print("============================")

for i in range(model.nu):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        i
    )

    print(i, name)

# body position and orientation
data=mujoco.MjData(model)
print(data.qpos[0:3])