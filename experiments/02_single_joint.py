from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))

from src.controllers.stand_controller import get_stand_command


MODEL_PATH = (
    PROJECT_ROOT
    / "assets"
    / "mujoco_menagerie"
    / "unitree_a1"
    / "scene.xml"
)


def main():

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # Load predefined home pose
    home_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home"
    )

    mujoco.mj_resetDataKeyframe(
        model,
        data,
        home_id
    )

    stand_command = get_stand_command()

    print("Standing command:")
    print(stand_command)

    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running():

            start = time.time()
            # desired joint position command 
            '''
            changes actuator position to 0.5
            
            '''
            stand_command[1] = 0.5

            mujoco.mj_step(model, data)

            viewer.sync()

            elapsed = time.time() - start

            remaining = model.opt.timestep - elapsed

            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()