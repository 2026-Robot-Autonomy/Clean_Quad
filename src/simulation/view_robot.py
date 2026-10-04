from pathlib import Path
import time

import mujoco
import mujoco.viewer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "assets"
    / "mujoco_menagerie"
    / "unitree_a1"
    / "scene.xml"
)


def main():

    print("Loading model from:")
    print(MODEL_PATH)
    # loads things that don't change during simulation
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    # contains the current state
    data = mujoco.MjData(model)

    print("Model loaded successfully!")

    print("Number of generalized coordinates:", model.nq)
    print("Number of velocities:", model.nv)
    print("Number of actuators:", model.nu)

    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running():

            step_start = time.time()
            # advances physics by one timestep
            mujoco.mj_step(model, data)

            viewer.sync()

            time_until_next_step = (
                model.opt.timestep - (time.time() - step_start)
            )

            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()