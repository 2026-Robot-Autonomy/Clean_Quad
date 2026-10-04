from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from src.controllers.trot_controller import TrotController


MODEL_PATH = (
    PROJECT_ROOT
    / "assets"
    / "mujoco_menagerie"
    / "unitree_a1"
    / "scene.xml"
)


def main():

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )

    data = mujoco.MjData(model)


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


    controller = TrotController(
        gait_frequency=1.0,
        step_length=0.00,
        step_height=0.03,
        nominal_height=-0.30,
    )


    simulation_time = 0.0


    with mujoco.viewer.launch_passive(
        model,
        data
    ) as viewer:

        while viewer.is_running():

            start = time.time()


            command = controller.get_command(
                simulation_time
            )


            data.ctrl[:] = command


            mujoco.mj_step(
                model,
                data
            )


            simulation_time += model.opt.timestep


            viewer.sync()


            elapsed = time.time() - start

            remaining = (
                model.opt.timestep
                - elapsed
            )

            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()