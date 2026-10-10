import numpy as np
import mujoco

from src.config import A1_XML_PATH


# =========================================================
# MODEL LOADING
# =========================================================
#
# a1.xml sets meshdir="assets", which MuJoCo resolves
# relative to the *main* model file. A world XML stored
# elsewhere that <include>s a1.xml therefore cannot find
# the meshes.
#
# Loading a1.xml through MjSpec keeps its own directory as
# the model directory, so worlds can be added in Python.
# =========================================================


def load_a1_spec():
    """
    Return an editable MjSpec containing only the A1 robot.
    """

    return mujoco.MjSpec.from_file(
        str(A1_XML_PATH)
    )


def add_flat_ground(spec):
    """
    Add the same floor and light as the Menagerie scene.xml.

    Physics matches scene.xml; the checker texture is
    replaced by a plain colour.
    """

    spec.worldbody.add_light(
        pos=[0.0, 0.0, 1.5],
        dir=[0.0, 0.0, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    )

    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.0, 0.0, 0.05],
        rgba=[0.2, 0.3, 0.4, 1.0],
    )

    return spec


HEAD_CAMERA = "head_camera"

HEAD_CAMERA_POS = (0.16, 0.0, 0.03)     # trunk frame [m], at the front face
HEAD_CAMERA_TILT_DEG = 25.0             # pitched down from horizontal
HEAD_CAMERA_FOVY_DEG = 70.0             # vertical field of view


def add_head_camera(spec):
    """
    Forward-looking RGB-D camera on the trunk.

    A MuJoCo camera looks along its -z axis with +y up in the
    image. Image right = trunk -y; the view direction is trunk
    +x pitched down by HEAD_CAMERA_TILT_DEG.
    """

    tilt = np.radians(HEAD_CAMERA_TILT_DEG)

    spec.body("trunk").add_camera(
        name=HEAD_CAMERA,
        pos=list(HEAD_CAMERA_POS),
        xyaxes=[0.0, -1.0, 0.0, np.sin(tilt), 0.0, np.cos(tilt)],
        fovy=HEAD_CAMERA_FOVY_DEG,
    )

    return spec


def build_flat_scene():
    """
    Compile A1 + flat ground from Python.

    Returns:
        model, spec
    """

    spec = add_flat_ground(
        load_a1_spec()
    )

    return spec.compile(), spec


# =========================================================
# RESET / SPAWN
# =========================================================


def yaw_to_quaternion(yaw):
    """
    MuJoCo quaternion [w, x, y, z] for a pure yaw rotation.
    """

    return np.array([
        np.cos(yaw / 2.0),
        0.0,
        0.0,
        np.sin(yaw / 2.0),
    ])


def reset_robot(
    model,
    data,
    x=0.0,
    y=0.0,
    yaw=0.0,
    ground_z=0.0,
    keyframe="home",
):
    """
    Reset to the standing keyframe at an arbitrary pose.

    x, y:
        world position of the trunk [m]

    yaw:
        heading [rad]

    ground_z:
        height of the surface under the robot [m]
        (e.g. raised pavement). The keyframe trunk height
        is added on top of it.

    With default arguments this is identical to the
    Stage-1 keyframe reset.
    """

    key_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        keyframe,
    )

    if key_id < 0:
        raise ValueError(
            f"keyframe '{keyframe}' not found"
        )

    mujoco.mj_resetDataKeyframe(
        model,
        data,
        key_id,
    )

    data.qpos[0] = x
    data.qpos[1] = y
    data.qpos[2] = model.key_qpos[key_id, 2] + ground_z
    data.qpos[3:7] = yaw_to_quaternion(yaw)

    mujoco.mj_forward(
        model,
        data,
    )
