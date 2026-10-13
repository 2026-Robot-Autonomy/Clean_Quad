"""Stage 7 cleaning tools: a suction hose under the chest and a back hopper.

Both are visual-only geoms on the trunk (contype/conaffinity 0). The trunk
has an explicit <inertial>, so they add no mass and the Stage 1-6 gait and
footprint measurements remain valid.

The geoms are in render group 3, which mujoco.Renderer hides by default, so
the head-camera RGB-D images used by Stage 4 perception are unchanged (no tool
pixels and no tool shadows). Viewers show them with show_tools(viewer).
"""

import numpy as np
import mujoco

NOZZLE_BASE = (0.15, 0.0, -0.05)        # trunk frame [m], behind the camera
NOZZLE_TIP = (0.24, 0.0, -0.12)         # trunk frame [m]
NOZZLE_RADIUS = 0.012
HOPPER_POS = (-0.07, 0.0, 0.095)        # on the back, behind the camera
HOPPER_SIZE = (0.075, 0.06, 0.04)       # box half-sizes [m]
TOOL_RGBA = (0.15, 0.15, 0.17, 1.0)
HOPPER_RGBA = (0.10, 0.55, 0.30, 1.0)
TOOL_GROUP = 3


def add_cleaning_tools(spec):
    """Add the visual nozzle and hopper to the robot trunk."""

    trunk = spec.body("trunk")
    trunk.add_geom(
        name="tool_nozzle_hose",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=list(NOZZLE_BASE) + list(NOZZLE_TIP),
        size=[NOZZLE_RADIUS, 0.0, 0.0],
        rgba=list(TOOL_RGBA),
        contype=0,
        conaffinity=0,
        mass=0.0,
        group=TOOL_GROUP,
    )
    trunk.add_geom(
        name="tool_nozzle_head",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=list(NOZZLE_TIP),
        size=[0.03, 0.006, 0.0],
        rgba=list(TOOL_RGBA),
        contype=0,
        conaffinity=0,
        mass=0.0,
        group=TOOL_GROUP,
    )
    trunk.add_geom(
        name="tool_hopper",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=list(HOPPER_POS),
        size=list(HOPPER_SIZE),
        rgba=list(HOPPER_RGBA),
        contype=0,
        conaffinity=0,
        mass=0.0,
        group=TOOL_GROUP,
    )
    return spec


def nozzle_tip(model, data):
    """World position of the nozzle tip (valid after mj_forward/mj_step)."""

    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    matrix = data.xmat[trunk].reshape(3, 3)
    return data.xpos[trunk] + matrix @ np.asarray(NOZZLE_TIP)


def show_tools(viewer):
    """Make the tool render group visible in a passive viewer."""

    with viewer.lock():
        viewer.opt.geomgroup[TOOL_GROUP] = 1
