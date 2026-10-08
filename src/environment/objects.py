"""
Static objects: boundary walls, obstacles, bins, base pad.
"""

import mujoco

from src.environment.layout import Circle
from src.environment.terrain import GROUND_BOTTOM


COLOURS = {
    "wall": [0.75, 0.72, 0.68, 1.0],
    "tree": [0.40, 0.26, 0.13, 1.0],
    "canopy": [0.20, 0.45, 0.20, 0.9],
    "bench": [0.50, 0.33, 0.18, 1.0],
    "pole": [0.30, 0.30, 0.32, 1.0],
    "bin": [0.10, 0.40, 0.65, 1.0],
    "base": [0.95, 0.55, 0.10, 0.6],
}


def add_walls(spec, layout):

    t = layout.wall_thickness
    top = layout.wall_height
    hz = 0.5 * (top - GROUND_BOTTOM)
    cz = 0.5 * (top + GROUND_BOTTOM)
    w, d = layout.width, layout.depth

    walls = {
        "south": ([0.5 * w, -0.5 * t], [0.5 * w + t, 0.5 * t]),
        "north": ([0.5 * w, d + 0.5 * t], [0.5 * w + t, 0.5 * t]),
        "west": ([-0.5 * t, 0.5 * d], [0.5 * t, 0.5 * d]),
        "east": ([w + 0.5 * t, 0.5 * d], [0.5 * t, 0.5 * d]),
    }

    for side, (center, half) in walls.items():

        spec.worldbody.add_geom(
            name=f"wall_{side}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[half[0], half[1], hz],
            pos=[center[0], center[1], cz],
            rgba=COLOURS["wall"],
        )


def add_obstacle(spec, layout, obstacle):

    cx, cy = obstacle.shape.center
    ground = layout.ground_height_at(cx, cy)
    rgba = COLOURS[obstacle.kind]

    if isinstance(obstacle.shape, Circle):

        spec.worldbody.add_geom(
            name=obstacle.name,
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[obstacle.shape.r, 0.5 * obstacle.height, 0.0],
            pos=[cx, cy, ground + 0.5 * obstacle.height],
            rgba=rgba,
        )

    else:

        hx, hy = obstacle.shape.half_size

        spec.worldbody.add_geom(
            name=obstacle.name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[hx, hy, 0.5 * obstacle.height],
            pos=[cx, cy, ground + 0.5 * obstacle.height],
            rgba=rgba,
        )

    if obstacle.kind == "tree":

        # Visual only: the trunk is the obstacle.
        spec.worldbody.add_geom(
            name=f"{obstacle.name}_canopy",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.6, 0.0, 0.0],
            pos=[cx, cy, ground + obstacle.height + 0.4],
            contype=0,
            conaffinity=0,
            rgba=COLOURS["canopy"],
        )


def add_base_pad(spec, layout, size=0.30):
    """Visual-only marker where the robot starts and returns."""

    x, y, _ = layout.base_pose

    spec.worldbody.add_geom(
        name="base_pad",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[size, size, 0.001],
        pos=[x, y, layout.ground_height_at(x, y) + 0.001],
        contype=0,
        conaffinity=0,
        rgba=COLOURS["base"],
    )


def add_objects(spec, layout):

    add_walls(spec, layout)

    for obstacle in layout.obstacles:
        add_obstacle(spec, layout, obstacle)

    add_base_pad(spec, layout)
