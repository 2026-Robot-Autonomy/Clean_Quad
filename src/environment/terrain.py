"""
Terrain geometry: ground, raised surfaces, ramp, stairs.

Every walkable piece is a box whose top face is the walking
surface and whose bottom sits at GROUND_BOTTOM, so heights
are exact and there are no gaps between neighbouring zones.
"""

import math

import mujoco


GROUND_BOTTOM = -0.10

COLOURS = {
    "road": [0.25, 0.25, 0.27, 1.0],
    "pavement": [0.62, 0.62, 0.60, 1.0],
    "plaza": [0.72, 0.68, 0.60, 1.0],
    "grass": [0.30, 0.55, 0.25, 1.0],
    "planter": [0.45, 0.35, 0.22, 1.0],
    "ramp": [0.80, 0.70, 0.35, 1.0],
    "stairs": [0.60, 0.50, 0.40, 1.0],
    "entrance": [0.55, 0.42, 0.32, 1.0],
    "underfloor": [0.15, 0.17, 0.20, 1.0],
}


# =========================================================
# CONTACT
# =========================================================
#
# Terrain uses MuJoCo's default contact, exactly like the
# Menagerie floor that Stage 1 was calibrated on:
#
# - foot spheres (priority=1) always contact with their own
#   parameters (friction 0.8, soft solimp, condim 6);
# - the lower calf capsules contact with the terrain's
#   parameters.
#
# Raising terrain priority would also change the calf
# contacts and measurably slows the calibrated gait, so
# surface friction only changes the non-foot contacts.
# =========================================================


def add_terrain_box(spec, name, rect, top, friction, rgba):

    hx, hy = rect.half_size
    cx, cy = rect.center

    return spec.worldbody.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[hx, hy, 0.5 * (top - GROUND_BOTTOM)],
        pos=[cx, cy, 0.5 * (top + GROUND_BOTTOM)],
        friction=[friction, 0.005, 0.0001],
        rgba=rgba,
    )


# =========================================================
# ELEMENTS
# =========================================================

def add_sky_and_light(spec, sun_dir=(-0.3, -0.2, -1.0), brightness=0.8):
    """
    Sky texture and one directional sun. Perception data varies
    sun_dir and brightness per scene.
    """

    spec.add_texture(
        name="sky",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.55, 0.70, 0.85],
        rgb2=[0.10, 0.12, 0.15],
        width=512,
        height=3072,
    )

    spec.worldbody.add_light(
        name="sun",
        pos=[4.0, 3.0, 6.0],
        dir=list(sun_dir),
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[brightness] * 3,
        castshadow=True,
    )


def add_underfloor(spec):
    """Large plane under everything, never walked on."""

    spec.worldbody.add_geom(
        name="terrain_underfloor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.0, 0.0, 0.05],
        pos=[0.0, 0.0, GROUND_BOTTOM],
        rgba=COLOURS["underfloor"],
    )


def add_surface(spec, surface):

    add_terrain_box(
        spec,
        name=f"terrain_{surface.kind}",
        rect=surface.rect,
        top=surface.height,
        friction=surface.friction,
        rgba=COLOURS.get(surface.kind, COLOURS["pavement"]),
    )


def add_ramp(spec, ramp, thickness=0.05):
    """
    Tilted box whose top face runs exactly from
    (y_bottom, z_bottom) to (y_top, z_top).

    """

    theta = math.radians(ramp.slope_deg)
    slope_length = ramp.length / math.cos(theta)

    # Midpoint of the top face, then move inward along the
    # face normal n = (0, -sin, cos) by the half thickness.
    mid_y = ramp.y_bottom + 0.5 * ramp.length
    mid_z = 0.5 * (ramp.z_bottom + ramp.z_top)

    spec.worldbody.add_geom(
        name=f"terrain_{ramp.kind}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.5 * (ramp.x1 - ramp.x0), 0.5 * slope_length, thickness],
        pos=[
            0.5 * (ramp.x0 + ramp.x1),
            mid_y + thickness * math.sin(theta),
            mid_z - thickness * math.cos(theta),
        ],
        quat=[math.cos(theta / 2.0), math.sin(theta / 2.0), 0.0, 0.0],
        friction=[ramp.friction, 0.005, 0.0001],
        rgba=COLOURS["ramp"],
    )


def add_stairs(spec, stairs):

    for k, (rect, top) in enumerate(stairs.steps()):

        add_terrain_box(
            spec,
            name=f"terrain_{stairs.kind}_{k}",
            rect=rect,
            top=top,
            friction=stairs.friction,
            rgba=COLOURS["stairs"],
        )


def add_terrain(spec, layout, sun_dir=(-0.3, -0.2, -1.0), brightness=0.8):

    add_sky_and_light(spec, sun_dir, brightness)
    add_underfloor(spec)

    for surface in layout.surfaces:
        add_surface(spec, surface)

    add_ramp(spec, layout.ramp)
    add_stairs(spec, layout.stairs)
