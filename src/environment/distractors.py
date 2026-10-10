"""
Distractors: small things on the ground that are NOT litter.

They are chosen to resemble litter colours and shapes (green
leaf vs green bottle, grey stone vs silver can, brown stick vs
brown bottle, yellow leaf vs yellow wrapper) so a detector is
tested on false alarms, not only on finding litter.

Static, non-colliding geoms: they never move or get collected.
"""

import mujoco

from src.environment.litter import ObjectKind, place_objects, scaled_size


DISTRACTOR_KINDS = {
    "leaf": ObjectKind(
        mujoco.mjtGeom.mjGEOM_ELLIPSOID, (0.05, 0.03, 0.003),
        (
            (0.30, 0.55, 0.20, 1.0),    # green
            (0.85, 0.75, 0.20, 1.0),    # yellow
            (0.50, 0.30, 0.12, 1.0),    # brown
            (0.85, 0.45, 0.15, 1.0),    # orange
        ),
        lying_probability=0.0,
    ),
    "stone": ObjectKind(
        mujoco.mjtGeom.mjGEOM_ELLIPSOID, (0.04, 0.03, 0.025),
        (
            (0.50, 0.50, 0.50, 1.0),
            (0.65, 0.63, 0.60, 1.0),
            (0.40, 0.38, 0.35, 1.0),
        ),
        lying_probability=0.0,
    ),
    "stick": ObjectKind(
        mujoco.mjtGeom.mjGEOM_CAPSULE, (0.008, 0.10, 0.0),
        (
            (0.45, 0.30, 0.15, 1.0),
            (0.35, 0.22, 0.10, 1.0),
        ),
        lying_probability=1.0,
    ),
}


def place_distractors(layout, rng, count=10, existing=()):
    return place_objects(layout, rng, DISTRACTOR_KINDS, count, "distractor", existing)


def add_distractors(spec, distractors):

    for obj in distractors:

        kind = DISTRACTOR_KINDS[obj.kind]

        spec.worldbody.add_geom(
            name=obj.geom_name,
            type=kind.geom,
            size=scaled_size(kind, obj.scale),
            pos=list(obj.position),
            quat=list(obj.quat),
            contype=0,
            conaffinity=0,
            rgba=list(obj.rgba),
        )
