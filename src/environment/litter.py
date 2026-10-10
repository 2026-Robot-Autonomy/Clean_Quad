"""
Small objects lying on the ground: litter (the targets) and,
via distractors.py, look-alikes that are not litter.

Every object gets a seeded random kind, colour, size, pose
and (for cylinders) lying/standing state, so perception sees
real variety.

Litter items are mocap bodies with non-colliding geoms:
visible to cameras, cannot be kicked by the robot, and
"collected" by moving them out of sight.
"""

from dataclasses import dataclass
import math

import numpy as np
import mujoco


# =========================================================
# KINDS
# =========================================================

@dataclass(frozen=True)
class ObjectKind:
    geom: int                   # mujoco.mjtGeom
    size: tuple                 # MuJoCo geom size at scale 1
    colours: tuple              # rgba choices
    lying_probability: float    # chance a cylinder/capsule lies on its side

    def rest_height(self, scale, lying):
        """Centre height above the ground."""

        if self.geom in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
            if lying:
                return self.size[0] * scale
            extra = self.size[0] if self.geom == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0
            return (self.size[1] + extra) * scale

        if self.geom == mujoco.mjtGeom.mjGEOM_SPHERE:
            return self.size[0] * scale

        return self.size[2] * scale     # box, ellipsoid


CYL = mujoco.mjtGeom.mjGEOM_CYLINDER
BOX = mujoco.mjtGeom.mjGEOM_BOX

LITTER_KINDS = {
    "bottle": ObjectKind(
        CYL, (0.035, 0.11, 0.0),
        (
            (0.55, 0.85, 0.60, 0.75),   # green PET
            (0.85, 0.90, 0.95, 0.60),   # clear
            (0.45, 0.25, 0.10, 0.90),   # brown glass
            (0.30, 0.50, 0.90, 0.80),   # blue
        ),
        lying_probability=0.8,
    ),
    "can": ObjectKind(
        CYL, (0.033, 0.06, 0.0),
        (
            (0.85, 0.15, 0.15, 1.0),    # red
            (0.15, 0.30, 0.80, 1.0),    # blue
            (0.75, 0.75, 0.78, 1.0),    # silver
            (0.20, 0.60, 0.25, 1.0),    # green
        ),
        lying_probability=0.5,
    ),
    "cup": ObjectKind(
        CYL, (0.045, 0.05, 0.0),
        (
            (0.95, 0.95, 0.92, 1.0),    # white
            (0.70, 0.55, 0.35, 1.0),    # paper brown
            (0.80, 0.15, 0.15, 1.0),    # red
        ),
        lying_probability=0.3,
    ),
    "wrapper": ObjectKind(
        BOX, (0.07, 0.045, 0.004),
        (
            (0.95, 0.80, 0.10, 1.0),    # yellow
            (0.95, 0.50, 0.10, 1.0),    # orange
            (0.20, 0.40, 0.85, 1.0),    # blue
            (0.95, 0.95, 0.95, 1.0),    # white
            (0.80, 0.80, 0.85, 1.0),    # foil
        ),
        lying_probability=0.0,
    ),
}

# Class index used by the detectors (YOLO label order).
LITTER_CLASSES = tuple(LITTER_KINDS)

# Zones where small objects may appear.
GROUND_OBJECT_ZONES = ("road", "pavement", "grass", "plaza")

# Where collected items are parked (below the underfloor).
HIDDEN_POSITION = (0.0, 0.0, -5.0)

SCALE_RANGE = (0.8, 1.2)


# =========================================================
# OBJECTS
# =========================================================

@dataclass
class GroundObject:
    name: str               # "litter_<kind>_<nn>" or "distractor_<kind>_<nn>"
    category: str           # "litter" | "distractor"
    kind: str
    position: np.ndarray    # resting centre (x, y, z)
    yaw: float
    scale: float
    lying: bool
    rgba: tuple
    collected: bool = False

    @property
    def geom_name(self):
        return f"{self.name}_geom"

    @property
    def quat(self):

        q_yaw = np.array([math.cos(self.yaw / 2), 0.0, 0.0, math.sin(self.yaw / 2)])

        if not self.lying:
            return q_yaw

        # Lay the long axis flat (90 deg about x), then yaw.
        q_side = np.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0])
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, q_yaw, q_side)
        return q


# Litter keeps its old name for the rest of the code base.
LitterItem = GroundObject


# =========================================================
# PLACEMENT
# =========================================================

def is_clear(layout, x, y, clearance):
    """Point and its surrounding ring all lie in object zones."""

    for k in range(9):
        if k == 0:
            px, py = x, y
        else:
            a = 2.0 * math.pi * (k - 1) / 8.0
            px, py = x + clearance * math.cos(a), y + clearance * math.sin(a)

        if layout.zone_at(px, py) not in GROUND_OBJECT_ZONES:
            return False

    return True


def place_objects(
    layout,
    rng,
    kinds,
    count,
    category,
    existing=(),
    clearance=0.30,
    spacing=0.40,
    base_clearance=0.80,
    max_tries=50000,
):
    """
    Random, reproducible placement of `count` objects.

    Kinds are used in turn so classes stay balanced; colour,
    size, yaw and lying state are random.

    clearance:      free distance to obstacles, walls, curb,
                    ramp and raised features
    spacing:        minimum distance to any other object
    base_clearance: keep the robot's start area clean
    """

    names = list(kinds)
    bx, by, _ = layout.base_pose
    placed = list(existing)
    objects = []

    for _ in range(max_tries):

        if len(objects) == count:
            break

        x = rng.uniform(0.0, layout.width)
        y = rng.uniform(0.0, layout.depth)

        if not is_clear(layout, x, y, clearance):
            continue

        if math.hypot(x - bx, y - by) < base_clearance:
            continue

        if any(math.hypot(x - o.position[0], y - o.position[1]) < spacing for o in placed):
            continue

        kind_name = names[len(objects) % len(names)]
        kind = kinds[kind_name]

        scale = rng.uniform(*SCALE_RANGE)
        lying = bool(rng.random() < kind.lying_probability)
        rgba = kind.colours[rng.integers(len(kind.colours))]
        z = layout.ground_height_at(x, y) + kind.rest_height(scale, lying)

        obj = GroundObject(
            name=f"{category}_{kind_name}_{len(objects):02d}",
            category=category,
            kind=kind_name,
            position=np.array([x, y, z]),
            yaw=rng.uniform(-math.pi, math.pi),
            scale=scale,
            lying=lying,
            rgba=tuple(rgba),
        )

        objects.append(obj)
        placed.append(obj)

    if len(objects) < count:
        raise RuntimeError(
            f"placed only {len(objects)}/{count} {category} objects; "
            "relax clearance/spacing or lower the count"
        )

    return objects


def place_litter(layout, rng, count=8, existing=()):
    return place_objects(layout, rng, LITTER_KINDS, count, "litter", existing)


# =========================================================
# MUJOCO
# =========================================================

def scaled_size(kind, scale):
    return [s * scale for s in kind.size]


def add_litter(spec, items):
    """Litter: mocap bodies, so they can be collected later."""

    for item in items:

        kind = LITTER_KINDS[item.kind]

        body = spec.worldbody.add_body(
            name=item.name,
            mocap=True,
            pos=list(item.position),
            quat=list(item.quat),
        )

        body.add_geom(
            name=item.geom_name,
            type=kind.geom,
            size=scaled_size(kind, item.scale),
            contype=0,
            conaffinity=0,
            rgba=list(item.rgba),
        )
