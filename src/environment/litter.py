"""
Litter: kinds, seeded random placement, MuJoCo bodies.

Each item is a mocap body with a non-colliding geom:
visible to cameras, cannot be kicked by the robot, and
"collected" by moving it out of sight.
"""

from dataclasses import dataclass
import math

import numpy as np
import mujoco


@dataclass(frozen=True)
class LitterKind:
    geom: int               # mujoco.mjtGeom
    size: tuple             # MuJoCo geom size
    rest_height: float      # centre height above ground when lying
    lying: bool             # cylinder rotated onto its side
    rgba: tuple


LITTER_KINDS = {
    "bottle": LitterKind(
        mujoco.mjtGeom.mjGEOM_CYLINDER, (0.035, 0.11, 0.0), 0.035, True,
        (0.55, 0.85, 0.60, 0.8),
    ),
    "can": LitterKind(
        mujoco.mjtGeom.mjGEOM_CYLINDER, (0.033, 0.06, 0.0), 0.06, False,
        (0.85, 0.15, 0.15, 1.0),
    ),
    "cup": LitterKind(
        mujoco.mjtGeom.mjGEOM_CYLINDER, (0.045, 0.05, 0.0), 0.05, False,
        (0.95, 0.95, 0.92, 1.0),
    ),
    "wrapper": LitterKind(
        mujoco.mjtGeom.mjGEOM_BOX, (0.07, 0.045, 0.004), 0.004, False,
        (0.95, 0.80, 0.10, 1.0),
    ),
}

# Zones where litter may appear.
LITTER_ZONES = ("road", "pavement", "grass", "plaza")

# Where collected items are parked (below the underfloor).
HIDDEN_POSITION = (0.0, 0.0, -5.0)


@dataclass
class LitterItem:
    name: str               # body name "litter_<kind>_<nn>"
    kind: str
    position: np.ndarray    # resting centre (x, y, z)
    yaw: float
    collected: bool = False

    @property
    def quat(self):
        return litter_quat(self.kind, self.yaw)


def litter_quat(kind, yaw):

    q_yaw = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])

    if not LITTER_KINDS[kind].lying:
        return q_yaw

    # Lay the cylinder on its side (90 deg about x), then yaw.
    q_side = np.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0])
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, q_yaw, q_side)
    return q


# =========================================================
# PLACEMENT
# =========================================================

def is_clear(layout, x, y, clearance):
    """Point and its surrounding ring all lie in litter zones."""

    for k in range(9):
        if k == 0:
            px, py = x, y
        else:
            a = 2.0 * math.pi * (k - 1) / 8.0
            px, py = x + clearance * math.cos(a), y + clearance * math.sin(a)

        if layout.zone_at(px, py) not in LITTER_ZONES:
            return False

    return True


def place_litter(
    layout,
    rng,
    count=8,
    clearance=0.35,
    spacing=0.60,
    base_clearance=0.80,
    max_tries=20000,
):
    """
    Random litter positions, reproducible through `rng`.

    clearance:      free distance to obstacles, walls, curb,
                    ramp and raised features
    spacing:        minimum distance between items
    base_clearance: keep the robot's start area clean
    """

    kinds = list(LITTER_KINDS)
    bx, by, _ = layout.base_pose
    items = []

    for _ in range(max_tries):

        if len(items) == count:
            break

        x = rng.uniform(0.0, layout.width)
        y = rng.uniform(0.0, layout.depth)

        if not is_clear(layout, x, y, clearance):
            continue

        if math.hypot(x - bx, y - by) < base_clearance:
            continue

        if any(math.hypot(x - it.position[0], y - it.position[1]) < spacing for it in items):
            continue

        kind = kinds[len(items) % len(kinds)]
        z = layout.ground_height_at(x, y) + LITTER_KINDS[kind].rest_height

        items.append(LitterItem(
            name=f"litter_{kind}_{len(items):02d}",
            kind=kind,
            position=np.array([x, y, z]),
            yaw=rng.uniform(-math.pi, math.pi),
        ))

    if len(items) < count:
        raise RuntimeError(
            f"placed only {len(items)}/{count} litter items; "
            "relax clearance/spacing"
        )

    return items


# =========================================================
# MUJOCO
# =========================================================

def add_litter(spec, items):

    for item in items:

        kind = LITTER_KINDS[item.kind]

        body = spec.worldbody.add_body(
            name=item.name,
            mocap=True,
            pos=list(item.position),
            quat=list(item.quat),
        )

        body.add_geom(
            name=f"{item.name}_geom",
            type=kind.geom,
            size=list(kind.size),
            contype=0,
            conaffinity=0,
            rgba=list(kind.rgba),
        )
