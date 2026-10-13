"""Whole-body postures for pickup and dumping, built on the Stage 1 leg IK.

A posture is a set of foot heights below the hips (front and rear pairs),
with the feet kept under the hips. Lowering the front pair pitches the trunk
nose-down, bringing the nozzle tip to the ground. Joint targets are blended
linearly from the current command, with the robot standing still.
"""

from dataclasses import dataclass
import math

import numpy as np

from src.controllers.leg_kinematics import solve_leg_ik
from src.utils.orientation import quaternion_to_euler

LEG_SIDES = (-1, 1, -1, 1)          # FR, FL, RR, RL (actuator order)


@dataclass(frozen=True)
class Posture:
    front_height: float
    rear_height: float

    def __post_init__(self):
        for value in (self.front_height, self.rear_height):
            if not math.isfinite(value) or not -0.32 <= value <= -0.15:
                raise ValueError("foot heights must lie in [-0.32, -0.15] m below the hips")


def posture_targets(posture, hip_offset):
    """Twelve joint targets for a posture."""

    targets = []
    for leg, side in enumerate(LEG_SIDES):
        z = posture.front_height if leg < 2 else posture.rear_height
        targets.extend(solve_leg_ik(x=0.0, y=side * hip_offset, z=z, side_sign=side))
    return np.array(targets, dtype=float)


class PostureFailure(RuntimeError):
    pass


def run_posture_sequence(campus, segments, *, step_hook=None, viewer=None,
                         min_trunk_height=0.15, max_tilt_deg=25.0, sample=None):
    """Blend through (targets, seconds) segments while standing still.

    Each segment blends from the previous command to its targets over its
    duration. Raises PostureFailure on low trunk height, excessive tilt, trunk
    contact or a closed viewer. `sample(campus)` is called every physics step
    and may record measurements (for example the nozzle tip height).
    """

    import mujoco
    model, data = campus.model, campus.data
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    current = data.ctrl.copy()
    next_sync = data.time
    stats = {"max_pitch_deg": 0.0, "max_roll_deg": 0.0, "min_trunk_height_m": math.inf}
    for targets, seconds in segments:
        steps = max(1, int(round(seconds / model.opt.timestep)))
        start = current
        for step in range(steps):
            blend = min(1.0, (step + 1) / steps)
            data.ctrl[:] = (1 - blend) * start + blend * targets
            mujoco.mj_step(model, data)
            if step_hook is not None:
                step_hook(campus)
            if sample is not None:
                sample(campus)
            roll, pitch, _ = quaternion_to_euler(data.qpos[3:7])
            height = data.qpos[2] - campus.layout.ground_height_at(data.qpos[0], data.qpos[1])
            stats["max_pitch_deg"] = max(stats["max_pitch_deg"], abs(math.degrees(pitch)))
            stats["max_roll_deg"] = max(stats["max_roll_deg"], abs(math.degrees(roll)))
            stats["min_trunk_height_m"] = min(stats["min_trunk_height_m"], float(height))
            if height < min_trunk_height or max(abs(roll), abs(pitch)) > math.radians(max_tilt_deg):
                raise PostureFailure("fall")
            for contact in data.contact[:data.ncon]:
                if trunk in (int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])):
                    raise PostureFailure("trunk_contact")
            if viewer is not None and data.time >= next_sync:
                if not viewer.is_running():
                    raise PostureFailure("viewer_closed")
                viewer.sync()
                next_sync += 0.1
        current = np.array(targets, dtype=float)
    return stats
