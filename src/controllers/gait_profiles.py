"""
Gait profiles: gait parameters + the calibration measured for
them. A calibration is only valid for the gait it was measured
on, so they are kept together.

V1_SLOW   the original Stage-1 gait (1 Hz, 3 cm foot lift),
          certified by experiments/15_final_locomotion_test.py.
          Kept unchanged for reproducibility.

V2_FAST   4 Hz, 7 cm foot lift. Calibrated by
          experiments/20_fast_gait_calibration.py and certified
          by experiments/21_fast_locomotion_test.py.
"""

from dataclasses import dataclass, asdict

import numpy as np

from src.controllers import velocity_calibration as v1_calibration


@dataclass(frozen=True)
class GaitProfile:

    name: str

    # Gait
    gait_frequency: float
    step_height: float
    nominal_height: float
    ramp_time: float

    # Heading feedback and turning
    k_yaw: float
    max_yaw_correction: float
    left_turn_gain: float
    right_turn_gain: float
    max_turn_correction: float

    # Limits
    max_step_length: float
    velocity_deadband: float
    min_moving_speed: float

    # Calibration: measured velocity [m/s] (increasing) -> stride
    calibration_velocities: tuple
    calibration_steps: tuple

    # Stride for rotation in place
    in_place_step_length: float

    # Validated envelope (LocomotionInterface clamps to it)
    max_forward_speed: float
    max_backward_speed: float
    max_yaw_rate: float

    def velocity_to_step_length(self, target_velocity):

        v = np.clip(
            target_velocity,
            self.calibration_velocities[0],
            self.calibration_velocities[-1],
        )

        return float(np.interp(
            v,
            self.calibration_velocities,
            self.calibration_steps,
        ))

    def controller_kwargs(self):
        """Keyword arguments for TrotController."""

        kwargs = asdict(self)

        for key in (
            "name",
            "calibration_velocities",
            "calibration_steps",
            "max_forward_speed",
            "max_backward_speed",
            "max_yaw_rate",
        ):
            kwargs.pop(key)

        kwargs["velocity_to_step_length"] = self.velocity_to_step_length

        return kwargs


# =========================================================
# V1: ORIGINAL STAGE-1 GAIT
# =========================================================

V1_SLOW = GaitProfile(
    name="v1_slow",
    gait_frequency=1.0,
    step_height=0.03,
    nominal_height=-0.30,
    ramp_time=2.0,
    k_yaw=0.03,
    max_yaw_correction=0.012,
    left_turn_gain=0.050,
    right_turn_gain=0.080,
    max_turn_correction=0.015,
    max_step_length=0.08,
    velocity_deadband=0.003,
    min_moving_speed=0.02,
    calibration_velocities=tuple(v1_calibration.MEASURED_VELOCITIES),
    calibration_steps=tuple(v1_calibration.STEP_LENGTHS),
    in_place_step_length=0.0,
    max_forward_speed=0.03,
    max_backward_speed=0.03,
    max_yaw_rate=0.05,
)


# =========================================================
# V2: FAST GAIT
# =========================================================
#
# Speed is almost linear in stride:
#     vx ~= -6.8 * stride - 0.19   [m/s]
# Stride 0 walks BACKWARD at ~0.19 m/s; stepping in place is
# at stride ~ -0.028.
# =========================================================

V2_FAST = GaitProfile(
    name="v2_fast",
    gait_frequency=4.0,
    step_height=0.07,
    nominal_height=-0.30,
    ramp_time=1.0,
    # Stride is ~4x more effective than in V1, and the open-loop
    # yaw bias (+0.027 rad/s) needs a stiffer heading loop.
    k_yaw=0.10,
    max_yaw_correction=0.02,
    left_turn_gain=0.1039,
    right_turn_gain=0.1021,
    max_turn_correction=0.06,
    max_step_length=0.20,
    velocity_deadband=0.01,
    min_moving_speed=0.05,
    calibration_velocities=(
        -0.6119, -0.5385, -0.4662, -0.3953, -0.3255, -0.2570, -0.1895,
        -0.1228, -0.0558, 0.0113, 0.0788, 0.1503, 0.2242, 0.2985,
        0.3734, 0.4472, 0.5105, 0.5763, 0.6352, 0.7021, 0.7716,
    ),
    calibration_steps=(
        0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.00,
        -0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07,
        -0.08, -0.09, -0.10, -0.11, -0.12, -0.13, -0.14,
    ),
    in_place_step_length=-0.0283,
    max_forward_speed=0.5,
    max_backward_speed=0.3,
    max_yaw_rate=0.5,
)


DEFAULT_PROFILE = V2_FAST
