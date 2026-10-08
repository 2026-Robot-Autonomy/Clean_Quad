import numpy as np

from src.controllers.gait_profiles import DEFAULT_PROFILE
from src.controllers.trot_controller import TrotController
from src.utils.orientation import quaternion_to_euler


def wrap_angle(angle):

    return (
        (angle + np.pi)
        % (2.0 * np.pi)
        - np.pi
    )


class LocomotionInterface:
    """
    Stage-1 locomotion as a reusable (vx, wz) interface.

    Wraps the frozen TrotController and adds the parts that
    previously lived only in 15_final_locomotion_test.py:

    - heading reference tracking and yaw error
    - a gait clock that restarts every time the robot
      leaves standing, so the smooth stand -> gait blend
      runs after every stop, not only at t = 0

    Usage, once per physics step:

        loco.set_velocity(vx, wz)        # any time
        data.ctrl[:] = loco.update(data, model.opt.timestep)
        mujoco.mj_step(model, data)

    The gait comes from a GaitProfile (gait_profiles.py).
    Commands are clamped to the profile's validated envelope:

        V2_FAST (default): vx in [-0.3, +0.5] m/s, |wz| <= 0.5 rad/s
        V1_SLOW:           vx in [-0.03, +0.03] m/s, |wz| <= 0.05 rad/s
    """

    def __init__(
        self,
        profile=DEFAULT_PROFILE,
    ):

        self.profile = profile

        self.controller = TrotController(
            **profile.controller_kwargs()
        )

        # Time since the robot last left standing.
        self.motion_time = 0.0

        # Heading reference:
        #
        # desired_yaw =
        #     anchor_yaw + wz * (motion_time - anchor_time)
        #
        # The anchor is the robot's actual yaw at the moment
        # a command starts, so every new command holds or
        # turns from the heading the robot really has. For a
        # constant command this is exactly the Stage-1
        # formula psi_0 + wz * t.

        self.anchor_yaw = 0.0
        self.anchor_time = 0.0
        self.reanchor = False

        self.moving = False

        self.desired_yaw = 0.0
        self.yaw_error = 0.0


    # =====================================================
    # COMMANDS
    # =====================================================

    def set_velocity(
        self,
        vx,
        wz=0.0,
    ):

        old_command = (
            self.controller.vx_command,
            self.controller.wz_command,
        )

        vx = float(np.clip(
            vx,
            -self.profile.max_backward_speed,
            self.profile.max_forward_speed,
        ))

        wz = float(np.clip(
            wz,
            -self.profile.max_yaw_rate,
            self.profile.max_yaw_rate,
        ))

        self.controller.set_velocity(
            vx,
            wz,
        )

        new_command = (
            self.controller.vx_command,
            self.controller.wz_command,
        )

        if self.moving and new_command != old_command:

            # Re-anchored on the next update(), where the
            # actual yaw is available.
            self.reanchor = True


    def stop(self):

        self.set_velocity(0.0, 0.0)


    def is_standing_command(self):
        """
        Same rule TrotController uses to enter stand mode.
        """

        return (
            abs(self.controller.vx_command)
            < self.controller.velocity_deadband

            and

            abs(self.controller.wz_command)
            < 1e-4
        )


    # =====================================================
    # PER-STEP UPDATE
    # =====================================================

    def update(
        self,
        data,
        dt,
    ):
        """
        Return the 12 joint targets for the current state,
        then advance the internal clock by dt.
        """

        _, _, yaw = quaternion_to_euler(
            data.qpos[3:7]
        )


        # -------------------------------------------------
        # STAND
        # -------------------------------------------------

        if self.is_standing_command():

            self.moving = False
            self.yaw_error = 0.0

            return self.controller.get_stand_command()


        # -------------------------------------------------
        # STAND -> MOVING: restart blend, hold this heading
        # -------------------------------------------------

        if not self.moving:

            self.moving = True
            self.motion_time = 0.0
            self.anchor_yaw = yaw
            self.anchor_time = 0.0
            self.reanchor = False


        # -------------------------------------------------
        # NEW COMMAND WHILE MOVING: no blend restart,
        # heading reference restarts from actual yaw
        # -------------------------------------------------

        if self.reanchor:

            self.anchor_yaw = yaw
            self.anchor_time = self.motion_time
            self.reanchor = False


        self.desired_yaw = (
            self.anchor_yaw
            +
            self.controller.wz_command
            * (self.motion_time - self.anchor_time)
        )

        self.yaw_error = wrap_angle(
            self.desired_yaw
            - yaw
        )

        command = self.controller.get_command(
            self.motion_time,
            yaw_error=self.yaw_error,
        )

        self.motion_time += dt

        return command
