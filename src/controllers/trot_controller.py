import numpy as np

from src.controllers.gait_generator import foot_trajectory
from src.controllers.leg_kinematics import solve_leg_ik
from src.controllers.velocity_calibration import (
    velocity_to_step_length,
)


class TrotController:

    def __init__(
        self,
        gait_frequency=1.0,
        step_height=0.03,
        nominal_height=-0.30,
        ramp_time=2.0,

        # -------------------------------------------------
        # YAW FEEDBACK
        # -------------------------------------------------
        k_yaw=0.03,

        # -------------------------------------------------
        # ASYMMETRIC TURN FEED-FORWARD
        # -------------------------------------------------
        #
        # LEFT TURN:
        #
        # Previous gain = 0.062
        #
        # Test:
        #
        # target wz = +0.050
        # actual wz = +0.0618
        #
        # New gain:
        #
        # 0.062 * (0.050 / 0.0618)
        # ≈ 0.050
        #
        # RIGHT TURN:
        #
        # Current value already passes:
        #
        # target = -0.050
        # actual = -0.0459
        #
        # So do NOT modify right_turn_gain.
        #
        left_turn_gain=0.050,
        right_turn_gain=0.080,

        # Maximum stride difference from turning
        max_turn_correction=0.015,

        # Maximum closed-loop yaw correction
        max_yaw_correction=0.012,

        # -------------------------------------------------
        # STRIDE SAFETY
        # -------------------------------------------------
        max_step_length=0.08,

        # -------------------------------------------------
        # TRANSLATIONAL LIMITS
        # -------------------------------------------------
        velocity_deadband=0.003,
        min_moving_speed=0.02,
    ):

        # =================================================
        # GAIT PARAMETERS
        # =================================================

        self.gait_frequency = gait_frequency
        self.step_height = step_height
        self.nominal_height = nominal_height
        self.ramp_time = ramp_time


        # =================================================
        # YAW FEEDBACK
        # =================================================

        self.k_yaw = k_yaw

        self.max_yaw_correction = (
            max_yaw_correction
        )


        # =================================================
        # TURN FEED-FORWARD
        # =================================================

        self.left_turn_gain = (
            left_turn_gain
        )

        self.right_turn_gain = (
            right_turn_gain
        )

        self.max_turn_correction = (
            max_turn_correction
        )


        # =================================================
        # ROBOT GEOMETRY
        # =================================================

        self.hip_offset = 0.085


        # =================================================
        # LIMITS
        # =================================================

        self.max_step_length = (
            max_step_length
        )

        self.velocity_deadband = (
            velocity_deadband
        )

        self.min_moving_speed = (
            min_moving_speed
        )


        # =================================================
        # HIGH-LEVEL COMMANDS
        # =================================================

        self.vx_command = 0.0

        self.wz_command = 0.0


    # =====================================================
    # HIGH-LEVEL VELOCITY INTERFACE
    # =====================================================

    def set_velocity(
        self,
        vx,
        wz=0.0,
    ):
        """
        Set desired body velocity.

        vx:
            translational velocity [m/s]

            positive -> forward
            negative -> backward

        wz:
            yaw rate [rad/s]

            positive -> left
            negative -> right
        """

        vx = float(vx)
        wz = float(wz)


        # -------------------------------------------------
        # TRANSLATIONAL DEAD BAND
        # -------------------------------------------------

        if (
            abs(vx)
            < self.velocity_deadband
        ):

            vx = 0.0


        # -------------------------------------------------
        # MINIMUM RELIABLE MOVING SPEED
        # -------------------------------------------------

        elif (
            abs(vx)
            < self.min_moving_speed
        ):

            vx = (
                np.sign(vx)
                * self.min_moving_speed
            )


        self.vx_command = vx

        self.wz_command = wz


    # =====================================================
    # STANDING COMMAND
    # =====================================================

    def get_stand_command(self):
        """
        Return symmetric stable standing joint targets.

        Actuator order:
            FR
            FL
            RR
            RL
        """

        commands = []


        leg_sides = [
            -1,     # FR
             1,     # FL
            -1,     # RR
             1,     # RL
        ]


        for side_sign in leg_sides:

            x = 0.0

            y = (
                side_sign
                * self.hip_offset
            )

            z = (
                self.nominal_height
            )


            q_hip, q_thigh, q_calf = (
                solve_leg_ik(
                    x=x,
                    y=y,
                    z=z,
                    side_sign=side_sign,
                )
            )


            commands.extend([
                q_hip,
                q_thigh,
                q_calf,
            ])


        return np.array(
            commands,
            dtype=float,
        )


    # =====================================================
    # SINGLE LEG COMMAND
    # =====================================================

    def get_leg_command(
        self,
        phase,
        step_length,
        side_sign,
    ):
        """
        Convert desired foot trajectory into joint targets.
        """

        # -------------------------------------------------
        # FOOT TRAJECTORY
        # -------------------------------------------------

        x, z = foot_trajectory(
            phase,
            step_length=step_length,
            step_height=self.step_height,
            nominal_height=self.nominal_height,
        )


        # -------------------------------------------------
        # LATERAL FOOT POSITION
        # -------------------------------------------------

        y = (
            side_sign
            * self.hip_offset
        )


        # -------------------------------------------------
        # 3D IK
        # -------------------------------------------------

        q_hip, q_thigh, q_calf = (
            solve_leg_ik(
                x=x,
                y=y,
                z=z,
                side_sign=side_sign,
            )
        )


        return np.array([
            q_hip,
            q_thigh,
            q_calf,
        ])


    # =====================================================
    # TURN FEED-FORWARD
    # =====================================================

    def get_turn_correction(self):
        """
        Convert desired yaw velocity into differential stride.

        Positive wz -> left turn
        Negative wz -> right turn

        Left/right gains are separately calibrated.
        """

        if self.wz_command > 0.0:

            correction = (
                self.left_turn_gain
                * self.wz_command
            )


        elif self.wz_command < 0.0:

            correction = (
                self.right_turn_gain
                * self.wz_command
            )


        else:

            correction = 0.0


        correction = np.clip(
            correction,
            -self.max_turn_correction,
            self.max_turn_correction,
        )


        return float(
            correction
        )


    # =====================================================
    # FULL ROBOT COMMAND
    # =====================================================

    def get_command(
        self,
        time,
        yaw_error=0.0,
    ):

        # =================================================
        # 1. TRUE STOP
        # =================================================

        if (
            abs(self.vx_command)
            < self.velocity_deadband

            and

            abs(self.wz_command)
            < 1e-4
        ):

            return (
                self.get_stand_command()
            )


        # =================================================
        # 2. SMOOTH STARTUP
        # =================================================

        if self.ramp_time > 0:

            r = np.clip(
                time
                / self.ramp_time,
                0.0,
                1.0,
            )

        else:

            r = 1.0


        # Smoothstep:
        #
        # alpha = 3r² - 2r³

        alpha = (
            r * r
            * (3.0 - 2.0 * r)
        )


        # =================================================
        # 3. BASE TRANSLATIONAL STRIDE
        # =================================================

        if (
            abs(self.vx_command)
            < self.velocity_deadband
        ):

            # Used for rotation-in-place.
            base_step_length = 0.0

        else:

            base_step_length = (
                velocity_to_step_length(
                    self.vx_command
                )
            )


        # =================================================
        # 4. OPEN-LOOP TURNING CORRECTION
        # =================================================

        turn_correction = (
            self.get_turn_correction()
        )


        left_step_length = (
            base_step_length
            + turn_correction
        )


        right_step_length = (
            base_step_length
            - turn_correction
        )


        # =================================================
        # 5. CLOSED-LOOP YAW FEEDBACK
        # =================================================

        yaw_correction = (
            self.k_yaw
            * yaw_error
        )


        yaw_correction = np.clip(
            yaw_correction,
            -self.max_yaw_correction,
            self.max_yaw_correction,
        )


        # Experimentally verified sign

        left_step_length += (
            yaw_correction
        )


        right_step_length -= (
            yaw_correction
        )


        # =================================================
        # 6. STRIDE SAFETY
        # =================================================

        left_step_length = np.clip(
            left_step_length,
            -self.max_step_length,
            self.max_step_length,
        )


        right_step_length = np.clip(
            right_step_length,
            -self.max_step_length,
            self.max_step_length,
        )


        # =================================================
        # 7. TROT PHASE
        # =================================================

        base_phase = (
            time
            * self.gait_frequency
        ) % 1.0


        # Diagonal pair:
        # FL + RR

        phase_fl = base_phase

        phase_rr = base_phase


        # Opposite diagonal pair:
        # FR + RL

        phase_fr = (
            base_phase
            + 0.5
        ) % 1.0


        phase_rl = (
            base_phase
            + 0.5
        ) % 1.0


        # =================================================
        # 8. LEG COMMANDS
        # =================================================

        command_fr = (
            self.get_leg_command(
                phase_fr,
                right_step_length,
                side_sign=-1,
            )
        )


        command_fl = (
            self.get_leg_command(
                phase_fl,
                left_step_length,
                side_sign=1,
            )
        )


        command_rr = (
            self.get_leg_command(
                phase_rr,
                right_step_length,
                side_sign=-1,
            )
        )


        command_rl = (
            self.get_leg_command(
                phase_rl,
                left_step_length,
                side_sign=1,
            )
        )


        # =================================================
        # 9. FULL GAIT TARGET
        # =================================================

        gait_command = np.concatenate([
            command_fr,
            command_fl,
            command_rr,
            command_rl,
        ])


        # =================================================
        # 10. STAND TARGET
        # =================================================

        stand_command = (
            self.get_stand_command()
        )


        # =================================================
        # 11. SMOOTH STAND -> GAIT BLEND
        # =================================================

        final_command = (
            (1.0 - alpha)
            * stand_command

            +

            alpha
            * gait_command
        )


        return final_command