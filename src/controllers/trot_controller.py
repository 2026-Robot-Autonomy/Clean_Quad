import numpy as np

from src.controllers.gait_generator import foot_trajectory
from src.controllers.leg_kinematics import solve_planar_ik


class TrotController:

    def __init__(
        self,
        gait_frequency=1.5,
        step_length=0.06,
        step_height=0.04,
        nominal_height=-0.30,
    ):

        self.gait_frequency = gait_frequency
        self.step_length = step_length
        self.step_height = step_height
        self.nominal_height = nominal_height


    def get_leg_command(self, phase):

        x, z = foot_trajectory(
            phase,
            step_length=self.step_length,
            step_height=self.step_height,
            nominal_height=self.nominal_height,
        )

        q_thigh, q_calf = solve_planar_ik(
            x,
            z
        )

        q_hip = 0.0

        return np.array([
            q_hip,
            q_thigh,
            q_calf,
        ])


    def get_command(self, time):

        base_phase = (
            time * self.gait_frequency
        ) % 1.0

        phase_fl = base_phase
        phase_rr = base_phase

        phase_fr = (base_phase + 0.5) % 1.0
        phase_rl = (base_phase + 0.5) % 1.0


        command_fr = self.get_leg_command(
            phase_fr
        )

        command_fl = self.get_leg_command(
            phase_fl
        )

        command_rr = self.get_leg_command(
            phase_rr
        )

        command_rl = self.get_leg_command(
            phase_rl
        )


        return np.concatenate([
            command_fr,
            command_fl,
            command_rr,
            command_rl,
        ])