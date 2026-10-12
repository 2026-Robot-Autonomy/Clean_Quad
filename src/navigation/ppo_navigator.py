"""Learned PPO speed/turning controller, with shared route/arrival semantics."""

import math
from pathlib import Path
import numpy as np

from src.controllers.locomotion_interface import wrap_angle
from src.navigation.follower import FollowerConfig
from src.navigation.observations import decode_action, encode_observation


class PPOFollower:
    def __init__(self, route, goal_yaw, planner, model, config=None):
        self.route, self.goal_yaw, self.planner = route, goal_yaw, planner
        self.model = model
        self.config = config or FollowerConfig()
        self.index = 1 if len(route.waypoints) > 1 else 0
        self.arrived = False
        self.motion = (0., 0.)
        self.previous_action = np.array([-1., 0.])
        self.inference_calls = 0
        self.last_raw_action = None

    def set_motion(self, forward, turn):
        self.motion = float(forward), float(turn)

    def command(self, x, y, yaw):
        final = len(self.route.waypoints) - 1
        while self.index < final and math.dist((x, y), self.route.waypoints[self.index]) <= self.config.waypoint_tolerance:
            self.index += 1
        target = self.route.waypoints[self.index]
        distance = math.dist((x, y), target)
        self.arrived = (self.index == final and distance <= self.config.goal_tolerance
                        and abs(wrap_angle(self.goal_yaw - yaw)) <= self.config.heading_tolerance)
        if self.arrived:
            return 0., 0.
        observation = encode_observation(self.planner, (x, y, yaw), target, self.goal_yaw,
                                         motion=self.motion, previous_action=self.previous_action,
                                         final=self.index == final)
        action, _ = self.model.predict(observation, deterministic=True)
        self.previous_action = np.asarray(action, dtype=float)
        self.last_raw_action = self.previous_action.tolist()
        self.inference_calls += 1
        vx, wz = decode_action(action)
        # A final alignment guard prohibits translation once in the goal disk.
        # Steering still comes from PPO; no classical-follower fallback exists.
        if self.index == final and distance <= self.config.goal_tolerance:
            vx = 0.
        return vx, wz


def load_policy(path):
    from stable_baselines3 import PPO
    if not Path(path).is_file():
        raise FileNotFoundError(f"PPO checkpoint not found: {path}")
    return PPO.load(str(path), device="cpu")
