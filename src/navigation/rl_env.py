"""Fast 2-D local-navigation training environment, with no rendering or CNN.

A* supplies global waypoints at deployment. Training samples locally visible
subgoals on the same clearance-aware semantic map. Dynamics include command
lag, the stand-to-gait blend and per-episode velocity/bias randomization.
"""

import math
import gymnasium as gym
import numpy as np

from src.controllers.locomotion_interface import wrap_angle
from src.environment.layout import default_layout
from src.mapping import SemanticMap
from src.navigation.observations import (OBSERVATION_SIZE, decode_action,
                                          encode_observation, teacher_action)
from src.navigation.planner import AStarPlanner, NavigationConfig


class NavigationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, planner=None, *, max_steps=450, randomize=True):
        super().__init__()
        self.planner = planner or AStarPlanner(SemanticMap.from_layout(
            default_layout(), seed=0, run_id="rl-training-static-layout"), NavigationConfig())
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self.max_steps, self.randomize = max_steps, randomize
        self.action_space = gym.spaces.Box(-1., 1., shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-1., 1., shape=(OBSERVATION_SIZE,), dtype=np.float32)
        self.free_cells = np.argwhere(~self.planner.blocked)
        if not len(self.free_cells):
            raise ValueError("map has no clearance-safe training starts")
        self.dt = .1
        self.pose = np.zeros(3)
        self.motion = np.zeros(2)
        self.previous_action = np.array([-1., 0.])
        self.target, self.goal_yaw = (0., 0.), 0.
        self.steps = 0
        self.finished = False

    def observation(self):
        return encode_observation(self.planner, self.pose, self.target, self.goal_yaw,
                                  motion=self.motion, previous_action=self.previous_action, final=self.final)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.final = bool(options.get("final", self.np_random.random() < .7))
        if "pose" in options:
            pose = np.asarray(options["pose"], dtype=float)
            target = np.asarray(options["target"], dtype=float)
            if pose.shape != (3,) or target.shape != (2,) or not np.all(np.isfinite(pose)) or not np.all(np.isfinite(target)):
                raise ValueError("reset needs finite pose[3] and target[2]")
            if not self.planner.segment_is_free(pose[:2], target):
                raise ValueError("local training goal must have clearance-safe line of sight")
            self.pose = pose.copy()
            self.target = tuple(target)
            self.goal_yaw = float(options.get("goal_yaw", pose[2]))
        else:
            for _ in range(300):
                cell = self.free_cells[self.np_random.integers(len(self.free_cells))]
                start = self.planner.grid.cell_center(*cell)
                angle = self.np_random.uniform(-math.pi, math.pi)
                distance = self.np_random.uniform(.15, 1.8)
                target = start + distance * np.array([math.cos(angle), math.sin(angle)])
                if self.planner.segment_is_free(start, target):
                    # Include both large turns and near-goal alignment states.
                    self.pose = np.array([*start, self.np_random.uniform(-math.pi, math.pi)])
                    self.target = tuple(target)
                    self.goal_yaw = float(self.np_random.uniform(-math.pi, math.pi))
                    break
            else:
                raise RuntimeError("could not sample a reachable local training goal")
        if not math.isfinite(self.goal_yaw):
            raise ValueError("goal_yaw must be finite")
        self.motion[:] = 0.
        self.previous_action[:] = [-1., 0.]
        self.steps, self.settle_steps, self.motion_time = 0, 0, 0.
        self.finished = False
        self.speed_scale = self.np_random.uniform(.96, 1.04) if self.randomize else 1.
        self.turn_scale = self.np_random.uniform(.97, 1.03) if self.randomize else 1.
        self.yaw_bias = self.np_random.uniform(-.008, .008) if self.randomize else 0.
        self.lag = self.np_random.uniform(.10, .22) if self.randomize else .15
        return self.observation(), {"target": list(self.target), "goal_yaw": self.goal_yaw}

    def step(self, action):
        if self.finished:
            raise RuntimeError("reset is required after a completed episode")
        vx, wz = decode_action(action)
        before = math.dist(self.pose[:2], self.target)
        before_heading = abs(wrap_angle(self.goal_yaw - self.pose[2]))
        moving = vx != 0 or wz != 0
        self.motion_time = self.motion_time + self.dt if moving else 0.
        blend = min(self.motion_time, 1.)
        blend = blend * blend * (3 - 2 * blend)
        wanted = np.array([vx * self.speed_scale, wz * self.turn_scale + (self.yaw_bias if moving else 0.)]) * blend
        self.motion += (wanted - self.motion) * min(self.dt / self.lag, 1.)
        old = self.pose[:2].copy()
        yaw_mid = self.pose[2] + self.motion[1] * self.dt / 2.
        self.pose[:2] += self.motion[0] * self.dt * np.array([math.cos(yaw_mid), math.sin(yaw_mid)])
        self.pose[2] = wrap_angle(self.pose[2] + self.motion[1] * self.dt)
        self.steps += 1
        distance = math.dist(self.pose[:2], self.target)
        heading = abs(wrap_angle(self.goal_yaw - self.pose[2]))
        collision = not self.planner.segment_is_free(old, self.pose[:2])
        ready = (distance <= .08 and heading <= math.radians(6) and abs(self.motion[0]) < .02) if self.final else distance <= .04
        self.settle_steps = self.settle_steps + 1 if ready else 0
        success = self.settle_steps >= (5 if self.final else 1)
        # Progress is to a locally visible subgoal, not through blocked terrain.
        reward = 12. * (before - distance) - .015
        if self.final and distance <= .12:
            reward += .5 * (before_heading - heading)
        observation = self.observation()
        bearing_error = abs(math.atan2(observation[3], observation[4]))
        reward -= .004 * bearing_error
        reward -= .001 * float(np.sum((np.asarray(action) - self.previous_action) ** 2))
        if collision:
            reward -= 25.
        if success:
            reward += 20.
        terminated = collision or success
        truncated = self.steps >= self.max_steps and not terminated
        self.finished = terminated or truncated
        self.previous_action = np.clip(np.asarray(action, dtype=float), -1, 1)
        return observation, float(reward), bool(terminated), bool(truncated), {
            "is_success": bool(success), "collision": bool(collision),
            "distance": distance, "heading_error": heading, "steps": self.steps}

    def teacher(self):
        return teacher_action(self.observation())
