"""A conservative stop/turn/drive follower for planned routes."""

from dataclasses import dataclass
import math

from src.controllers.locomotion_interface import wrap_angle


@dataclass(frozen=True)
class FollowerConfig:
    forward_speed: float = 0.30
    near_speed: float = 0.10
    yaw_rate: float = 0.30
    waypoint_tolerance: float = 0.04
    goal_tolerance: float = 0.08
    start_turn: float = math.radians(12)
    stop_turn: float = math.radians(3)
    heading_tolerance: float = math.radians(6)

    def __post_init__(self):
        values = tuple(self.__dict__.values())
        if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("follower settings must be positive and finite")
        if not self.near_speed <= self.forward_speed <= 0.5 or self.yaw_rate > 0.5:
            raise ValueError("follower commands exceed the locomotion envelope")
        if self.stop_turn >= self.start_turn:
            raise ValueError("turn hysteresis must be ordered")


class RouteFollower:
    def __init__(self, route, goal_yaw, config=None):
        if not route.waypoints:
            raise ValueError("route needs at least one waypoint")
        self.route, self.goal_yaw = route, goal_yaw
        self.config = config or FollowerConfig()
        self.index = 1 if len(route.waypoints) > 1 else 0
        self.turning = False
        self.arrived = False

    def command(self, x, y, yaw):
        cfg = self.config
        final = len(self.route.waypoints) - 1
        while self.index < final and math.dist((x, y), self.route.waypoints[self.index]) <= cfg.waypoint_tolerance:
            self.index += 1
            self.turning = False
        target = self.route.waypoints[self.index]
        distance = math.dist((x, y), target)
        if self.index == final and distance <= cfg.goal_tolerance:
            error = wrap_angle(self.goal_yaw - yaw)
            self.arrived = abs(error) <= cfg.heading_tolerance
            return (0.0, 0.0) if self.arrived else (0.0, math.copysign(cfg.yaw_rate, error))
        self.arrived = False
        error = wrap_angle(math.atan2(target[1] - y, target[0] - x) - yaw)
        if abs(error) > cfg.start_turn:
            self.turning = True
        if self.turning and abs(error) <= cfg.stop_turn:
            self.turning = False
        if self.turning:
            return 0.0, math.copysign(cfg.yaw_rate, error)
        speed = cfg.near_speed if distance < 0.30 else cfg.forward_speed
        # Existing curved walking is validated; small corrections reduce drift.
        turn = max(-cfg.yaw_rate, min(cfg.yaw_rate, 1.2 * error))
        if abs(error) < math.radians(1):
            turn = 0.0
        return speed, turn
