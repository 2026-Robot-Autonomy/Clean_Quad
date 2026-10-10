"""
Scripted waypoint following for demos (Stages 2-3).

A bang-bang heading follower using validated V2_FAST commands
(vx = 0.3 m/s, wz in {-0.5, 0, +0.5} rad/s). No planning and
no obstacle avoidance: real navigation is Stage 6.
"""

import math

from src.controllers.locomotion_interface import wrap_angle


# Hand-picked campus loop: road -> up the ramp -> pavement ->
# grass around the planter -> down the ramp -> base.
DEMO_ROUTE = [
    (3.3, 1.0),     # east along the road
    (4.0, 2.6),     # up the ramp onto the pavement
    (4.4, 4.4),     # grass, east of the planter
    (2.0, 4.8),     # north of the planter
    (1.0, 3.6),     # grass, west
    (2.6, 2.7),     # pavement
    (3.4, 2.7),     # line up above the ramp
    (4.0, 1.1),     # down the ramp
    (0.8, 1.0),     # home
]

VX = 0.30
WZ = 0.50

REACHED_DISTANCE = 0.25

# Hysteresis on heading error [rad]
START_TURN = math.radians(8.0)
STOP_TURN = math.radians(3.0)


class HeadingFollower:
    """Bang-bang steering towards a point, with hysteresis."""

    def __init__(self, vx=VX, wz=WZ):
        self.vx = vx
        self.wz_max = wz
        self.wz = 0.0

    def command(self, x, y, yaw, target):

        error = wrap_angle(math.atan2(target[1] - y, target[0] - x) - yaw)

        if self.wz == 0.0 and abs(error) > START_TURN:
            self.wz = math.copysign(self.wz_max, error)
        elif self.wz != 0.0 and (abs(error) < STOP_TURN or error * self.wz < 0.0):
            self.wz = 0.0

        return self.vx, self.wz


def reached(x, y, target):
    return math.dist((x, y), target) < REACHED_DISTANCE
