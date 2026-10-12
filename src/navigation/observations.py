"""Shared, normalized local-navigation contract for 2-D training and MuJoCo."""

import math
import numpy as np

from src.controllers.locomotion_interface import wrap_angle

OBSERVATION_VERSION = 1
OBSERVATION_SIZE = 25
RAY_ANGLES = np.linspace(-math.pi, math.pi, 12, endpoint=False)
RAY_DISTANCES = np.linspace(.05, 2., 24)
MAX_FORWARD = .30
MAX_TURN = .30


def obstacle_rays(planner, pose):
    """Distances to the padded known map, not readings from a physical lidar."""
    x, y, yaw = pose
    angles = RAY_ANGLES + yaw
    px = x + np.cos(angles)[:, None] * RAY_DISTANCES
    py = y + np.sin(angles)[:, None] * RAY_DISTANCES
    grid = planner.grid
    col = np.floor((px - grid.origin[0]) / grid.resolution).astype(int)
    row = np.floor((py - grid.origin[1]) / grid.resolution).astype(int)
    valid = (row >= 0) & (col >= 0) & (row < grid.rows) & (col < grid.columns)
    hits = ~valid
    hits[valid] = planner.blocked[row[valid], col[valid]]
    first = np.argmax(hits, axis=1)
    return np.where(np.any(hits, axis=1), RAY_DISTANCES[first], 2.) / 2.


def encode_observation(planner, pose, target, goal_yaw, *, motion=(0., 0.),
                       previous_action=(-1., 0.), final=True):
    x, y, yaw = pose
    dx, dy = target[0] - x, target[1] - y
    distance = math.hypot(dx, dy)
    aligning = final and distance <= .08
    bearing = goal_yaw if aligning else math.atan2(dy, dx)
    error = wrap_angle(bearing - yaw)
    local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
    local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
    heading = wrap_angle(goal_yaw - yaw)
    try:
        cell = planner.grid.world_to_cell(x, y)
        terrain = planner.map.layers["terrain"][cell]
        height = float(planner.map.layers["height"][cell])
    except IndexError:
        terrain, height = "outside", 0.
    values = [np.clip(local_x / 2., -1, 1), np.clip(local_y / 2., -1, 1),
              min(distance / 2., 1), math.sin(error), math.cos(error),
              math.sin(heading), math.cos(heading), float(aligning), float(final),
              np.clip(motion[0] / MAX_FORWARD, -1, 1), np.clip(motion[1] / MAX_TURN, -1, 1),
              float(terrain == "ramp"), np.clip(height / .1, -1, 1)]
    # Previous actions are intentionally not included: the policy is a memoryless
    # controller and receives measured motion. Kept as an API argument for runners.
    result = np.asarray(values + obstacle_rays(planner, pose).tolist(), dtype=np.float32)
    if result.shape != (OBSERVATION_SIZE,) or not np.all(np.isfinite(result)):
        raise ValueError("invalid local navigation observation")
    return result


def decode_action(action):
    action = np.asarray(action, dtype=float)
    if action.shape != (2,) or not np.all(np.isfinite(action)):
        raise ValueError("navigation action must contain two finite values")
    action = np.clip(action, -1., 1.)
    vx, wz = (action[0] + 1.) * MAX_FORWARD / 2., action[1] * MAX_TURN
    # Respect the gait's minimum useful translation and suppress tiny yaw noise.
    return (0. if vx < .05 else float(vx), 0. if abs(wz) < .015 else float(wz))


def teacher_action(observation):
    """Smooth feedback teacher, used only to initialize the policy's weights."""
    obs = np.asarray(observation)
    error = math.atan2(obs[3], obs[4])
    distance = obs[2] * 2.
    aligning = bool(obs[7] > .5)
    if aligning:
        speed = 0.
    else:
        limit = .10 if distance < .30 else MAX_FORWARD
        speed = limit * np.clip((.35 - abs(error)) / .25, 0., 1.)
        if speed < .05:
            speed = 0.
    turn = np.clip(1.2 * error, -MAX_TURN, MAX_TURN)
    if abs(error) < math.radians(1):
        turn = 0.
    return np.array([2. * speed / MAX_FORWARD - 1., turn / MAX_TURN], dtype=np.float32)
