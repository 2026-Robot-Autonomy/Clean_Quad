"""Stage 7 pickup skill: dock, crouch, suction, rise, back off, verify.

The robot arrives at a Stage 6 approach pose facing the item. The skill then

1. checks with the camera that the item is really there (when a `look`
   callback is supplied),
2. drives straight toward the item until the nozzle is over it (DOCK),
3. lowers its front to bring the nozzle tip to the ground (LOWER),
4. applies suction (SUCTION): the simulator decides the physical outcome,
5. stands back up (RAISE) and reverses to the approach pose (BACK OFF),
6. looks again: the pickup is verified only if the item is no longer seen.

Suction is abstract: a litter item is removed (Campus.collect) only if the
true object lies inside the nozzle capture zone at the lowest point. A wrong
map estimate therefore produces a real miss, never a fake success. If a
non-litter distractor lies in the zone the attempt is aborted.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np

from src.controllers.locomotion_interface import LocomotionInterface, wrap_angle
from src.manipulation.posture import Posture, PostureFailure, posture_targets, run_posture_sequence
from src.manipulation.tools import nozzle_tip
from src.navigation.evaluation import execute_approach
from src.navigation.follower import FollowerConfig
from src.navigation.planner import ApproachPlan, RoutePlan


@dataclass(frozen=True)
class PickupConfig:
    # Standing trunk centre to crouched nozzle tip, along the heading [m].
    # Measured by experiments/33_pickup_trials.py --calibrate.
    dock_offset: float = 0.245
    dock_speed: float = 0.12
    dock_tolerance: float = 0.025
    dock_time: float = 25.0
    capture_radius: float = 0.07
    max_tip_height: float = 0.10        # nozzle tip above item centre [m]
    pickup: Posture = Posture(-0.17, -0.27)
    stand: Posture = Posture(-0.30, -0.30)
    lower_seconds: float = 1.2
    hold_seconds: float = 0.8
    raise_seconds: float = 1.2
    settle_seconds: float = 0.6
    attempts: int = 3

    def __post_init__(self):
        for name in ("dock_offset", "dock_speed", "dock_tolerance", "dock_time", "capture_radius",
                     "max_tip_height", "lower_seconds", "hold_seconds", "raise_seconds", "settle_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.dock_speed > 0.3:
            raise ValueError("docking speed exceeds the validated envelope")
        if not isinstance(self.attempts, int) or self.attempts < 1:
            raise ValueError("attempts must be a positive integer")


class LineFollower:
    """Drive to a goal point, then turn in place to a final heading.

    The trunk points along the line of travel (forward for docking, backward
    when reversing) so the trot does not need to side-step; after the goal the
    robot rotates to goal_yaw. Shares the route-follower interface used by
    execute_approach (command, arrived, index).
    """

    def __init__(self, start, goal, goal_yaw, *, reverse=False, speed=0.12, tolerance=0.025,
                 config=None, min_speed=0.06, max_turn=0.25):
        delta = np.subtract(goal, start)
        if np.linalg.norm(delta) < 1e-9:
            raise ValueError("line follower needs distinct start and goal")
        self.start, self.goal = tuple(start), tuple(goal)
        self.goal_yaw = goal_yaw
        self.direction = delta / np.linalg.norm(delta)
        self.reverse = reverse
        self.speed, self.tolerance, self.min_speed = speed, tolerance, min_speed
        self.max_turn = max_turn
        self.config = config or FollowerConfig()
        self.index = 1
        self.arrived = False

    def remaining(self, x, y):
        return float(np.subtract(self.goal, (x, y)) @ self.direction)

    def turn(self, error):
        magnitude = min(self.max_turn, max(0.08, 1.5 * abs(error)))
        return math.copysign(magnitude, error)

    def command(self, x, y, yaw):
        along = self.remaining(x, y)
        if abs(along) <= self.tolerance:
            heading_error = wrap_angle(self.goal_yaw - yaw)
            self.arrived = abs(heading_error) <= self.config.heading_tolerance
            return (0.0, 0.0) if self.arrived else (0.0, self.turn(heading_error))
        self.arrived = False
        # An overshoot is corrected by stepping back along the line, not by
        # turning around.
        backwards = self.reverse != (along < 0)
        offset = np.subtract(self.goal, (x, y))
        if np.linalg.norm(offset) > 0.06:
            travel = math.atan2(offset[1], offset[0])
        else:
            travel = math.atan2(self.direction[1], self.direction[0]) + (math.pi if along < 0 else 0.0)
        heading_error = wrap_angle(travel + (math.pi if backwards else 0.0) - yaw)
        if abs(heading_error) > math.radians(15):
            return 0.0, self.turn(heading_error)
        speed = min(self.speed, max(self.min_speed, 0.8 * abs(along)))
        vx = -speed if backwards else speed
        return vx, float(np.clip(1.5 * heading_error, -self.max_turn, self.max_turn))


def choose_dock(planner, start, belief, offset, *, margin=0.02, max_skew_deg=30, step_deg=3):
    """Dock pose (point, heading) that puts the nozzle tip on `belief`.

    The heading is searched outward from the bearing to the item. The dock
    point, a margin beyond it and the straight line from `start` must all be
    free in the padded map, so the motion shield never sees the goal as unsafe.
    """

    bearing = math.atan2(belief[1] - start[1], belief[0] - start[0])
    deltas = sorted(range(-max_skew_deg, max_skew_deg + 1, step_deg), key=abs)
    for delta in deltas:
        heading = bearing + math.radians(delta)
        dock = (belief[0] - offset * math.cos(heading), belief[1] - offset * math.sin(heading))
        motion = np.subtract(dock, start)
        length = float(np.linalg.norm(motion))
        if length < 1e-6:
            return dock, heading
        ahead = tuple(np.asarray(dock) + margin * motion / length)
        if planner.is_free(dock) and planner.is_free(ahead) and planner.segment_is_free(start, ahead):
            return dock, heading
    return None


def line_plan(start, goal, yaw):
    route = RoutePlan((tuple(start), tuple(goal)), (), 0.0, math.dist(start, goal))
    return ApproachPlan(0, tuple(goal), yaw, 0.0, route)


def suction(campus, tip, config):
    """Physical outcome of suction at the tip (simulator ground truth)."""

    def in_zone(obj):
        return (math.dist(tip[:2], obj.position[:2]) <= config.capture_radius
                and tip[2] - obj.position[2] <= config.max_tip_height)

    distractors = [obj.name for obj in campus.distractors if in_zone(obj)]
    captured = [obj.name for obj in campus.remaining_litter if in_zone(obj)]
    if distractors:
        return {"status": "distractor_in_zone", "captured": [], "distractors_in_zone": distractors}
    for name in captured:
        campus.collect(name)
    return {"status": "captured" if captured else "empty", "captured": captured,
            "distractors_in_zone": []}


def nearest_litter_error(campus, tip):
    """Diagnostic only: horizontal tip distance to the closest remaining item."""

    distances = [math.dist(tip[:2], obj.position[:2]) for obj in campus.remaining_litter]
    return min(distances, default=None)


class PickupSkill:
    def __init__(self, campus, planner, config=None, *, step_hook=None, viewer=None, on_capture=None):
        self.campus, self.planner = campus, planner
        self.on_capture = on_capture
        self.config = config or PickupConfig()
        self.step_hook, self.viewer = step_hook, viewer
        hip_offset = LocomotionInterface().controller.hip_offset
        self.pickup_targets = posture_targets(self.config.pickup, hip_offset)
        self.stand_targets = posture_targets(self.config.stand, hip_offset)

    # ----------------------------------------------------- motion primitives

    def drive_line(self, goal, yaw, label, *, reverse=False):
        start = self.campus.robot_pose()[:2]
        if math.dist(start, goal) <= self.config.dock_tolerance:
            return {"status": "success", "success": True, "label": label, "skipped": True}
        if not self.planner.segment_is_free(start, goal):
            return {"status": "line_blocked", "success": False, "label": label}
        cfg = FollowerConfig()
        factory = lambda route, goal_yaw: LineFollower(start, goal, yaw, reverse=reverse, speed=self.config.dock_speed,
                                                       tolerance=self.config.dock_tolerance, config=cfg)
        metrics, _ = execute_approach(self.campus, self.planner, line_plan(start, goal, yaw),
                                      controller_factory=factory, reset=False, follower_config=cfg,
                                      max_time=self.config.dock_time, step_hook=self.step_hook,
                                      viewer=self.viewer)
        metrics["controller"] = "line follower"
        metrics["label"] = label
        return metrics

    def crouch_and_suck(self, *, dump=False):
        """Lower, act (suction or dump), raise. Returns posture/suction metrics."""

        cfg = self.config
        lowest = {"tip": None}

        def sample(campus):
            tip = nozzle_tip(campus.model, campus.data)
            if lowest["tip"] is None or tip[2] < lowest["tip"][2]:
                lowest["tip"] = tip.copy()

        result = {}
        try:
            down = run_posture_sequence(self.campus, [(self.pickup_targets, cfg.lower_seconds),
                                                      (self.pickup_targets, cfg.hold_seconds)],
                                        step_hook=self.step_hook, viewer=self.viewer, sample=sample)
            tip = nozzle_tip(self.campus.model, self.campus.data)
            result["tip_at_suction"] = tip.tolist()
            result["tip_ground_clearance_m"] = float(
                tip[2] - self.campus.layout.ground_height_at(tip[0], tip[1]))
            if dump:
                result["suction"] = None
            else:
                result["nearest_litter_error_m"] = nearest_litter_error(self.campus, tip)
                result["suction"] = suction(self.campus, tip, cfg)
                if self.on_capture is not None and result["suction"]["captured"]:
                    self.on_capture(result["suction"]["captured"])
            up = run_posture_sequence(self.campus, [(self.stand_targets, cfg.raise_seconds),
                                                    (self.stand_targets, cfg.settle_seconds)],
                                      step_hook=self.step_hook, viewer=self.viewer)
        except PostureFailure as failure:
            result["status"] = str(failure)
            return result
        result["status"] = "success"
        result["lower"], result["raise"] = down, up
        return result

    # ----------------------------------------------------------- the skill

    def pick(self, belief, *, look=None, refresh=None):
        """Pick up the item believed to be at `belief` (x, y, z).

        The robot must be standing at an approach pose facing the item.
        look(position) -> {"in_view": bool, "seen": bool, "position": list|None}
        uses only the camera and detector. refresh() returns the latest map
        position of the item, or None to keep the current belief.
        """

        cfg = self.config
        approach = self.campus.robot_pose()
        attempts = []
        status = "pickup_failed"
        for attempt in range(cfg.attempts):
            if refresh is not None:
                belief = refresh() or belief
            record = {"attempt": attempt + 1, "belief": list(map(float, belief))}
            attempts.append(record)
            if look is not None:
                record["pre_look"] = look(belief)
                pre = record["pre_look"]
                if pre["in_view"] and not pre["seen"]:
                    # Nothing to pick up here: a false map record, or the
                    # previous attempt already succeeded without a confirmation.
                    status = "not_found" if attempt == 0 else "verified_gone"
                    break
                if pre["seen"] and pre.get("position") is not None:
                    belief = pre["position"]
                    record["belief"] = list(map(float, belief))
            pose = self.campus.robot_pose()
            if math.dist(pose[:2], belief[:2]) <= cfg.dock_offset + cfg.dock_tolerance:
                record["status"] = "too_close"
                status = "too_close"
                break
            choice = choose_dock(self.planner, pose[:2], belief[:2], cfg.dock_offset)
            if choice is None:
                status = "no_dock_pose"
                break
            dock, heading = choice
            record["dock_target"], record["dock_heading_rad"] = list(dock), heading
            record["dock"] = self.drive_line(dock, heading, "dock")
            if not record["dock"]["success"]:
                status = record["dock"]["status"]
                break
            record["dock_error_m"] = math.dist(self.campus.robot_pose()[:2], dock)
            record["action"] = self.crouch_and_suck()
            if record["action"]["status"] != "success":
                status = record["action"]["status"]
                break
            facing = math.atan2(belief[1] - approach[1], belief[0] - approach[0])
            back = self.drive_line(approach[:2], facing, "back_off", reverse=True)
            record["back_off"] = back
            if not back["success"]:
                status = back["status"]
                break
            captured = record["action"]["suction"]["captured"]
            if look is not None:
                record["post_look"] = look(belief)
                post = record["post_look"]
                if post["in_view"] and not post["seen"]:
                    status = "verified_gone"
                    break
            elif captured:
                # Headless trials have no camera; report the simulator outcome.
                status = "captured_unverified"
                break
        collected = [name for record in attempts
                     for name in ((record.get("action") or {}).get("suction") or {}).get("captured", [])]
        return {"status": status, "success": status in ("verified_gone", "captured_unverified"),
                "captured": collected, "attempts": attempts, "approach_pose": list(approach),
                "config": asdict(cfg)}

    def dump(self, hopper):
        """Lean toward a bin and empty the hopper (abstract)."""

        result = self.crouch_and_suck(dump=True)
        result["dumped"] = list(hopper) if result["status"] == "success" else []
        return result
