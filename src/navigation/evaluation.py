"""Headless MuJoCo execution and independent baseline safety measurements."""

from dataclasses import asdict
import math
import time

import numpy as np

from src.controllers.locomotion_interface import LocomotionInterface, wrap_angle
from src.navigation.follower import FollowerConfig, RouteFollower
from src.utils.orientation import quaternion_to_euler


def robot_collision_geoms(model):
    import mujoco
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    bodies = {trunk}
    for body in range(trunk + 1, model.nbody):
        if int(model.body_parentid[body]) in bodies:
            bodies.add(body)
    return tuple(g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies
                 and (model.geom_contype[g] or model.geom_conaffinity[g]))


def projected_robot_radius(model, data, geoms):
    """Conservative XY circle enclosing every robot collision primitive.

    Capsule/cylinder endpoint circles bound their projections; box corners are
    transformed exactly. Visual meshes and nonrobot world geoms are excluded.
    This is a measurement of sampled poses, not a bound for every possible gait.
    """
    import mujoco
    centre, maximum = data.qpos[:2], 0.0
    for geom in geoms:
        position, matrix = data.geom_xpos[geom], data.geom_xmat[geom].reshape(3, 3)
        size, kind = model.geom_size[geom], int(model.geom_type[geom])
        if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
            corners = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * size
            points = corners @ matrix.T + position
            radius = float(np.max(np.linalg.norm(points[:, :2] - centre, axis=1)))
        elif kind in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
            endpoints = position + np.array([-1, 1])[:, None] * size[1] * matrix[:, 2]
            radius = float(np.max(np.linalg.norm(endpoints[:, :2] - centre, axis=1)) + size[0])
        elif kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            radius = float(np.linalg.norm(position[:2] - centre) + size[0])
        else:
            raise ValueError(f"unsupported robot collision primitive {kind}")
        maximum = max(maximum, radius)
    return maximum


def measure_footprint(campus, duration=3.0):
    import mujoco
    model, data = campus.model, campus.data
    geoms = robot_collision_geoms(model)
    cases = []
    for name, vx, wz in (("stand", 0, 0), ("forward", .3, 0),
                         ("left_curve", .3, .3), ("right_curve", .3, -.3),
                         ("left_rotation", 0, .3), ("right_rotation", 0, -.3)):
        campus.reset()
        loco = LocomotionInterface()
        loco.set_velocity(vx, wz)
        next_sample, maximum = 0.0, 0.0
        while data.time < duration:
            data.ctrl[:] = loco.update(data, model.opt.timestep)
            mujoco.mj_step(model, data)
            if data.time >= next_sample:
                mujoco.mj_forward(model, data)
                maximum = max(maximum, projected_robot_radius(model, data, geoms))
                next_sample += .02
        cases.append({"case": name, "vx": vx, "wz": wz,
                      "simulated_seconds": duration, "max_projected_radius_m": maximum})
    campus.reset()
    return {"scope": "sampled collision-geometry bounds, flat road; six command cases",
            "sample_rate_hz": 50, "cases": cases,
            "max_projected_radius_m": max(c["max_projected_radius_m"] for c in cases)}


def command_is_safe(planner, pose, vx, wz, horizon=.5):
    """Check a short unicycle centre trajectory in the padded static map.

    This filter does not model balance, drift, or unknown obstacles. Execution
    monitors actual clearance, contacts and body attitude independently.
    """
    x, y, yaw = pose
    if not planner.is_free((x, y)):
        return False
    previous = x, y
    for elapsed in np.linspace(0, horizon, 6)[1:]:
        if abs(wz) < 1e-9:
            point = x + vx * elapsed * math.cos(yaw), y + vx * elapsed * math.sin(yaw)
        else:
            point = (x + vx / wz * (math.sin(yaw + wz * elapsed) - math.sin(yaw)),
                     y - vx / wz * (math.cos(yaw + wz * elapsed) - math.cos(yaw)))
        if not planner.segment_is_free(previous, point):
            return False
        previous = point
    return True


def contact_is_forbidden(name, position, normal, robot_body, layout):
    """Allow leg support on walkable terrain, including ramp-end contacts.

    Ramp end contacts below the road surface are expected with the A1 calf
    capsules. Ramp lateral sides, curb faces outside the cut, trunk/ground,
    solids and no-go terrain remain failures.
    """
    if (name in {obstacle.name for obstacle in layout.obstacles}
            or name in ("terrain_planter", "terrain_entrance")
            or name.startswith(("wall_", "terrain_stairs_"))):
        return True
    if robot_body == "trunk" and name.startswith("terrain_"):
        return True
    if name == "terrain_ramp":
        return abs(normal[0]) > .5
    raised = name in ("terrain_pavement", "terrain_grass", "terrain_plaza")
    outside_cut = not layout.ramp.x0 < position[0] < layout.ramp.x1
    return (raised and abs(normal[2]) < .5 and outside_cut
            and abs(position[1] - layout.curb_y) < .1)


def execute_approach(campus, planner, approach, *, max_time=180.0, control_hz=10.0,
                     follower_config=None, start_pose=None, controller_factory=None,
                     reset=True, step_hook=None, goal_provider=None, viewer=None):
    import mujoco
    if not math.isfinite(max_time) or max_time <= 0 or not math.isfinite(control_hz) or control_hz <= 0:
        raise ValueError("execution time and control rate must be positive and finite")
    if reset:
        campus.reset()
    model, data = campus.model, campus.data
    if start_pose is not None:
        from src.simulation.robot_setup import reset_robot
        if len(start_pose) != 3 or not all(math.isfinite(v) for v in start_pose):
            raise ValueError("start_pose must contain finite x, y and yaw")
        if not planner.is_free(start_pose[:2]):
            raise ValueError("start_pose is not clear for the robot footprint")
        reset_robot(model, data, x=start_pose[0], y=start_pose[1], yaw=start_pose[2],
                    ground_z=campus.layout.ground_height_at(*start_pose[:2]))
    initial_pose = campus.robot_pose()
    initial_time = float(data.time)
    cfg = follower_config or FollowerConfig()
    def make_follower(route, yaw):
        return (controller_factory(route, yaw) if controller_factory
                else RouteFollower(route, yaw, cfg))
    follower = make_follower(approach.route, approach.yaw)
    loco = LocomotionInterface()
    robot_geoms = robot_collision_geoms(model)
    robot_set = set(robot_geoms)
    geom_names = {g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(model.ngeom)}
    robot_bodies = {g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) for g in robot_geoms}
    started = time.perf_counter()
    next_control = next_log = next_measure = initial_time
    settled_since = shield_since = None
    log, contacts, contact_examples = [], set(), []
    support_transition_samples, support_examples = 0, []
    status = "timeout"
    minimum_clearance, maximum_radius = math.inf, 0.0
    max_roll = max_pitch = 0.0
    shield_interventions = 0
    replans = []
    last_replan = -math.inf
    vx = wz = 0.0
    goal_updates = []
    while data.time - initial_time < max_time:
        if viewer is not None and not viewer.is_running():
            status = "viewer_closed"
            break
        pose = campus.robot_pose()
        if data.time >= next_control:
            next_control += 1 / control_hz
            if goal_provider is not None:
                updated = goal_provider(campus, approach)
                if updated is not None:
                    approach = updated
                    follower = make_follower(approach.route, approach.yaw)
                    goal_updates.append({"time": float(data.time - initial_time),
                                         "goal_position": list(approach.position)})
            if hasattr(follower, "set_motion"):
                velocity = data.qvel[:2]
                forward = float(velocity[0] * math.cos(pose[2]) + velocity[1] * math.sin(pose[2]))
                follower.set_motion(forward, float(data.qvel[5]))
            vx, wz = follower.command(*pose)
            if not command_is_safe(planner, pose, vx, wz):
                shield_interventions += 1
                # Tracking error can make a formerly safe shortcut clip a
                # padded corner. Replan from the observed pose before retrying.
                if vx != 0 and planner.is_free(pose[:2]) and len(replans) < 3 and data.time - last_replan > 1:
                    from src.navigation.planner import NoPathError
                    try:
                        revised = planner.plan(pose[:2], approach.position)
                        follower = make_follower(revised, approach.yaw)
                        last_replan = data.time
                        replans.append({"time": float(data.time), "pose": list(pose),
                                        "reason": "predicted padded-corner intersection",
                                        "waypoints": [list(p) for p in revised.waypoints]})
                        vx, wz = follower.command(*pose)
                    except NoPathError:
                        vx = wz = 0.0
                if not command_is_safe(planner, pose, vx, wz):
                    vx = wz = 0.0
                    if shield_since is None:
                        shield_since = data.time
                else:
                    shield_since = None
            else:
                shield_since = None
            loco.set_velocity(vx, wz)
        data.ctrl[:] = loco.update(data, model.opt.timestep)
        mujoco.mj_step(model, data)
        if step_hook is not None:
            step_hook(campus)
        x, y, yaw = campus.robot_pose()
        roll, pitch, _ = quaternion_to_euler(data.qpos[3:7])
        max_roll, max_pitch = max(max_roll, abs(roll)), max(max_pitch, abs(pitch))
        if data.qpos[2] - campus.layout.ground_height_at(x, y) < .20 or max(abs(roll), abs(pitch)) > math.radians(30):
            status = "fall"
            break
        if not planner.is_free((x, y), raw=True):
            status = "forbidden_entry"
            break
        # Leg support contacts are expected, including the ramp's lower edge.
        for contact in data.contact[:data.ncon]:
            a, b = int(contact.geom1), int(contact.geom2)
            if (a in robot_set) == (b in robot_set):
                continue
            world = b if a in robot_set else a
            name = geom_names[world]
            robot = a if a in robot_set else b
            body = robot_bodies[robot]
            example = None
            if abs(contact.frame[2]) < .5 or name.startswith(("obstacle_", "bin_", "wall_")) or body == "trunk":
                example = {"world_geom": name, "robot_geom": robot, "robot_body": body,
                           "position": contact.pos.tolist(), "normal": contact.frame[:3].tolist()}
            if contact_is_forbidden(name, contact.pos, contact.frame[:3], body, campus.layout):
                contacts.add(name)
                if len(contact_examples) < 5:
                    contact_examples.append(example or {"world_geom": name, "robot_body": body,
                                                         "position": contact.pos.tolist(), "normal": contact.frame[:3].tolist()})
            elif name.startswith("terrain_") and abs(contact.frame[2]) < .5:
                support_transition_samples += 1
                if len(support_examples) < 5:
                    support_examples.append(example)
        if contacts:
            status = "collision"
            break
        if data.time >= next_measure:
            next_measure += .02
            clearance = planner.clearance_at((x, y))
            minimum_clearance = min(minimum_clearance, clearance)
            mujoco.mj_forward(model, data)
            maximum_radius = max(maximum_radius, projected_robot_radius(model, data, robot_geoms))
            if clearance < planner.config.clearance_radius - 1e-6:
                status = "clearance_violation"
                break
            if maximum_radius > planner.config.clearance_radius - .02:
                status = "footprint_exceeds_padding"
                break
        error = math.dist((x, y), approach.position)
        heading_error = abs(wrap_angle(approach.yaw - yaw))
        speed = float(np.linalg.norm(data.qvel[:2]))
        ready = (follower.arrived and vx == 0 and wz == 0 and error <= cfg.goal_tolerance
                 and heading_error <= cfg.heading_tolerance and speed < .02)
        if ready:
            if settled_since is None:
                settled_since = data.time
            if data.time - settled_since >= .5:
                status = "success"
                break
        else:
            settled_since = None
        if shield_since is not None and data.time - shield_since > 1:
            status = "safety_stop"
            break
        if data.time >= next_log:
            next_log += .1
            if viewer is not None:
                viewer.sync()
            log.append({"time": float(data.time - initial_time), "x": x, "y": y, "yaw": yaw,
                        "roll_deg": math.degrees(roll), "pitch_deg": math.degrees(pitch),
                        "vx_command": vx, "wz_command": wz, "waypoint": follower.index,
                        "goal_error_m": error, "heading_error_deg": math.degrees(heading_error),
                        "speed_m_s": speed, "zone": campus.layout.zone_at(x, y)})
    loco.stop()
    data.ctrl[:] = loco.update(data, model.opt.timestep)
    pose = campus.robot_pose()
    metrics = {"track_id": approach.track_id, "status": status, "success": status == "success",
               "simulated_seconds": float(data.time - initial_time), "wall_seconds": time.perf_counter() - started,
               "goal_position": list(approach.position), "goal_yaw_rad": approach.yaw,
               "planned_standoff_m": approach.standoff, "planned_length_m": approach.route.length,
               "waypoints": [list(p) for p in approach.route.waypoints],
               "final_pose": list(pose), "final_position_error_m": math.dist(pose[:2], approach.position),
               "final_heading_error_deg": math.degrees(abs(wrap_angle(approach.yaw - pose[2]))),
               "final_speed_m_s": float(np.linalg.norm(data.qvel[:2])),
               "min_grid_clearance_m": None if math.isinf(minimum_clearance) else minimum_clearance,
               "max_projected_robot_radius_m": maximum_radius,
               "max_roll_deg": math.degrees(max_roll), "max_pitch_deg": math.degrees(max_pitch),
               "collision_geoms": sorted(contacts), "collision_examples": contact_examples,
               "support_transition_contact_samples": support_transition_samples,
               "support_transition_examples": support_examples,
               "shield_interventions": shield_interventions,
               "replanning_events": replans,
               "live_goal_updates": goal_updates,
               "controller": "PPO" if controller_factory else "classical follower",
               "ppo_inference_calls": getattr(follower, "inference_calls", 0),
               "reset_before_route": reset,
               "follower_config": asdict(cfg), "control_hz": control_hz,
               "start_pose": list(initial_pose)}
    return metrics, log
