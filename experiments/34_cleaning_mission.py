"""Stage 7/8 cleaning mission: live CNN mapping, PPO navigation, pickup, bins, battery.

python -B experiments/34_cleaning_mission.py --mode stage8 --require-pass [--viewer]

--mode stage7  Pickup integration: map items in fixed track-ID order, an
               unlimited hopper and no battery limit; the hopper is emptied
               into the best bin once at the end, then the robot returns home.
--mode stage8  Task planning: the TaskPlanner chooses every next action from
               distance, terrain risk and class priority; a hopper capacity
               forces bin visits and a simulated battery forces charging
               returns to the base pad. The plan is recomputed after every
               action, so newly detected items and failures are handled.

The prior Stage 5 map is starting knowledge; live RGB-D/CNN tracks update it
as in Stage 6. Pickups are confirmed only by the camera: an item is marked
collected in the map when, after suction, it is in view and no longer
detected. Simulator truth decides the physical suction outcome and is read
for the post-mission evaluation, never for decisions.
"""

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.controllers.locomotion_interface import wrap_angle
from src.environment import build_campus
from src.localization.geometry import LocalizationConfig
from src.localization.pipeline import LocalizationPipeline
from src.manipulation.display import CarryDisplay
from src.manipulation.pickup import PickupConfig, PickupSkill
from src.mapping import load_map, save_map
from src.navigation import AStarPlanner, ApproachPlan, NoPathError
from src.navigation.evaluation import execute_approach
from src.navigation.live_map import LiveMapBridge
from src.navigation.ppo_navigator import PPOFollower, load_policy
from src.tasks.planner import TaskConfig, TaskPlanner

DOCK_FAILURES = ("no_dock_pose", "line_blocked", "safety_stop", "too_close", "timeout")
UNSAFE = ("fall", "collision", "forbidden_entry", "trunk_contact", "viewer_closed",
          "clearance_violation", "footprint_exceeds_padding")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=float) + "\n", encoding="utf-8")


def save_rows(path, rows):
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def match_truth(campus, position, kind=None, radius=0.25):
    candidates = [t for t in campus.litter if (kind is None or t.kind == kind)
                  and math.dist(t.position[:2], position[:2]) <= radius]
    return min(candidates, key=lambda t: math.dist(t.position[:2], position[:2]), default=None)


def main():
    import mujoco
    import torch
    from src.perception.camera import RobotCamera
    from src.perception.yolo_detector import MODEL_PATH, YoloDetector
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("stage7", "stage8"), default="stage8")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/navigation/ppo_full/seed_0/best.zip")
    parser.add_argument("--prior-map", type=Path, default=ROOT / "results/logs/semantic_mapping/cnn/seed_0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--camera-rate", type=float, default=1.)
    parser.add_argument("--max-route-time", type=float, default=180.)
    parser.add_argument("--hopper-capacity", type=int, default=3)
    parser.add_argument("--battery-drain", type=float, default=TaskConfig().drain_per_second,
                        help="percent per simulated second")
    parser.add_argument("--max-actions", type=int, default=60)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.camera_rate) or not 0 < args.camera_rate <= 10:
        parser.error("camera rate must be in (0, 10] Hz")
    stage8 = args.mode == "stage8"
    output = args.output or ROOT / f"results/logs/cleaning/{args.mode}_seed_0"
    output.mkdir(parents=True, exist_ok=True)

    prior = load_map(args.prior_map)
    campus = build_campus(seed=prior.seed, cleaning_tools=True)
    bridge = LiveMapBridge(prior, campus.layout, run_id=f"{args.mode}-mission-{prior.seed}-{time.time_ns()}")
    planner = AStarPlanner(bridge.map)
    pickup_config = PickupConfig()
    task_config = TaskConfig(hopper_capacity=args.hopper_capacity if stage8 else 100,
                             drain_per_second=args.battery_drain)
    tasks = TaskPlanner(planner, task_config)
    checkpoint_bytes = args.checkpoint.read_bytes()
    (output / "policy_used.zip").write_bytes(checkpoint_bytes)
    policy = load_policy(output / "policy_used.zip")
    detector = YoloDetector(device="cpu")
    pipeline = LocalizationPipeline(config=LocalizationConfig(min_score=.38))
    started = time.perf_counter()

    # Expected outcome for the evaluation: prior items with a pickup approach.
    expected, initially_unreachable = [], []
    for item in prior.active_items():
        try:
            planner.approach_item(campus.layout.base_pose[:2], item, dock_offset=pickup_config.dock_offset)
            expected.append(item["track_id"])
        except NoPathError as error:
            initially_unreachable.append({"track_id": item["track_id"], "reason": str(error)})

    stats = {"frames": 0, "detections": 0, "localization_observations": 0, "look_frames": 0,
             "detector_seconds": 0., "render_seconds": 0.}
    state = {"battery": task_config.battery_capacity, "battery_min": task_config.battery_capacity,
             "battery_time": 0.0, "charges": [], "hopper": [], "hopper_truth": [], "max_load": 0,
             "next_camera": 0.0, "aborted": None, "action": "starting"}
    deposits, actions, decisions, trajectory = [], [], [], []
    excluded = {}               # track_id -> reason (unreachable, not_found, failed)
    target_positions = {}

    with ExitStack() as resources:
        camera = resources.enter_context(RobotCamera(campus.model, campus.data))
        viewer = None
        if args.viewer:
            import mujoco.viewer
            from src.manipulation.tools import show_tools
            viewer = resources.enter_context(mujoco.viewer.launch_passive(campus.model, campus.data))
            show_tools(viewer)
            with viewer.lock():
                viewer.cam.lookat[:] = [4., 3., 0.]
                viewer.cam.distance = 9.
                viewer.cam.elevation = -50.

        def account():
            elapsed = float(campus.data.time) - state["battery_time"]
            state["battery_time"] = float(campus.data.time)
            state["battery"] -= task_config.energy(elapsed)
            state["battery_min"] = min(state["battery_min"], state["battery"])

        def process(frame):
            tick = time.perf_counter()
            detections = [d for d in detector.detect(frame.rgb) if d.score >= .38]
            stats["detector_seconds"] += time.perf_counter() - tick
            result = pipeline.update(frame, detections)
            bridge.update(pipeline.tracker.confirmed_tracks)
            stats["detections"] += len(detections)
            stats["localization_observations"] += len(result.observations)
            return result

        def draw_markers():
            if viewer is None:
                return
            with viewer.lock():
                scene = viewer.user_scn
                if scene is None:
                    return
                scene.ngeom = 0
                # Blue: pending target. Red: real litter the robot cannot reach.
                # Records rejected by the camera (not_found) or failed are not drawn.
                for item in bridge.map.active_items():
                    reason = excluded.get(item["track_id"], "")
                    if scene.ngeom >= scene.maxgeom or (reason and not reason.startswith("unreachable")):
                        continue
                    colour = [.9, .15, .1, .85] if reason else [.12, .48, 1., .3]
                    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                        np.full(3, .035), np.asarray(item["position"]), np.eye(3).ravel(),
                                        np.array(colour))
                    scene.ngeom += 1
            battery = state["battery"] - task_config.energy(float(campus.data.time) - state["battery_time"])
            capacity = task_config.hopper_capacity if stage8 else "-"
            viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT, "Mode\nAction\nHopper\nBattery\nCollected",
                              f"{args.mode}\n{state['action']}\n{len(state['hopper'])} / {capacity}\n"
                              f"{battery:.0f}%\n{sum(t.collected for t in campus.litter)}"))

        def capture():
            mujoco.mj_forward(campus.model, campus.data)
            tick = time.perf_counter()
            frame = camera.capture(segmentation=False)
            stats["render_seconds"] += time.perf_counter() - tick
            return frame

        def observe(current):
            if current.data.time < state["next_camera"]:
                return
            state["next_camera"] = float(current.data.time) + 1. / args.camera_rate
            process(capture())
            stats["frames"] += 1
            draw_markers()

        def hold(seconds):
            for _ in range(max(1, int(round(seconds / campus.model.opt.timestep)))):
                if display is not None:
                    display.update()
                mujoco.mj_step(campus.model, campus.data)
            if viewer is not None:
                viewer.sync()

        def look(position, frames=3):
            """Camera-only check: is litter still detected at `position`?"""
            in_view, measured = True, []
            for _ in range(frames):
                hold(0.25)
                frame = capture()
                u, v, depth = frame.project(np.asarray(position, dtype=float))
                k = frame.intrinsics
                in_view &= bool(depth > 0.1 and 0 <= u < k.width and 0 <= v < k.height)
                result = process(frame)
                stats["look_frames"] += 1
                for observation in result.observations:
                    if np.linalg.norm(observation.position[:2] - np.asarray(position[:2])) <= .15:
                        measured.append(observation.position)
            draw_markers()
            return {"in_view": in_view, "seen": bool(measured), "frames": frames,
                    "position": np.mean(measured, axis=0).tolist() if measured else None}

        def refresh_goal(current, approach):
            if approach.track_id == 0 or approach.track_id not in target_positions:
                return None
            item = bridge.map.item(approach.track_id)
            if np.linalg.norm(np.asarray(item["position"]) - target_positions[approach.track_id]) < .08:
                return None
            try:
                revised = planner.approach_item(current.robot_pose()[:2], item, dock_offset=pickup_config.dock_offset)
            except NoPathError:
                return None
            target_positions[approach.track_id] = np.asarray(item["position"])
            return revised

        factory = lambda route, yaw: PPOFollower(route, yaw, planner, policy)
        display = CarryDisplay(campus) if viewer is not None else None

        def tick(current):
            if display is not None:
                display.update()
            observe(current)

        skill = PickupSkill(campus, planner, pickup_config, step_hook=tick, viewer=viewer,
                            on_capture=display.pick if display is not None else None)

        def navigate(node, label, track_id=0):
            pose = campus.robot_pose()
            if math.dist(pose[:2], node.position) <= .05 and abs(wrap_angle(node.yaw - pose[2])) <= math.radians(6):
                return {"status": "success", "success": True, "already_there": True}
            try:
                route = planner.plan(pose[:2], node.position)
            except NoPathError as error:
                return {"status": "no_path", "success": False, "reason": str(error)}
            approach = ApproachPlan(track_id, node.position, node.yaw, 0., route)
            metrics, rows = execute_approach(campus, planner, approach, controller_factory=factory, reset=False,
                                             max_time=args.max_route_time, step_hook=tick,
                                             goal_provider=refresh_goal if track_id else None, viewer=viewer)
            leg = output / f"legs/{len(actions):02d}_{label}"
            save_rows(leg / "route.csv", rows)
            write_json(leg / "metrics.json", metrics)
            trajectory.extend({"x": r["x"], "y": r["y"], "action": len(actions), "label": label} for r in rows[::5])
            return metrics

        def charge():
            state["action"] = "charging"
            draw_markers()
            needed = task_config.battery_capacity - state["battery"]
            seconds = max(0.0, needed / task_config.charge_per_second)
            hold(seconds)
            state["battery_time"] = float(campus.data.time)
            state["charges"].append({"time": float(campus.data.time), "from": state["battery"],
                                     "seconds": seconds})
            state["battery"] = task_config.battery_capacity

        def do(step, reason):
            kind, node = step
            state["action"] = f"{kind} {node.name}"
            draw_markers()
            record = {"index": len(actions), "kind": kind, "target": node.name, "reason": reason,
                      "time": float(campus.data.time), "battery_before": state["battery"],
                      "load_before": len(state["hopper"])}
            if kind == "pick":
                track_id = node.item["track_id"]
                target_positions[track_id] = np.asarray(node.item["position"])
                record["navigation"] = navigate(node, node.name, track_id)
                if record["navigation"]["success"]:
                    item = bridge.map.item(track_id)
                    result = skill.pick(item["position"], look=look,
                                        refresh=lambda: bridge.map.item(track_id)["position"])
                    if result["status"] in DOCK_FAILURES and not result["captured"]:
                        # Re-approach once from a freshly planned pose, then retry.
                        record["first_pickup"] = result
                        try:
                            retry = tasks.item_node(bridge.map.item(track_id), campus.robot_pose()[:2],
                                                    pickup_config.dock_offset)
                            record["re_approach"] = navigate(retry, f"{node.name}_retry", track_id)
                        except NoPathError as error:
                            record["re_approach"] = {"status": "no_path", "success": False, "reason": str(error)}
                        if record["re_approach"]["success"]:
                            result = skill.pick(bridge.map.item(track_id)["position"], look=look,
                                                refresh=lambda: bridge.map.item(track_id)["position"])
                    record["pickup"] = result
                    state["hopper_truth"].extend(result["captured"])
                    if result["status"] == "verified_gone":
                        bridge.map.mark_collected(track_id)
                        state["hopper"].append(track_id)
                    elif result["status"] == "not_found":
                        excluded[track_id] = "not_found: no detection at the mapped position"
                    else:
                        excluded[track_id] = f"pickup failed: {result['status']}"
                    status = result["status"]
                else:
                    excluded[track_id] = f"navigation failed: {record['navigation']['status']}"
                    status = record["navigation"]["status"]
            elif kind == "dump":
                record["navigation"] = navigate(node, node.name)
                status = record["navigation"]["status"]
                if record["navigation"]["success"]:
                    state["action"] = f"dumping into {node.name}"
                    draw_markers()
                    if display is not None:
                        display.start_dump(next(b for b in bridge.map.landmarks["bins"] if b["name"] == node.name))
                    result = skill.dump(state["hopper"])
                    record["dump"] = result
                    status = result["status"]
                    if status == "success":
                        deposits.append({"bin": node.name, "map_ids": list(state["hopper"]),
                                         "objects": list(state["hopper_truth"]), "time": float(campus.data.time)})
                        state["hopper"], state["hopper_truth"] = [], []
            else:
                record["navigation"] = navigate(node, "base")
                status = record["navigation"]["status"]
            state["max_load"] = max(state["max_load"], len(state["hopper"]))
            account()
            if kind == "base" and status == "success" and reason == "battery":
                charge()
                record["charged"] = True
            record.update({"status": status, "battery_after": state["battery"],
                           "load_after": len(state["hopper"]), "duration": float(campus.data.time) - record["time"]})
            actions.append(record)
            print(f"[{campus.data.time:7.1f} s] {kind:4s} {node.name:8s} -> {status}; battery {state['battery']:5.1f}%; "
                  f"hopper {len(state['hopper'])}; frames {stats['frames']}", flush=True)
            write_json(output / "progress.json", {"actions": actions, "decisions": decisions, "excluded": excluded})
            draw_markers()
            for metrics in (record.get("navigation"), *[(a.get("dock") or {}) for a in (record.get("pickup") or {}).get("attempts", [])]):
                if metrics and metrics.get("status") in UNSAFE:
                    state["aborted"] = metrics["status"]
            if (record.get("pickup") or {}).get("status") in UNSAFE:
                state["aborted"] = record["pickup"]["status"]
            if (record.get("dump") or {}).get("status") in UNSAFE:
                state["aborted"] = record["dump"]["status"]

        observe(campus)
        finished = False
        while len(actions) < args.max_actions and state["aborted"] is None:
            pose = campus.robot_pose()
            nodes = []
            for item in bridge.map.active_items():
                if item["track_id"] in excluded:
                    continue
                try:
                    nodes.append(tasks.item_node(item, pose[:2], pickup_config.dock_offset))
                except NoPathError as error:
                    excluded[item["track_id"]] = f"unreachable: {error}"
            if stage8:
                plan = tasks.plan(pose[:2], nodes, load=len(state["hopper"]))
                step = plan["steps"][0]
            else:
                plan = tasks.plan(pose[:2], [], load=len(state["hopper"])) if not nodes else None
                step = (("pick", min(nodes, key=lambda n: n.item["track_id"])) if nodes else plan["steps"][0])
            decision = {"time": float(campus.data.time), "battery": state["battery"], "load": len(state["hopper"]),
                        "candidates": [n.name for n in nodes], "chosen": f"{step[0]} {step[1].name}"}
            if plan is not None:
                decision["plan"] = [f"{k} {n.name}" for k, n in plan["steps"]]
                decision["estimated_seconds"] = plan["estimated_seconds"]
            reason = "plan" if stage8 else "fixed track-ID order"
            if stage8 and step[0] != "base" and not tasks.can_afford(state["battery"], pose[:2], step):
                if state["battery"] >= task_config.battery_capacity - 1e-6:
                    # Even a full battery cannot cover this task and the trip home.
                    if step[0] == "pick":
                        excluded[step[1].item["track_id"]] = "beyond battery range"
                    decisions.append({**decision, "chosen": "exclude (beyond battery range)"})
                    continue
                step, reason = ("base", tasks.base), "battery"
                decision["chosen"] = "base (battery)"
            decisions.append(decision)
            do(step, reason)
            if step[0] == "base" and reason != "battery":
                finished = not nodes
                if finished:
                    break

    account()
    save_map(bridge.map, output / "final_map")
    write_json(output / "trajectory.json", trajectory)

    # ------------------------------------------------ post-mission evaluation
    collected_truth = [t.name for t in campus.litter if t.collected]
    expected_truth = []
    for track_id in expected:
        item = prior.item(track_id)
        truth = match_truth(campus, item["position"], item["class"])
        if truth is not None:
            expected_truth.append(truth.name)
    map_collected = [i for i in bridge.map.items.values() if i["collected"]]
    consistent = []
    for item in map_collected:
        truth = match_truth(campus, item["position"], radius=.15)
        consistent.append(truth is not None and truth.collected)
    deposited = [name for d in deposits for name in d["objects"]]
    final_pose = campus.robot_pose()
    statuses = [a["status"] for a in actions]
    checks = {
        "all_reachable_litter_collected": bool(expected_truth) and set(expected_truth) <= set(collected_truth),
        "map_collection_matches_simulator": all(consistent) and len(map_collected) == len(collected_truth),
        "hopper_emptied_into_bins": not state["hopper_truth"] and sorted(deposited) == sorted(collected_truth),
        "returned_to_base": finished and math.dist(final_pose[:2], campus.layout.base_pose[:2]) <= .1,
        "no_unsafe_events": state["aborted"] is None and not any(s in UNSAFE for s in statuses),
        "source_prior_map_unchanged": prior.items == load_map(args.prior_map).items,
    }
    if stage8:
        checks["hopper_capacity_respected"] = state["max_load"] <= task_config.hopper_capacity
        checks["battery_never_empty"] = state["battery_min"] > 0
    report = {
        "mission_passed": all(checks.values()), "checks": checks, "mode": args.mode, "scope": __doc__,
        "scene_seed": prior.seed, "camera_hz": args.camera_rate, "viewer_enabled": args.viewer,
        "expected_map_ids": expected, "initially_unreachable": initially_unreachable,
        "collected_map_ids": [i["track_id"] for i in map_collected], "collected_objects": collected_truth,
        "expected_objects": expected_truth, "excluded": {str(k): v for k, v in excluded.items()},
        "deposits": deposits, "charges": state["charges"], "battery_min": state["battery_min"],
        "battery_final": state["battery"], "max_hopper_load": state["max_load"],
        "pickups": sum(a["kind"] == "pick" for a in actions),
        "pickup_attempts": sum(len((a.get("pickup") or {}).get("attempts", [])) for a in actions),
        "actions": actions, "decisions": decisions, "online": stats, "bridge": bridge.summary(),
        "task_config": asdict(task_config), "pickup_config": asdict(pickup_config),
        "final_pose": list(final_pose), "aborted": state["aborted"],
        "ppo_checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "detector_weights_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__), ROOT / "src/manipulation/pickup.py", ROOT / "src/tasks/planner.py",
                                    ROOT / "src/navigation/planner.py", ROOT / "src/navigation/evaluation.py")},
        "total_simulated_seconds": float(campus.data.time), "total_wall_seconds": time.perf_counter() - started}
    write_json(output / "results.json", report)
    print(f"{args.mode} mission {'PASS' if report['mission_passed'] else 'FAIL'}: "
          f"{len(collected_truth)}/{len(expected_truth)} expected items collected, {len(deposits)} bin dumps, "
          f"{len(state['charges'])} charges, min battery {state['battery_min']:.1f}%, "
          f"{campus.data.time:.0f} s simulated", flush=True)
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}", flush=True)
    return 1 if args.require_pass and not report["mission_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
