"""Continuous Stage 6 mission with real CNN/RGB-D updates and PPO navigation.

python -B experiments/31_live_navigation.py --require-pass
The prior Stage 5 map is starting knowledge. Online tracks have a fresh
namespace and are reconciled explicitly. Targets are visited in fixed ID order
for testing; neither collection nor Stage 8 task optimization is performed.
"""

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.environment import build_campus
from src.mapping import load_map, save_map
from src.localization.geometry import LocalizationConfig
from src.localization.pipeline import LocalizationPipeline
from src.navigation import AStarPlanner, ApproachPlan, NoPathError
from src.navigation.evaluation import execute_approach
from src.navigation.live_map import LiveMapBridge
from src.navigation.ppo_navigator import PPOFollower, load_policy


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def evaluate_map(semantic_map, campus):
    """Simulator object truth is read only after the navigation mission."""
    from scipy.optimize import linear_sum_assignment
    items = semantic_map.active_items()
    truth = campus.litter
    cost = np.full((len(items), len(truth)), 999.)
    for i, item in enumerate(items):
        for j, actual in enumerate(truth):
            if item["class"] == actual.kind:
                cost[i, j] = np.linalg.norm(np.asarray(item["position"]) - actual.position)
    matches = []
    for i, j in zip(*linear_sum_assignment(cost)):
        if cost[i, j] <= .25:
            matches.append({"map_id": items[i]["track_id"], "truth_name": truth[j].name,
                            "error_cm": float(cost[i, j] * 100)})
    return {"scope": "post-mission all-scene-object diagnostic; prior knowledge included",
            "map_items": len(items), "scene_items": len(truth), "matched": len(matches),
            "unmatched_map_records": len(items) - len(matches), "unmatched_scene_items": len(truth) - len(matches),
            "max_error_cm": max((m["error_cm"] for m in matches), default=None), "matches": matches}


def main():
    import mujoco
    import torch
    from src.perception.camera import RobotCamera
    from src.perception.yolo_detector import YoloDetector
    from src.perception.yolo_detector import MODEL_PATH
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/navigation/ppo_full/seed_0/best.zip")
    parser.add_argument("--prior-map", type=Path, default=ROOT / "results/logs/semantic_mapping/cnn/seed_0")
    parser.add_argument("--output", type=Path, default=ROOT / "results/logs/navigation/live/seed_0")
    parser.add_argument("--camera-rate", type=float, default=3.)
    parser.add_argument("--max-route-time", type=float, default=180.)
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.camera_rate) or args.camera_rate <= 0 or args.camera_rate > 10:
        parser.error("camera rate must be in (0, 10] Hz")
    if args.max_targets is not None and args.max_targets < 1:
        parser.error("max-targets must be positive")
    prior = load_map(args.prior_map)
    campus = build_campus(seed=prior.seed)
    bridge = LiveMapBridge(prior, campus.layout, run_id=f"stage6-live-{prior.seed}-{time.time_ns()}")
    planner = AStarPlanner(bridge.map)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_bytes = args.checkpoint.read_bytes()
    frozen_checkpoint = args.output / "policy_used.zip"
    frozen_checkpoint.write_bytes(checkpoint_bytes)
    policy = load_policy(frozen_checkpoint)
    detector = YoloDetector(device="cpu")
    config = LocalizationConfig(min_score=.38)
    pipeline = LocalizationPipeline(config=config)
    stats = {"frames": 0, "detections": 0, "localization_observations": 0,
             "detector_seconds": 0., "render_seconds": 0., "updates": []}
    started = time.perf_counter()
    next_camera = 0.
    visited, unreachable, routes, target_positions = [], [], [], {}
    bootstrap_ids = [item["track_id"] for item in prior.active_items()]
    if args.max_targets is not None:
        bootstrap_ids = bootstrap_ids[:args.max_targets]
    with ExitStack() as resources:
        camera = resources.enter_context(RobotCamera(campus.model, campus.data))
        viewer = None
        if args.viewer:
            import mujoco.viewer
            viewer = resources.enter_context(mujoco.viewer.launch_passive(campus.model, campus.data))
            with viewer.lock():
                viewer.cam.lookat[:] = [4., 3., 0.]
                viewer.cam.distance = 9.
                viewer.cam.elevation = -50.
        def observe(current):
            nonlocal next_camera
            if current.data.time < next_camera:
                return
            next_camera += 1. / args.camera_rate
            # Synchronize camera pose with the integrated robot state.
            mujoco.mj_forward(current.model, current.data)
            tick = time.perf_counter()
            frame = camera.capture(segmentation=False)
            stats["render_seconds"] += time.perf_counter() - tick
            tick = time.perf_counter()
            detections = [d for d in detector.detect(frame.rgb) if d.score >= .38]
            stats["detector_seconds"] += time.perf_counter() - tick
            result = pipeline.update(frame, detections)
            count = bridge.update(pipeline.tracker.confirmed_tracks)
            stats["frames"] += 1
            stats["detections"] += len(detections)
            stats["localization_observations"] += len(result.observations)
            stats["updates"].append({"time": float(frame.time), "detections": len(detections),
                                     "confirmed_tracks": len(pipeline.tracker.confirmed_tracks),
                                     "map_records_updated": count, "robot_pose": list(current.robot_pose())})
            if viewer is not None:
                with viewer.lock():
                    scene = viewer.user_scn
                    if scene is not None:
                        scene.ngeom = 0
                        for item in bridge.map.active_items()[:scene.maxgeom]:
                            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                                np.full(3, .035), np.asarray(item["position"]), np.eye(3).ravel(),
                                                np.array([.12, .48, 1., .8]))
                            scene.ngeom += 1
        def refresh_goal(current, approach):
            if approach.track_id == 0:
                return None
            item = bridge.map.item(approach.track_id)
            previous = target_positions[approach.track_id]
            if np.linalg.norm(np.asarray(item["position"]) - previous) < .08:
                return None
            try:
                revised = planner.approach_item(current.robot_pose()[:2], item)
            except NoPathError:
                return None
            target_positions[approach.track_id] = np.asarray(item["position"])
            return revised
        factory = lambda route, yaw: PPOFollower(route, yaw, planner, policy)
        observe(campus)
        for map_id in bootstrap_ids:
            item = bridge.map.item(map_id)
            target_positions[map_id] = np.asarray(item["position"])
            try:
                approach = planner.approach_item(campus.robot_pose()[:2], item)
            except NoPathError as error:
                unreachable.append({"track_id": map_id, "reason": str(error)})
                print(f"Live target {map_id}: unreachable", flush=True)
                continue
            metrics, rows = execute_approach(campus, planner, approach, controller_factory=factory,
                                             reset=False, max_time=args.max_route_time, step_hook=observe,
                                             goal_provider=refresh_goal, viewer=viewer)
            metrics.update({"class": item["class"], "target_source": "prior CNN map, refined by online RGB-D/CNN tracks"})
            routes.append(metrics)
            # Route CSV format is shared with the baseline, without importing its CLI.
            import csv
            folder = args.output / f"track_{map_id}"
            folder.mkdir(parents=True, exist_ok=True)
            if rows:
                with (folder / "route.csv").open("w", newline="", encoding="utf-8") as output:
                    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            write_json(folder / "metrics.json", metrics)
            if metrics["success"]:
                visited.append(map_id)
            print(f"Live target {map_id}: {metrics['status']}; frames {stats['frames']}; confirmed {len(pipeline.tracker.confirmed_tracks)}", flush=True)
            write_json(args.output / "progress.json", {"routes": routes, "unreachable": unreachable, "stats": stats, "bridge": bridge.summary()})
            if not metrics["success"]:
                break
        return_metrics = None
        if routes and all(route["success"] for route in routes):
            route = planner.plan(campus.robot_pose()[:2], campus.layout.base_pose[:2])
            base = ApproachPlan(0, campus.layout.base_pose[:2], campus.layout.base_pose[2], 0., route)
            return_metrics, rows = execute_approach(campus, planner, base, controller_factory=factory,
                                                    reset=False, max_time=args.max_route_time, step_hook=observe, viewer=viewer)
            write_json(args.output / "return_to_base/metrics.json", return_metrics)
            folder = args.output / "return_to_base"
            if rows:
                with (folder / "route.csv").open("w", newline="", encoding="utf-8") as output:
                    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            print(f"Live return to base: {return_metrics['status']}", flush=True)
    save_map(bridge.map, args.output / "final_map")
    final_map_diagnostic = evaluate_map(bridge.map, campus)
    checks = {"all_prior_targets_classified": len(routes) + len(unreachable) == len(bootstrap_ids),
              "all_reachable_targets_visited": bool(routes) and all(r["success"] for r in routes),
              "return_to_base": return_metrics is not None and return_metrics["success"],
              "actual_online_mapping_updates": bridge.records_updated > 0 and len(pipeline.tracker.confirmed_tracks) > 0,
              "continuous_execution_without_reset": all(not r["reset_before_route"] for r in routes),
              "source_prior_map_unchanged": prior.items == load_map(args.prior_map).items,
              "no_pickup_side_effects": all(not item.collected for item in campus.litter)}
    report = {"mission_passed": all(checks.values()), "checks": checks,
              "scope": "continuous PPO navigation with live RGB-D/CNN updates; prior map starting knowledge; simulator pose; no pickup",
              "scene_seed": prior.seed, "camera_hz": args.camera_rate, "viewer_enabled": args.viewer,
              "requested_map_ids": bootstrap_ids, "visited_map_ids": visited, "unreachable": unreachable,
              "routes": routes, "return_to_base": return_metrics, "online": stats, "bridge": bridge.summary(),
              "post_mission_map_diagnostic": final_map_diagnostic,
              "ppo_checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
              "detector_weights_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (Path(__file__), ROOT / "src/navigation/live_map.py", ROOT / "src/navigation/evaluation.py")},
              "total_simulated_seconds": float(campus.data.time), "total_wall_seconds": time.perf_counter() - started}
    write_json(args.output / "results.json", report)
    print(f"Live mission {'PASS' if report['mission_passed'] else 'FAIL'}; {len(visited)} visited, {len(unreachable)} unreachable, {stats['frames']} camera frames", flush=True)
    return 1 if args.require_pass and not report["mission_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

