"""Run Stage 4 on the existing campus loop and evaluate both estimators."""

from dataclasses import dataclass
from contextlib import ExitStack
import math
import time

import numpy as np

from src.localization.evaluation import LocalizationEvaluator
from src.localization.geometry import LocalizationConfig
from src.localization.pipeline import LocalizationPipeline


@dataclass
class RouteResult:
    campus: object
    pipelines: dict
    evaluators: dict
    metrics: dict
    map_matches: dict
    path: np.ndarray
    video: list
    route: dict


def run_localization_route(
    detector, seed=0, config=None, methods=("depth", "ground_plane"),
    rate=5.0, max_time=120.0, association_radius=0.20, min_observations=3,
    match_radius=0.25, target_error_cm=10.0, save_video=False, class_switch_radius=0.08,
    show_viewer=False,
):
    """detector=None is a labelled-box diagnostic, never a detector benchmark."""

    import mujoco
    from src.controllers.locomotion_interface import LocomotionInterface
    from src.environment import build_campus
    from src.environment.litter import LITTER_CLASSES
    from src.navigation.waypoint_follower import DEMO_ROUTE, HeadingFollower, reached
    from src.perception.camera import RobotCamera
    from src.perception.detection import Detection
    from src.perception.labels import ground_truth_boxes, object_geom_ids
    from src.utils.orientation import quaternion_to_euler

    if not math.isfinite(rate) or rate <= 0 or not math.isfinite(max_time) or max_time <= 0:
        raise ValueError("rate and max_time must be positive and finite")
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("choose at least one distinct localization method")
    config = config or LocalizationConfig()
    campus = build_campus(seed=seed)
    model, data = campus.model, campus.data
    geom_map = object_geom_ids(model, campus.litter + campus.distractors)
    objects = [{
        "name": o.name, "class_id": LITTER_CLASSES.index(o.kind), "position": o.position.copy(),
    } for o in campus.litter]
    pipelines = {m: LocalizationPipeline(m, config, association_radius, min_observations, class_switch_radius) for m in methods}
    evaluators = {m: LocalizationEvaluator(objects, match_radius, target_error_cm) for m in methods}
    loco, follower = LocomotionInterface(), HeadingFollower()
    waypoint, frames = 0, 0
    next_frame, detector_seconds = 0.0, 0.0
    path, video = [], []
    fall = False
    started = time.perf_counter()

    with ExitStack() as resources:
        camera = resources.enter_context(RobotCamera(model, data))
        viewer = None
        if show_viewer:
            import mujoco.viewer
            viewer = resources.enter_context(mujoco.viewer.launch_passive(model, data))
            with viewer.lock():
                viewer.cam.lookat[:] = [4.0, 3.0, 0.0]
                viewer.cam.distance = 9.0
                viewer.cam.elevation = -50.0
                viewer.cam.azimuth = 90.0
            viewer.sync()
            print("Stage 4 viewer opened; blue markers show confirmed litter positions. Close it to stop.", flush=True)
        viewer_started, next_sync = time.perf_counter(), 0.0
        while waypoint < len(DEMO_ROUTE) and data.time < max_time:
            if viewer is not None:
                if not viewer.is_running():
                    break
                if data.time >= next_sync:
                    viewer.sync()
                    next_sync = data.time + 1.0 / 30.0
                remaining = data.time - (time.perf_counter() - viewer_started)
                if remaining > 0:
                    time.sleep(remaining)
            x, y, yaw = campus.robot_pose()
            if reached(x, y, DEMO_ROUTE[waypoint]):
                waypoint += 1
                continue
            loco.set_velocity(*follower.command(x, y, yaw, DEMO_ROUTE[waypoint]))
            data.ctrl[:] = loco.update(data, model.opt.timestep)
            mujoco.mj_step(model, data)
            px, py, z = data.qpos[:3]
            roll, pitch, _ = quaternion_to_euler(data.qpos[3:7])
            if z - campus.layout.ground_height_at(px, py) < 0.20 or max(abs(roll), abs(pitch)) > math.radians(30):
                fall = True
                break
            if data.time < next_frame:
                continue
            next_frame += 1.0 / rate
            # mj_step integrates qpos after computing kinematics. Refresh before
            # rendering so RGB, depth, camera pose and timestamp describe one state.
            mujoco.mj_forward(model, data)
            frame = camera.capture()
            truths = [b for b in ground_truth_boxes(frame, geom_map) if b.category == "litter"]
            tick = time.perf_counter()
            if detector is None:
                detections = [Detection(b.class_id, 1.0, b.box) for b in truths]
            else:
                detections = [d for d in detector.detect(frame.rgb) if d.score >= config.min_score]
            detector_seconds += time.perf_counter() - tick
            for method, pipeline in pipelines.items():
                result = pipeline.update(frame, detections)
                evaluators[method].observe(frame, detections, truths, result)
            if viewer is not None:
                display_pipeline = pipelines.get("depth", next(iter(pipelines.values())))
                tracks = display_pipeline.tracker.confirmed_tracks
                with viewer.lock():
                    scene = viewer.user_scn
                    if scene is not None:
                        scene.ngeom = 0
                        for track in tracks[:scene.maxgeom]:
                            mujoco.mjv_initGeom(
                                scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.full(3, .035), track.position, np.eye(3).ravel(),
                                np.array([.12, .48, 1., .8]),
                            )
                            scene.ngeom += 1
                if hasattr(viewer, "set_texts"):
                    viewer.set_texts((None, None, "Stage 4 localization",
                                     f"t={frame.time:.1f}s  waypoints={waypoint}/{len(DEMO_ROUTE)}  tracks={len(tracks)}"))
            path.append((float(px), float(py)))
            frames += 1
            if save_video:
                from PIL import Image, ImageDraw
                image = Image.fromarray(frame.rgb)
                draw = ImageDraw.Draw(image)
                for d in detections:
                    draw.rectangle(d.box, outline="#2a78d6", width=2)
                counts = ", ".join(f"{m}: {len(p.tracker.confirmed_tracks)}" for m, p in pipelines.items())
                draw.rectangle((0, 0, 640, 28), fill="black")
                draw.text((8, 8), f"t={frame.time:.1f}s  tracks {counts}", fill="white")
                video.append(image.resize((320, 240)))
            if frames % 100 == 0:
                print(f"seed {seed}: {frames} frames, {data.time:.1f}s simulated, waypoint {waypoint}/{len(DEMO_ROUTE)}", flush=True)
    loco.stop()
    data.ctrl[:] = loco.update(data, model.opt.timestep)
    route = {
        "seed": seed, "waypoints_reached": waypoint, "waypoints_total": len(DEMO_ROUTE),
        "completed": waypoint == len(DEMO_ROUTE) and not fall, "fall": fall,
        "simulation_seconds": float(data.time), "frames": frames,
        "wall_seconds": time.perf_counter() - started, "detector_seconds": detector_seconds,
        "camera_rate_hz": rate,
        "viewer_enabled": show_viewer,
    }
    metrics, map_matches = {}, {}
    for method, pipeline in pipelines.items():
        report, matches = evaluators[method].summarize(pipeline.tracker.confirmed_tracks)
        report["rejected"] = dict(pipeline.rejection_counts)
        report["route"] = route
        report["stage4_passed"] = bool(route["completed"] and report["map"]["passed"])
        metrics[method], map_matches[method] = report, matches
    return RouteResult(campus, pipelines, evaluators, metrics, map_matches, np.asarray(path, dtype=float).reshape(-1, 2), video, route)
