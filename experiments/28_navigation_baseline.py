"""Stage 6 configuration 1: A* + classical follower, without RL or pickup.

python -B experiments/28_navigation_baseline.py --require-pass
python -B experiments/28_navigation_baseline.py --offline-only --require-pass

Default inputs are the saved Stage 5 seed-0 CNN map. Each item route starts at
base independently. Additional seeds, when requested, use explicitly labelled
true-position diagnostic targets; they are not CNN/map validation results.
"""

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.environment.layout import default_layout
from src.mapping import SemanticMap, load_map
from src.navigation import AStarPlanner, ApproachPlan, NavigationConfig, NoPathError


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def offline_checks(planner, layout, approaches):
    checks = {"positive_robot_padding": planner.config.clearance_radius > 0,
              "at_least_one_reachable_approach": bool(approaches),
              "all_path_segments_clear": all(planner.segment_is_free(a, b)
                  for p in approaches for a, b in zip(p.route.waypoints, p.route.waypoints[1:])),
              "all_goals_clear": all(planner.is_free(p.position) for p in approaches)}
    checks["ramp_passage"] = True
    try:
        planner.plan(layout.base_pose[:2], (4, 2.7))
    except NoPathError:
        checks["ramp_passage"] = False
    closed_layout = replace(layout, traversable={**layout.traversable, "ramp": False})
    closed_map = SemanticMap.from_layout(closed_layout, seed=planner.map.seed, run_id="closed-ramp")
    closed = AStarPlanner(closed_map, planner.config)
    checks["unreachable_when_ramp_closed"] = False
    try:
        closed.plan(layout.base_pose[:2], (4, 2.7))
    except NoPathError:
        checks["unreachable_when_ramp_closed"] = True
    checks["occupied_goal_rejected"] = False
    try:
        planner.plan(layout.base_pose[:2], layout.obstacles[0].shape.center)
    except NoPathError:
        checks["occupied_goal_rejected"] = True
    return {name: bool(value) for name, value in checks.items()}


def save_route(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(path, layout, planner, cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.environment.plotting import plot_campus
    count = len(cases)
    columns = min(4, max(1, count))
    rows = max(1, (count + columns - 1) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4.2 * rows), squeeze=False,
                             layout="constrained")
    for ax, case in zip(axes.flat, cases):
        plot_campus(ax, layout)
        if ax.get_legend():
            ax.get_legend().remove()
        overlay = np.zeros((*planner.grid.shape, 4))
        overlay[planner.blocked & ~planner.raw_blocked] = (.65, .1, .8, .16)
        ax.imshow(overlay, origin="lower", extent=planner.grid.extent, interpolation="nearest", zorder=2)
        route = np.asarray(case["waypoints"])
        ax.plot(route[:, 0], route[:, 1], "--", color="#147ad6", lw=1.5, label="planned")
        actual = case.get("path", [])
        if actual:
            ax.plot([r["x"] for r in actual], [r["y"] for r in actual], color="#ef7d00", lw=1.2, label="actual")
        ax.plot(*case["goal_position"], "x", color="black", ms=7, label="approach goal")
        ax.plot(*case["item_position"][:2], "*", color="#dc2222", ms=9, label="mapped litter")
        ax.set_title(f"Track {case['track_id']} ({case['class']}) | {case['status']}")
        ax.legend(loc="upper left", fontsize=7)
    for ax in list(axes.flat)[count:]:
        ax.set_visible(False)
    fig.suptitle(f"Stage 6 A* baseline | clearance {planner.config.clearance_radius:.2f} m | independent base-to-item routes")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map", type=Path, default=ROOT / "results/logs/semantic_mapping/cnn/seed_0")
    parser.add_argument("--output", type=Path, default=ROOT / "results/logs/navigation/astar/seed_0")
    parser.add_argument("--clearance-radius", type=float, default=.45)
    parser.add_argument("--track-ids", nargs="+", type=int)
    parser.add_argument("--diagnostic-seeds", nargs="*", type=int, default=[])
    parser.add_argument("--max-time", type=float, default=180.)
    parser.add_argument("--offline-only", action="store_true")
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.max_time) or args.max_time <= 0:
        parser.error("--max-time must be positive and finite")
    started = time.perf_counter()
    try:
        config = NavigationConfig(clearance_radius=args.clearance_radius)
        semantic_map = load_map(args.map)
        layout = default_layout()
        if semantic_map.grid.width != layout.width or semantic_map.grid.depth != layout.depth:
            raise ValueError("this runner requires the default campus layout")
        reference = SemanticMap.from_layout(layout, seed=semantic_map.seed, run_id="layout-check",
                                            resolution=semantic_map.grid.resolution)
        if any(not np.array_equal(semantic_map.layers[name], reference.layers[name])
               for name in ("terrain", "height", "occupancy", "forbidden", "boundary")):
            raise ValueError("saved map static layers do not match the simulated campus")
        planner = AStarPlanner(semantic_map, config)
        items = semantic_map.active_items()
        if args.track_ids:
            if len(set(args.track_ids)) != len(args.track_ids) or not set(args.track_ids).issubset({i["track_id"] for i in items}):
                raise ValueError("--track-ids must contain distinct active map identities")
            selected = set(args.track_ids)
            items = [item for item in items if item["track_id"] in selected]
        approaches, failures = [], []
        for item in items:
            try:
                approaches.append(planner.approach_item(layout.base_pose[:2], item))
            except NoPathError as error:
                failures.append({"track_id": item["track_id"], "reason": str(error)})
        checks = offline_checks(planner, layout, approaches)
        checks["all_items_classified"] = len(approaches) + len(failures) == len(items)
    except (ValueError, IndexError, KeyError, OSError) as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {"algorithm": "A* + stop/turn/drive follower", "rl_training": False,
                "scope": "offline saved-CNN-map navigation; simulator robot pose; no live mapping or pickup",
                "seed": semantic_map.seed, "run_id": semantic_map.run_id,
                "navigation_config": asdict(config), "selected_track_ids": [i["track_id"] for i in items],
                "map_json_sha256": hashlib.sha256((args.map / "map.json").read_bytes()).hexdigest(),
                "layers_sha256": hashlib.sha256((args.map / "layers.npz").read_bytes()).hexdigest(),
                "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in (Path(__file__), ROOT / "src/navigation/planner.py",
                                               ROOT / "src/navigation/follower.py", ROOT / "src/navigation/evaluation.py")},
                "python": platform.python_version(), "offline_only": args.offline_only}
    write_json(args.output / "plans.json", {"config": asdict(config), "approaches": [asdict(p) for p in approaches],
                                           "failures": failures, "checks": checks})
    cases, footprint, diagnostics, terrain_probes = [], None, [], []
    if not args.offline_only:
        import mujoco
        from src.environment import build_campus
        from src.navigation.evaluation import execute_approach, measure_footprint
        metadata["mujoco_version"] = mujoco.__version__
        campus = build_campus(seed=semantic_map.seed)
        footprint = measure_footprint(campus)
        checks["sampled_footprint_plus_margin"] = footprint["max_projected_radius_m"] + .04 <= config.clearance_radius
        print(f"Footprint radius bound {footprint['max_projected_radius_m']:.3f} m; planning radius {config.clearance_radius:.3f} m", flush=True)
        if checks["sampled_footprint_plus_margin"]:
            for approach in approaches:
                metrics, rows = execute_approach(campus, planner, approach, max_time=args.max_time)
                item = semantic_map.item(approach.track_id)
                metrics.update({"class": item["class"], "item_position": item["position"], "target_source": "Stage 4 CNN depth track"})
                cases.append(metrics)
                save_route(args.output / f"track_{approach.track_id}/route.csv", rows)
                write_json(args.output / f"track_{approach.track_id}/metrics.json", metrics)
                metrics["path"] = rows
                print(f"Track {approach.track_id}: {metrics['status']}; error {metrics['final_position_error_m']:.3f} m; sim {metrics['simulated_seconds']:.1f} s; wall {metrics['wall_seconds']:.1f} s", flush=True)
            # Separate downhill execution diagnostic. This spawn does not
            # claim a continuous multi-item mission or a perception result.
            downhill_start = (4., 2.9, -math.pi / 2)
            downhill_route = planner.plan(downhill_start[:2], layout.base_pose[:2])
            downhill = ApproachPlan(0, layout.base_pose[:2], layout.base_pose[2], 0., downhill_route)
            metrics, rows = execute_approach(campus, planner, downhill, max_time=args.max_time,
                                             start_pose=downhill_start)
            metrics["target_source"] = "known-pose downhill-to-base diagnostic"
            terrain_probes.append(metrics)
            write_json(args.output / "terrain_probes/downhill/metrics.json", metrics)
            save_route(args.output / "terrain_probes/downhill/route.csv", rows)
            checks["downhill_route_successful"] = metrics["success"]
            print(f"Downhill-to-base diagnostic: {metrics['status']}", flush=True)
            for seed in args.diagnostic_seeds:
                other = build_campus(seed=seed)
                diagnostic_map = SemanticMap.from_layout(layout, seed=seed, run_id=f"truth-diagnostic-{seed}")
                diagnostic_planner = AStarPlanner(diagnostic_map, config)
                for index, target in enumerate(other.litter, 1):
                    item = {"track_id": index, "position": target.position.tolist(), "collected": False}
                    try:
                        approach = diagnostic_planner.approach_item(layout.base_pose[:2], item)
                        metrics, rows = execute_approach(other, diagnostic_planner, approach, max_time=args.max_time)
                        metrics.update({"seed": seed, "class": target.kind, "item_position": item["position"],
                                        "target_source": "simulator true-position diagnostic; not CNN validation"})
                        save_route(args.output / f"diagnostics/seed_{seed}/item_{index}/route.csv", rows)
                    except NoPathError as error:
                        metrics = {"seed": seed, "track_id": index, "success": False,
                                   "status": "unreachable", "reason": str(error), "target_source": "simulator true-position diagnostic"}
                    diagnostics.append(metrics)
                    write_json(args.output / f"diagnostics/seed_{seed}/item_{index}/metrics.json", metrics)
                    print(f"Diagnostic seed {seed}, item {index}: {metrics['status']}", flush=True)
            if args.diagnostic_seeds:
                checks["diagnostic_reachable_routes_successful"] = bool(diagnostics) and all(
                    c["status"] in ("success", "unreachable") for c in diagnostics)
        checks["all_reachable_routes_successful"] = bool(cases) and len(cases) == len(approaches) and all(c["success"] for c in cases)
    if not args.no_figure and approaches:
        plotted = cases if cases else [{"track_id": p.track_id, "waypoints": p.route.waypoints,
                                       "goal_position": p.position, "item_position": semantic_map.item(p.track_id)["position"],
                                       "class": semantic_map.item(p.track_id)["class"], "status": "planned"} for p in approaches]
        save_plot(args.output / "routes.png", layout, planner, plotted)
    summary_cases = [{key: value for key, value in c.items() if key != "path"} for c in cases]
    result = {"baseline_passed": all(checks.values()), "execution_validated": not args.offline_only and bool(cases),
              "checks": checks, "configuration": metadata,
              "footprint": footprint, "planning_failures": failures, "routes": summary_cases,
              "diagnostics": diagnostics, "terrain_probes": terrain_probes,
              "routes_successful": sum(c["success"] for c in cases),
              "routes_attempted": len(cases), "mapped_targets": len(items),
              "reachable_targets": len(approaches), "unreachable_targets": len(failures),
              "total_wall_seconds": time.perf_counter() - started,
              "remaining_scope": ["live perception/map updates", "continuous multi-item mission", "PPO navigator", "physical pickup"]}
    write_json(args.output / "results.json", result)
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    print(f"Saved results to {args.output}", flush=True)
    return 1 if args.require_pass and not result["baseline_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
