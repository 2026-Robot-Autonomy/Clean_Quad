"""Stage 7 pickup-skill trials: dock, crouch, suction, back off (headless).

python -B experiments/33_pickup_trials.py --require-pass

Each trial starts the robot standing at a planned pickup approach pose,
optionally perturbed sideways and in heading, then runs PickupSkill.pick on
the believed item position. Scene 0 uses the saved Stage 5 CNN map positions
(real perception error). Scenes 1/2 have no CNN map, so beliefs are the
simulator positions plus seeded Gaussian noise; that is a skill diagnostic,
not a perception result. There is no camera here, so the outcome is the
simulator's suction result; live verification is in 34_cleaning_mission.py.
"""

import argparse
from collections import defaultdict
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
from src.mapping import SemanticMap, load_map
from src.navigation import AStarPlanner, NoPathError
from src.manipulation.pickup import PickupConfig, PickupSkill

# (name, sideways offset [m], heading offset [deg]) applied to the start pose.
PERTURBATIONS = (("nominal", 0.0, 0.0), ("left", 0.04, 4.0), ("right", -0.04, -4.0))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def settle(campus, seconds=0.5):
    import mujoco
    from src.controllers.locomotion_interface import LocomotionInterface
    loco = LocomotionInterface()
    for _ in range(int(seconds / campus.model.opt.timestep)):
        campus.data.ctrl[:] = loco.update(campus.data, campus.model.opt.timestep)
        mujoco.mj_step(campus.model, campus.data)


def scene_items(campus, scene_seed, noise):
    if scene_seed == 0:
        semantic_map = load_map(ROOT / "results/logs/semantic_mapping/cnn/seed_0")
        return semantic_map, semantic_map.active_items(), "saved Stage 4 CNN depth tracks"
    semantic_map = SemanticMap.from_layout(campus.layout, seed=scene_seed, run_id=f"pickup-diagnostic-{scene_seed}")
    rng = np.random.default_rng(1000 + scene_seed)
    items = []
    for index, truth in enumerate(campus.litter, 1):
        position = truth.position.copy()
        position[:2] += rng.normal(0.0, noise, 2)
        items.append({"track_id": index, "class": truth.kind, "position": position.tolist()})
    return semantic_map, items, f"simulator positions + N(0, {noise} m) noise"


def target_truth(campus, belief):
    return min(campus.litter, key=lambda t: math.dist(t.position[:2], belief[:2]))


def main():
    from src.simulation.robot_setup import reset_robot
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--noise", type=float, default=0.03, help="belief noise std for scenes without a CNN map [m]")
    parser.add_argument("--output", type=Path, default=ROOT / "results/logs/pickup/trials")
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    config = PickupConfig()
    started = time.perf_counter()
    trials, unreachable, skipped = [], [], []
    for scene_seed in args.scene_seeds:
        campus = build_campus(seed=scene_seed, cleaning_tools=True)
        semantic_map, items, source = scene_items(campus, scene_seed, args.noise)
        planner = AStarPlanner(semantic_map)
        for item in items:
            try:
                approach = planner.approach_item(campus.layout.base_pose[:2], item, dock_offset=config.dock_offset)
            except NoPathError as error:
                unreachable.append({"scene_seed": scene_seed, "track_id": item["track_id"],
                                    "class": item["class"], "reason": str(error)})
                print(f"Scene {scene_seed} item {item['track_id']}: unreachable", flush=True)
                continue
            for name, sideways, turn in PERTURBATIONS:
                campus.reset()
                left = (-math.sin(approach.yaw), math.cos(approach.yaw))
                x = approach.position[0] + sideways * left[0]
                y = approach.position[1] + sideways * left[1]
                # A valid start keeps the 5 cm margin a planned arrival at a
                # free cell centre has; the gait start-up can shift the trunk.
                margin = [(x + .05 * math.cos(a), y + .05 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
                if not all(planner.is_free(point) for point in [(x, y)] + margin):
                    skipped.append({"scene_seed": scene_seed, "track_id": item["track_id"], "perturbation": name})
                    continue
                reset_robot(campus.model, campus.data, x=x, y=y, yaw=approach.yaw + math.radians(turn),
                            ground_z=campus.layout.ground_height_at(x, y))
                settle(campus)
                truth = target_truth(campus, item["position"])
                tick, sim_start = time.perf_counter(), float(campus.data.time)
                result = PickupSkill(campus, planner, config).pick(item["position"])
                final = result["attempts"][-1]
                action = final.get("action") or {}
                trial = {"scene_seed": scene_seed, "track_id": item["track_id"], "class": item["class"],
                         "zone": campus.layout.zone_at(*truth.position[:2]), "perturbation": name,
                         "belief_source": source,
                         "belief_error_m": math.dist(item["position"][:2], truth.position[:2]),
                         "target": truth.name, "status": result["status"],
                         "success": truth.name in result["captured"],
                         "wrong_captures": [c for c in result["captured"] if c != truth.name],
                         "attempts": len(result["attempts"]),
                         "dock_error_m": final.get("dock_error_m"),
                         "tip_to_target_m": math.dist(action["tip_at_suction"][:2], truth.position[:2]) if "tip_at_suction" in action else None,
                         "tip_ground_clearance_m": action.get("tip_ground_clearance_m"),
                         "max_pitch_deg": (action.get("lower") or {}).get("max_pitch_deg"),
                         "min_trunk_height_m": (action.get("lower") or {}).get("min_trunk_height_m"),
                         "simulated_seconds": float(campus.data.time) - sim_start,
                         "wall_seconds": time.perf_counter() - tick}
                trials.append(trial)
                folder = args.output / f"scene_{scene_seed}/track_{item['track_id']}_{name}"
                write_json(folder / "result.json", {**trial, "skill": result})
                print(f"Scene {scene_seed} item {item['track_id']} ({item['class']}, {trial['zone']}) {name}: "
                      f"{'OK' if trial['success'] else 'MISS'} {result['status']}; attempts {trial['attempts']}; "
                      f"tip error {trial['tip_to_target_m'] if trial['tip_to_target_m'] is None else round(trial['tip_to_target_m'], 3)} m", flush=True)

    def summarize(key):
        groups = defaultdict(list)
        for trial in trials:
            groups[trial[key]].append(trial)
        return {str(k): {"trials": len(v), "successful": sum(t["success"] for t in v),
                         "first_attempt": sum(t["success"] and t["attempts"] == 1 for t in v)}
                for k, v in sorted(groups.items())}

    successes = [t for t in trials if t["success"]]
    errors = [t["tip_to_target_m"] for t in trials if t["tip_to_target_m"] is not None]
    checks = {"all_trials_captured_target": bool(trials) and len(successes) == len(trials),
              "no_wrong_captures": all(not t["wrong_captures"] for t in trials),
              "posture_stable": all(t["status"] not in ("fall", "trunk_contact") for t in trials)}
    result = {"passed": all(checks.values()), "checks": checks, "scope": __doc__,
              "config": asdict(config), "scene_seeds": args.scene_seeds,
              "trials": len(trials), "successful": len(successes),
              "first_attempt_successes": sum(t["attempts"] == 1 for t in successes),
              "mean_tip_error_m": float(np.mean(errors)) if errors else None,
              "max_tip_error_m": max(errors, default=None),
              "mean_simulated_seconds": float(np.mean([t["simulated_seconds"] for t in trials])) if trials else None,
              "by_class": summarize("class"), "by_zone": summarize("zone"),
              "by_scene": summarize("scene_seed"), "by_perturbation": summarize("perturbation"),
              "unreachable": unreachable, "skipped_starts_without_margin": skipped, "cases": trials,
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (Path(__file__), ROOT / "src/manipulation/pickup.py",
                                          ROOT / "src/manipulation/posture.py", ROOT / "src/manipulation/tools.py")},
              "total_wall_seconds": time.perf_counter() - started}
    write_json(args.output / "results.json", result)
    print(f"Pickup trials {'PASS' if result['passed'] else 'FAIL'}: {len(successes)}/{len(trials)} captured; "
          f"{result['first_attempt_successes']} on the first attempt; {len(unreachable)} unreachable; "
          f"{len(skipped)} perturbed starts skipped (inside the 5 cm margin)", flush=True)
    return 1 if args.require_pass and not result["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
