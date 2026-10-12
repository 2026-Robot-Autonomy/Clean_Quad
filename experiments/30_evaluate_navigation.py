"""Fair MuJoCo comparison of A*+classical and A*+learned PPO navigators.

python -B experiments/30_evaluate_navigation.py --policy-seeds 0 1 2 --scene-seeds 0 1 2 --require-pass
All policies run the saved CNN-map targets. The first requested policy also
runs extra true-position scene diagnostics. No live CNN result is implied.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.environment import build_campus
from src.mapping import SemanticMap, load_map
from src.navigation import AStarPlanner, ApproachPlan, NoPathError
from src.navigation.evaluation import execute_approach
from src.navigation.ppo_navigator import PPOFollower, load_policy


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_rows(path, rows):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    import torch
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--scene-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--models", type=Path, default=ROOT / "models/navigation/ppo_full")
    parser.add_argument("--output", type=Path, default=ROOT / "results/logs/navigation/comparison_full/full")
    parser.add_argument("--max-time", type=float, default=180.)
    parser.add_argument("--track-ids", nargs="*", type=int)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    if not args.policy_seeds or len(set(args.policy_seeds)) != len(args.policy_seeds) or len(set(args.scene_seeds)) != len(args.scene_seeds):
        parser.error("use distinct nonempty policy seeds and distinct scene seeds")
    started = time.perf_counter()
    policies = {}
    manifests = {}
    for seed in args.policy_seeds:
        path = args.models / f"seed_{seed}/best.zip"
        policies[seed] = load_policy(path)
        manifests[seed] = json.loads((path.parent / "manifest.json").read_text())
        if manifests[seed]["best_ppo_step"] <= 0:
            parser.error("comparison requires a policy with positive PPO training steps")
    cases, classifications, plans = [], [], []
    for scene_seed in args.scene_seeds:
        campus = build_campus(seed=scene_seed)
        if scene_seed == 0:
            semantic_map = load_map(ROOT / "results/logs/semantic_mapping/cnn/seed_0")
            items = semantic_map.active_items()
            source = "saved Stage 4 CNN depth tracks"
        else:
            semantic_map = SemanticMap.from_layout(campus.layout, seed=scene_seed, run_id=f"navigation-diagnostic-{scene_seed}")
            items = [{"track_id": i, "class": t.kind, "position": t.position.tolist()}
                     for i, t in enumerate(campus.litter, 1)]
            source = "simulator true-position navigation diagnostic"
        planner = AStarPlanner(semantic_map)
        if args.track_ids:
            items = [item for item in items if item["track_id"] in args.track_ids]
        selected_policies = args.policy_seeds if scene_seed == 0 else args.policy_seeds[:1]
        for item in items:
            try:
                approach = planner.approach_item(campus.layout.base_pose[:2], item)
            except NoPathError as error:
                classifications.append({"scene_seed": scene_seed, "track_id": item["track_id"],
                                        "status": "unreachable", "reason": str(error), "target_source": source})
                continue
            plans.append({"scene_seed": scene_seed, "track_id": item["track_id"], "approach": asdict(approach)})
            for controller, seed in [("classical", None)] + [("ppo", seed) for seed in selected_policies]:
                factory = None if controller == "classical" else lambda route, yaw, seed=seed: PPOFollower(route, yaw, planner, policies[seed])
                metrics, rows = execute_approach(campus, planner, approach, controller_factory=factory, max_time=args.max_time)
                metrics.update({"scene_seed": scene_seed, "policy_seed": seed, "algorithm": controller,
                                "class": item["class"], "item_position": item["position"], "target_source": source})
                label = "classical" if seed is None else f"ppo_seed_{seed}"
                folder = args.output / f"scene_{scene_seed}/{label}/track_{item['track_id']}"
                write_json(folder / "metrics.json", metrics)
                save_rows(folder / "route.csv", rows)
                cases.append(metrics)
                write_json(args.output / "progress.json", {"cases": cases, "classifications": classifications})
                print(f"Scene {scene_seed} track {item['track_id']} {label}: {metrics['status']}; error {metrics['final_position_error_m']:.3f} m; sim {metrics['simulated_seconds']:.1f} s", flush=True)
        if scene_seed == 0 and not args.track_ids:
            start = (4., 2.9, -math.pi / 2)
            route = planner.plan(start[:2], campus.layout.base_pose[:2])
            approach = ApproachPlan(0, campus.layout.base_pose[:2], 0., 0., route)
            for seed in selected_policies:
                factory = lambda route, yaw, seed=seed: PPOFollower(route, yaw, planner, policies[seed])
                metrics, rows = execute_approach(campus, planner, approach, controller_factory=factory,
                                                start_pose=start, max_time=args.max_time)
                metrics.update({"scene_seed": scene_seed, "policy_seed": seed, "algorithm": "ppo", "target_source": "downhill-to-base diagnostic"})
                cases.append(metrics)
                folder = args.output / f"scene_0/ppo_seed_{seed}/downhill"
                write_json(folder / "metrics.json", metrics)
                save_rows(folder / "route.csv", rows)
                print(f"PPO seed {seed} downhill: {metrics['status']}", flush=True)
    groups = []
    for label in ["classical"] + [f"ppo_seed_{seed}" for seed in args.policy_seeds]:
        group = [case for case in cases if (case["algorithm"] == "classical" and label == "classical")
                 or (case["algorithm"] == "ppo" and label == f"ppo_seed_{case['policy_seed']}")]
        groups.append({"controller": label, "attempted": len(group), "successful": sum(c["success"] for c in group),
                       "max_final_error_m": max((c["final_position_error_m"] for c in group), default=None),
                       "simulation_seconds": sum(c["simulated_seconds"] for c in group),
                       "wall_seconds": sum(c["wall_seconds"] for c in group),
                       "shield_interventions": sum(c["shield_interventions"] for c in group),
                       "replans": sum(len(c["replanning_events"]) for c in group)})
    passed = bool(cases) and all(c["success"] for c in cases)
    result = {"comparison_passed": passed, "scope": __doc__, "scene_seeds": args.scene_seeds,
              "policy_seeds": args.policy_seeds, "groups": groups, "cases": cases,
              "classifications": classifications, "plans": plans,
              "model_sha256": {str(seed): hashlib.sha256((args.models / f"seed_{seed}/best.zip").read_bytes()).hexdigest() for seed in args.policy_seeds},
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (Path(__file__), ROOT / "src/navigation/evaluation.py", ROOT / "src/navigation/ppo_navigator.py", ROOT / "src/navigation/observations.py")},
              "total_wall_seconds": time.perf_counter() - started}
    write_json(args.output / "results.json", result)
    print(f"Comparison {'PASS' if passed else 'FAIL'}: {sum(c['success'] for c in cases)}/{len(cases)} executed routes", flush=True)
    return 1 if args.require_pass and not passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
