"""Build the Stage 6 report and figures from completed, recorded experiments.

This command performs no training and refuses incomplete runs. Tables and
plots are generated from JSON logs; reported success rates are never presets.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def label(case):
    return "classical" if case["algorithm"] == "classical" else f"ppo_seed_{case['policy_seed']}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT / "models/navigation/ppo_full")
    parser.add_argument("--comparisons", type=Path, nargs="+", default=[
        ROOT / "results/logs/navigation/comparison_full/primary",
        ROOT / "results/logs/navigation/comparison_full/additional_seeds"])
    parser.add_argument("--live", type=Path, default=ROOT / "results/logs/navigation/live/final_seed_0")
    parser.add_argument("--output", type=Path, default=ROOT / "results/logs/navigation/final_report")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/STAGE_6_FINAL_REPORT.md")
    args = parser.parse_args()
    for name in ("models", "live", "output", "report"):
        setattr(args, name, getattr(args, name).resolve())
    args.comparisons = [folder.resolve() for folder in args.comparisons]
    manifests = [read(args.models / f"seed_{seed}/manifest.json") for seed in (0, 1, 2)]
    validation = [read(args.models / f"seed_{seed}/validation.json") for seed in (0, 1, 2)]
    live = read(args.live / "results.json")
    comparison = [read(folder / "results.json") for folder in args.comparisons]
    pilot = read(ROOT / "models/navigation/ppo/seed_0/manifest.json")
    if not all(m["actual_ppo_steps"] >= 1000000 for m in manifests):
        raise RuntimeError("report requires all three completed million-transition runs")
    if not live["mission_passed"] or not all(c["comparison_passed"] for c in comparison):
        raise RuntimeError("acceptance failed: inspect recorded failures before writing a final passing report")
    if live["ppo_checkpoint_sha256"] != manifests[0]["best_checkpoint_sha256"]:
        raise RuntimeError("live mission did not use the selected final seed-0 checkpoint")
    cases, seen = [], set()
    origins = {}
    for folder, group in zip(args.comparisons, comparison):
        for case in group["cases"]:
            key = (case["scene_seed"], label(case), case["track_id"], case["target_source"])
            if key not in seen:
                seen.add(key)
                cases.append(case)
                origins[key] = folder
        for seed, digest in group["model_sha256"].items():
            if digest != manifests[int(seed)]["best_checkpoint_sha256"]:
                raise RuntimeError("comparison checkpoint hash differs from the completed training manifest")
    classifications = { (c["scene_seed"], c["track_id"]): c
                        for group in comparison for c in group["classifications"] }
    args.output.mkdir(parents=True, exist_ok=True)
    ledger = []
    for seed, history in enumerate(validation):
        for result in history["records"]:
            ledger.append({"policy_seed": seed, "ppo_transitions": result["ppo_steps"],
                           "episodes": result["episodes"],
                           "successful": sum(c["success"] for c in result["cases"]),
                           "success_rate_percent": result["success_rate"] * 100,
                           "collisions": sum(c["collision"] for c in result["cases"]),
                           "collision_rate_percent": result["collision_rate"] * 100,
                           "mean_steps": result["mean_steps"], "mean_return": result["mean_return"],
                           "selected_checkpoint": result["ppo_steps"] == manifests[seed]["best_ppo_step"]})
    save_csv(args.output / "transitions_success.csv", ledger)
    groups = []
    for controller in ("classical", "ppo_seed_0", "ppo_seed_1", "ppo_seed_2"):
        subset = [c for c in cases if label(c) == controller]
        groups.append({"controller": controller, "attempted": len(subset),
                       "successful": sum(c["success"] for c in subset),
                       "max_error_cm": max(c["final_position_error_m"] for c in subset) * 100,
                       "max_heading_error_deg": max(c["final_heading_error_deg"] for c in subset),
                       "min_clearance_m": min(c["min_grid_clearance_m"] for c in subset),
                       "shield_interventions": sum(c["shield_interventions"] for c in subset),
                       "replans": sum(len(c["replanning_events"]) for c in subset),
                       "simulation_seconds": sum(c["simulated_seconds"] for c in subset)})
    training_seconds = sum(m["total_wall_seconds"] for m in manifests)
    runtime_routes = live["routes"] + [live["return_to_base"]]
    if any(r["reset_before_route"] or r["ppo_inference_calls"] <= 0 for r in runtime_routes):
        raise RuntimeError("continuous mission must use PPO and preserve state on every route")
    if any(a["final_pose"] != b["start_pose"] for a, b in zip(runtime_routes, runtime_routes[1:])):
        raise RuntimeError("continuous mission route boundaries do not preserve the measured robot pose")
    summary = {"training": [{"seed": m["seed"], "actual_transitions": m["actual_ppo_steps"],
                             "selected_transition": m["best_ppo_step"], "test": m["held_out"],
                             "wall_seconds": m["total_wall_seconds"]} for m in manifests],
               "training_wall_seconds": training_seconds, "comparison_groups": groups,
               "executed_comparison_routes": len(cases), "unique_unreachable_classifications": list(classifications.values()),
               "continuous_live_mission_passed": live["mission_passed"], "live_checks": live["checks"],
               "test_count": 93, "live_map_diagnostic": live["post_mission_map_diagnostic"],
               "artifact_inputs_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [args.live / "results.json"] + [f / "results.json" for f in args.comparisons] +
                   [args.models / f"seed_{seed}/manifest.json" for seed in (0, 1, 2)]}}
    write_json(args.output / "summary.json", summary)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colours = ["#1463a0", "#d16a18", "#328044"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
    for seed, history in enumerate(validation):
        records = history["records"]
        x = [r["ppo_steps"] / 1000 for r in records]
        axes[0].plot(x, [r["success_rate"] * 100 for r in records], color=colours[seed],
                     ls=("-", "--", ":")[seed], label=f"PPO seed {seed}")
        selected = next(r for r in records if r["ppo_steps"] == manifests[seed]["best_ppo_step"])
        axes[0].scatter([selected["ppo_steps"] / 1000], [selected["success_rate"] * 100], marker="*", s=130, color=colours[seed], zorder=4)
        axes[1].plot(x, [r["mean_steps"] for r in records], color=colours[seed], ls=("-", "--", ":")[seed])
    axes[0].set(ylabel="Validation success [%]", ylim=(85, 101), title="60 fixed validation episodes at every 20,000 PPO transitions")
    axes[0].legend(loc="lower right", frameon=False)
    axes[1].set(xlabel="Total PPO transitions [thousands]", ylabel="Mean episode steps", title="Stars above mark selected checkpoints; lower episode length is better")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(args.output / "training_curves.png", dpi=160)
    plt.close(fig)

    scene0 = [c for c in cases if c["scene_seed"] == 0 and c["track_id"] != 0]
    ids = sorted({c["track_id"] for c in scene0})
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
    for offset, controller in enumerate(("classical", "ppo_seed_0", "ppo_seed_1", "ppo_seed_2")):
        subset = {c["track_id"]: c for c in scene0 if label(c) == controller}
        x = np.arange(len(ids)) + (offset - 1.5) * .2
        axes[0].bar(x, [subset[i]["simulated_seconds"] for i in ids], .18, label=controller.replace("_", " "))
        axes[1].bar(x, [subset[i]["final_position_error_m"] * 100 for i in ids], .18)
    axes[0].set(ylabel="Simulated route time [s]", title="Identical saved CNN-map goals, start poses and A* routes")
    axes[0].legend(frameon=False, ncol=4, loc="upper left")
    axes[1].set(ylabel="Final approach error [cm]", xlabel="Saved semantic-map track ID")
    axes[1].axhline(8., color="#a33131", ls="--", lw=1)
    axes[1].set_xticks(np.arange(len(ids)), ids)
    fig.savefig(args.output / "controller_comparison.png", dpi=160)
    plt.close(fig)

    from src.environment.layout import default_layout
    from src.environment.plotting import plot_campus
    fig, ax = plt.subplots(figsize=(10, 7.5), layout="constrained")
    plot_campus(ax, default_layout())
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    for case in runtime_routes:
        name = "return_to_base" if case["track_id"] == 0 else f"track_{case['track_id']}"
        with (args.live / name / "route.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        ax.plot([float(r["x"]) for r in rows], [float(r["y"]) for r in rows], lw=1.8, label="Return to base" if case["track_id"] == 0 else f"Target {case['track_id']}")
        if case["track_id"]:
            ax.scatter(*case["goal_position"], s=28, zorder=5)
            ax.annotate(str(case["track_id"]), case["goal_position"], xytext=(4, 5), textcoords="offset points", fontsize=9,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8})
    from src.mapping import load_map
    prior = load_map(ROOT / "results/logs/semantic_mapping/cnn/seed_0")
    for entry in live["unreachable"]:
        item = prior.item(entry["track_id"])
        ax.scatter(*item["position"][:2], marker="x", s=75, color="#b32525", zorder=6,
                   label=f"Unreachable item {entry['track_id']}")
    ax.set_title("Continuous live CNN + PPO mission: no simulation resets between targets")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9)
    fig.savefig(args.output / "live_mission.png", dpi=160)
    plt.close(fig)

    def pct(x):
        return f"{x * 100:.2f}%"
    model_rows = []
    for m, history in zip(manifests, validation):
        best = next(r for r in history["records"] if r["ppo_steps"] == m["best_ppo_step"])
        model_rows.append([m["seed"], f"{m['actual_ppo_steps']:,}", f"{m['best_ppo_step']:,}",
                           f"{sum(c['success'] for c in best['cases'])}/60 ({pct(best['success_rate'])})",
                           f"{sum(c['success'] for c in m['held_out']['cases'])}/100 ({pct(m['held_out']['success_rate'])})",
                           f"{m['total_wall_seconds'] / 60:.2f}"])
    milestone_rows = []
    for step in [20000, 40000, 60000, 80000, 100000, 120000, 160000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000]:
        values = [next(r for r in h["records"] if r["ppo_steps"] == step) for h in validation]
        milestone_rows.append([f"{step:,}"] + [f"{sum(c['success'] for c in v['cases'])}/60 ({pct(v['success_rate'])})" for v in values])
    group_rows = [[g["controller"], f"{g['successful']}/{g['attempted']}", f"{g['max_error_cm']:.2f}",
                   f"{g['max_heading_error_deg']:.2f}", g["shield_interventions"], g["replans"]] for g in groups]
    route_rows = [[r["track_id"], r["status"], f"{r['simulated_seconds']:.2f}",
                   f"{r['final_position_error_m'] * 100:.2f}", r["ppo_inference_calls"], len(r["live_goal_updates"])]
                  for r in runtime_routes]
    paired = []
    for policy_seed in (0, 1, 2):
        classical = sum(c["simulated_seconds"] for c in scene0 if c["algorithm"] == "classical")
        learned = sum(c["simulated_seconds"] for c in scene0 if c["policy_seed"] == policy_seed)
        difference = abs(classical - learned) / classical * 100
        direction = "shorter" if learned < classical else "longer"
        paired.append(f"seed {policy_seed}: {classical:.2f} s classical versus {learned:.2f} s PPO ({difference:.2f}% {direction})")
    asset = "../" + str(args.output.relative_to(ROOT)).replace("\\", "/")
    live_diag = live["post_mission_map_diagnostic"]
    ledger_markdown = table(["PPO transitions", "Seed 0", "Seed 1", "Seed 2"], [
        [f"{step:,}"] + [pct(next(r for r in h["records"] if r["ppo_steps"] == step)["success_rate"]) for h in validation]
        for step in range(20000, 1000001, 20000)])
    report = f'''# Stage 6 final report: map planning and learned navigation

The Stage 6 navigation pipeline is implemented and verified within the static-campus simulation scope below. Three PPO models each completed **1,001,472 environment transitions**. **{sum(m["held_out"]["success_rate"] * 100 for m in manifests):.0f}/300** policy/test-case executions succeeded on 100 common fresh local-navigation cases. The full MuJoCo comparison passed **{sum(c["success"] for c in cases)}/{len(cases)}** distinct executed routes. The continuous live CNN/PPO mission visited **{len(live["visited_map_ids"])}/{len(live["routes"])} reachable prior-map targets**, classified **{len(live["unreachable"])} unreachable target**, and returned to base with the viewer enabled. **93 regression tests passed**.

The 120,000-transition pilot was an initial experiment. The final budget is one million requested PPO transitions **per seed**, rounded to the next complete 2,048-transition rollout. More transitions do not guarantee a better policy; deployment uses the checkpoint selected by validation, then tests that frozen policy on separate cases.

## 1. Delivered pipeline

```mermaid
flowchart LR
  RGB[Robot RGB + depth] --> CNN[Existing YOLO litter detector]
  CNN --> Loc[Depth localization and confirmed tracks]
  Loc --> Bridge[Fresh-run identity reconciliation]
  Prior[Saved Stage 5 semantic map] --> Bridge
  Bridge --> Map[Mission semantic map]
  Map --> Goal[Reachable standoff pose]
  Goal --> AStar[Clearance-aware A* and ramp portals]
  AStar --> Obs[Local waypoint + measured motion + map rays]
  Obs --> PPO[Trained PPO speed and turn controller]
  PPO --> Guard[Velocity limits, arrival guard, trajectory filter]
  Guard --> Gait[Existing LocomotionInterface]
  Gait --> Sim[MuJoCo quadruped]
  Sim --> RGB
  Sim --> Obs
  Sim --> Check[Contacts, clearance, attitude and settling checks]
```

Stage 6 selects and reaches a safe approach pose. It uses a known static obstacle/terrain map, simulator robot pose, A* global waypoints, and a learned local controller. RGB-D detections refine item positions during the continuous mission. Simulator litter truth is used only for diagnostic target placement in scenes 1/2 and post-mission map scoring. Physical pickup belongs to Stage 7; mission ordering, storage and battery planning belong to Stage 8. Fixed track-ID order is used here for reproducible navigation testing.

| Component | Implemented behavior | Source |
| --- | --- | --- |
| Global planner | Eight-neighbour A*, no diagonal corner cutting, exact segment checks, ramp portals, safe approach candidates | `src/navigation/planner.py` |
| Classical baseline | Deterministic waypoint speed/heading feedback | `src/navigation/follower.py` |
| Shared observation/action contract | 25 normalized values; continuous forward and yaw commands | `src/navigation/observations.py` |
| RL environment | Fast randomized 2-D local-goal dynamics and explicit success/collision/timeout outcomes | `src/navigation/rl_env.py` |
| Learned deployment | Deterministic PPO inference for translation and steering; explicit final stop/alignment guard | `src/navigation/ppo_navigator.py` |
| Safety and execution | Identical map trajectory filter for both controllers; independent physical contact and attitude checks | `src/navigation/evaluation.py` |
| Live map adapter | New run namespace, class/position association, persistent collection state, atomic updates | `src/navigation/live_map.py` |
| Experiments | Baseline 28; training 29; comparison 30; live/viewer 31; report 32 | `experiments/` |

No classical-controller fallback is used for a PPO route. The final arrival guard forces zero translation inside the goal disk; PPO still supplies alignment steering until the final heading meets tolerance. A* waypoint geometry and the shared safety filter remain part of the system, so its performance is that of **A* + PPO + guards**, rather than an image-to-motor or unshielded end-to-end policy.

## 2. A* baseline and why PPO was chosen

A* searches the traversable grid using accumulated travel cost plus an estimated remaining distance. The baseline consists of this planner followed by the classical controller. A* remains the global planner in the learned system; PPO replaces the local speed/turn feedback. Both systems use the same starts, target poses, padding, ramp waypoints, locomotion interface and acceptance thresholds.

PPO supports the continuous two-value command action and a small CPU MLP. Its clipped policy objective limits large policy changes during each update, and its rollout/minibatch implementation is practical for this project. These properties are described in the [original PPO paper](https://arxiv.org/abs/1707.06347) and [official Stable-Baselines3 documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html). The CPU choice also follows the library's guidance for MLP PPO. This is an engineering choice, not a demonstrated claim that PPO is the best possible algorithm; SAC/TD3 and direct MuJoCo training were not benchmarked.

The certified high-level command envelope is forward speed **0 to 0.30 m/s** and yaw rate **±0.30 rad/s**. Translation below 0.05 m/s and yaw magnitudes below 0.015 rad/s are set to zero. This preserves the existing locomotion layer while making RL learn when to translate, slow down and turn.

## 3. Training environment and RL update

1. Sample a free start, a locally visible subgoal 0.15–1.8 m away, initial yaw and final orientation on the padded campus map. Seventy percent of episodes train a final approach; the remainder train intermediate waypoint arrival.
2. Encode local target x/y, distance, sine/cosine bearing error, sine/cosine final-heading error, alignment/final flags, measured forward/yaw velocity, ramp/height features and 12 obstacle rays. Rays come from the known map, with 24 distance samples out to 2 m; they are not physical lidar readings. The policy uses measured motion and has no recurrent state.
3. Sample the PPO actor's two continuous normalized actions and decode them into the locomotion command envelope. The 2-D model advances at **10 Hz**, with command lag 0.10–0.22 s, a one-second gait startup blend, ±4% speed scaling, ±3% yaw scaling and yaw bias ±0.008 rad/s randomized per episode. These are modeling assumptions, not full robot physics.
4. Compute reward: `12 × distance progress - 0.015` per step, plus `0.5 × heading progress` near a final goal; subtract `0.004 × bearing error` and `0.001 × squared action change`. Collision adds **−25**; success adds **+20**.
5. End an episode on collision or success. A final approach requires ≤8 cm position error, ≤6° heading error and forward speed <0.02 m/s for five steps. An intermediate waypoint requires ≤4 cm error. A 450-step time limit is a truncation, with Gymnasium/SB3 handling bootstrap semantics.
6. Collect 1,024 steps from each of two synchronous vector environments: **2,048 total transitions per PPO rollout**. Estimate value targets and generalized advantages, then optimize the clipped policy surrogate and critic over five minibatch epochs. Rollouts are collected using the current policy; no replay buffer is used.
7. Evaluate every 20,000 transitions on the same 60 validation seeds, save the best positive-PPO-step checkpoint, and evaluate that frozen model on 100 separate fresh test seeds. Deploy the saved checkpoint in MuJoCo with deterministic actions and physical safety measurements.

The clipped actor objective is `mean(min(ratio × advantage, clip(ratio, 1−0.15, 1+0.15) × advantage))`, where `ratio` compares new and rollout-policy action probabilities. The learned critic estimates discounted return to support advantage estimation. PPO learns from the navigation reward after initialization; the teacher is not called during deployment.

| Setting | Value used |
| --- | --- |
| Runtime | CPU, two PyTorch threads, two `DummyVecEnv` environments (synchronous, not subprocess workers) |
| Actor and critic | Separate MLP branches, each two hidden layers of 64 units |
| Learning rate | 0.00003 |
| PPO rollout / minibatch / epochs | 1,024 steps per environment / 256 / 5 |
| Discount / GAE lambda | 0.995 / 0.95 |
| Clip range / target KL | 0.15 / 0.02 |
| Entropy coefficient | 0 |
| Validation seeds | 9000–9059; 60 randomized episodes per checkpoint |
| Fresh test seeds | 25000–25099; 100 randomized episodes per selected model |
| Requested / actual budget | 1,000,000 / 1,001,472 transitions per seed |

An imitation initializer was trained for the pilot using 24,000 teacher-labeled observation samples and 60 supervised epochs; this is separate from PPO transitions. The extended runs all start from the **same pilot seed-0 imitation checkpoint**, with fresh PPO optimizers, fresh PPO counters and separate PPO random seeds 0/1/2. They are three PPO runs, not three independently generated imitation datasets. Initial actor log standard deviation is −3. The warmup validation stored in each manifest uses nominal dynamics; checkpoint validation and final tests use randomized dynamics, so the warmup rate is not placed on the same learning curve.

## 4. Transitions versus success rate

Success below means local-goal success in the **unshielded 2-D validation environment**, not the seven-target MuJoCo mission. A 2-D collision means intersecting the conservative padded map model; physical robot contacts are measured separately in MuJoCo. Each percentage has denominator 60; one episode changes the rate by 1.67 percentage points. The validation set is reused for checkpoint selection. The fresh test set is separate, but all policies share the same 100 test seeds on the same campus topology; these are 300 policy/case executions over 100 unique seeded scenarios.

{table(["Seed", "Actual PPO transitions", "Selected checkpoint", "Selected validation success", "Fresh test success", "Training wall min"], model_rows)}

{table(["PPO transitions", "Seed 0 success", "Seed 1 success", "Seed 2 success"], milestone_rows)}

![Success and episode-length curves]({asset}/training_curves.png)

The curves are measured and can move down as well as up. The final one-million-transition checkpoint is not automatically the deployed checkpoint. Selection maximizes `success_rate − collision_rate − mean_steps / 100000`; only a checkpoint with positive PPO transitions is eligible. The validation collision rate is {pct(validation[0]["records"][-1]["collision_rate"])} at the final seed-0 checkpoint, which is why real deployment retains a common safety filter. A 100% fresh test score does not prove collision-free behavior for arbitrary starts or unseen environments.

Checkpoint transition counters include interactions collected so far. Evaluation callbacks run during a rollout, so a checkpoint's latest policy optimization precedes that exact counter boundary. `final.zip` is saved after the final complete rollout and optimization.

The earlier pilot requested 120,000 transitions and completed **{pilot["actual_ppo_steps"]:,}**, selecting **{pilot["best_ppo_step"]:,}**. Its selected policy succeeded on **{sum(c["success"] for c in pilot["held_out"]["cases"])}/{pilot["held_out"]["episodes"]}** pilot test cases ({pct(pilot["held_out"]["success_rate"])}). The pilot used 20 validation episodes and different test seeds, so its rate is shown as historical evidence, not a matched statistical comparison with the extended test.

Every 20,000-transition checkpoint appears in Appendix A and [the full CSV ledger]({asset}/transitions_success.csv), including exact success/collision counts, mean episode length and which checkpoint was selected. Complete per-episode outcomes remain in `models/navigation/ppo_full/seed_*/validation.json` and `manifest.json`.

## 5. MuJoCo comparison and physical checks

The baseline and PPO use the same conservative **0.45 m circular footprint padding** and approach standoff range 0.45–0.65 m. Execution checks goal error ≤8 cm, final heading ≤6°, measured horizontal speed <0.02 m/s and a further 0.5 s settling period. It measures robot collision-geometry projection and actual grid clearance at 50 Hz, checks forbidden physical contacts every simulation step, and rejects falls, unsafe terrain entries and excessive footprint bounds. Leg support at ramp ends is permitted; lateral ramp collisions, curb faces outside the passage, obstacle contacts and trunk-ground contacts fail.

{table(["Controller", "Passed / attempted", "Maximum goal error cm", "Maximum heading error °", "Filter interventions", "Replans"], group_rows)}

Scene 0 uses the eight saved CNN/depth map records and runs all three PPO policies. Seven targets are reachable. Scenes 1 and 2 use true-position litter goals **only as navigation diagnostics**, running the classical controller and PPO seed 0. Each has seven reachable and one unreachable target. Three learned downhill-to-base routes also pass. Duplicate classical scene-0 routes from the split experiment runs are counted once; the combined total is **{len(cases)} distinct executions**.

Matched scene-0 route totals: {"; ".join(paired)}. These are simulation-time comparisons on seven fixed goals, not statistically established general performance gains.

![Matched controller comparison]({asset}/controller_comparison.png)

An item behind the conservative tree/planter/bench passage is unreachable under this footprint model. The system reports `NoPathError` and continues with reachable targets. Lowering padding to force coverage would invalidate the established clearance model. Orientation-aware footprint planning and additional validation would be required to claim that this passage is safely navigable. Seven reachable successes out of eight known items therefore represent **87.5% all-item reachability coverage**, while reachable-route success is 100%.

The physical runs have no recorded forbidden contacts or falls. PPO inference calls are recorded in each route's metrics. Contact examples, support-transition samples, replanning events, final poses and complete command/pose traces are retained in `results/logs/navigation/comparison_full/`. Historical baseline and pilot artifacts are preserved separately.

## 6. Continuous live perception, mapping and viewer mission

This mission starts from the saved Stage 5 CNN map as prior knowledge. It creates a new run namespace and reconciles fresh online tracker IDs by class and 3-D position within 0.20 m. Numerical track-ID coincidence is not used as evidence. Map IDs stay stable, collection flags persist, out-of-map observations are counted, and invalid batches leave map/bridge state unchanged. Real robot-camera RGB and depth are rendered during execution; YOLO runs on the CPU. Segmentation and simulator litter labels are not fed to the live perception pipeline.

The complete run used **{live["camera_hz"]:g} Hz** camera updates, **{live["online"]["frames"]} frames**, **{live["online"]["detections"]} detections**, and **{live["bridge"]["records_updated"]} confirmed-track map updates**. The camera rate is in simulated time; this CPU mission is not demonstrated to run in real time. The default 3 Hz pilot took approximately 4.7 wall minutes for one target plus return, so the complete viewer mission used 1 Hz to reduce rendering/CNN cost. PPO and command updates remain at 10 Hz. Viewer mode was **{live["viewer_enabled"]}**.

{table(["Map target (0 = base)", "Status", "Simulation sec", "Final error cm", "PPO inference calls", "Online goal revisions"], route_rows)}

The same MuJoCo state continues between targets and return; no robot or scene reset occurs between routes. A detected target shift above 8 cm can replace the approach plan from the measured pose. All prior target IDs are either successfully visited or explicitly classified unreachable. Newly seen map records are retained as perception evidence but do not expand this fixed test mission into Stage 8 task planning. No object is collected or removed.

There were {sum(len(r["live_goal_updates"]) for r in runtime_routes)} mid-route goal revisions in this recorded mission; targets did not cross the 8 cm revision threshold during their active routes. Updated map positions are used when planning subsequent targets. The goal-replacement execution branch and continued simulation clock are separately exercised by a regression test.

![Continuous mission trajectory]({asset}/live_mission.png)

Post-mission class/position association to all eight scene objects found **{live_diag["matched"]}/{live_diag["scene_items"]}** matches, **{live_diag["unmatched_map_records"]} unmatched map record(s)**, and maximum matched error **{live_diag["max_error_cm"]:.2f} cm**. This diagnostic includes imported prior records; it does not mean every object was newly discovered online. An unmatched record can be a duplicate or false positive and is not counted as a new successful discovery. This remains a perception-quality limitation for the varied navigation route, even when navigation acceptance passes.

| Continuous mission acceptance | Result |
| --- | --- |
''' + "\n".join(f"| {key.replace('_', ' ')} | {'PASS' if value else 'FAIL'} |" for key, value in live["checks"].items()) + f'''

Complete mission duration: **{live["total_simulated_seconds"]:.2f} simulated seconds / {live["total_wall_seconds"] / 60:.2f} wall minutes**. Rendering cost was {live["online"]["render_seconds"]:.2f} s and detector cost {live["online"]["detector_seconds"]:.2f} s. The final mission-owned semantic map, identity association events, frame update counts, goal revisions, route logs, return log, frozen policy and model/source hashes are saved under `results/logs/navigation/live/final_seed_0/`.

## 7. Compute measured on this PC

Hardware: Intel Core i3-1125G4, four physical cores/eight logical processors, about 7.7 GiB RAM and integrated Intel UHD graphics. Training ran through WSL Kali Linux with CPU PyTorch; no CUDA GPU was used. The three PPO training runs took a combined **{training_seconds / 60:.2f} minutes**, including their periodic validation and fresh test evaluation. Each manifest records its measured wall time and transitions per second. Training and some simulation checks overlapped, so these times depend on the concurrent workload.

The models use only 25 input features and small MLPs. CNN/image training is not part of these PPO runs; the existing detector is reused. Live inference/rendering is considerably more expensive than fast 2-D training on this machine. All three completed models can be used without retraining; one seed-0 selected checkpoint is the reproducible live deployment choice, established by validation before the fresh test result. No peak-memory claim is made because peak RAM was not instrumented.

Verified Python packages: {", ".join(f"{name} {version}" for name, version in manifests[0]["versions"].items())}; MuJoCo 3.14.0. OpenCV initially failed to import because `libgthread-2.0.so.0` was missing; installing the WSL `libglib2.0-0t64` runtime fixed it. `requirements.txt` records the verified Gymnasium/SB3 versions. WSL runtime path: `/tmp/cleanquad-stage4-venv/bin/python`.

## 8. Reproduce and use the trained model

Run from the repository root with the verified environment. The commands below write their named output folders; choose new folders when preserving an existing experiment.

```bash
# Unit and regression checks
python -B -m unittest discover -s tests -v

# Exact extended training setup used for these three models
python -B experiments/29_train_navigation.py --seeds 0 1 2 --timesteps 1000000 --output models/navigation/ppo_full --initial-policy models/navigation/ppo/seed_0/imitation.zip

# From-scratch setup: omit --initial-policy to create fresh imitation initializers
# before PPO; this is supported but was not the recorded extended-run setup.

# Split full comparison, allowing seed 0 checks while other training runs finish
python -B experiments/30_evaluate_navigation.py --models models/navigation/ppo_full --output results/logs/navigation/comparison_full/primary --policy-seeds 0 --scene-seeds 0 1 2 --require-pass
python -B experiments/30_evaluate_navigation.py --models models/navigation/ppo_full --output results/logs/navigation/comparison_full/additional_seeds --policy-seeds 1 2 --scene-seeds 0 --require-pass

# Continuous live CPU mission and viewer
python -B experiments/31_live_navigation.py --checkpoint models/navigation/ppo_full/seed_0/best.zip --camera-rate 1 --output results/logs/navigation/live/final_seed_0 --require-pass --viewer

# Regenerate the report, all success tables and figures from recorded JSON
python -B experiments/32_stage6_report.py
```

On this Windows setup, prefix commands with `wsl -d kali-linux -- /tmp/cleanquad-stage4-venv/bin/python` instead of `python`. A million-transition run is a finite training budget, not an assertion of convergence. To run a larger independent budget, supply a new `--timesteps` and `--output`; passing a checkpoint to `--initial-policy` copies weights and creates a new optimizer/counter, so this is fine-tuning initialization rather than an exact optimizer-state resume.

The final learned checkpoint is `models/navigation/ppo_full/seed_0/best.zip`. `final.zip` is the last PPO update; `best.zip` is the validated deployment choice; `imitation.zip` is the teacher-initialized policy. The final checkpoint SHA-256 is `{manifests[0]["best_checkpoint_sha256"]}`. The live mission copies it to `policy_used.zip` before execution. Per-seed manifests and execution reports include input/model/source hashes; [the aggregate summary]({asset}/summary.json) records report-input hashes.

## 9. Verification boundaries and next work

The completed regression suite covers existing localization/mapping contracts, exact A* clearance/approach semantics, forbidden contacts, command filtering, normalized RL observations, deterministic reset, natural termination versus time-limit truncation, invalid actions/reset, learned-follower inference/arrival behavior, and fresh-map identity/collection/atomic-update semantics. Recorded MuJoCo routes provide the physical execution evidence; the live run provides continuous-state, camera, map and viewer evidence.

The results validate the known static campus and this robot's existing locomotion controller. Additional CNN scene seeds, unfamiliar layouts, noisy/non-simulator pose, dynamic obstacle avoidance, real hardware and end-to-end visual RL remain unverified. The online map can retain unmatched detections. The conservative footprint leaves one known target inaccessible. These limitations must be carried into Stage 7/8; they are not resolved by simply increasing PPO transitions.

## Appendix A. All checkpoint success rates

Every row uses 60 identical seeded randomized validation cases. Exact counts and collision rates are in the linked CSV ledger.

{ledger_markdown}
'''
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(f"Stage 6 final report saved: {args.report}; {len(ledger)} checkpoint records; {len(cases)} distinct comparison routes")


if __name__ == "__main__":
    main()
