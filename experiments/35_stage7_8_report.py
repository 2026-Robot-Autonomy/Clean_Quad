"""Build the Stage 7/8 figures and summary from recorded experiment logs.

python -B experiments/35_stage7_8_report.py

Reads results/logs/pickup/trials and results/logs/cleaning/{stage7,stage8}_seed_0.
Nothing is simulated here; every number comes from the JSON logs.
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

FIGURES = ROOT / "results/figures/stage7_8"
LOGS = ROOT / "results/logs"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mission_panel(ax, result, trajectory, title):
    from src.environment.layout import default_layout
    from src.environment.plotting import plot_campus
    plot_campus(ax, default_layout())
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    points = np.array([[p["x"], p["y"]] for p in trajectory])
    actions = np.array([p["action"] for p in trajectory])
    ax.scatter(points[:, 0], points[:, 1], c=actions, cmap="viridis", s=4, zorder=4)
    for action in result["actions"]:
        if action["kind"] == "pick" and action.get("pickup"):
            x, y = action["pickup"]["attempts"][-1]["belief"][:2]
            done = action["pickup"]["status"] == "verified_gone"
            ax.plot(x, y, "o" if done else "x", ms=9, mfc="#2ca02c" if done else "none",
                    mec="black" if done else "#7f7f7f", mew=1.5, zorder=6)
            ax.annotate(f"{action['index'] + 1}", (x, y), xytext=(5, 5), textcoords="offset points",
                        fontsize=8, bbox=dict(facecolor="white", alpha=.8, edgecolor="none", pad=1), zorder=7)
        elif action["kind"] == "dump":
            label = action["target"]
            ax.annotate(f"{action['index'] + 1}: dump", (7.35, 2.6 if label == "bin_1" else 3.9),
                        xytext=(-60, 10 + 12 * (action["index"] % 3)), textcoords="offset points", fontsize=7,
                        arrowprops=dict(arrowstyle="-", lw=.5))
    from src.mapping import load_map
    final_map = load_map(LOGS / f"cleaning/{result['mode']}_seed_0/final_map")
    for key, reason in result["excluded"].items():
        if reason.startswith("unreachable"):
            x, y = final_map.item(int(key))["position"][:2]
            ax.plot(x, y, "X", ms=10, color="#d62728", mec="black", zorder=6)
            ax.annotate("unreachable", (x, y), xytext=(6, -10), textcoords="offset points", fontsize=7, color="#d62728")
    ax.set_title(title, fontsize=10)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    trials = read(LOGS / "pickup/trials/results.json")
    missions = {mode: read(LOGS / f"cleaning/{mode}_seed_0/results.json") for mode in ("stage7", "stage8")}
    paths = {mode: read(LOGS / f"cleaning/{mode}_seed_0/trajectory.json") for mode in missions}

    # Mission paths
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.0), layout="constrained")
    for ax, mode in zip(axes, missions):
        r = missions[mode]
        title = (f"{mode.upper()}: {len(r['collected_objects'])}/{len(r['expected_objects'])} collected, "
                 f"{len(r['deposits'])} dump(s), {len(r['charges'])} charge(s), {r['total_simulated_seconds']:.0f} s")
        mission_panel(ax, r, paths[mode], title)
    fig.suptitle("Cleaning missions, scene 0 (path colour = action order; green = picked up and verified, "
                 "grey x = no litter found)", fontsize=10)
    fig.savefig(FIGURES / "mission_paths.png", dpi=150)
    plt.close(fig)

    # Battery and hopper timelines
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")
    for mode, colour in (("stage7", "#7f7f7f"), ("stage8", "#1f77b4")):
        r = missions[mode]
        times, battery, load = [0.0], [100.0], [0]
        for action in r["actions"]:
            end = action["time"] + action["duration"]
            if action.get("charged"):
                charge = next(c for c in r["charges"] if abs(c["time"] - end) < 1e3 and c["time"] >= action["time"])
                times += [charge["time"] - charge["seconds"], charge["time"]]
                battery += [charge["from"], 100.0]
                load += [action["load_after"]] * 2
            else:
                times.append(end)
                battery.append(action["battery_after"])
                load.append(action["load_after"])
        axes[0].plot(times, battery, "-o", ms=3, color=colour, label=f"{mode} ({'not managed' if mode == 'stage7' else 'planned'})")
        axes[1].step(times, load, where="post", color=colour, label=mode)
    reserve = missions["stage8"]["task_config"]["battery_reserve"]
    axes[0].axhline(reserve, ls="--", color="#d62728", lw=1, label=f"reserve {reserve:g}%")
    axes[0].axhline(0, color="black", lw=.8)
    axes[0].set(ylabel="battery [%]", title="Simulated battery: Stage 8 returns to charge before the reserve")
    axes[0].legend(frameon=False)
    capacity = missions["stage8"]["task_config"]["hopper_capacity"]
    axes[1].axhline(capacity, ls="--", color="#d62728", lw=1, label=f"Stage 8 capacity {capacity}")
    axes[1].set(xlabel="simulated time [s]", ylabel="items in hopper", title="Hopper load")
    axes[1].legend(frameon=False)
    fig.savefig(FIGURES / "battery_hopper.png", dpi=150)
    plt.close(fig)

    # Pickup trials
    cases = trials["cases"]
    classes = sorted({c["class"] for c in cases})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for index, kind in enumerate(classes):
        errors = [c["tip_to_target_m"] * 100 for c in cases if c["class"] == kind and c["tip_to_target_m"] is not None]
        axes[0].scatter(np.full(len(errors), index) + np.random.default_rng(index).uniform(-.15, .15, len(errors)),
                        errors, s=18, alpha=.8)
    axes[0].axhline(trials["config"]["capture_radius"] * 100, ls="--", color="#d62728", lw=1, label="capture radius")
    axes[0].set(xticks=range(len(classes)), xticklabels=classes, ylabel="nozzle tip to true item [cm]",
                title=f"Pickup trials: {trials['successful']}/{trials['trials']} captured")
    axes[0].legend(frameon=False)
    zones = trials["by_zone"]
    names = list(zones)
    axes[1].bar(names, [zones[z]["trials"] for z in names], color="#c7c7c7", label="trials")
    axes[1].bar(names, [zones[z]["successful"] for z in names], color="#2ca02c", width=.5, label="captured")
    axes[1].set(ylabel="trials", title="By terrain zone")
    axes[1].legend(frameon=False)
    fig.savefig(FIGURES / "pickup_trials.png", dpi=150)
    plt.close(fig)

    summary = {
        "pickup_trials": {k: trials[k] for k in ("passed", "trials", "successful", "first_attempt_successes",
                                                 "mean_tip_error_m", "max_tip_error_m", "mean_simulated_seconds",
                                                 "by_class", "by_zone", "by_scene")},
        "missions": {mode: {k: r[k] for k in ("mission_passed", "checks", "collected_map_ids", "excluded",
                                              "battery_min", "battery_final", "max_hopper_load",
                                              "pickups", "pickup_attempts", "total_simulated_seconds",
                                              "total_wall_seconds")}
                     | {"dumps": len(r["deposits"]), "charges": len(r["charges"]),
                        "collected": len(r["collected_objects"]), "expected": len(r["expected_objects"])}
                     for mode, r in missions.items()},
        "figures": [str(p.relative_to(ROOT)) for p in sorted(FIGURES.glob("*.png"))]}
    (FIGURES / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({mode: {k: v for k, v in m.items() if k not in ("checks", "excluded")}
                      for mode, m in summary["missions"].items()}, indent=1))
    print("Figures:", ", ".join(summary["figures"]))


if __name__ == "__main__":
    main()
