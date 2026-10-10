"""Stage 4 route evaluation and depth-versus-ground-plane ablation.

python experiments/26_localization_pipeline.py --detector cnn --seeds 0 1 2
python experiments/26_localization_pipeline.py --detector oracle --seeds 0
python experiments/26_localization_pipeline.py --detector cnn --device cpu --viewer

Oracle mode supplies simulator 2-D boxes only. It isolates localization and
tracking, and cannot certify the CNN perception-to-localization pipeline.
"""

from pathlib import Path
import argparse
import csv
from dataclasses import asdict
import hashlib
from importlib import metadata
import json
import platform
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.localization.geometry import LocalizationConfig
from src.localization.route import run_localization_route


OUTPUT_DIR = PROJECT_ROOT / "results" / "logs" / "localization"


def operating_threshold(name, default):
    path = PROJECT_ROOT / "results" / "logs" / "perception" / "detector_comparison.csv"
    if path.exists():
        with path.open(newline="") as file:
            for row in csv.DictReader(file):
                if row["detector"] == name:
                    return float(row["threshold"])
    return default


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_figures(result, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.environment.plotting import plot_campus

    methods = list(result.pipelines)
    fig, axes = plt.subplots(1, len(methods), figsize=(8 * len(methods), 6), squeeze=False)
    for ax, method in zip(axes[0], methods):
        plot_campus(ax, result.campus.layout, path=result.path)
        pipeline = result.pipelines[method]
        evaluator = result.evaluators[method]
        if evaluator.visible:
            truth_positions = np.array([evaluator.objects[n]["position"] for n in sorted(evaluator.visible)])
            ax.scatter(truth_positions[:, 0], truth_positions[:, 1], s=65, facecolors="none", edgecolors="black", label="visible true centres")
        tracks = pipeline.tracker.confirmed_tracks
        if tracks:
            points = np.array([t.position for t in tracks])
            ax.scatter(points[:, 0], points[:, 1], marker="x", color="#2a78d6", s=65, label="fused estimates")
            for track in tracks:
                ax.text(*track.position[:2], f" {track.track_id}", color="#155398", fontsize=8)
        for match in result.map_matches[method]:
            track = next(t for t in tracks if t.track_id == match["track_id"])
            truth = evaluator.objects[match["truth_name"]]["position"]
            ax.plot([track.position[0], truth[0]], [track.position[1], truth[1]], color="#2a78d6", lw=1)
        metric = result.metrics[method]["map"]
        error = metric["p95_error_cm"]
        error_text = "n/a" if error is None else f"{error:.1f} cm"
        ax.set_title(f"{method}: {metric['matched_items']}/{metric['visible_items']} mapped, P95 {error_text}")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7)
    fig.tight_layout()
    fig.savefig(destination / "map.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for method in methods:
        rows = result.evaluators[method].rows
        errors = sorted(r["error_cm"] for r in rows if r["error_cm"] is not None)
        if errors:
            axes[0].plot(errors, np.arange(1, len(errors) + 1) / len(errors), label=method)
        bins = result.metrics[method]["by_distance"]
        axes[1].plot(list(bins), [v["p95_error_cm"] if v["p95_error_cm"] is not None else np.nan for v in bins.values()], "o-", label=method)
    axes[0].axvline(10, color="black", ls="--", lw=1, label="10 cm target")
    axes[0].set(xlabel="3-D observation error (cm)", ylabel="fraction of observations", ylim=(0, 1.02))
    axes[1].set(xlabel="camera-to-object distance", ylabel="P95 3-D observation error (cm)")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(destination / "errors.png", dpi=140)
    plt.close(fig)


def save_result(result, detector_name, config, destination, rate, figures=True, device=None):
    destination.mkdir(parents=True, exist_ok=True)
    versions = {"python": platform.python_version()}
    for package in ("numpy", "mujoco", "matplotlib", "scipy", "torch", "torchvision", "ultralytics"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    model_name = {"YOLOv8n (CNN)": "litter_yolov8n.pt", "colour baseline": "color_baseline.npz"}.get(detector_name)
    model_path = PROJECT_ROOT / "models" / model_name if model_name else None
    manifest = {
        "seed": result.campus.seed, "detector": detector_name, "device": device,
        "oracle_boxes_diagnostic": detector_name == "oracle_boxes_diagnostic",
        "model": f"models/{model_name}" if model_name else None,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest() if model_path else None,
        "config": asdict(config), "versions": versions, "route": result.route,
        "methods": {
            name: {"association_radius_m": p.tracker.association_radius,
                   "class_switch_radius_m": p.tracker.class_switch_radius,
                   "association_window_s": p.tracker.association_window,
                   "confirmation": "winning_class_frames", "min_observations": p.tracker.min_observations}
            for name, p in result.pipelines.items()
        },
    }
    (destination / "run.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for method, pipeline in result.pipelines.items():
        folder = destination / method
        folder.mkdir(exist_ok=True)
        tracks = {
            "schema_version": 1, "units": "metres", "frame": "mujoco_world_xyz_z_up",
            "seed": result.campus.seed, "detector": detector_name, "method": method,
            "config": asdict(config), "association_radius_m": pipeline.tracker.association_radius,
            "class_switch_radius_m": pipeline.tracker.class_switch_radius,
            "association_window_s": pipeline.tracker.association_window,
            "confirmation": "winning_class_frames",
            "min_observations": pipeline.tracker.min_observations,
            "tracks": [t.as_dict() for t in pipeline.tracker.confirmed_tracks],
        }
        (folder / "tracks.json").write_text(json.dumps(tracks, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (folder / "metrics.json").write_text(json.dumps(result.metrics[method], indent=2, allow_nan=False) + "\n", encoding="utf-8")
        observations = result.evaluators[method].rows
        columns = ["time", "detection_index", "track_id", "class_id", "score", "x", "y", "z", "camera_distance", "depth_pixels", "depth_mad", "truth_name", "truth_distance", "error_cm", "xy_error_cm"]
        write_csv(folder / "observations.csv", observations, columns)
        write_csv(folder / "map_matches.csv", result.map_matches[method], ["track_id", "truth_name", "class_id", "error_cm", "xy_error_cm"])
    write_csv(destination / "route.csv", [dict(x=x, y=y) for x, y in result.path], ["x", "y"])
    if figures:
        save_figures(result, destination)
    if result.video:
        result.video[0].save(
            destination / "localization.gif", save_all=True, append_images=result.video[1:],
            duration=round(1000 / rate), loop=0,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detector", choices=("cnn", "color", "oracle"), default="cnn")
    parser.add_argument("--device", default=None, help="YOLO device, e.g. cpu or 0")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--methods", nargs="+", choices=("depth", "ground_plane"), default=["depth", "ground_plane"])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--association-radius", type=float, default=0.20)
    parser.add_argument("--class-switch-radius", type=float, default=0.08)
    parser.add_argument("--min-observations", type=int, default=3)
    parser.add_argument("--depth-crop", type=float, default=0.6)
    parser.add_argument("--max-extent", type=float, default=0.35)
    parser.add_argument("--ground-z", type=float, default=0.0)
    parser.add_argument("--rate", type=float, default=5.0)
    parser.add_argument("--max-time", type=float, default=120.0)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--viewer", action="store_true", help="open a live MuJoCo viewer with confirmed litter markers")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--require-pass", action="store_true", help="exit 1 if the depth pipeline misses the Stage 4 target")
    args = parser.parse_args()
    if args.require_pass and "depth" not in args.methods:
        parser.error("--require-pass requires the depth estimator")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct")
    try:
        if args.detector == "cnn":
            from src.perception.yolo_detector import YoloDetector
            detector = YoloDetector(device=args.device)
        elif args.detector == "color":
            from src.perception.color_detector import ColorDetector
            detector = ColorDetector.load()
        else:
            detector = None
    except (ImportError, FileNotFoundError) as error:
        parser.error(f"{error}. Install requirements.txt and use the saved Stage 3 weights.")
    detector_name = detector.name if detector is not None else "oracle_boxes_diagnostic"
    threshold = args.threshold
    if threshold is None:
        threshold = operating_threshold(detector_name, 0.38 if args.detector == "cnn" else 0.25)
    try:
        config = LocalizationConfig(min_score=threshold, depth_crop=args.depth_crop, max_extent=args.max_extent, ground_z=args.ground_z)
    except ValueError as error:
        parser.error(str(error))
    print(f"STAGE 4: {detector_name}; methods {', '.join(args.methods)}; threshold {threshold:.2f}", flush=True)
    if detector is None:
        print("Diagnostic only: simulator 2-D boxes isolate geometry/fusion; CNN performance is not measured.", flush=True)
    summary, passed = [], True
    for seed in args.seeds:
        result = run_localization_route(
            detector, seed, config, tuple(args.methods), args.rate, args.max_time,
            args.association_radius, args.min_observations, save_video=args.save_video,
            class_switch_radius=args.class_switch_radius,
            show_viewer=args.viewer,
        )
        destination = args.output / args.detector / f"seed_{seed}"
        save_result(result, detector_name, config, destination, args.rate, not args.no_figures, args.device)
        for method, metrics in result.metrics.items():
            observed, mapped = metrics["per_observation"], metrics["map"]
            row = {
                "seed": seed, "detector": detector_name, "method": method,
                "route_completed": result.route["completed"], "visible_items": mapped["visible_items"],
                "mapped_items": mapped["matched_items"], "confirmed_tracks": mapped["confirmed_tracks"],
                "observation_mean_cm": observed["mean_error_cm"], "observation_p95_cm": observed["p95_error_cm"],
                "map_mean_cm": mapped["mean_error_cm"], "map_p95_cm": mapped["p95_error_cm"],
                "map_max_cm": mapped["max_error_cm"], "duplicate_rate": mapped["duplicate_rate"],
                "missed_item_rate": mapped["missed_item_rate"], "false_positives": mapped["false_positives"],
                "stage4_passed": metrics["stage4_passed"],
            }
            summary.append(row)
            mean = "n/a" if row["map_mean_cm"] is None else f"{row['map_mean_cm']:.2f}cm"
            print(f"seed {seed} {method}: mapped {mapped['matched_items']}/{mapped['visible_items']}, mean {mean}, duplicates {mapped['duplicates']}, false positives {mapped['false_positives']}, passed={metrics['stage4_passed']}", flush=True)
        passed &= result.metrics.get("depth", {"stage4_passed": False})["stage4_passed"]
        print(f"Saved {destination}", flush=True)
    folder = args.output / args.detector
    write_csv(folder / "summary.csv", summary, list(summary[0]))
    return 1 if args.require_pass and not passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
