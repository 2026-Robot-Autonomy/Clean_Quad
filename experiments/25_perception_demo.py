"""
Stage 3 live demo: the robot walks the campus route and runs the
litter detector on its head camera at 5 Hz.

Reports which litter items came into view and which were
detected, frame-level precision / recall, and detector speed.
Saves an annotated animation (GIF) of the camera view.

    python experiments/25_perception_demo.py                   # CNN
    python experiments/25_perception_demo.py --detector color  # baseline
"""

from pathlib import Path
import argparse
import csv
import sys
import time

import mujoco
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from src.controllers.locomotion_interface import LocomotionInterface
from src.environment import build_campus
from src.navigation.waypoint_follower import DEMO_ROUTE, HeadingFollower, reached
from src.perception.camera import RobotCamera
from src.perception.color_detector import ColorDetector
from src.perception.detection import match_image
from src.perception.labels import ground_truth_boxes, object_geom_ids
from src.perception.yolo_detector import YoloDetector


DETECTION_RATE = 5.0                # Hz
MAX_TIME = 120.0

COMPARISON_CSV = PROJECT_ROOT / "results" / "logs" / "perception" / "detector_comparison.csv"
VIDEO_DIR = PROJECT_ROOT / "results" / "videos"

BOX_COLOURS = {"YOLOv8n (CNN)": "#2a78d6", "colour baseline": "#eb6834"}
GIF_SCALE = 0.5


def operating_threshold(name, default=0.25):
    """Validation-tuned threshold from experiment 24, if available."""

    if COMPARISON_CSV.exists():
        with open(COMPARISON_CSV, newline="") as f:
            for row in csv.DictReader(f):
                if row["detector"] == name:
                    return float(row["threshold"])
    return default


def annotate(rgb, detections, colour, text):

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)

    for d in detections:
        x0, y0, x1, y1 = d.box
        draw.rectangle((x0, y0, x1, y1), outline=colour, width=2)
        label = f"{d.kind} {d.score:.2f}"
        draw.rectangle(draw.textbbox((x0, y0 - 12), label), fill=colour)
        draw.text((x0, y0 - 12), label, fill="white")

    draw.text((8, 8), text, fill="white")

    w, h = image.size
    return image.resize((int(w * GIF_SCALE), int(h * GIF_SCALE)))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", choices=("cnn", "color"), default="cnn")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    detector = YoloDetector() if args.detector == "cnn" else ColorDetector.load()
    threshold = operating_threshold(detector.name)
    colour = BOX_COLOURS[detector.name]

    print("=" * 80)
    print(f"CLEANQUAD STAGE 3: LIVE PERCEPTION DEMO  ({detector.name}, threshold {threshold:.2f})")
    print("=" * 80)

    campus = build_campus(seed=args.seed)
    model, data = campus.model, campus.data
    geom_map = object_geom_ids(model, campus.litter + campus.distractors)

    loco = LocomotionInterface()
    follower = HeadingFollower()
    waypoint = 0

    animation = []

    visible, detected = set(), set()
    tp = fp = n_gt = 0
    detect_time = 0.0
    frames = 0
    next_frame = 0.0

    with RobotCamera(model, data) as camera:

        while waypoint < len(DEMO_ROUTE) and data.time < MAX_TIME:

            x, y, yaw = campus.robot_pose()

            if reached(x, y, DEMO_ROUTE[waypoint]):
                waypoint += 1
                continue

            loco.set_velocity(*follower.command(x, y, yaw, DEMO_ROUTE[waypoint]))
            data.ctrl[:] = loco.update(data, model.opt.timestep)
            mujoco.mj_step(model, data)

            if data.time < next_frame:
                continue
            next_frame += 1.0 / DETECTION_RATE

            frame = camera.capture(depth=False)

            start = time.perf_counter()
            detections = [d for d in detector.detect(frame.rgb) if d.score >= threshold]
            detect_time += time.perf_counter() - start
            frames += 1

            truths = [b for b in ground_truth_boxes(frame, geom_map) if b.category == "litter"]
            flags, matched = match_image(detections, [(b.class_id, b.box) for b in truths])

            visible.update(b.name for b in truths)
            detected.update(truths[j].name for j in matched if j >= 0)
            tp += sum(flags)
            fp += len(flags) - sum(flags)
            n_gt += len(truths)

            animation.append(annotate(
                frame.rgb, detections, colour,
                f"t={data.time:5.1f}s  {detector.name}  seen {len(detected)}/{len(campus.litter)}",
            ))

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    video_path = VIDEO_DIR / f"perception_demo_{args.detector}.gif"
    animation[0].save(
        video_path, save_all=True, append_images=animation[1:],
        duration=int(1000 / DETECTION_RATE), loop=0,
    )

    print(f"\nroute: {waypoint}/{len(DEMO_ROUTE)} waypoints in {data.time:.1f} s, {frames} camera frames")
    print(f"litter items in the campus     : {len(campus.litter)}")
    print(f"came into view (>= 40 px)      : {len(visible)}  {sorted(visible)}")
    print(f"detected at least once         : {len(detected)}  {sorted(detected)}")
    print(f"missed                         : {sorted(visible - detected) or 'none'}")
    print(f"frame-level precision / recall : {tp / max(tp + fp, 1):.3f} / {tp / max(n_gt, 1):.3f}")
    print(f"detector speed                 : {frames / max(detect_time, 1e-9):.1f} frames/s")
    print(f"\nVideo: {video_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
