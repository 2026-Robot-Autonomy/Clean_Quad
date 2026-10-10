# CleanQuad — Progress Report, Stages 1–4

*Simulated Unitree A1 quadruped for campus litter collection (MuJoCo). Status as of 10 Oct 2026.*

| Stage | Title | Status |
|---|---|---|
| 1 | Quadruped locomotion | **Done** (V1 certified; V2 fast gait added and certified) |
| 2 | Campus / street environment | **Done** |
| 3 | Litter perception | **Done** (CNN + classical baseline, evaluated) |
| 4 | 3-D litter localization | **Done** (CNN seed-0 headless route passes; 31 unit tests pass) |
| 5 | Semantic mapping | **Next** (implementation planned) |
| 6 | Autonomous navigation (RL navigator) | Planned |
| 7 | Cleaning / pickup | Planned |
| 8 | Task planning & optimization | Planned |

---

## 1. Stage 1 — Quadruped Locomotion

### 1.1 Approach
A model-based (non-learned) trot controller drives the 12 position actuators of the Unitree A1
(MuJoCo Menagerie model):

- **Trot gait generator.** Diagonal leg pairs swing alternately; each foot follows a swing/stance trajectory.
- **3-D leg inverse kinematics.** Converts each foot target into hip/thigh/calf angles.
- **Yaw stabilisation.** A proportional heading term keeps straight walking straight.
- **Velocity → stride calibration.** Measured tables map commanded `vx` to stride length, and `wz` to a left/right stride difference.
- **`LocomotionInterface(vx, wz)`.** The single entry point used by every later stage. Commands are clamped to the certified envelope, and the heading reference is re-anchored to the actual yaw whenever the command changes.

Two gait profiles are kept in `src/controllers/gait_profiles.py`:

| Profile | Step freq. | Foot lift | Speed envelope | Use |
|---|---|---|---|---|
| `V1_SLOW` (original) | 1 Hz | 3 cm | ≈ 0.03 m/s | Original certification (experiment 15); kept bit-identical |
| `V2_FAST` (default) | 4 Hz | 7 cm | vx −0.3 … 0.5 m/s, \|wz\| ≤ 0.5 rad/s | All Stage 2–3 work |

V2 was added because V1 was too slow for a campus task and could not climb even a 1 cm step.
Experiment 16 confirms that V1 still reproduces its original certification exactly.

### 1.2 V2 certification (experiment 21)
**Result: 16/16 required tests and 4/4 optional tests pass, with no falls.**

| Test | Target | Measured | Error |
|---|---|---|---|
| Forward | 0.10 / 0.30 / 0.50 m/s | 0.101 / 0.300 / 0.491 m/s | 0.7 / 0.2 / 1.8 % |
| Backward | −0.10 / −0.30 m/s | −0.099 / −0.299 m/s | 0.6 / 0.2 % |
| Curve | vx 0.3, wz ±0.5 rad/s | wz ±0.503 rad/s | 0.6 % |
| Rotate in place | wz ±0.3 rad/s | ±0.303 rad/s | ≤ 1.3 % |

- Maximum roll is 5.2° and maximum pitch is 3.4° across all tests.
- Stopping from 0.5 m/s coasts about 2 cm.

### 1.3 Terrain capability (experiment 18, V2 at 0.3 m/s)

| Feature | Pass | Unstable | Fail |
|---|---|---|---|
| Ramp up / down | 3°–15° all pass | — | — |
| Step **up** | 1 cm | 2–3 cm | ≥ 4 cm (falls or blocked) |
| Step **down** | 1–3 cm | 4–8 cm | — |
| Friction change (road → grass, μ 0.6) | pass | — | — |

**Design consequence:** curbs are treated as **no-go lines**, and level changes are reached through
an 8° ramp (a curb cut).

---

## 2. Stage 2 — Campus Environment

![Campus map](../results/figures/campus_map.png)

### 2.1 World
The campus is an 8 × 6 m area inspired by IIT-KGP, built procedurally with `MjSpec` in `src/environment/`.

- **Ground surfaces:**
  - Road at z = 0.
  - Pavement, grass and plaza raised **6 cm** behind a curb.
  - An **8° ramp** (curb cut) at x 3.4–4.6 m connecting the two levels.
- **No-go areas:** planter, stairs, building entrance.
- **Obstacles:** tree, bench, lamp pole, 2 dustbins.
- **Litter:** 4 classes (bottle, can, cup, wrapper), with randomised colour and size, lying or standing.
  Items are mocap bodies, so they can be "collected" (removed) in Stage 7.
- **Distractors:** leaves, stones and sticks. They are litter-like clutter that must *not* be detected.
- **Per-seed randomisation:** litter and distractor placement, plus lighting.
- **Map API:** `zone_at`, `ground_height_at` and `is_traversable` in `layout.py`, ready for Stage 5/6.

### 2.2 Integration test (experiment 19)
A scripted route of 9 waypoints goes road → ramp → pavement → grass around the planter → back down the ramp → home.

![Campus route](../results/figures/campus_route.png)

**Result: 9/9 waypoints in 63.4 s, no falls and no curb contacts.** The route was driven by a simple
heading follower with no planning; real navigation is Stage 6.

**Note on terrain contact:** each A1 leg touches the ground with both the foot and the calf
capsule. Giving the terrain a higher contact priority slowed the gait, so the terrain uses MuJoCo's
default contact with μ = 1.0 (and μ = 0.6 on grass).

---

## 3. Stage 3 — Litter Perception

### 3.1 Sensor
An RGB-D head camera is mounted on the trunk at (0.16, 0, 0.03) m and tilted 25° down.

- Image: 640 × 480, vertical field of view 70°.
- Outputs per frame: RGB, depth and a segmentation image.
- The camera's position and intrinsics are recorded with every frame, for Stage 4.

### 3.2 Dataset (experiment 22)
Labels are generated **automatically** from the simulator's segmentation render, so no manual
annotation is needed. An object counts as labelled if at least 40 visible pixels of it are in view.
Labels are in YOLO format.

| Split | Scenes (seeds) | Images | Purpose |
|---|---|---|---|
| train | 600 (1000–1599) | 3000 | Training |
| val | 100 (2000–2099) | 500 | Threshold tuning, early checks |
| test | 100 (3000–3099) | 500 | Final numbers only |

- Each scene is rendered from 5 robot viewpoints.
- Splits use disjoint seeds, so the test set contains completely unseen layouts.

![Dataset preview](../results/figures/dataset_preview.png)

### 3.3 Detectors (experiment 23)

| | **Main: YOLOv8n CNN** | **Baseline: classical colour** |
|---|---|---|
| Method | COCO-pretrained YOLOv8n, fine-tuned | HSV colour-histogram Bayes classifier, then connected blobs, then a per-blob class vote |
| Training | 50 epochs, 640 px, batch 16, about 74 min on a GTX 1650 (4 GB) | A few seconds (histograms from training masks) |
| Model | `models/litter_yolov8n.pt` | `models/color_baseline.npz` |

### 3.4 Results on the test set (experiment 24)
For each detector, the operating threshold is the confidence that maximises F1 on the **validation** split.
Boxes count as matched at IoU ≥ 0.5.

| Metric | **YOLOv8n (CNN)** | Colour baseline |
|---|---|---|
| mAP@0.5 | **0.965** | 0.342 |
| mAP@0.5:0.95 | **0.818** | 0.181 |
| Precision | **0.948** | 0.671 |
| Recall | **0.939** | 0.420 |
| F1 | **0.944** | 0.516 |
| False positives / image | **0.22** | 0.86 |
| False positives on distractors / image | **0.00** | 0.02 |
| Latency (per image) | **8 ms** (148 img/s, GPU) | 133 ms (7.6 img/s, CPU) |

AP@0.5 per class:

| Class | CNN | Baseline |
|---|---|---|
| bottle | 0.967 | 0.491 |
| can | 0.956 | 0.391 |
| cup | 0.962 | 0.296 |
| wrapper | 0.974 | 0.191 |

Recall by distance:

| Distance | CNN | Baseline |
|---|---|---|
| < 1 m | 0.958 | 0.633 |
| 1–2 m | 0.980 | 0.458 |
| > 2 m | 0.919 | 0.358 |

![Detector comparison](../results/figures/detector_comparison.png)
![Detection examples](../results/figures/detection_examples.png)

**Interpretation:**
- **The colour baseline works only when colour is unique.** Colour alone cannot separate litter
  from same-coloured scenery or other litter classes. It is weakest on wrappers and cups, whose
  colours vary most, and it degrades quickly with distance as objects shrink to a few pixels.
- **The CNN uses shape and context.** It stays above 0.9 recall even beyond 2 m and ignores the
  distractors completely.
- **This justifies using ML here.** The classical method is clearly insufficient, which satisfies
  the brief's rule "classical baseline first, ML only where it measurably helps".

### 3.5 Closed-loop demo (experiment 25)
The robot walks the Stage 2 route while running the detector on its head camera at 5 Hz.
Each frame is scored against ground truth from segmentation.

| | CNN | Colour baseline |
|---|---|---|
| Litter items detected (of 8 in view) | **8 / 8** | 6 / 8 |
| Frame-level precision / recall | **0.976 / 0.976** | 0.792 / 0.679 |
| Detector rate | 71.6 frames/s | ≈ 7 frames/s |

Animations: `results/videos/perception_demo_cnn.gif` and `results/videos/perception_demo_color.gif`.

### 3.6 Known limitations
1. **Blue dustbins are sometimes detected as "can".** Bins were never labelled as negatives. Fix:
   add a `bin` / "not litter" class, or more bin views, to the dataset.
2. **Simulation-to-real gap.** The model is trained and tested only on rendered images, so these
   numbers do not transfer to a real camera.
3. **The detector outputs 2-D boxes.** Stage 4 now supplies world positions through depth
   back-projection and temporal fusion; uncertain robot pose and real-camera transfer remain unverified.

---

## 4. Stage 4 — 3-D Litter Localization

**Status: complete for the verified CNN seed-0 headless route.** The original milestone was
one completed route producing a class-labelled world-position list for every visible item,
with final position error below about 10 cm. The saved result meets that milestone.

### 4.1 Implemented pipeline

Experiment 26 combines the saved Stage 3 CNN, synchronized RGB-D frames and camera pose,
central-crop median depth, world-coordinate back-projection, and persistent track fusion.
Recent observations help associate changing visible surface points with their established
identity. Confidence-weighted class votes and repeated class support determine the confirmed
litter list. The same route evaluates a separate no-depth, flat-ground estimate.

Runtime localization uses no simulator litter identities or true positions. Those remain in
the evaluator. Camera pose is supplied by the simulator, and litter is assumed stationary.
Implementation is in `src/localization/`; the runner is
[experiments/26_localization_pipeline.py](../experiments/26_localization_pipeline.py).

### 4.2 Localization methods and why their results differ

Both methods receive the same CNN detections, camera intrinsics and pose, and sampled
route frames. Each method has its own tracker using the same association, fusion, and
confirmation rules. Simulator truth is used only to evaluate their estimates.

**Depth-based localization (`depth`) — the method used for the Stage 4 litter list.**
Take the median valid depth from the central 60% of each detection box, back-project the
box centre into camera coordinates, and transform that point into world coordinates using
the camera pose. The measured depth supplies the object's distance without assuming a
ground height. The central crop and median reduce background contamination; temporal
position fusion and class voting produce the confirmed item records.

**Flat-ground ray-intersection baseline (`ground_plane`) — the no-depth comparison.**
Take the bottom-centre pixel of the detection box and construct a ray from the camera
through that pixel. Estimate position where the ray intersects a fixed horizontal plane,
`z = 0`. This method uses neither the depth image nor the terrain-height map. It tests how
well localization can work using camera geometry and a simple flat-ground assumption.

**Why depth succeeds on the verified route.** Measured depth locates the visible object
surface in 3-D on the road, raised pavement, grass, plaza, and ramp without forcing all
positions onto one plane. Fusion reduces observation variation, and the corrected
association/confirmation rules preserve one identity per item. The final seed-0 depth
map contains **8/8 items**, with **zero duplicates, misses, or false positives**, mean error
**3.36 cm**, and maximum error **4.92 cm**. It meets the 10 cm final-position requirement.
This is the verified simulation result, not a claim of real-camera performance.

**Why the flat-ground baseline fails.** Pavement, grass, and plaza are raised 6 cm, and
the ramp changes height along its length. Litter centres are also above their support
surface. Intersecting every viewing ray with `z = 0` therefore introduces vertical and
horizontal position bias. For elevated objects, that ray generally reaches the assumed
plane beyond the actual support surface; shallow viewing angles can amplify the error.
The resulting estimates can vary as the robot moves, making association less reliable.
Repeated averaging cannot remove the incorrect ground-plane assumption.

The baseline maps only **7/8 items** and produces **19 confirmed tracks**, of which
**12 are unmatched final estimates** counted as false positives by the evaluator. Its
matched-map mean error is **10.11 cm** and maximum error **19.70 cm**. It fails the
completion criterion because one item is missed, unmatched tracks remain, and matched
positions exceed 10 cm. These counts are measured; attributing every unmatched track
to a specific cause would require further analysis of individual tracks.

**Impact on robot performance.** The flat-ground estimate is an independent evaluation
baseline, not the localization input intended for Stage 5. Its failure does not invalidate
the passing depth map or the observed walking stability: both methods were evaluated
during the same **9/9-waypoint route without a fall**. The current scripted locomotion
controller does not use either litter map for steering. Stage 5 should consume the verified
depth `tracks.json`; navigation and pickup using flat-ground positions have not been validated.

### 4.3 Verified CNN seed-0 result

| Metric | Depth pipeline | Flat-ground baseline |
|---|---|---|
| Visible items mapped | **8/8** | 7/8 |
| Confirmed tracks | **8** | 19 |
| Mean / P95 final error | **3.36 / 4.89 cm** | 10.11 / 19.04 cm |
| Maximum final error | **4.92 cm** | 19.70 cm |
| Duplicates / missed items / false positives | **0 / 0 / 0** | 0 / 1 / 12 |
| Route | **9/9 waypoints, no fall** | Same route |
| Completion criterion | **Passed** | Failed |

The route processed 318 frames at 5 Hz over 63.412 simulated seconds. The command with
`--require-pass` returned exit code 0. All **31 unit tests passed** in the last executed suite.
The 10 cm completion gate applies to the fused final positions; individual observations
have mean error 3.86 cm and P95 error 6.53 cm and can exceed 10 cm.

Evidence: [summary CSV](../results/logs/localization/cnn/summary.csv),
[depth metrics](../results/logs/localization/cnn/seed_0/depth/metrics.json),
[flat-ground metrics](../results/logs/localization/cnn/seed_0/ground_plane/metrics.json),
[confirmed litter list](../results/logs/localization/cnn/seed_0/depth/tracks.json),
[map matches](../results/logs/localization/cnn/seed_0/depth/map_matches.csv), and
[run metadata](../results/logs/localization/cnn/seed_0/run.json). The metadata records the
model checksum, dependency versions, and `viewer_enabled: false`.

![Stage 4 depth and flat-ground maps](../results/logs/localization/cnn/seed_0/map.png)

### 4.4 Duplicate-tracking challenge and mitigation

An earlier full CNN route matched all eight items but confirmed nine tracks. At about
18 seconds, an extra wrapper detection started a tentative track beside an established
bottle. The old nearest-mean association assigned two later bottle observations to the
new track because its fresh position was closer to the currently visible surface. Three
mixed-label observations then confirmed it as another bottle, yielding one duplicate and
a failed completion check despite adequate final position accuracy.

The fix added recent-position evidence to association, required three frames supporting
the winning class before confirmation, and extended duplicate-box suppression to small
contained fragments with matching 3-D positions. One observation per track per frame and
the spatial gates remain enforced. Tests preserve separate nearby objects and objects at
different depths. The corrected saved log assigns the later bottle observations to the
original track; the extra wrapper remains tentative and expires.

Seven new regressions brought the suite from 24 to 31 tests. Four exposed failures in
the old implementation, and all passed after the changes. The observed failure sequence,
mitigation details, test names, and evidence are in the
[Stage 4 localization guide](STAGE_4_LOCALIZATION.md#duplicate-track-failure-and-mitigation).
The earlier failed outputs were overwritten by the successful rerun; the current saved
files report the corrected result.

### 4.5 Validation limits

The full headless seed-0 route is verified. The live viewer received a short startup check
only: `--viewer --max-time 0.6`, four frames, 0.6 simulated seconds, and 0/9 waypoints.
That check opened the window and exercised the litter markers but did not validate a full
viewer route. A subsequent full viewer run is now saved separately: its
[metadata](../results/logs/localization_viewer/cnn/seed_0/run.json) records
`viewer_enabled: true`, 318 frames, 9/9 waypoints, no fall, and about 408 seconds wall time.
Its [depth metrics](../results/logs/localization_viewer/cnn/seed_0/depth/metrics.json) also
report 8/8 items, eight tracks, zero duplicates/misses/false positives, maximum error
4.92 cm, and `stage4_passed: true`. These saved records verify the full viewer route
independently of the earlier short check.

Additional seeds and real-camera transfer remain unverified. The flat-ground baseline's
recorded failure is an ablation result, not a failure of the depth completion gate.
Stage 5 semantic mapping has not yet been implemented.

---

## 5. How to Reproduce

Run from `Robot-Autonomy/` using a working Python environment with the repository's
dependencies and MuJoCo rendering support. For this Windows checkout, reuse the existing
WSL CPU environment through the PowerShell commands in the
[Stage 4 guide](STAGE_4_LOCALIZATION.md#local-windowswsl-execution). The commands below
describe each experiment; the saved Stage 3 models are already available.

```bash
python3 experiments/16_locomotion_interface_check.py   # V1 unchanged
python3 experiments/21_fast_locomotion_test.py         # V2 certification
python3 experiments/18_terrain_traversal_test.py       # curbs / ramps
python3 experiments/17_view_campus.py                  # look at the campus (viewer)
python3 experiments/19_campus_route_demo.py --viewer   # robot walks the route
python3 experiments/22_generate_dataset.py             # ~4000 labelled images
python3 experiments/23_train_detectors.py              # baseline + YOLOv8n (~75 min)
python3 experiments/24_evaluate_detectors.py           # tables + figures above
python3 experiments/25_perception_demo.py              # live demo (add --detector color)
python3 -m unittest discover -s tests -v                # 31 Stage 4 unit tests
python3 experiments/26_localization_pipeline.py --detector cnn --device cpu --seeds 0 --require-pass
```

Outputs:
- `results/figures/`: figures.
- `results/logs/`: CSV logs.
- `results/videos/`: animations.
- `models/`: trained detectors.

---

## 6. Next Steps

### Stage 5 — Semantic mapping (immediate next)

**Goal from the existing roadmap:** create a 2-D grid storing occupancy and terrain type
from `layout.py`, together with Stage 4 litter records containing class, confidence, and
collected / uncollected state. Stage 5 prepares the map consumed by Stage 6 navigation
and the collection-state updates used by Stage 7.

Begin offline with the verified
[depth tracks](../results/logs/localization/cnn/seed_0/depth/tracks.json) and
[`CampusLayout`](../src/environment/layout.py), keeping the run's seed and world frame.
Define the grid indexing and boundary policy, rasterize terrain and obstacle/no-go
geometry, and attach each confirmed item by its track identity while preserving its
full 3-D position. Add collection-state updates, export/reload support, and a figure
showing the grid, blocked regions, base, bins, and litter classes.

The [Stage 5 semantic mapping plan](STAGE_5_SEMANTIC_MAPPING_PLAN.md) lists required inputs,
existing and proposed files, the implementation sequence, expected artifacts, and proposed
acceptance criteria. The first checkpoint is a deterministic seed-0 map that retains all
eight confirmed items, preserves the ramp connection, blocks curbs and obstacles, and
round-trips through serialization. These are planned checks, not completed validation.

### Later stages (planned design)
- **Stage 6 — Navigation.**
  - Classical global planner: A* on the map.
  - **RL (PPO) local navigator**, trained in a fast 2-D abstraction that uses the certified
    (vx, wz) envelope, then tested in MuJoCo against the classical planner.
- **Stage 7 — Pickup.** A simulated scoop or suction action: when the robot is close enough, the
  litter body is removed (`Campus.collect`) and the map is updated.
- **Stage 8 — Task planning.** Choose the next item by distance, terrain risk and priority; add
  battery and storage limits that force returns to base.

### Remaining validation and documentation

- Validate additional CNN seeds; seed-0 headless and full-viewer results are recorded separately.
- Keep the Stage 4 input records and run metadata alongside the Stage 5 map for reproducibility.
- The README now contains the project overview, Stage 4 status, and run commands; this checkout is already a Git repository.
