# CleanQuad: campus litter collection

MuJoCo simulation of a Unitree A1 quadruped that detects and localizes campus
litter. The existing locomotion controller, campus environment, and trained
detectors and localization results are described in the
[complete report through Stage 5](docs/CLEANQUAD_COMPLETE_REPORT_STAGES_1-5.md).

Stage 4 adds a reusable RGB-D localization and tracking pipeline. It exports
one world-position estimate per confirmed litter track, and evaluates depth
against a flat-ground estimate on the same walking route.

Use Python with the dependencies in `requirements.txt` installed and an OpenGL
backend supported by MuJoCo. The saved CNN weights are
`models/litter_yolov8n.pt`; retraining is unnecessary for Stage 4.

```bash
python -m unittest discover -s tests -v
python experiments/26_localization_pipeline.py --detector cnn --seeds 0 --require-pass
python experiments/27_semantic_mapping.py --require-pass
```

Results are written to `results/logs/localization/cnn/`, including tracks,
error metrics, observation logs, and comparison figures. See the
[Stage 4 pipeline reference](docs/CLEANQUAD_COMPLETE_REPORT_STAGES_1-5.md#stage4-technical-reference) for the workflow,
validation criteria, multiple-seed runs, and integration with Stage 5.

The verified CPU CNN run on seed 0 passes Stage 4: 8/8 visible items mapped,
zero duplicates, misses, or false positives, and maximum final position error
4.92 cm. A separately saved full viewer run also passes
on seed 0. Other seeds have not yet been validated.

Stage 5's offline seed-0 map is implemented and verified: an 80 × 60 grid at
10 cm resolution stores terrain, height, occupancy, traversability, boundaries,
and optional clearance padding, with all eight precise litter records and
collection state. All 54 unit tests and 12 offline acceptance checks pass.
The map, input snapshots, validation results, and figure are saved under
`results/logs/semantic_mapping/cnn/seed_0/`. See the
[Stage 5 implementation reference](docs/CLEANQUAD_COMPLETE_REPORT_STAGES_1-5.md#stage5-implementation) for the API, reproduction
commands, coordinate contract, and validation limits. Stage 6 now connects this
map to continuous live RGB-D/CNN updates and navigation.

Stage 6 is implemented and verified for the known static campus: clearance-aware
A*, the classical baseline, and an imitation-initialized PPO local navigator.
Three CPU PPO runs each completed 1,001,472 transitions, and each selected model
passed 100/100 fresh local-goal test cases. The combined MuJoCo comparison passed
59/59 distinct routes, and the continuous live CNN/map/PPO viewer mission visited
seven reachable targets and returned to base without resetting the simulation.
All 93 regression tests pass. One of eight known items is unreachable under the
conservative 0.45 m circular footprint; the live map retains one unmatched
detection. Robot pose is supplied by the simulator; pickup and task planning are
Stages 7 and 8. See the [Stage 6 final report](docs/STAGE_6_FINAL_REPORT.md) for the
complete pipeline, every checkpoint's transitions/success rate, compute,
comparison figures, live mission results and limitations. The original
[baseline report](docs/STAGE_6_NAVIGATION_BASELINE.md) remains historical evidence.

```bash
python -B experiments/28_navigation_baseline.py --diagnostic-seeds 1 2 --require-pass
python -B experiments/29_train_navigation.py --seeds 0 1 2 --timesteps 1000000 --output models/navigation/ppo_full --initial-policy models/navigation/ppo/seed_0/imitation.zip
python -B experiments/30_evaluate_navigation.py --models models/navigation/ppo_full --output results/logs/navigation/comparison_full/full --policy-seeds 0 1 2 --scene-seeds 0 1 2 --require-pass
python -B experiments/31_live_navigation.py --checkpoint models/navigation/ppo_full/seed_0/best.zip --camera-rate 1 --viewer --require-pass
```

The trained models already exist under `models/navigation/ppo_full/seed_*/`.
Use `best.zip` for deployment; the last training update is `final.zip` and the
supervised initializer is `imitation.zip`. The final report includes the exact
split comparison commands used for its recorded results. The RL environments
are fast 2-D abstractions; MuJoCo supplies the independent robot-physics checks.
