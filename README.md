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
commands, coordinate contract, and validation limits. Full live map integration
and Stage 6 navigation remain later work.
