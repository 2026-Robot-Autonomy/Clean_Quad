# CleanQuad: campus litter collection

MuJoCo simulation of a Unitree A1 quadruped that detects and localizes campus
litter. The existing locomotion controller, campus environment, and trained
detectors and localization results are described in the
[Stages 1–4 progress report](docs/PROGRESS_REPORT_STAGES_1-3.md).

Stage 4 adds a reusable RGB-D localization and tracking pipeline. It exports
one world-position estimate per confirmed litter track, and evaluates depth
against a flat-ground estimate on the same walking route.

Use Python with the dependencies in `requirements.txt` installed and an OpenGL
backend supported by MuJoCo. The saved CNN weights are
`models/litter_yolov8n.pt`; retraining is unnecessary for Stage 4.

```bash
python -m unittest discover -s tests -v
python experiments/26_localization_pipeline.py --detector cnn --seeds 0 --require-pass
```

Results are written to `results/logs/localization/cnn/`, including tracks,
error metrics, observation logs, and comparison figures. See the
[Stage 4 pipeline guide](docs/STAGE_4_LOCALIZATION.md) for the workflow,
validation criteria, multiple-seed runs, and integration with Stage 5.

The verified CPU CNN run on seed 0 passes Stage 4: 8/8 visible items mapped,
zero duplicates, misses, or false positives, and maximum final position error
4.92 cm. All 31 unit tests pass. A separately saved full viewer run also passes
on seed 0. Other seeds have not yet been validated.

Stage 5 is next: a semantic grid combining the known campus layout with the
confirmed litter records and collection state. See the
[Stage 5 mapping plan](docs/STAGE_5_SEMANTIC_MAPPING_PLAN.md) for inputs,
implementation steps, deliverables, and proposed acceptance checks.
