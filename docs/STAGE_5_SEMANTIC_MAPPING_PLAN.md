# Stage 5: semantic mapping plan

**Status: offline seed-0 implementation complete; acceptance checks passed on 11 Oct 2026.**
See the [Stage 5 implementation report](STAGE_5_SEMANTIC_MAPPING.md) for the
executed checks, API, artifacts, and remaining integration limits. This design
record defines the semantic-map scope already specified in the
[overall progress report](PROGRESS_REPORT_STAGES_1-3.md#6-next-steps).
The original proposed file structure, initial grid resolution, and acceptance
criteria below have now been implemented for the offline milestone.

## Objective

Create a 2-D campus grid containing occupancy, terrain type, height, and
traversability, plus an item registry containing the Stage 4 litter identities,
classes, world positions, confidence information, and collected/uncollected
state. Keep solid-obstacle occupancy separate from traversability: a curb or
stair can be forbidden to the robot even when it is not a solid obstacle.

The roadmap supplies static terrain and obstacle geometry from `CampusLayout`.
This first map therefore uses a known layout and known simulator pose. It does
not establish a sensor-built map or SLAM capability. Litter records come from
Stage 4 estimates; simulator litter identities and true positions are reserved
for evaluation. A* and PPO navigation belong to Stage 6, while physical pickup
belongs to Stage 7.

## Required inputs

| Input | Source and use |
| --- | --- |
| Confirmed litter estimates | [Seed-0 depth tracks](../results/logs/localization/cnn/seed_0/depth/tracks.json). Use the eight confirmed records, retaining `track_id`, class, full position, observation counts, timestamps, scores, class evidence, and position spread. |
| Coordinate contract | Track metadata specifies metres and `mujoco_world_xyz_z_up`. The layout uses x east, y north, z up. Index the grid using x/y while keeping each item's original z. |
| Static campus layout | [layout.py](../src/environment/layout.py): dimensions, surfaces, obstacle shapes, ramp, stairs, curb geometry, base pose, and bins. Use `zone_at`, `ground_height_at`, and `is_traversable` as label/query references. |
| Scene and run provenance | [run.json](../results/logs/localization/cnn/seed_0/run.json): seed 0, detector, model checksum, configuration, and versions. Keep identities scoped to that scene/run; track IDs from different seeds are not interchangeable. |
| Validation evidence | [Depth metrics](../results/logs/localization/cnn/seed_0/depth/metrics.json) and [map matches](../results/logs/localization/cnn/seed_0/depth/map_matches.csv) confirm the Stage 4 input. Use truth matches only for evaluation, not runtime map assembly. |
| Initial item state | Default each imported record to uncollected. Stage 4 has not observed pickup; the existing `Campus.collect` operation is a later Stage 7 integration point. |

Start with the saved depth output, not the failed flat-ground baseline. This
allows map development without running another walking route or retraining the
detector. The map should also accept the same record schema from the live
`LocalizationPipeline.tracker.confirmed_tracks` interface later.

## Relevant files

| Existing file | Role |
| --- | --- |
| [src/environment/layout.py](../src/environment/layout.py) | Pure layout data, zone/height/traversability queries, and static footprints. |
| [src/environment/campus.py](../src/environment/campus.py) | Scene construction and seed context; `Campus.collect` is the later pickup hook. |
| [src/localization/tracking.py](../src/localization/tracking.py) | Confirmed item fields exported by `LitterTrack.as_dict()`. |
| [src/localization/pipeline.py](../src/localization/pipeline.py) | Live source of confirmed litter records. |
| [experiments/26_localization_pipeline.py](../experiments/26_localization_pipeline.py) | Saved-input schema, run manifest, and current evaluation output. |
| [src/environment/plotting.py](../src/environment/plotting.py) | Campus plot to reuse when displaying map layers and item positions. |
| [src/navigation/waypoint_follower.py](../src/navigation/waypoint_follower.py) | Existing scripted-route reference; autonomous map-based planning remains Stage 6. |

Implemented additions are `src/mapping/grid.py`, `src/mapping/semantic_map.py`,
`src/mapping/serialization.py`, `src/mapping/plotting.py`,
`experiments/27_semantic_mapping.py`, and `tests/test_semantic_mapping.py`.
Run `python experiments/27_semantic_mapping.py --require-pass` to build and
validate the saved-input map.

## Implementation design

1. **Define the map and input contracts.** Validate the track schema, finite
   positions, unique IDs, units, frame, and scene seed. Preserve the original
   item records. Use an item registry keyed by track ID within a run, and let
   each grid cell refer to a list of IDs so multiple real items can share a cell.
   Keep detector score, class-vote fraction, and position spread as separate
   fields; none is a calibrated probability or error bound.

2. **Implement coordinates and indexing.** A proposed starting resolution is
   0.10 m at origin `(0, 0)`, yielding 80 columns and 60 rows for the 8 by 6 m
   layout. Columns increase east and rows increase north. Document a half-open
   bounds policy, world-to-cell conversion, cell-centre conversion, and handling
   of invalid/outside points. Grid indexing must not replace an item's accurate
   3-D position with its quantized cell centre.

3. **Rasterize the static layers.** Query terrain labels and heights at cell
   centres, then conservatively rasterize solid footprints and forbidden regions
   using cell/geometry intersections. Centre-only sampling can miss the thin
   lamp pole and curb bands. Preserve the ramp opening while blocking the curb
   outside it, plus stairs, planter, entrance, obstacles, and map boundaries.
   Keep any robot-footprint clearance padding explicit and separate from the
   raw occupancy layer so later planning can inspect both.

4. **Attach litter and state.** Import every confirmed Stage 4 item once and
   associate it with a cell. Preserve class votes, scores, timestamps, counts,
   and the source position. Repeated updates to the same identity should update
   its record, not append another object. Keep collection state through later
   observation updates. A `mark_collected(track_id)`-style map operation should
   update active-item queries without altering terrain or deleting provenance;
   physical removal through `Campus.collect` is deferred to Stage 7.

5. **Export and visualize the map.** Save metadata and items as JSON, grid
   layers in an array format such as NPZ, a summary of counts, and a PNG showing
   terrain, blocked cells, ramp, base, bins, and class-labelled items. Include
   resolution, dimensions, origin, frame, schema version, seed, and Stage 4 run
   provenance. Support reload with the same indexing and item-state semantics.

6. **Validate offline before live integration.** Build a deterministic map from
   the verified seed-0 records, check the criteria below, and compare its display
   with the existing campus plot. Add a live update adapter after the offline
   contract works. Additional layouts and live semantic-map/viewer integration
   remain separate checks; no Stage 5 success should be inferred from Stage 4 alone.

## Expected deliverables

- A reusable semantic-map API for coordinate conversion, terrain/occupancy
  lookup, item updates, active-item queries, collection state, and export/reload.
- An offline experiment consuming the saved Stage 4 records and known layout.
- A serialized map, grid arrays, summary, and figure, proposed under
  `results/logs/semantic_mapping/cnn/seed_0/`.
- Tests for coordinates, boundaries, conservative rasterization, ramp
  connectivity, identity updates, collection state, and serialization.
- A Stage 5 report stating the executed checks, their outcomes, and remaining
  limitations before the overall progress report marks this stage complete.

## Acceptance criteria

These checks make the existing roadmap concrete. All pass for the offline
seed-0 map; see the implementation report and saved `validation.json`.

| Check | Required result |
| --- | --- |
| Grid contract | Correct dimensions, origin, units, frame, axis directions, explicit boundary behavior, and world/cell-centre round trips. |
| Static semantics | Interior terrain labels/heights agree with layout queries. Obstacles and forbidden regions are represented conservatively, including the thin pole and curb. |
| Connectivity | The traversable road connects to the raised pavement through the ramp. A map connectivity check cannot cross the curb outside the opening or enter blocked regions. This validates map topology, not navigation performance. |
| Litter preservation | Exactly eight distinct confirmed seed-0 items with the same IDs, classes, and full 3-D positions as the Stage 4 input. No duplicate records, dropped items, or cell-centre substitution. Preserve all source confidence fields. |
| Updates and collection state | Reimporting an identity is idempotent; two identities in one cell remain separate. Marking an item collected changes active-item queries and survives later updates without changing the static map. |
| Persistence and visualization | Export/reload preserves layers, coordinate metadata, provenance, item records, and collection state. The figure places all eight records in the correct world locations with readable terrain and blocked-region legends. |

The first Stage 5 milestone is complete: the offline seed-0 map passes those checks.
Record additional-seed, live integration, or viewer tests only after they have
actually been executed. Learned navigation, task ordering, battery/storage
limits, and pickup remain later-stage work in the existing roadmap.
