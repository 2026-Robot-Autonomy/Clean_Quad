# Stage 5: semantic mapping

**Status: offline seed-0 milestone complete and verified on 11 Oct 2026.**
The map combines the known campus layout with all eight confirmed Stage 4 CNN
depth tracks. All 12 offline acceptance checks and all 54 unit tests pass
(31 existing localization tests plus 23 mapping tests).

## Reproduce

From the repository root with NumPy and Matplotlib installed:

```bash
python -B experiments/27_semantic_mapping.py --require-pass
python -B -m unittest discover -s tests -v
```

The full test suite also uses the existing Stage 4 dependencies. No MuJoCo
rendering, detector inference, walking route, or model training is needed to
build the offline map. In this Windows checkout, the executed commands were:

```powershell
wsl -d kali-linux -- /tmp/cleanquad-stage4-venv/bin/python -B experiments/27_semantic_mapping.py --require-pass
wsl -d kali-linux -- /tmp/cleanquad-stage4-venv/bin/python -B -m unittest discover -s tests -v
```

Defaults read `results/logs/localization/cnn/seed_0/depth/tracks.json` and that
run's `run.json`. Options include `--tracks`, `--run`, `--output`, `--resolution`,
`--clearance-radius`, `--expected-items` (default 8), and `--no-figure`.
Dimensions must be whole multiples of resolution. `--require-pass` exits 1 if
any acceptance check fails; invalid input exits with an argument error.

## Map contract

The default 8 by 6 m domain has 80 columns and 60 rows at 0.10 m resolution.
The origin is `(0, 0)`, units are metres, and the frame is
`mujoco_world_xyz_z_up`: x east, y north, z up. Arrays use `[row, column]`;
rows increase north and columns east. World bounds are half-open:
`0 <= x < 8`, `0 <= y < 6`. Invalid/non-finite coordinates raise `ValueError`;
outside points and outside cell indices raise `IndexError`.

| Layer | Meaning |
| --- | --- |
| `terrain` | `CampusLayout.zone_at` label at each cell centre. |
| `height` | Ground height in metres at the centre, ignoring obstacle tops. |
| `occupancy` | Solid obstacle footprints, including trees, bench, pole, and bins. |
| `forbidden` | Nontraversable layout queries, conservatively extended across curb bands, stairs, planter, and entrance. |
| `boundary` | Explicit nontraversable perimeter cell ring. Simulator walls lie outside the domain, so this policy is separate from occupancy. |
| `traversable` | Complement of occupancy, forbidden regions, and boundary. |
| `clearance_blocked` | Optional conservative expansion of blocked cell squares by a clearance radius. Default radius 0; raw occupancy stays unchanged. |

Rectangle footprints require positive-area cell overlap. Shared edges do not
block neighbouring cells, preserving the ramp opening. Circle intersections
include tangencies. Rasterization uses footprint intersections rather than
centre-only occupancy sampling, so even a pole invisible to all cell centres
is represented. Terrain labels and heights still describe the cell centre;
an occupied cell need not have an `obstacle` terrain label.

The perimeter policy reserves one cell inside the domain. Clearance padding
is an explicit planning option, not a validation of the A1's body footprint.
The current default map has 92 occupied cells, 484 forbidden cells, 276
boundary cells, 764 cells blocked in their union, and 4,036 traversable cells.

## Litter identities, evidence, and collection

Each map belongs to one seed/run namespace. Offline `run_id` defaults to the
SHA-256 of the canonical Stage 4 run manifest, which includes model checksum,
configuration, and versions. The metadata checks the schema, frame, units,
seed, detector, localization configuration, and depth tracking settings.
The failed flat-ground input is rejected.

Every item retains its original track ID, class ID/name, full 3-D position,
position spread, observation counts, first/last timestamps, mean/max detector
scores, class confidence, and class evidence. It also has a grid cell and a
map-owned `collected` flag, initially false. The grid association never
quantizes the source position. Multiple distinct items may share a cell.

Repeated identities update their record and cell membership; they do not add
duplicates or reset collection state. Updates validate the whole batch before
changing map state. Duplicate IDs, invalid fields, unconfirmed live tracks,
and different seed/run namespaces are rejected. Item queries return copies,
and grid arrays are read-only. Detector scores, class-vote fractions, and
position spread remain separate evidence fields; they are not calibrated
probabilities or error bounds.

```python
import json
from pathlib import Path
from src.environment.layout import default_layout
from src.mapping import SemanticMap, load_map, save_map

source = Path("results/logs/localization/cnn/seed_0")
tracks = json.loads((source / "depth/tracks.json").read_text())
run = json.loads((source / "run.json").read_text())
semantic_map = SemanticMap.from_tracks(default_layout(), tracks, run)

cell = semantic_map.grid.world_to_cell(4.0, 2.5)  # (25, 40)
terrain = semantic_map.lookup(4.0, 2.5)
items_here = semantic_map.items_in_cell(*cell)
targets = semantic_map.active_items()
semantic_map.mark_collected(targets[0]["track_id"])
save_map(semantic_map, "results/logs/my_map")
restored = load_map("results/logs/my_map")
```

`mark_collected` changes map state only. Physical pickup through
`Campus.collect` remains Stage 7 work.

The update interface also accepts confirmed `LitterTrack` objects directly:

```python
live_map = SemanticMap.from_layout(default_layout(), seed=0, run_id="episode-1")
# After the existing LocalizationPipeline processes a frame:
live_map.update_tracks(pipeline.tracker.confirmed_tracks,
                       seed=0, run_id="episode-1")
```

Use a new run ID and new map after resetting the tracker or scene; track IDs
can be reused in later episodes. The confirmed-track adapter has a synthetic
tracker regression test. A full live semantic-map route/viewer has not been run.

## Saved artifacts

Artifacts are under `results/logs/semantic_mapping/cnn/seed_0/`:

| File | Contents |
| --- | --- |
| [map.json](../results/logs/semantic_mapping/cnn/seed_0/map.json) | Versioned coordinates, grid dimensions, layers checksum, seed/run namespace, complete source provenance, layout/landmarks, and item records/state. |
| [layers.npz](../results/logs/semantic_mapping/cnn/seed_0/layers.npz) | The seven grid layers; reload uses `allow_pickle=False`. |
| [summary.json](../results/logs/semantic_mapping/cnn/seed_0/summary.json) | Terrain/layer cell counts and active/collected item counts. |
| [validation.json](../results/logs/semantic_mapping/cnn/seed_0/validation.json) | Results of all 12 acceptance checks and `stage5_passed: true`. |
| [map.png](../results/logs/semantic_mapping/cnn/seed_0/map.png) | Semantic grid beside the existing campus plot, with blocked cells, base, bins, ramp, and class-labelled litter. |
| `stage4_tracks.json`, `stage4_run.json` | Byte-for-byte input snapshots; their SHA-256 hashes are recorded in provenance. |

Reload validates the coordinate contract, array checksum, shapes/dtypes,
traversability/clearance consistency, source scene, cell associations, and
collection-state types. Collection state and original evidence survive reload.

## Executed validation

The final full suite executed 54 tests in 10.272 seconds and exited 0.
The standalone offline experiment also exited 0, with these results:

| Check | Result |
| --- | --- |
| Grid coordinates, dimensions, and all 4,800 cell-centre round trips | Pass |
| Terrain and height agreement with the layout at every centre | Pass |
| Conservative coverage of every solid footprint | Pass |
| Forbidden-region coverage | Pass |
| Boundaries blocked and occupied cells nontraversable | Pass |
| Four-neighbour road-to-pavement connectivity through ramp | Pass |
| No curb bypass when the ramp passage is closed | Pass |
| Exact preservation of all eight confirmed litter records | Pass |
| Export/reload equality of layers, metadata, provenance, and items | Pass |
| Idempotent reimport of track IDs | Pass |
| Collected state persists through observation updates | Pass |
| Multiple identities in one grid cell stay distinct | Pass |

The unit suite additionally covers invalid coordinates and fractional origins,
thin off-centre poles, clearance separation, atomic invalid updates, namespace
mixing, malformed metadata, position/class updates, synthetic live tracks,
collected-state persistence after reload, tampered artifacts, and CLI failure
when the expected item count is wrong. The generated figure was visually
inspected: all eight labelled items align with the known campus plot and the
ramp opening remains visible.

## Remaining scope

This is the Stage 5 offline known-layout milestone. Static terrain comes from
`CampusLayout`; litter positions come from Stage 4 estimates. Simulator litter
identities and true positions are not used to assemble the map. Known simulator
pose remains an assumption. Additional scene seeds/layouts, moving litter,
uncertain pose, and a full live map/viewer integration remain unverified.
SLAM, A*/PPO navigation, physical pickup, and task ordering belong to later work.
