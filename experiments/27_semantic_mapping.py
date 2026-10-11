"""Build and check the offline Stage 5 map; no walking or retraining required.

python experiments/27_semantic_mapping.py --require-pass
"""

import argparse
from collections import deque
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.environment.layout import Rect, default_layout
from src.mapping import SemanticMap, load_map, save_map
from src.mapping.grid import footprint_mask
from src.mapping.plotting import save_figure


INPUT_ROOT = PROJECT_ROOT / "results/logs/localization/cnn/seed_0"
OUTPUT_ROOT = PROJECT_ROOT / "results/logs/semantic_mapping/cnn/seed_0"


def reachable(mask, start):
    """Four-neighbour topology check, not a navigation controller."""
    if not mask[start]:
        return set()
    visited, queue = {start}, deque([start])
    while queue:
        row, column = queue.popleft()
        for cell in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)):
            if (0 <= cell[0] < mask.shape[0] and 0 <= cell[1] < mask.shape[1]
                    and mask[cell] and cell not in visited):
                visited.add(cell)
                queue.append(cell)
    return visited


def acceptance_checks(semantic_map, reloaded, layout, tracks, expected_items):
    grid, layers = semantic_map.grid, semantic_map.layers
    checks = {}
    checks["grid_contract"] = (grid.origin == (0, 0)
        and grid.shape == (round(layout.depth / grid.resolution), round(layout.width / grid.resolution))
        and all(grid.world_to_cell(*grid.cell_center(*cell)) == cell for cell in np.ndindex(grid.shape)))
    checks["terrain_and_height"] = all(
        layers["terrain"][cell] == layout.zone_at(*grid.cell_center(*cell))
        and layers["height"][cell] == layout.ground_height_at(*grid.cell_center(*cell))
        for cell in np.ndindex(grid.shape))
    checks["solid_footprints"] = all(
        np.any(mask := footprint_mask(grid, obstacle.shape)) and np.all(layers["occupancy"][mask])
        for obstacle in layout.obstacles)
    forbidden_shapes = [surface.rect for surface in layout.surfaces if not layout.traversable[surface.kind]]
    forbidden_shapes.append(layout.stairs.rect)
    forbidden_shapes.extend(Rect(x0, layout.curb_y - layout.curb_band, x1, layout.curb_y + layout.curb_band)
                            for x0, x1 in ((0, layout.ramp.x0), (layout.ramp.x1, layout.width)))
    checks["forbidden_regions"] = all(np.all(~layers["traversable"][footprint_mask(grid, shape)])
                                      for shape in forbidden_shapes)
    checks["boundaries_and_occupancy"] = bool(
        np.all(layers["boundary"][[0, -1], :]) and np.all(layers["boundary"][:, [0, -1]])
        and not np.any(layers["traversable"] & layers["occupancy"]))
    start = grid.world_to_cell(*layout.base_pose[:2])
    target = grid.world_to_cell((layout.ramp.x0 + layout.ramp.x1) / 2, layout.curb_y + 0.3)
    reachable_cells = reachable(layers["traversable"], start)
    checks["road_to_pavement_via_ramp"] = target in reachable_cells
    ramp_cut = layers["traversable"].copy()
    ramp_cut[footprint_mask(grid, Rect(layout.ramp.x0, layout.curb_y - layout.curb_band,
                                     layout.ramp.x1, layout.curb_y + layout.curb_band))] = False
    checks["no_curb_bypass"] = target not in reachable(ramp_cut, start)
    checks["litter_preserved"] = (
        len(semantic_map.items) == expected_items
        and semantic_map.source_records == sorted(tracks["tracks"], key=lambda item: item["track_id"])
        and len(semantic_map.active_items()) == expected_items)
    checks["export_reload"] = (
        reloaded.grid == grid and reloaded.items == semantic_map.items
        and reloaded.seed == semantic_map.seed and reloaded.run_id == semantic_map.run_id
        and reloaded.provenance == semantic_map.provenance and reloaded.landmarks == semantic_map.landmarks
        and reloaded.clearance_radius == semantic_map.clearance_radius
        and all(np.array_equal(reloaded.layers[name], layer) for name, layer in layers.items()))
    reloaded.update_tracks(tracks["tracks"], seed=reloaded.seed, run_id=reloaded.run_id)
    checks["idempotent_import"] = reloaded.items == semantic_map.items
    checks["collection_state"] = False
    checks["multiple_items_per_cell"] = False
    if tracks["tracks"]:
        identity = tracks["tracks"][0]["track_id"]
        reloaded.mark_collected(identity)
        reloaded.update_tracks(tracks["tracks"], seed=reloaded.seed, run_id=reloaded.run_id)
        checks["collection_state"] = (
            reloaded.item(identity)["collected"] and len(reloaded.active_items()) == expected_items - 1
            and all(np.array_equal(reloaded.layers[name], layer) for name, layer in layers.items()))
        second = deepcopy(tracks["tracks"][0])
        second["track_id"] = max(reloaded.items) + 1
        reloaded.update_tracks([second], seed=reloaded.seed, run_id=reloaded.run_id)
        cell = reloaded.item(identity)["cell"]
        checks["multiple_items_per_cell"] = {identity, second["track_id"]}.issubset(
            {item["track_id"] for item in reloaded.items_in_cell(*cell)})
    return {name: bool(value) for name, value in checks.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tracks", type=Path, default=INPUT_ROOT / "depth/tracks.json")
    parser.add_argument("--run", type=Path, default=INPUT_ROOT / "run.json")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--resolution", type=float, default=0.10)
    parser.add_argument("--clearance-radius", type=float, default=0.0)
    parser.add_argument("--expected-items", type=int, default=8)
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    if args.expected_items < 1:
        parser.error("--expected-items must be positive")
    try:
        tracks_bytes, run_bytes = args.tracks.read_bytes(), args.run.read_bytes()
        tracks = json.loads(tracks_bytes.decode("utf-8"))
        run = json.loads(run_bytes.decode("utf-8"))
        source = {"tracks_path": str(args.tracks.resolve()), "run_path": str(args.run.resolve()),
                  "tracks_sha256": hashlib.sha256(tracks_bytes).hexdigest(),
                  "run_sha256": hashlib.sha256(run_bytes).hexdigest(),
                  "tracks_snapshot": "stage4_tracks.json", "run_snapshot": "stage4_run.json"}
        layout = default_layout()
        semantic_map = SemanticMap.from_tracks(layout, tracks, run, resolution=args.resolution,
                                               clearance_radius=args.clearance_radius, source=source)
        save_map(semantic_map, args.output)
        (args.output / "stage4_tracks.json").write_bytes(tracks_bytes)
        (args.output / "stage4_run.json").write_bytes(run_bytes)
        reloaded = load_map(args.output)
        checks = acceptance_checks(semantic_map, reloaded, layout, tracks, args.expected_items)
        if not args.no_figure:
            save_figure(semantic_map, layout, args.output / "map.png")
        validation = {"stage5_passed": all(checks.values()), "checks": checks,
                      "scope": "offline known-layout map; seed/run scoped; no navigation or pickup",
                      "summary": semantic_map.summary()}
        (args.output / "validation.json").write_text(json.dumps(validation, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (ValueError, IndexError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    print(f"Saved {len(semantic_map.items)} items in {semantic_map.grid.rows}x{semantic_map.grid.columns} grid to {args.output}")
    return 1 if args.require_pass and not validation["stage5_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
