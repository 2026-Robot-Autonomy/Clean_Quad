"""Portable JSON metadata/items and non-pickled NumPy grid layers."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from src.mapping.grid import GridSpec
from src.mapping.semantic_map import FRAME, LAYER_NAMES, UNITS, SemanticMap


def save_map(semantic_map, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    arrays_path = destination / "layers.npz"
    np.savez_compressed(arrays_path, **semantic_map.layers)
    document = {
        "schema_version": 1, "units": UNITS, "frame": FRAME,
        "axis_order": "row_y_north_column_x_east", "bounds": "half_open",
        "boundary_policy": "nontraversable_perimeter_cell_ring",
        "grid": asdict(semantic_map.grid), "seed": semantic_map.seed,
        "run_id": semantic_map.run_id, "clearance_radius_m": semantic_map.clearance_radius,
        "provenance": semantic_map.provenance, "landmarks": semantic_map.landmarks,
        "layers_sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
        "items": [{"track": record, "collected": semantic_map.item(record["track_id"])["collected"],
                   "cell": semantic_map.item(record["track_id"])["cell"]}
                  for record in semantic_map.source_records],
    }
    (destination / "map.json").write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (destination / "summary.json").write_text(json.dumps(semantic_map.summary(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return destination / "map.json"


def load_map(destination):
    destination = Path(destination)
    document = json.loads((destination / "map.json").read_text(encoding="utf-8"))
    expected = {"schema_version": 1, "units": UNITS, "frame": FRAME,
                "axis_order": "row_y_north_column_x_east", "bounds": "half_open",
                "boundary_policy": "nontraversable_perimeter_cell_ring"}
    if not isinstance(document, dict) or any(document.get(k) != v for k, v in expected.items()):
        raise ValueError("unsupported semantic-map coordinate/schema contract")
    if type(document["schema_version"]) is not int:
        raise ValueError("schema_version must be an integer")
    arrays_path = destination / "layers.npz"
    if hashlib.sha256(arrays_path.read_bytes()).hexdigest() != document.get("layers_sha256"):
        raise ValueError("grid array checksum mismatch")
    grid = GridSpec(**document["grid"])
    with np.load(arrays_path, allow_pickle=False) as arrays:
        if set(arrays.files) != set(LAYER_NAMES):
            raise ValueError("serialized map has missing or extra layers")
        layers = {name: arrays[name] for name in LAYER_NAMES}
    result = SemanticMap(grid, layers, seed=document["seed"], run_id=document["run_id"],
                         clearance_radius=document["clearance_radius_m"],
                         provenance=document["provenance"], landmarks=document["landmarks"])
    entries = document["items"]
    if not isinstance(entries, list):
        raise ValueError("serialized items must be a list")
    result.update_tracks([entry["track"] for entry in entries], seed=result.seed, run_id=result.run_id)
    for entry in entries:
        identity = entry["track"]["track_id"]
        if entry["cell"] != result.item(identity)["cell"] or type(entry["collected"]) is not bool:
            raise ValueError("invalid serialized cell or collection state")
        if entry["collected"]:
            result.mark_collected(identity)
    return result
