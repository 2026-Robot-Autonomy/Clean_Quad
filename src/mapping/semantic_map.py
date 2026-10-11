"""Known-layout semantic layers and run-scoped Stage 4 litter records."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math

import numpy as np

from src.environment.layout import Rect
from src.mapping.grid import GridSpec, clearance_mask, footprint_mask


FRAME = "mujoco_world_xyz_z_up"
UNITS = "metres"
CLASSES = ("bottle", "can", "cup", "wrapper")
LAYER_NAMES = ("terrain", "height", "occupancy", "forbidden", "boundary",
               "traversable", "clearance_blocked")


def json_copy(value):
    """Reject non-JSON or non-finite evidence before changing any map state."""
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("map records and provenance must be finite JSON data") from error


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def finite_number(value, name, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is out of range")


class SemanticMap:
    """A map belongs to one scene/run. IDs from another run cannot be merged.

    Public item queries return copies. Array layers are read-only. Collection
    state belongs to the map and survives subsequent track observations.
    """

    def __init__(self, grid, layers, *, seed, run_id, provenance=None,
                 landmarks=None, clearance_radius=0.0):
        integer(seed, "seed")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a nonempty scene/run identifier")
        finite_number(clearance_radius, "clearance_radius", minimum=0)
        if set(layers) != set(LAYER_NAMES):
            raise ValueError("map must contain all semantic layers")
        for name, layer in layers.items():
            if layer.shape != grid.shape:
                raise ValueError(f"wrong shape for {name}")
            if name == "terrain":
                if layer.dtype.kind != "U":
                    raise ValueError("terrain must contain Unicode labels")
            elif name == "height":
                if layer.dtype.kind != "f" or not np.all(np.isfinite(layer)):
                    raise ValueError("height must contain finite floats")
            elif layer.dtype != np.dtype(bool):
                raise ValueError(f"{name} must contain booleans")
        blocked = layers["occupancy"] | layers["forbidden"] | layers["boundary"]
        if not np.array_equal(layers["traversable"], ~blocked):
            raise ValueError("traversability contradicts blocked layers")
        if not np.array_equal(layers["clearance_blocked"], clearance_mask(blocked, grid.resolution, clearance_radius)):
            raise ValueError("clearance layer contradicts its radius")
        self.grid, self.seed, self.run_id = grid, seed, run_id
        self.clearance_radius = clearance_radius
        self.provenance = json_copy(provenance or {})
        self.landmarks = json_copy(landmarks or {})
        if not isinstance(self.provenance, dict) or not isinstance(self.landmarks, dict):
            raise ValueError("provenance and landmarks must be JSON objects")
        if "stage4_run" in self.provenance or "track_metadata" in self.provenance:
            run = self.provenance.get("stage4_run")
            metadata = self.provenance.get("track_metadata")
            if not isinstance(run, dict) or not isinstance(metadata, dict):
                raise ValueError("Stage 4 provenance requires both run and track metadata")
            if run.get("seed") != seed or metadata.get("seed") != seed:
                raise ValueError("provenance belongs to a different scene seed")
            if metadata.get("frame") != FRAME or metadata.get("units") != UNITS:
                raise ValueError("provenance uses a different coordinate contract")
            if run.get("detector") != metadata.get("detector") or run.get("config") != metadata.get("config"):
                raise ValueError("provenance track/run detector or config mismatch")
            integer(metadata.get("min_observations"), "min_observations", 1)
        self.layers = {}
        for name, layer in layers.items():
            self.layers[name] = layer.copy()
            self.layers[name].flags.writeable = False
        self._records, self._collected, self._cell_items = {}, {}, {}

    @classmethod
    def from_layout(cls, layout, *, seed, run_id, resolution=0.10,
                    clearance_radius=0.0, provenance=None):
        grid = GridSpec(layout.width, layout.depth, resolution)
        terrain = np.empty(grid.shape, dtype="U16")
        height = np.empty(grid.shape, dtype=float)
        forbidden = np.zeros(grid.shape, dtype=bool)
        occupancy = np.zeros(grid.shape, dtype=bool)
        for row, column in np.ndindex(grid.shape):
            x, y = grid.cell_center(row, column)
            terrain[row, column] = layout.zone_at(x, y)
            height[row, column] = layout.ground_height_at(x, y)
            forbidden[row, column] = not layout.is_traversable(x, y)
        for obstacle in layout.obstacles:
            occupancy |= footprint_mask(grid, obstacle.shape)
        # Positive-area intersections catch raised no-go regions at their edges.
        for surface in layout.surfaces:
            if not layout.traversable[surface.kind]:
                forbidden |= footprint_mask(grid, surface.rect)
        if not layout.traversable["stairs"]:
            forbidden |= footprint_mask(grid, layout.stairs.rect)
        if not layout.traversable["ramp"]:
            forbidden |= footprint_mask(grid, layout.ramp.rect)
        if not layout.traversable["curb"]:
            for x0, x1 in ((0, layout.ramp.x0), (layout.ramp.x1, layout.width)):
                forbidden |= footprint_mask(grid, Rect(x0, layout.curb_y - layout.curb_band,
                                                      x1, layout.curb_y + layout.curb_band))
        # Simulator walls lie outside the domain. Reserve the perimeter cell
        # ring as an explicit boundary policy rather than inventing occupancy.
        boundary = np.zeros(grid.shape, dtype=bool)
        boundary[[0, -1], :] = True
        boundary[:, [0, -1]] = True
        blocked = occupancy | forbidden | boundary
        layers = dict(terrain=terrain, height=height, occupancy=occupancy,
                      forbidden=forbidden, boundary=boundary, traversable=~blocked,
                      clearance_blocked=clearance_mask(blocked, resolution, clearance_radius))
        layout_data = asdict(layout)
        for obstacle, data in zip(layout.obstacles, layout_data["obstacles"]):
            data["shape_type"] = type(obstacle.shape).__name__
        landmarks = {"base_pose": list(layout.base_pose),
                     "bins": [o for o in layout_data["obstacles"] if o["kind"] == "bin"],
                     "ramp": asdict(layout.ramp), "layout": layout_data}
        return cls(grid, layers, seed=seed, run_id=run_id, provenance=provenance,
                   landmarks=landmarks, clearance_radius=clearance_radius)

    @classmethod
    def from_tracks(cls, layout, document, run, *, run_id=None, resolution=0.10,
                    clearance_radius=0.0, source=None):
        document, run = json_copy(document), json_copy(run)
        if not isinstance(document, dict) or not isinstance(run, dict):
            raise ValueError("tracks and run manifest must be JSON objects")
        if document.get("schema_version") != 1 or type(document.get("schema_version")) is not int:
            raise ValueError("unsupported Stage 4 track schema")
        if document.get("units") != UNITS or document.get("frame") != FRAME:
            raise ValueError("Stage 4 tracks must use metres and the MuJoCo world frame")
        if document.get("method") != "depth":
            raise ValueError("Stage 5 requires confirmed depth tracks")
        integer(document.get("seed"), "seed")
        integer(run.get("seed"), "run seed")
        for key in ("seed", "detector", "config"):
            if key not in document or key not in run or document[key] != run[key]:
                raise ValueError(f"track/run {key} mismatch")
        if not isinstance(document["detector"], str) or not document["detector"]:
            raise ValueError("detector must be named")
        methods = run.get("methods")
        if not isinstance(methods, dict) or not isinstance(methods.get("depth"), dict):
            raise ValueError("run manifest requires depth method metadata")
        if not isinstance(document["config"], dict):
            raise ValueError("localization config must be an object")
        method = methods["depth"]
        for key in ("association_radius_m", "class_switch_radius_m", "association_window_s",
                    "confirmation", "min_observations"):
            if key not in document or key not in method or document[key] != method[key]:
                raise ValueError(f"track/run depth {key} mismatch")
        integer(document["min_observations"], "min_observations", 1)
        if document["confirmation"] != "winning_class_frames":
            raise ValueError("unsupported Stage 4 confirmation policy")
        for name in ("association_radius_m", "class_switch_radius_m", "association_window_s"):
            finite_number(document[name], name, minimum=0)
            if document[name] == 0:
                raise ValueError(f"{name} must be positive")
        records = document.get("tracks")
        if not isinstance(records, list):
            raise ValueError("tracks must be a list")
        if run_id is None:
            run_id = "stage4:" + hashlib.sha256(json.dumps(run, sort_keys=True).encode()).hexdigest()
        provenance = {"stage4_run": run,
                      "track_metadata": {k: v for k, v in document.items() if k != "tracks"},
                      "source": source or {}}
        result = cls.from_layout(layout, seed=document["seed"], run_id=run_id,
                                 resolution=resolution, clearance_radius=clearance_radius,
                                 provenance=provenance)
        result.update_tracks(records, seed=result.seed, run_id=result.run_id)
        return result

    def _validate_record(self, record):
        required = {"track_id", "class_id", "class", "position", "position_std", "observations",
                    "class_observations", "mean_score", "max_score", "first_seen", "last_seen",
                    "class_confidence", "class_evidence"}
        if not isinstance(record, dict) or not required.issubset(record):
            raise ValueError("incomplete Stage 4 track record")
        if "cell" in record or "collected" in record:
            raise ValueError("map state must not be supplied as observation evidence")
        integer(record["track_id"], "track_id", 1)
        integer(record["class_id"], "class_id")
        if record["class_id"] >= len(CLASSES) or record["class"] != CLASSES[record["class_id"]]:
            raise ValueError("class name and ID must agree with Stage 4")
        for name in ("position", "position_std"):
            values = record[name]
            if not isinstance(values, list) or len(values) != 3:
                raise ValueError(f"{name} must have three components")
            for value in values:
                finite_number(value, name, minimum=0 if name == "position_std" else None)
        cell = self.grid.world_to_cell(*record["position"][:2])
        for name in ("observations", "class_observations"):
            integer(record[name], name, 1)
        minimum = self.provenance.get("track_metadata", {}).get("min_observations", 1)
        if not minimum <= record["class_observations"] <= record["observations"]:
            raise ValueError("track lacks confirmation evidence or has inconsistent counts")
        for name in ("mean_score", "max_score", "class_confidence"):
            finite_number(record[name], name, minimum=0, maximum=1)
        if record["mean_score"] > record["max_score"] + 1e-12:
            raise ValueError("mean score exceeds max score")
        for name in ("first_seen", "last_seen"):
            finite_number(record[name], name, minimum=0)
        if record["first_seen"] > record["last_seen"]:
            raise ValueError("first_seen must precede last_seen")
        evidence = record["class_evidence"]
        if not isinstance(evidence, dict) or record["class"] not in evidence:
            raise ValueError("class_evidence must include the selected class")
        for name, weight in evidence.items():
            if name not in CLASSES:
                raise ValueError("unknown class evidence")
            finite_number(weight, "class evidence weight", minimum=0)
        if evidence[record["class"]] <= 0:
            raise ValueError("selected class must have positive evidence")
        return cell

    def update_tracks(self, records, *, seed, run_id):
        """Validate an entire batch before updating; repeated identities replace.

        Records may be dictionaries or confirmed LitterTrack objects. A caller
        must retain the same run_id and start a new map after tracker.reset().
        """
        integer(seed, "seed")
        if seed != self.seed or run_id != self.run_id:
            raise ValueError("cannot mix track identities from different scenes/runs")
        normalized = []
        for record in records:
            if hasattr(record, "as_dict"):
                if not getattr(record, "confirmed", False):
                    raise ValueError("only confirmed live tracks may enter the map")
                record = record.as_dict()
            normalized.append(json_copy(record))
        cells = [self._validate_record(record) for record in normalized]
        ids = [r["track_id"] for r in normalized]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate track IDs in an update batch")
        for record, cell in zip(normalized, cells):
            identity = record["track_id"]
            if identity in self._records:
                old_cell = self.grid.world_to_cell(*self._records[identity]["position"][:2])
                self._cell_items[old_cell].remove(identity)
                if not self._cell_items[old_cell]:
                    del self._cell_items[old_cell]
            self._records[identity] = record
            self._collected.setdefault(identity, False)
            self._cell_items.setdefault(cell, set()).add(identity)

    def item(self, track_id):
        record = deepcopy(self._records[track_id])
        return {**record, "cell": list(self.grid.world_to_cell(*record["position"][:2])),
                "collected": self._collected[track_id]}

    @property
    def items(self):
        return {identity: self.item(identity) for identity in sorted(self._records)}

    @property
    def source_records(self):
        return [deepcopy(self._records[i]) for i in sorted(self._records)]

    def active_items(self):
        return [self.item(i) for i in sorted(self._records) if not self._collected[i]]

    def items_in_cell(self, row, column):
        self.grid.validate_cell(row, column)
        return [self.item(i) for i in sorted(self._cell_items.get((row, column), ()))]

    def mark_collected(self, track_id):
        if track_id not in self._records:
            raise KeyError(track_id)
        self._collected[track_id] = True

    def lookup(self, x, y):
        cell = self.grid.world_to_cell(x, y)
        return {**{name: layer[cell].item() for name, layer in self.layers.items()},
                "cell": list(cell),
                "item_ids": sorted(self._cell_items.get(cell, ()))}

    def summary(self):
        labels, counts = np.unique(self.layers["terrain"], return_counts=True)
        return {"seed": self.seed, "run_id": self.run_id, "rows": self.grid.rows,
                "columns": self.grid.columns, "resolution_m": self.grid.resolution,
                "items": len(self._records), "active_items": len(self.active_items()),
                "collected_items": sum(self._collected.values()),
                "terrain_cells": dict(zip(labels.tolist(), counts.tolist())),
                "layer_cells": {name: int(self.layers[name].sum()) for name in LAYER_NAMES[2:]}}
