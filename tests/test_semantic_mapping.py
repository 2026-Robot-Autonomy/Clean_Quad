"""Stage 5 acceptance and regression checks using the saved Stage 4 input."""

from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from src.environment.layout import Circle, Obstacle, Rect, default_layout
from src.localization.geometry import LocalizedDetection
from src.localization.tracking import LitterTracker
from src.mapping import GridSpec, SemanticMap, load_map, save_map
from src.mapping.grid import footprint_mask


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/logs/localization/cnn/seed_0"
TRACKS = json.loads((INPUT / "depth/tracks.json").read_text(encoding="utf-8"))
RUN = json.loads((INPUT / "run.json").read_text(encoding="utf-8"))
SPEC = importlib.util.spec_from_file_location("stage5_experiment", ROOT / "experiments/27_semantic_mapping.py")
EXPERIMENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPERIMENT)


def build(**kwargs):
    return SemanticMap.from_tracks(default_layout(), TRACKS, RUN, **kwargs)


class GridTests(unittest.TestCase):
    def test_default_axes_and_all_cell_round_trips(self):
        grid = GridSpec(8, 6)
        self.assertEqual(grid.shape, (60, 80))
        self.assertEqual(grid.world_to_cell(0.3, 0.2), (2, 3))
        for cell in np.ndindex(grid.shape):
            self.assertEqual(grid.world_to_cell(*grid.cell_center(*cell)), cell)

    def test_half_open_edges_and_invalid_indices(self):
        grid = GridSpec(8, 6)
        self.assertEqual(grid.world_to_cell(0, 0), (0, 0))
        self.assertEqual(grid.world_to_cell(np.nextafter(8., 0), np.nextafter(6., 0)), (59, 79))
        self.assertEqual(grid.world_to_cell(np.nextafter(.3, 0), .2), (2, 2))
        for point in ((8, 1), (1, 6), (-1e-8, 0), (0, -1e-8)):
            with self.assertRaises(IndexError):
                grid.world_to_cell(*point)
        for point in ((np.nan, 0), (0, np.inf), (True, 0), ("0", 0)):
            with self.assertRaises(ValueError):
                grid.world_to_cell(*point)
        for cell, error in (((-1, 0), IndexError), ((60, 0), IndexError), ((1.2, 0), ValueError), ((True, 0), ValueError)):
            with self.assertRaises(error):
                grid.cell_center(*cell)

    def test_nonzero_origin_and_bad_grid_dimensions(self):
        grid = GridSpec(2, 1, .25, (-1, 3))
        self.assertEqual(grid.world_to_cell(-.5, 3.25), (1, 2))
        self.assertEqual(grid.cell_center(1, 2), (-.375, 3.375))
        for args in ((0, 6), (8, -1), (8, 6, 0), (8, 6, np.nan), (8, 6, .3), (8, 6, .1, (np.inf, 0))):
            with self.assertRaises(ValueError):
                GridSpec(*args)

    def test_fractional_origin_does_not_admit_an_extra_boundary_cell(self):
        grid = GridSpec(.2, .2, .1, (.1, .1))
        self.assertEqual(grid.world_to_cell(np.nextafter(.3, 0), .2), (1, 1))
        with self.assertRaises(IndexError):
            grid.world_to_cell(.3, .2)
        for origin in (None, (False, 0), ("0", 0)):
            with self.assertRaises(ValueError):
                GridSpec(1, 1, origin=origin)


class StaticLayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layout, cls.map = default_layout(), build()

    def test_labels_and_heights_agree_with_layout_everywhere(self):
        for cell in np.ndindex(self.map.grid.shape):
            x, y = self.map.grid.cell_center(*cell)
            self.assertEqual(self.map.layers["terrain"][cell], self.layout.zone_at(x, y))
            self.assertEqual(self.map.layers["height"][cell], self.layout.ground_height_at(x, y))
            if self.map.layers["traversable"][cell]:
                self.assertTrue(self.layout.is_traversable(x, y))

    def test_circle_and_rectangle_intersections_have_expected_cells(self):
        grid = GridSpec(1, 1)
        rect = footprint_mask(grid, Rect(.205, .205, .215, .215))
        self.assertEqual(np.argwhere(rect).tolist(), [[2, 2]])
        pole = footprint_mask(grid, Circle(.4, .4, .005))
        self.assertEqual(np.argwhere(pole).tolist(), [[3, 3], [3, 4], [4, 3], [4, 4]])
        aligned = footprint_mask(grid, Rect(.3, .2, .4, .3))
        self.assertEqual(np.argwhere(aligned).tolist(), [[2, 3]])
        with self.assertRaises(TypeError):
            footprint_mask(grid, object())

    def test_thin_obstacle_is_blocked_when_no_cell_centre_sees_it(self):
        pole = Obstacle("tiny_pole", "pole", Circle(5.4, 2.7, .005), 2)
        layout = replace(self.layout, obstacles=(pole,))
        mapped = SemanticMap.from_layout(layout, seed=0, run_id="thin-pole")
        self.assertFalse(np.any(mapped.layers["terrain"] == "obstacle"))
        self.assertEqual(int(mapped.layers["occupancy"].sum()), 4)
        self.assertFalse(np.any(mapped.layers["traversable"] & mapped.layers["occupancy"]))

    def test_curb_is_forbidden_but_not_solid_occupancy(self):
        for x in (1., 2., 5., 6.):
            for y in (1.975, 2.025):
                cell = self.map.grid.world_to_cell(x, y)
                self.assertTrue(self.map.layers["forbidden"][cell])
                self.assertFalse(self.map.layers["occupancy"][cell])
                self.assertFalse(self.map.layers["traversable"][cell])
        for x, y in ((2.5, 4), (6.5, 4.9), (6.5, 5.6)):
            self.assertFalse(self.map.lookup(x, y)["traversable"])

    def test_ramp_connects_road_and_pavement_and_is_only_curb_passage(self):
        start = self.map.grid.world_to_cell(.8, 1)
        target = self.map.grid.world_to_cell(4, 2.3)
        mask = self.map.layers["traversable"]
        self.assertIn(target, EXPERIMENT.reachable(mask, start))
        blocked_ramp = mask.copy()
        blocked_ramp[19:21, 34:46] = False
        self.assertNotIn(target, EXPERIMENT.reachable(blocked_ramp, start))
        self.assertTrue(np.all(mask[19:21, 34:46]))
        self.assertFalse(np.any(mask[19:21, :34]))
        self.assertFalse(np.any(mask[19:21, 46:]))

    def test_boundaries_and_clearance_are_separate_from_occupancy(self):
        padded = build(clearance_radius=.15)
        for name in ("terrain", "height", "occupancy", "forbidden", "boundary", "traversable"):
            np.testing.assert_array_equal(padded.layers[name], self.map.layers[name])
        self.assertGreater(padded.layers["clearance_blocked"].sum(), (~padded.layers["traversable"]).sum())
        self.assertTrue(np.all(padded.layers["boundary"][[0, -1], :]))
        self.assertTrue(np.all(padded.layers["boundary"][:, [0, -1]]))
        with self.assertRaises(ValueError):
            build(clearance_radius=-.1)


class ItemTests(unittest.TestCase):
    def setUp(self):
        self.map = build()

    def update(self, records):
        self.map.update_tracks(records, seed=self.map.seed, run_id=self.map.run_id)

    def test_all_eight_records_are_preserved_without_quantizing_positions(self):
        self.assertEqual(len(self.map.items), 8)
        self.assertEqual(self.map.source_records, TRACKS["tracks"])
        for record in TRACKS["tracks"]:
            item = self.map.item(record["track_id"])
            self.assertFalse(item["collected"])
            self.assertEqual(item["cell"], list(self.map.grid.world_to_cell(*record["position"][:2])))
            self.assertNotEqual(record["position"][:2], list(self.map.grid.cell_center(*item["cell"])))
        exposed = self.map.item(1)
        exposed["position"][0] = 0
        exposed["class_evidence"]["wrapper"] = 0
        self.assertEqual(self.map.source_records, TRACKS["tracks"])
        with self.assertRaises(ValueError):
            self.map.layers["occupancy"][0, 0] = True

    def test_idempotent_reimport_and_two_items_sharing_a_cell(self):
        self.update(TRACKS["tracks"])
        self.assertEqual(len(self.map.items), 8)
        record = deepcopy(TRACKS["tracks"][0])
        record["track_id"] = 99
        self.update([record])
        cell = self.map.item(1)["cell"]
        self.assertEqual([item["track_id"] for item in self.map.items_in_cell(*cell)], [1, 99])
        self.assertEqual(self.map.lookup(*record["position"][:2])["item_ids"], [1, 99])

    def test_collection_survives_position_and_class_updates(self):
        self.map.mark_collected(1)
        self.map.mark_collected(1)
        old_cell = self.map.item(1)["cell"]
        updated = deepcopy(TRACKS["tracks"][0])
        updated.update(position=[2.5, 1.5, .035], class_id=0, **{"class": "bottle", "class_evidence": {"bottle": 5.}})
        self.update([updated])
        self.assertTrue(self.map.item(1)["collected"])
        self.assertEqual(len(self.map.active_items()), 7)
        self.assertEqual(self.map.items_in_cell(*old_cell), [])
        self.assertEqual(self.map.lookup(2.5, 1.5)["item_ids"], [1])
        with self.assertRaises(KeyError):
            self.map.mark_collected(999)

    def test_bad_batch_is_atomic_and_duplicate_ids_are_rejected(self):
        before = self.map.items
        good = deepcopy(TRACKS["tracks"][0])
        good["position"] = [2, 1, .01]
        bad = deepcopy(TRACKS["tracks"][1])
        bad["position"] = [8, 1, .01]
        with self.assertRaises(IndexError):
            self.update([good, bad])
        self.assertEqual(self.map.items, before)
        with self.assertRaises(ValueError):
            self.update([good, good])
        self.assertEqual(self.map.items, before)

    def test_bad_record_fields_are_rejected(self):
        invalid = ({"track_id": True}, {"class_id": 2}, {"position": [np.nan, 1, 0]},
                   {"position": [1, 0]}, {"position_std": [-1, 0, 0]}, {"observations": 0},
                   {"class_observations": 1}, {"last_seen": -1}, {"mean_score": 1.5},
                   {"class_confidence": np.inf}, {"class_evidence": {"unknown": 1}}, {"collected": True})
        for values in invalid:
            with self.subTest(values=values):
                record = {**deepcopy(TRACKS["tracks"][0]), **values}
                with self.assertRaises(ValueError):
                    self.update([record])

    def test_cross_seed_and_cross_run_imports_are_rejected(self):
        for seed, run_id in ((1, self.map.run_id), (0, "different-run")):
            with self.assertRaises(ValueError):
                self.map.update_tracks(TRACKS["tracks"], seed=seed, run_id=run_id)

    def test_bad_input_metadata_and_manifest_mismatches_are_rejected(self):
        for values in ({"schema_version": 2}, {"units": "cm"}, {"frame": "camera"}, {"seed": 1},
                       {"method": "ground_plane"}, {"detector": "other"}, {"min_observations": 4}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                SemanticMap.from_tracks(default_layout(), {**TRACKS, **values}, RUN)
        run = deepcopy(RUN)
        run["model_sha256"] = "another-model"
        self.assertNotEqual(build().run_id, SemanticMap.from_tracks(default_layout(), TRACKS, run).run_id)

    def test_malformed_manifest_and_confirmation_policy_are_rejected(self):
        for methods in (None, [], {"depth": []}):
            with self.subTest(methods=methods), self.assertRaises(ValueError):
                SemanticMap.from_tracks(default_layout(), TRACKS, {**RUN, "methods": methods})
        tracks, run = deepcopy(TRACKS), deepcopy(RUN)
        tracks["confirmation"] = run["methods"]["depth"]["confirmation"] = "anything"
        with self.assertRaises(ValueError):
            SemanticMap.from_tracks(default_layout(), tracks, run)

    def test_confirmed_tracker_adapter_preserves_evidence_and_state(self):
        tracker = LitterTracker(min_observations=3)
        live_map = SemanticMap.from_layout(default_layout(), seed=0, run_id="live-episode-1")
        for time in range(3):
            tracker.update([LocalizedDetection(0, 0, .9, np.array([1., 1., .02]), float(time), 1., "depth")], float(time))
        live_map.update_tracks(tracker.confirmed_tracks, seed=0, run_id="live-episode-1")
        self.assertEqual(live_map.source_records, [tracker.confirmed_tracks[0].as_dict()])
        live_map.mark_collected(1)
        tracker.update([LocalizedDetection(0, 0, .8, np.array([1.01, 1., .02]), 3., 1., "depth")], 3.)
        live_map.update_tracks(tracker.confirmed_tracks, seed=0, run_id="live-episode-1")
        self.assertTrue(live_map.item(1)["collected"])
        with self.assertRaises(ValueError):
            live_map.update_tracks([replace(tracker.confirmed_tracks[0], confirmed=False)], seed=0, run_id="live-episode-1")


class PersistenceTests(unittest.TestCase):
    def test_round_trip_preserves_layers_provenance_and_collection_state(self):
        original = build(clearance_radius=.1)
        original.mark_collected(1)
        with tempfile.TemporaryDirectory() as folder:
            save_map(original, folder)
            restored = load_map(folder)
        self.assertEqual(restored.grid, original.grid)
        self.assertEqual(restored.items, original.items)
        self.assertEqual(restored.provenance, original.provenance)
        self.assertEqual(restored.landmarks, original.landmarks)
        self.assertEqual(restored.summary(), original.summary())
        for name in original.layers:
            np.testing.assert_array_equal(restored.layers[name], original.layers[name])
        restored.update_tracks(TRACKS["tracks"], seed=restored.seed, run_id=restored.run_id)
        self.assertTrue(restored.item(1)["collected"])

    def test_tampered_contract_cells_and_array_checksum_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            save_map(build(), folder)
            path = Path(folder) / "map.json"
            original = json.loads(path.read_text())
            for mutate in (lambda d: d.update(frame="camera"),
                           lambda d: d.update(seed=1),
                           lambda d: d["items"][0].update(cell=[0, 0]),
                           lambda d: d["items"][0].update(collected="yes")):
                damaged = deepcopy(original)
                mutate(damaged)
                path.write_text(json.dumps(damaged))
                with self.assertRaises(ValueError):
                    load_map(folder)
            path.write_text(json.dumps(original))
            with (Path(folder) / "layers.npz").open("ab") as file:
                file.write(b"bad")
            with self.assertRaises(ValueError):
                load_map(folder)

    def test_offline_seed0_acceptance_checks(self):
        mapped = build()
        with tempfile.TemporaryDirectory() as folder:
            save_map(mapped, folder)
            checks = EXPERIMENT.acceptance_checks(mapped, load_map(folder), default_layout(), TRACKS, 8)
        self.assertTrue(all(checks.values()), checks)

    def test_cli_writes_artifacts_and_fails_when_count_is_wrong(self):
        with tempfile.TemporaryDirectory() as folder:
            command = [sys.executable, "-B", str(ROOT / "experiments/27_semantic_mapping.py"),
                       "--output", folder, "--no-figure", "--require-pass"]
            result = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(json.loads((Path(folder) / "validation.json").read_text())["stage5_passed"])
            for name in ("map.json", "layers.npz", "summary.json", "validation.json", "stage4_tracks.json", "stage4_run.json"):
                self.assertTrue((Path(folder) / name).is_file())
            self.assertEqual((Path(folder) / "stage4_tracks.json").read_bytes(), (INPUT / "depth/tracks.json").read_bytes())
            self.assertEqual((Path(folder) / "stage4_run.json").read_bytes(), (INPUT / "run.json").read_bytes())
            result = subprocess.run(command + ["--expected-items", "9"], capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
