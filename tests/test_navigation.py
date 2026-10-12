"""Planning geometry, unreachable goals, approach semantics and command checks."""

from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from src.environment.layout import default_layout
from src.mapping import GridSpec, SemanticMap, load_map
from src.mapping.grid import clearance_mask
from src.navigation import AStarPlanner, NavigationConfig, NoPathError, RouteFollower, RoutePlan
from src.navigation.evaluation import command_is_safe, contact_is_forbidden


ROOT = Path(__file__).resolve().parents[1]


def toy_map(blocked=None):
    grid = GridSpec(2, 2)
    blocked = np.zeros(grid.shape, dtype=bool) if blocked is None else blocked
    layers = {"terrain": np.full(grid.shape, "road", dtype="U16"),
              "height": np.zeros(grid.shape), "occupancy": blocked.copy(),
              "forbidden": np.zeros(grid.shape, dtype=bool),
              "boundary": np.zeros(grid.shape, dtype=bool),
              "traversable": ~blocked, "clearance_blocked": clearance_mask(blocked, .1, 0)}
    return SemanticMap(grid, layers, seed=0, run_id="toy")


class PlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layout = default_layout()
        cls.map = load_map(ROOT / "results/logs/semantic_mapping/cnn/seed_0")
        cls.planner = AStarPlanner(cls.map)

    def test_open_grid_cost_matches_octile_distance_and_exact_endpoints(self):
        planner = AStarPlanner(toy_map(), NavigationConfig(clearance_radius=.01))
        route = planner.plan((.25, .25), (1.25, .75))
        self.assertAlmostEqual(route.grid_cost, .5 + .5 * math.sqrt(2))
        self.assertEqual(route.waypoints, ((.25, .25), (1.25, .75)))

    def test_same_cell_and_same_point_do_not_move_to_cell_centre(self):
        route = self.planner.plan((.8, 1), (.8, 1))
        self.assertEqual(route.waypoints, ((.8, 1),))
        self.assertEqual(route.length, 0)

    def test_ramp_crossing_uses_map_derived_straight_passage(self):
        route = self.planner.plan((.8, 1), (4, 3))
        self.assertIn((4., self.layout.ramp.y_bottom - .65), route.waypoints)
        self.assertIn((4., 2.65), route.waypoints)
        self.assertTrue(all(self.planner.segment_is_free(a, b)
                            for a, b in zip(route.waypoints, route.waypoints[1:])))
        self.assertTrue(any(self.map.layers["terrain"][cell] == "ramp" for cell in route.cells))

    def test_closed_ramp_has_no_curb_bypass(self):
        layout = replace(self.layout, traversable={**self.layout.traversable, "ramp": False})
        mapped = SemanticMap.from_layout(layout, seed=0, run_id="closed")
        with self.assertRaises(NoPathError):
            AStarPlanner(mapped).plan((.8, 1), (4, 3))

    def test_oversized_robot_cannot_pass_the_ramp(self):
        planner = AStarPlanner(self.map, NavigationConfig(clearance_radius=.65))
        with self.assertRaises(NoPathError):
            planner.plan((.8, 1), (4, 3))

    def test_occupied_start_and_goal_are_rejected_without_snapping(self):
        for start, goal in (((1, 5), (4, 3)), ((.8, 1), (1, 5))):
            with self.assertRaises(NoPathError):
                self.planner.plan(start, goal)

    def test_world_bounds_and_nonfinite_input_are_rejected(self):
        with self.assertRaises(IndexError):
            self.planner.plan((8, 1), (4, 3))
        with self.assertRaises(ValueError):
            self.planner.plan((float("nan"), 1), (4, 3))

    def test_exact_segment_check_rejects_corner_tangency(self):
        blocked = np.zeros((20, 20), dtype=bool)
        blocked[5, 5] = True
        planner = AStarPlanner(toy_map(blocked), NavigationConfig(clearance_radius=.01))
        self.assertFalse(planner.segment_is_free((.4, .6), (.6, .4), raw=True))
        self.assertTrue(planner.segment_is_free((.4, .59), (.59, .4), raw=True))

    def test_diagonal_gap_between_two_regions_is_unreachable(self):
        blocked = np.zeros((20, 20), dtype=bool)
        blocked[:10, 10:] = True
        blocked[10:, :10] = True
        planner = AStarPlanner(toy_map(blocked), NavigationConfig(clearance_radius=.01))
        with self.assertRaises(NoPathError):
            planner.plan((.25, .25), (1.75, 1.75))

    def test_obstacle_detour_is_longer_than_straight_line(self):
        start, goal = (1.2, 3.2), (4.6, 4.45)
        route = self.planner.plan(start, goal)
        self.assertGreater(route.length, math.dist(start, goal))
        self.assertTrue(all(self.planner.segment_is_free(a, b)
                            for a, b in zip(route.waypoints, route.waypoints[1:])))

    def test_saved_reachable_litter_approaches_are_clear_and_face_the_item(self):
        for item in self.map.active_items():
            with self.subTest(track_id=item["track_id"]):
                if item["track_id"] == 9:
                    # The circular footprint closes the tree/planter/bench gap.
                    with self.assertRaises(NoPathError):
                        self.planner.approach_item((.8, 1), item)
                    continue
                plan = self.planner.approach_item((.8, 1), item)
                self.assertTrue(self.planner.is_free(plan.position))
                self.assertGreaterEqual(plan.standoff, .45)
                self.assertLessEqual(plan.standoff, .65)
                expected = math.atan2(item["position"][1] - plan.position[1], item["position"][0] - plan.position[0])
                self.assertAlmostEqual(plan.yaw, expected)

    def test_approach_does_not_reach_across_curb_or_mutate_source_map(self):
        evidence = json.dumps(self.map.items, sort_keys=True)
        raw = self.map.layers["traversable"].copy()
        item = {"track_id": 99, "position": [2., 2.2, .08], "collected": False}
        approach = self.planner.approach_item((.8, 1), item)
        self.assertGreater(approach.position[1], 2)
        self.assertEqual(json.dumps(self.map.items, sort_keys=True), evidence)
        np.testing.assert_array_equal(self.map.layers["traversable"], raw)

    def test_collected_or_forbidden_litter_is_rejected(self):
        for item in ({"track_id": 99, "position": [2, 2.2, .08], "collected": True},
                     {"track_id": 99, "position": [2.5, 4, .12], "collected": False}):
            with self.assertRaises(NoPathError):
                self.planner.approach_item((.8, 1), item)

    def test_command_filter_rejects_a_future_boundary_collision(self):
        self.assertTrue(command_is_safe(self.planner, (.8, 1., 0.), .3, 0.))
        self.assertFalse(command_is_safe(self.planner, (.65, 1., math.pi), .3, 0.))

    def test_replanning_from_tracking_offset_preserves_clearance(self):
        start, goal = (3.812698, 2.573466), (1.75, 2.65)
        route = self.planner.plan(start, goal)
        self.assertTrue(all(self.planner.segment_is_free(a, b)
                            for a, b in zip(route.waypoints, route.waypoints[1:])))
        self.assertGreater(len(route.waypoints), 2)

    def test_invalid_configuration_is_rejected(self):
        for kwargs in ({"clearance_radius": 0}, {"clearance_radius": float("nan")},
                       {"approach_max": .2}, {"grass_cost": .5}):
            with self.assertRaises(ValueError):
                NavigationConfig(**kwargs)

    def test_calf_ramp_end_support_is_allowed_but_lateral_side_is_not(self):
        self.assertFalse(contact_is_forbidden("terrain_ramp", (4.126, 1.574, -.00055),
                                             (0, .894, -.448), "RR_calf", self.layout))
        self.assertTrue(contact_is_forbidden("terrain_ramp", (3.4, 1.8, .03),
                                            (1, 0, 0), "FR_calf", self.layout))

    def test_curb_solids_no_go_and_trunk_ground_contacts_fail(self):
        self.assertTrue(contact_is_forbidden("terrain_pavement", (2, 2, .02),
                                            (0, 1, 0), "FR_calf", self.layout))
        self.assertFalse(contact_is_forbidden("terrain_pavement", (4, 2, .02),
                                             (0, 1, 0), "FR_calf", self.layout))
        for name in ("obstacle_tree", "terrain_stairs_0", "terrain_planter", "wall_north"):
            self.assertTrue(contact_is_forbidden(name, (1, 1, .1), (0, 0, 1), "FR_calf", self.layout))
        self.assertTrue(contact_is_forbidden("terrain_road", (1, 1, 0), (0, 0, 1), "trunk", self.layout))


class FollowerTests(unittest.TestCase):
    def setUp(self):
        self.route = RoutePlan(((.8, 1), (2, 1)), (), 1.2, 1.2)

    def test_large_heading_error_rotates_before_driving(self):
        follower = RouteFollower(self.route, 0)
        vx, wz = follower.command(.8, 1, math.pi / 2)
        self.assertEqual(vx, 0)
        self.assertAlmostEqual(wz, -.3)
        vx, wz = follower.command(.8, 1, .01)
        self.assertGreater(vx, 0)
        self.assertLessEqual(abs(wz), .3)

    def test_arrival_requires_final_heading_and_returns_stop(self):
        follower = RouteFollower(self.route, math.pi / 2)
        vx, wz = follower.command(1.99, 1, 0)
        self.assertEqual(vx, 0)
        self.assertGreater(wz, 0)
        self.assertFalse(follower.arrived)
        self.assertEqual(follower.command(1.99, 1, math.pi / 2), (0, 0))
        self.assertTrue(follower.arrived)

    def test_near_goal_reduces_speed_and_waypoints_are_not_skipped(self):
        route = RoutePlan(((.8, 1), (1, 1), (1, 2)), (), 1.2, 1.2)
        follower = RouteFollower(route, math.pi / 2)
        self.assertEqual(follower.command(.8, 1, 0)[0], .1)
        self.assertEqual(follower.index, 1)
        follower.command(.99, 1, 0)
        self.assertEqual(follower.index, 2)
        self.assertEqual(follower.command(.99, 1, 0)[0], 0)


class ExperimentTests(unittest.TestCase):
    def test_offline_cli_reports_unreachable_item_and_execution_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, "-B", str(ROOT / "experiments/28_navigation_baseline.py"),
                       "--offline-only", "--no-figure", "--require-pass", "--output", directory]
            run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            result = json.loads((Path(directory) / "results.json").read_text())
            self.assertFalse(result["execution_validated"])
            self.assertEqual(result["mapped_targets"], 8)
            self.assertEqual(result["reachable_targets"], 7)
            self.assertEqual(result["planning_failures"][0]["track_id"], 9)
            self.assertFalse(result["configuration"]["rl_training"])
            self.assertEqual(result["routes_attempted"], 0)


if __name__ == "__main__":
    unittest.main()
