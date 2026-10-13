"""Stage 7 pickup: tools, docking geometry, line follower, suction and one live pick."""

from dataclasses import dataclass
import math
from pathlib import Path
import unittest

import numpy as np

from src.mapping import load_map
from src.navigation import AStarPlanner, NavigationConfig, NoPathError
from src.navigation.planner import dock_point
from src.manipulation.pickup import LineFollower, PickupConfig, PickupSkill, choose_dock, suction
from src.manipulation.posture import Posture, posture_targets
from tests.test_navigation import toy_map


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class FakeObject:
    name: str
    position: np.ndarray
    collected: bool = False


class FakeCampus:
    def __init__(self, litter, distractors=()):
        self.litter, self.distractors = list(litter), list(distractors)

    @property
    def remaining_litter(self):
        return [item for item in self.litter if not item.collected]

    def collect(self, name):
        next(item for item in self.litter if item.name == name).collected = True


class LineFollowerTests(unittest.TestCase):
    def test_forward_drive_turn_in_place_and_arrival(self):
        follower = LineFollower((0, 0), (0.3, 0), 0.0)
        vx, _ = follower.command(0, 0, 0)
        self.assertGreater(vx, 0)
        self.assertFalse(follower.arrived)
        vx, wz = follower.command(0.3, 0, math.radians(20))
        self.assertEqual(vx, 0)
        self.assertLess(wz, 0)
        self.assertEqual(follower.command(0.3, 0, 0), (0.0, 0.0))
        self.assertTrue(follower.arrived)

    def test_reverse_drives_backwards_facing_away_from_goal(self):
        follower = LineFollower((0, 0), (-0.3, 0), 0.0, reverse=True)
        vx, wz = follower.command(0, 0, 0)
        self.assertLess(vx, 0)
        self.assertAlmostEqual(wz, 0)

    def test_overshoot_steps_back_without_turning_around(self):
        follower = LineFollower((0, 0), (0.3, 0), 0.0)
        vx, wz = follower.command(0.36, 0, 0)
        self.assertLess(vx, 0)
        self.assertAlmostEqual(wz, 0)

    def test_misaligned_start_turns_before_driving(self):
        follower = LineFollower((0, 0), (0, 0.3), math.pi / 2)
        vx, wz = follower.command(0, 0, 0)
        self.assertEqual(vx, 0)
        self.assertGreater(wz, 0)

    def test_degenerate_line_is_rejected(self):
        with self.assertRaises(ValueError):
            LineFollower((1, 1), (1, 1), 0.0)


class DockGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.map = load_map(ROOT / "results/logs/semantic_mapping/cnn/seed_0")
        cls.planner = AStarPlanner(cls.map)

    def test_dock_point_is_offset_short_of_target(self):
        self.assertEqual(dock_point((0, 0), (1, 0), 0.25), (0.75, 0.0))
        with self.assertRaises(ValueError):
            dock_point((1, 1), (1, 1), 0.1)

    def test_open_space_dock_is_on_the_bearing(self):
        planner = AStarPlanner(toy_map(), NavigationConfig(clearance_radius=.01))
        dock, heading = choose_dock(planner, (0.3, 0.5), (1.3, 0.5), 0.25)
        self.assertAlmostEqual(heading, 0.0)
        self.assertAlmostEqual(dock[0], 1.05)
        self.assertAlmostEqual(dock[1], 0.5)

    def test_dock_heading_skews_away_from_blocked_cells_or_gives_up(self):
        blocked = np.zeros((20, 20), dtype=bool)
        planner = AStarPlanner(toy_map(blocked), NavigationConfig(clearance_radius=.01))
        planner.blocked = planner.blocked.copy()
        planner.blocked[5, 11] = True       # the cell of the straight-ahead dock point
        dock, heading = choose_dock(planner, (0.35, 0.55), (1.35, 0.55), 0.25)
        self.assertNotAlmostEqual(heading, 0.0)
        self.assertTrue(planner.is_free(dock))
        self.assertAlmostEqual(math.dist(dock, (1.35, 0.55)), 0.25)
        planner.blocked[:, 9:13] = True
        self.assertIsNone(choose_dock(planner, (0.35, 0.55), (1.35, 0.55), 0.25))

    def test_pickup_approaches_leave_a_free_dock_line(self):
        offset = PickupConfig().dock_offset
        for item in self.map.active_items():
            try:
                approach = self.planner.approach_item((.8, 1), item, dock_offset=offset)
            except NoPathError:
                continue
            dock = dock_point(approach.position, item["position"][:2], offset)
            self.assertTrue(self.planner.segment_is_free(approach.position, dock))

    def test_curbside_item_is_reachable_with_the_forward_nozzle(self):
        self.assertIsNotNone(self.planner.approach_item((.8, 1), self.map.item(3), dock_offset=PickupConfig().dock_offset))

    def test_bin_approach_faces_bin_from_outside_its_padding(self):
        for record in self.map.landmarks["bins"]:
            approach = self.planner.approach_bin((.8, 1), record)
            centre = (record["shape"]["x"], record["shape"]["y"])
            self.assertTrue(self.planner.is_free(approach.position))
            self.assertGreaterEqual(math.dist(approach.position, centre),
                                    record["shape"]["r"] + self.planner.config.clearance_radius)
            bearing = math.atan2(centre[1] - approach.position[1], centre[0] - approach.position[0])
            self.assertAlmostEqual(approach.yaw, bearing)


class SuctionTests(unittest.TestCase):
    def setUp(self):
        self.config = PickupConfig()

    def test_captures_only_litter_inside_the_zone(self):
        near = FakeObject("near", np.array([1.03, 1.0, 0.02]))
        far = FakeObject("far", np.array([1.2, 1.0, 0.02]))
        campus = FakeCampus([near, far])
        result = suction(campus, np.array([1.0, 1.0, 0.05]), self.config)
        self.assertEqual(result["captured"], ["near"])
        self.assertTrue(near.collected)
        self.assertFalse(far.collected)

    def test_tip_too_high_or_item_already_collected_captures_nothing(self):
        item = FakeObject("item", np.array([1.0, 1.0, 0.02]))
        campus = FakeCampus([item])
        self.assertEqual(suction(campus, np.array([1.0, 1.0, 0.3]), self.config)["status"], "empty")
        item.collected = True
        self.assertEqual(suction(campus, np.array([1.0, 1.0, 0.05]), self.config)["captured"], [])

    def test_distractor_in_zone_aborts_without_collecting(self):
        item = FakeObject("item", np.array([1.0, 1.0, 0.02]))
        leaf = FakeObject("leaf", np.array([1.02, 1.0, 0.0]))
        result = suction(FakeCampus([item], [leaf]), np.array([1.0, 1.0, 0.05]), self.config)
        self.assertEqual(result["status"], "distractor_in_zone")
        self.assertFalse(item.collected)


class ConfigTests(unittest.TestCase):
    def test_invalid_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            PickupConfig(dock_speed=0.4)
        with self.assertRaises(ValueError):
            PickupConfig(attempts=0)
        with self.assertRaises(ValueError):
            Posture(-0.10, -0.30)

    def test_stand_posture_matches_the_locomotion_stand(self):
        from src.controllers.locomotion_interface import LocomotionInterface
        loco = LocomotionInterface()
        targets = posture_targets(PickupConfig().stand, loco.controller.hip_offset)
        np.testing.assert_allclose(targets, loco.controller.get_stand_command())


class SimulationTests(unittest.TestCase):
    """Short MuJoCo checks: the tools are invisible and one pickup works."""

    @classmethod
    def setUpClass(cls):
        from src.environment import build_campus
        cls.plain = build_campus(seed=0)
        cls.tooled = build_campus(seed=0, cleaning_tools=True)

    def test_tools_add_no_mass_and_do_not_change_camera_images(self):
        from src.perception.camera import RobotCamera
        np.testing.assert_array_equal(self.plain.model.body_mass, self.tooled.model.body_mass)
        with RobotCamera(self.plain.model, self.plain.data) as a, RobotCamera(self.tooled.model, self.tooled.data) as b:
            first, second = a.capture(segmentation=False), b.capture(segmentation=False)
        np.testing.assert_array_equal(first.rgb, second.rgb)
        np.testing.assert_array_equal(first.depth, second.depth)

    def test_one_pickup_collects_the_target_and_nothing_else(self):
        import mujoco
        from src.controllers.locomotion_interface import LocomotionInterface
        from src.simulation.robot_setup import reset_robot
        campus = self.tooled
        campus.reset()
        semantic_map = load_map(ROOT / "results/logs/semantic_mapping/cnn/seed_0")
        planner = AStarPlanner(semantic_map)
        item = semantic_map.item(1)
        approach = planner.approach_item((.8, 1), item, dock_offset=PickupConfig().dock_offset)
        reset_robot(campus.model, campus.data, x=approach.position[0], y=approach.position[1], yaw=approach.yaw)
        loco = LocomotionInterface()
        for _ in range(250):
            campus.data.ctrl[:] = loco.update(campus.data, campus.model.opt.timestep)
            mujoco.mj_step(campus.model, campus.data)
        result = PickupSkill(campus, planner).pick(item["position"])
        self.assertEqual(result["status"], "captured_unverified")
        self.assertEqual(result["captured"], ["litter_wrapper_07"])
        self.assertEqual([t.name for t in campus.litter if t.collected], ["litter_wrapper_07"])
        self.assertLess(math.dist(campus.robot_pose()[:2], approach.position), 0.08)
        campus.reset()


if __name__ == "__main__":
    unittest.main()
