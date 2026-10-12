"""RL contracts, episode semantics, transfer decoding and continued execution."""

import math
import unittest
import numpy as np

from src.navigation.observations import decode_action, encode_observation, teacher_action
from src.navigation.planner import RoutePlan
from src.navigation.ppo_navigator import PPOFollower
from src.navigation.planner import ApproachPlan

try:
    from src.navigation.rl_env import NavigationEnv
except ImportError:
    NavigationEnv = None


@unittest.skipIf(NavigationEnv is None, "Stage 6 RL dependencies not installed")
class NavigationRLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = NavigationEnv(randomize=False)

    def test_reset_is_seeded_and_observation_is_normalized(self):
        a, _ = self.env.reset(seed=12)
        b, _ = self.env.reset(seed=12)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (25,))
        self.assertTrue(self.env.observation_space.contains(a))

    def test_teacher_can_reach_and_align_a_local_goal(self):
        self.env.reset(seed=1, options={"pose": [.8, 1., 0.], "target": [1.5, 1.], "goal_yaw": .5, "final": True})
        for _ in range(450):
            _, _, terminated, truncated, info = self.env.step(self.env.teacher())
            if terminated or truncated:
                break
        self.assertTrue(info["is_success"], info)
        self.assertFalse(info["collision"])
        with self.assertRaises(RuntimeError):
            self.env.step(np.zeros(2))

    def test_time_limit_is_truncation_not_natural_termination(self):
        env = NavigationEnv(planner=self.env.planner, max_steps=1, randomize=False)
        env.reset(seed=1, options={"pose": [.8, 1., 0.], "target": [1.5, 1.], "goal_yaw": 0.})
        _, _, terminated, truncated, _ = env.step(np.array([-1., 0.]))
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_collision_terminates_without_success(self):
        self.env.reset(seed=1, options={"pose": [.65, 1., math.pi], "target": [1.5, 1.], "goal_yaw": 0.})
        for _ in range(100):
            _, _, terminated, _, info = self.env.step(np.array([1., 0.]))
            if terminated:
                break
        self.assertTrue(info["collision"])
        self.assertFalse(info["is_success"])

    def test_decode_enforces_velocity_envelope_and_deadbands(self):
        self.assertEqual(decode_action([-1., 0.]), (0., 0.))
        self.assertEqual(decode_action([3., -3.]), (.3, -.3))
        self.assertEqual(decode_action([-.8, .01]), (0., 0.))
        with self.assertRaises(ValueError):
            decode_action([float("nan"), 0.])

    def test_rays_distinguish_open_road_and_boundary(self):
        planner = self.env.planner
        a = encode_observation(planner, (4., 1., 0.), (5., 1.), 0.)
        b = encode_observation(planner, (.65, 1., 0.), (1.5, 1.), 0.)
        self.assertLess(b[13], a[13])  # first ray looks behind the robot

    def test_learned_follower_uses_policy_and_explicit_arrival_guard(self):
        class FakePolicy:
            calls = 0
            def predict(self, observation, deterministic):
                self.calls += 1
                return teacher_action(observation), None
        policy = FakePolicy()
        route = RoutePlan(((.8, 1.), (1.5, 1.)), (), .7, .7)
        follower = PPOFollower(route, 0., self.env.planner, policy)
        self.assertGreater(follower.command(.8, 1., 0.)[0], 0.)
        self.assertEqual(policy.calls, 1)
        self.assertEqual(follower.command(1.49, 1., 0.), (0., 0.))
        self.assertTrue(follower.arrived)

    def test_blocked_or_invalid_reset_is_rejected(self):
        with self.assertRaises(ValueError):
            self.env.reset(options={"pose": [1., 5., 0.], "target": [4., 3.]})
        with self.assertRaises(ValueError):
            self.env.reset(options={"pose": [.8, 1.], "target": [4., 3.]})

    def test_continuous_clock_and_live_goal_revision(self):
        from src.environment import build_campus
        from src.navigation.evaluation import execute_approach
        campus = build_campus(seed=0)
        campus.data.time = 10.
        pose = campus.robot_pose()
        planner = self.env.planner
        route = planner.plan(pose[:2], pose[:2])
        approach = ApproachPlan(1, pose[:2], pose[2], .5, route)
        revised = ApproachPlan(1, pose[:2], pose[2] + .2, .5, route)
        seen = []
        def provider(current, goal):
            return revised if goal is approach else None
        metrics, _ = execute_approach(campus, planner, approach, max_time=.12,
                                      reset=False, step_hook=lambda c: seen.append(c.data.time),
                                      goal_provider=provider)
        self.assertFalse(metrics["reset_before_route"])
        self.assertEqual(len(metrics["live_goal_updates"]), 1)
        self.assertGreaterEqual(metrics["simulated_seconds"], .12)
        self.assertLess(metrics["simulated_seconds"], .13)
        self.assertGreater(min(seen), 10.)
        self.assertEqual(metrics["goal_yaw_rad"], revised.yaw)


if __name__ == "__main__":
    unittest.main()
