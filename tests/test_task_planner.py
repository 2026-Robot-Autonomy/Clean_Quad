"""Stage 8 task planning: optimality, hopper capacity, priority and battery."""

from itertools import permutations, product
import math
from pathlib import Path
import unittest

from src.mapping import load_map
from src.navigation import AStarPlanner
from src.tasks.planner import Node, TaskConfig, TaskPlanner


ROOT = Path(__file__).resolve().parents[1]


class EuclideanTasks(TaskPlanner):
    """Straight-line legs at 1 m/s and fixed pickup times, for exact checks."""

    def leg(self, a, b):
        seconds = math.dist(a, b) / self.config.planning_speed
        return seconds, seconds

    def pickup_cost(self, node):
        return self.config.pickup_seconds


def item(track_id, x, y, kind="can"):
    record = {"track_id": track_id, "class": kind, "position": [x, y, 0.0]}
    return Node(f"item_{track_id}", (x, y), 0.0, "item", record)


def objective(tasks, start, items, order, dumps_before, bins, final_bin):
    """Brute-force objective for an explicit sequence (mirrors the DP)."""

    cfg = tasks.config
    weights = {node.name: cfg.priority(node.item["class"]) for node in items}
    waiting = sum(weights.values())
    here, total, load = start, 0.0, 0
    bin_iter = iter(bins)
    for index, node in enumerate(order):
        if dumps_before[index]:
            b = next(bin_iter)
            for a, c in ((here, b.position), (b.position, node.position)):
                total += tasks.leg(a, c)[1] * (1 + cfg.latency_weight * waiting)
            total += cfg.dump_seconds * (1 + cfg.latency_weight * waiting)
            load = 0
        else:
            total += tasks.leg(here, node.position)[1] * (1 + cfg.latency_weight * waiting)
        load += 1
        if load > cfg.hopper_capacity:
            return math.inf
        total += cfg.pickup_seconds * (1 + cfg.latency_weight * waiting)
        waiting -= weights[node.name]
        here = node.position
    if final_bin is None:
        return total + tasks.leg(here, tasks.base.position)[1] if load == 0 else math.inf
    return (total + tasks.leg(here, final_bin.position)[1] + tasks.leg(final_bin.position, tasks.base.position)[1]
            + cfg.dump_seconds)


class TaskPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.map = load_map(ROOT / "results/logs/semantic_mapping/cnn/seed_0")
        cls.planner = AStarPlanner(cls.map)

    def make(self, **settings):
        defaults = dict(planning_speed=1.0, leg_overhead_s=0.0, pickup_seconds=2.0, dump_seconds=1.0)
        return EuclideanTasks(self.planner, TaskConfig(**{**defaults, **settings}))

    def test_dynamic_program_matches_brute_force(self):
        tasks = self.make(hopper_capacity=2, latency_weight=0.05)
        items = [item(1, 2, 1, "wrapper"), item(2, 5, 1), item(3, 3, 4, "cup"), item(4, 6, 4)]
        start = tasks.base.position
        best = math.inf
        for order in permutations(items):
            for dumps in product((False, True), repeat=len(items)):
                if dumps[0]:
                    continue
                count = sum(dumps)
                for bins in product(tasks.bins, repeat=count):
                    for final in (None, *tasks.bins):
                        best = min(best, objective(tasks, start, items, order, dumps, bins, final))
        plan = tasks.plan(start, items)
        self.assertAlmostEqual(plan["objective"], best, places=6)

    def test_capacity_forces_bin_visits(self):
        tasks = self.make(hopper_capacity=1)
        items = [item(1, 2, 1), item(2, 3, 1), item(3, 4, 1)]
        steps = tasks.plan(tasks.base.position, items)["steps"]
        load = peak = 0
        for kind, _ in steps:
            load = 0 if kind == "dump" else load + (kind == "pick")
            peak = max(peak, load)
        self.assertEqual(peak, 1)
        self.assertEqual([k for k, _ in steps].count("dump"), 3)
        self.assertEqual(steps[-1][0], "base")

    def test_priority_pulls_wrapper_earlier(self):
        # The can lies on the way to the wrapper, so distance alone takes it first.
        items = [item(1, 1.5, 1.0, "can"), item(2, 3.5, 1.0, "wrapper")]
        neutral = self.make(latency_weight=0.0, hopper_capacity=3, class_priority={"can": 1.0, "wrapper": 1.0})
        urgent = self.make(latency_weight=0.5, hopper_capacity=3, class_priority={"can": 1.0, "wrapper": 20.0})
        first = lambda tasks: tasks.plan((0.8, 1.0), items)["steps"][0][1].name
        self.assertEqual(first(neutral), "item_1")
        self.assertEqual(first(urgent), "item_2")

    def test_empty_task_list_dumps_only_when_loaded(self):
        tasks = self.make()
        self.assertEqual([k for k, _ in tasks.plan((3, 1), [], load=0)["steps"]], ["base"])
        self.assertEqual([k for k, _ in tasks.plan((3, 1), [], load=2)["steps"]], ["dump", "base"])

    def test_full_hopper_must_dump_before_the_next_pickup(self):
        tasks = self.make(hopper_capacity=2)
        steps = tasks.plan((3, 1), [item(1, 2, 1)], load=2)["steps"]
        self.assertEqual(steps[0][0], "dump")
        with self.assertRaises(ValueError):
            tasks.plan((3, 1), [], load=3)

    def test_battery_check_includes_the_trip_home(self):
        tasks = self.make(drain_per_second=1.0, battery_reserve=10.0)
        far = ("pick", item(1, 7.0, 4.0))
        needed = tasks.step_seconds(tasks.base.position, far) + tasks.leg(far[1].position, tasks.base.position)[0]
        self.assertTrue(tasks.can_afford(10.0 + needed + 1e-6, tasks.base.position, far))
        self.assertFalse(tasks.can_afford(10.0 + needed - 1.0, tasks.base.position, far))
        self.assertTrue(tasks.can_afford(0.0, tasks.base.position, ("base", tasks.base)))

    def test_real_legs_penalize_the_ramp_and_reject_bad_config(self):
        tasks = TaskPlanner(self.planner)
        flat = tasks.leg((1.0, 1.0), (3.0, 1.0))
        self.assertTrue(tasks.crosses_ramp((1.0, 1.0), (4.0, 3.0)))
        self.assertFalse(tasks.crosses_ramp((1.0, 1.0), (3.0, 1.0)))
        self.assertGreater(flat[0], 2.0 / tasks.config.planning_speed)
        for bad in (dict(hopper_capacity=0), dict(battery_reserve=100.0), dict(drain_per_second=0.0),
                    dict(class_priority={"can": 0.0})):
            with self.assertRaises(ValueError):
                TaskConfig(**bad)


if __name__ == "__main__":
    unittest.main()
