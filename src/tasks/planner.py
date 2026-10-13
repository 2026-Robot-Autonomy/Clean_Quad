"""Stage 8 task planning: which item next, when to empty the hopper, when to charge.

Nodes are the base pad, the bin approach poses and the litter approach poses.
Leg costs come from the Stage 6 clearance-aware A* (terrain-weighted grid
cost) plus a ramp-crossing penalty, converted to time with a conservative
mean speed measured in the Stage 6 live mission.

The order is optimized exactly by dynamic programming over
(picked set, current node, hopper load). The objective is total mission time
plus a priority-weighted latency term, so high-priority litter (light,
wind-blown wrappers) is collected earlier when the detour is small. A full
hopper forces a bin visit; the DP may also empty a partly full hopper when
that is cheaper.

Battery is enforced while executing: before every task the robot checks that
it can do the task and still reach the base with the reserve left. If not, it
returns to the base, charges, and replans.
"""

from dataclasses import dataclass, field
import math

from src.navigation.planner import NoPathError

BASE = "base"


@dataclass(frozen=True)
class TaskConfig:
    hopper_capacity: int = 3
    battery_capacity: float = 100.0         # percent
    battery_reserve: float = 15.0           # percent kept for the trip home
    drain_per_second: float = 0.25          # percent per simulated second of work
    charge_per_second: float = 5.0          # percent per simulated second on the pad
    planning_speed: float = 0.15            # m/s, conservative mean including turns
    leg_overhead_s: float = 5.0             # start, final alignment and settling
    pickup_seconds: float = 14.0            # dock, crouch, back off, look
    dump_seconds: float = 4.0
    ramp_penalty_m: float = 1.0             # risk of the curb ramp, per crossing
    grass_pickup_penalty_m: float = 0.5     # softer footing while crouching
    latency_weight: float = 0.02            # cost per priority-second of waiting
    class_priority: dict = field(default_factory=lambda: {
        "wrapper": 1.5, "cup": 1.2, "bottle": 1.0, "can": 1.0})

    def __post_init__(self):
        if not isinstance(self.hopper_capacity, int) or self.hopper_capacity < 1:
            raise ValueError("hopper capacity must be a positive integer")
        for name in ("battery_capacity", "drain_per_second", "charge_per_second", "planning_speed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("battery_reserve", "leg_overhead_s", "pickup_seconds", "dump_seconds",
                     "ramp_penalty_m", "grass_pickup_penalty_m", "latency_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be non-negative and finite")
        if self.battery_reserve >= self.battery_capacity:
            raise ValueError("battery reserve must be below capacity")
        if any(not math.isfinite(v) or v <= 0 for v in self.class_priority.values()):
            raise ValueError("class priorities must be positive")

    def priority(self, kind):
        return self.class_priority.get(kind, 1.0)

    def energy(self, seconds):
        return seconds * self.drain_per_second


@dataclass(frozen=True)
class Node:
    name: str
    position: tuple
    yaw: float
    kind: str                   # "base" | "bin" | "item"
    item: dict = None


class TaskPlanner:
    def __init__(self, planner, config=None):
        self.planner = planner
        self.config = config or TaskConfig()
        landmarks = planner.map.landmarks
        base = landmarks["base_pose"]
        self.base = Node(BASE, tuple(base[:2]), base[2], "base")
        self.bins = []
        for record in landmarks["bins"]:
            approach = planner.approach_bin(self.base.position, record)
            self.bins.append(Node(record["name"], approach.position, approach.yaw, "bin"))
        self._legs = {}
        ramp = landmarks.get("ramp")
        self._ramp_top = None
        if ramp:
            self._ramp_top = ramp["y_bottom"] + (ramp["z_top"] - ramp["z_bottom"]) / math.tan(
                math.radians(ramp["slope_deg"]))

    # ------------------------------------------------------------ geometry

    def item_node(self, item, start, dock_offset):
        approach = self.planner.approach_item(start, item, dock_offset=dock_offset)
        return Node(f"item_{item['track_id']}", approach.position, approach.yaw, "item", item)

    def crosses_ramp(self, a, b):
        return self._ramp_top is not None and (a[1] - self._ramp_top) * (b[1] - self._ramp_top) < 0

    def leg(self, a, b):
        """(seconds, cost) of driving between two positions; inf if unreachable."""

        key = (tuple(round(v, 4) for v in a), tuple(round(v, 4) for v in b))
        if key not in self._legs:
            if math.dist(a, b) < 1e-6:
                self._legs[key] = (0.0, 0.0)
            else:
                try:
                    route = self.planner.plan(a, b)
                except NoPathError:
                    self._legs[key] = (math.inf, math.inf)
                else:
                    seconds = route.length / self.config.planning_speed + self.config.leg_overhead_s
                    penalty = self.config.ramp_penalty_m * self.crosses_ramp(a, b)
                    seconds_cost = (route.grid_cost + penalty) / self.config.planning_speed
                    self._legs[key] = (seconds, seconds_cost + self.config.leg_overhead_s)
        return self._legs[key]

    def pickup_cost(self, node):
        terrain = self.planner.map.lookup(*node.item["position"][:2])["terrain"]
        penalty = self.config.grass_pickup_penalty_m / self.config.planning_speed if terrain == "grass" else 0.0
        return self.config.pickup_seconds + penalty

    # ---------------------------------------------------------- optimizer

    def plan(self, start, items, load=0):
        """Optimal task sequence from `start` for item nodes, given hopper load.

        Returns a dict with "steps" (list of ("pick", node) / ("dump", node) /
        ("base", node)), the objective value and the estimated seconds.
        """

        cfg = self.config
        n = len(items)
        if load < 0 or load > cfg.hopper_capacity:
            raise ValueError("hopper load outside capacity")
        priority = [cfg.priority(node.item["class"]) for node in items]
        full = (1 << n) - 1

        waiting = [sum(p for i, p in enumerate(priority) if not mask >> i & 1) for mask in range(full + 1)]

        def edge(a, b, mask):
            seconds, cost = self.leg(a.position, b.position)
            return seconds, cost + cfg.latency_weight * waiting[mask] * seconds

        start_node = Node("start", tuple(start), 0.0, "start")
        # best[(mask, last, load)] = (objective, seconds, previous state, action)
        best = {(0, -1, load): (0.0, 0.0, None, None)}
        buckets = {0: [(0, -1, load)]}
        for mask in sorted(range(full + 1), key=lambda m: bin(m).count("1")):
            for state in buckets.get(mask, ()):
                objective, seconds, _, _ = best[state]
                _, last, carried = state
                here = start_node if last == -1 else items[last]
                scale = 1 + cfg.latency_weight * waiting[mask]
                for j in range(n):
                    if mask >> j & 1:
                        continue
                    work = self.pickup_cost(items[j])
                    options = []
                    if carried < cfg.hopper_capacity:
                        s, c = edge(here, items[j], mask)
                        options.append((c, s, carried + 1, None))
                    if carried > 0:
                        for b in self.bins:
                            s1, c1 = edge(here, b, mask)
                            s2, c2 = edge(b, items[j], mask)
                            options.append((c1 + c2 + cfg.dump_seconds * scale,
                                            s1 + s2 + cfg.dump_seconds, 1, b))
                    for cost, travel, new_load, via in options:
                        if not math.isfinite(cost):
                            continue
                        value = objective + cost + work * scale
                        key = (mask | 1 << j, j, new_load)
                        if key not in best:
                            buckets.setdefault(key[0], []).append(key)
                        if key not in best or value < best[key][0] - 1e-9:
                            best[key] = (value, seconds + travel + work, state, (j, via))
        finals = []
        for (mask, last, carried), (objective, seconds, _, _) in best.items():
            if mask != full:
                continue
            here = start_node if last == -1 else items[last]
            if carried == 0:
                s, c = edge(here, self.base, full)
                finals.append((objective + c, seconds + s, (mask, last, carried), None))
            for b in self.bins:
                s1, c1 = edge(here, b, full)
                s2, c2 = edge(b, self.base, full)
                finals.append((objective + c1 + c2 + cfg.dump_seconds,
                               seconds + s1 + s2 + cfg.dump_seconds, (mask, last, carried), b))
        finals = [f for f in finals if math.isfinite(f[0])]
        if not finals:
            raise NoPathError("no feasible task sequence")
        objective, seconds, state, final_bin = min(finals, key=lambda f: f[0])
        steps = []
        if final_bin is not None:
            steps.append(("dump", final_bin))
        steps.append(("base", self.base))
        tail = []
        while best[state][2] is not None:
            _, _, previous, (j, via) = best[state]
            tail.append(("pick", items[j]))
            if via is not None:
                tail.append(("dump", via))
            state = previous
        steps = list(reversed(tail)) + steps
        return {"steps": steps, "objective": objective, "estimated_seconds": seconds}

    # ------------------------------------------------------------ battery

    def nearest_bin(self, position):
        return min(self.bins, key=lambda b: self.leg(position, b.position)[0])

    def step_seconds(self, start, step):
        kind, node = step
        travel, _ = self.leg(start, node.position)
        work = {"pick": self.pickup_cost(node) if kind == "pick" else 0.0,
                "dump": self.config.dump_seconds, "base": 0.0}[kind]
        return travel + work

    def can_afford(self, battery, start, step):
        """True if the step leaves enough charge to then reach the base."""

        if step[0] == "base":
            return True
        seconds = self.step_seconds(start, step) + self.leg(step[1].position, self.base.position)[0]
        return battery - self.config.energy(seconds) >= self.config.battery_reserve
