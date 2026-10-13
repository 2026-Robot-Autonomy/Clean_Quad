"""Clearance-aware A* and litter approach poses on the Stage 5 map.

Navigation owns its padded mask; the saved semantic map and item evidence are
never changed. Units are metres and radians, in the existing world frame.
"""

from dataclasses import dataclass
import heapq
import math

import numpy as np

from src.mapping.grid import clearance_mask


class NoPathError(ValueError):
    """A valid query has no clearance-safe route or approach pose."""


@dataclass(frozen=True)
class NavigationConfig:
    clearance_radius: float = 0.45
    approach_min: float = 0.45
    approach_max: float = 0.65
    approach_preferred: float = 0.50
    grass_cost: float = 1.15

    def __post_init__(self):
        for name in ("clearance_radius", "approach_min", "approach_max",
                     "approach_preferred", "grass_cost"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not self.approach_min <= self.approach_preferred <= self.approach_max:
            raise ValueError("approach distances must be ordered")
        if self.grass_cost < 1:
            raise ValueError("terrain costs must be >= 1 for the A* heuristic")


@dataclass(frozen=True)
class RoutePlan:
    waypoints: tuple
    cells: tuple
    grid_cost: float
    length: float


@dataclass(frozen=True)
class ApproachPlan:
    track_id: int
    position: tuple
    yaw: float
    standoff: float
    route: RoutePlan


class AStarPlanner:
    """Eight-neighbour A*, with no diagonal corner cutting.

Line-of-sight pruning uses exact segment/cell-square intersections, including
corner tangencies. It preserves clearance, but the pruned route is not claimed
to minimize the original terrain-weighted grid cost.
"""

    def __init__(self, semantic_map, config=None):
        self.map = semantic_map
        self.grid = semantic_map.grid
        self.config = config or NavigationConfig()
        self.raw_blocked = (~semantic_map.layers["traversable"]).copy()
        self.blocked = clearance_mask(self.raw_blocked, self.grid.resolution,
                                      self.config.clearance_radius)
        self.cost = np.ones(self.grid.shape)
        self.cost[semantic_map.layers["terrain"] == "grass"] = self.config.grass_cost
        self._bounds = {}
        for raw, mask in ((False, self.blocked), (True, self.raw_blocked)):
            row, col = np.nonzero(mask)
            low = np.column_stack((self.grid.origin[0] + col * self.grid.resolution,
                                   self.grid.origin[1] + row * self.grid.resolution))
            self._bounds[raw] = (low, low + self.grid.resolution)
        self.blocked.flags.writeable = False

    def is_free(self, point, *, raw=False):
        try:
            cell = self.grid.world_to_cell(*point)
        except IndexError:
            return False
        return not (self.raw_blocked if raw else self.blocked)[cell]

    def segment_is_free(self, start, end, *, raw=False):
        if not self.is_free(start, raw=raw) or not self.is_free(end, raw=raw):
            return False
        a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        lower, upper = self._bounds[raw]
        # Slab intersection: t in [0,1] where a + t*(b-a) touches each square.
        entry, leave = np.zeros(len(lower)), np.ones(len(lower))
        intersects = np.ones(len(lower), dtype=bool)
        for axis in range(2):
            delta = b[axis] - a[axis]
            if abs(delta) < 1e-12:
                intersects &= (a[axis] >= lower[:, axis] - 1e-12) & (a[axis] <= upper[:, axis] + 1e-12)
            else:
                t0 = (lower[:, axis] - a[axis]) / delta
                t1 = (upper[:, axis] - a[axis]) / delta
                entry = np.maximum(entry, np.minimum(t0, t1))
                leave = np.minimum(leave, np.maximum(t0, t1))
        return not np.any(intersects & (entry <= leave + 1e-12))

    def clearance_at(self, point):
        """Distance to raw blocked cell squares; conservative versus geometry."""
        if not self.is_free(point, raw=True):
            return 0.0
        low, high = self._bounds[True]
        if not len(low):
            return math.inf
        delta = np.maximum(np.maximum(low - point, np.asarray(point) - high), 0)
        return float(np.min(np.linalg.norm(delta, axis=1)))

    def _neighbours(self, cell):
        row, col = cell
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nxt = row + dr, col + dc
            if not (0 <= nxt[0] < self.grid.rows and 0 <= nxt[1] < self.grid.columns):
                continue
            if self.blocked[nxt]:
                continue
            if dr and dc and (self.blocked[row + dr, col] or self.blocked[row, col + dc]):
                continue
            distance = math.hypot(dr, dc) * self.grid.resolution
            yield nxt, distance * (self.cost[cell] + self.cost[nxt]) / 2

    def _search(self, start, goal=None):
        distance, parent = {start: 0.0}, {}
        queue = [(0.0, 0.0, start)]
        while queue:
            _, travelled, cell = heapq.heappop(queue)
            if travelled > distance[cell] + 1e-12:
                continue
            if cell == goal:
                break
            for nxt, edge in self._neighbours(cell):
                proposed = travelled + edge
                if proposed + 1e-12 < distance.get(nxt, math.inf):
                    distance[nxt], parent[nxt] = proposed, cell
                    heuristic = 0.0 if goal is None else math.dist(nxt, goal) * self.grid.resolution
                    heapq.heappush(queue, (proposed + heuristic, proposed, nxt))
        return distance, parent

    def _plan_direct(self, start, goal):
        start, goal = tuple(start), tuple(goal)
        source = self.grid.world_to_cell(*start)
        destination = self.grid.world_to_cell(*goal)
        if self.blocked[source] or self.blocked[destination]:
            raise NoPathError("start or goal is blocked after robot-clearance padding")
        if start == goal:
            return RoutePlan((start,), (source,), 0.0, 0.0)
        distance, parent = self._search(source, destination)
        if destination not in distance:
            raise NoPathError("goal is unreachable on the clearance-aware grid")
        cells = [destination]
        while cells[-1] != source:
            cells.append(parent[cells[-1]])
        cells.reverse()
        points = [start] + [self.grid.cell_center(*cell) for cell in cells] + [goal]
        points = [p for i, p in enumerate(points) if i == 0 or math.dist(p, points[i - 1]) > 1e-9]
        pruned, index = [points[0]], 0
        while index < len(points) - 1:
            next_index = len(points) - 1
            while next_index > index and not self.segment_is_free(points[index], points[next_index]):
                next_index -= 1
            if next_index == index:
                raise NoPathError("grid route has no safe world-coordinate connection")
            pruned.append(points[next_index])
            index = next_index
        length = sum(math.dist(a, b) for a, b in zip(pruned, pruned[1:]))
        return RoutePlan(tuple(pruned), tuple(cells), float(distance[destination]), length)

    def plan(self, start, goal):
        """Preserve a straight ramp crossing using map-derived entry/exit poses.

        The ramp portals are terrain execution constraints, not hand-picked
        litter routes. All sections still go through A* and clearance checks.
        """
        start, goal = tuple(start), tuple(goal)
        # Validate public coordinates even when no ramp crossing is needed.
        self.grid.world_to_cell(*start)
        self.grid.world_to_cell(*goal)
        ramp = self.map.landmarks.get("ramp")
        portals = []
        if ramp:
            top = ramp["y_bottom"] + (ramp["z_top"] - ramp["z_bottom"]) / math.tan(math.radians(ramp["slope_deg"]))
            if (start[1] - top) * (goal[1] - top) < 0:
                x = (ramp["x0"] + ramp["x1"]) / 2
                # Leave room to settle before turning beyond either ramp end.
                margin = self.config.clearance_radius + 2 * self.grid.resolution
                portals = [(x, ramp["y_bottom"] - margin), (x, top + margin)]
                if start[1] > goal[1]:
                    portals.reverse()
        endpoints = [start] + portals + [goal]
        sections = [self._plan_direct(a, b) for a, b in zip(endpoints, endpoints[1:])]
        points = list(sections[0].waypoints)
        cells = list(sections[0].cells)
        for section in sections[1:]:
            points.extend(section.waypoints[1:])
            cells.extend(section.cells[1:])
        return RoutePlan(tuple(points), tuple(cells), sum(p.grid_cost for p in sections),
                         sum(p.length for p in sections))

    def approach_item(self, start, item, *, dock_offset=None):
        """Approach pose facing the item.

        With dock_offset, the pose must also leave a padded-free straight line
        to the dock point, dock_offset short of the item, so a pickup tool
        mounted ahead of the trunk centre can be driven over the litter.
        """
        if item.get("collected", False):
            raise NoPathError("item has already been collected")
        source = self.grid.world_to_cell(*start)
        if self.blocked[source]:
            raise NoPathError("start is blocked after robot-clearance padding")
        target = tuple(item["position"][:2])
        target_cell = self.grid.world_to_cell(*target)
        if self.raw_blocked[target_cell]:
            raise NoPathError("litter is in a forbidden or occupied cell")
        distance, _ = self._search(source)
        candidates = []
        target_height = self.map.layers["height"][target_cell]
        for cell, cost in distance.items():
            position = self.grid.cell_center(*cell)
            standoff = math.dist(position, target)
            if not self.config.approach_min <= standoff <= self.config.approach_max:
                continue
            # Avoid approaching litter across the raised curb or through solids.
            if abs(self.map.layers["height"][cell] - target_height) > 0.025:
                continue
            if not self.segment_is_free(position, target, raw=True):
                continue
            if dock_offset is not None:
                dock = dock_point(position, target, dock_offset)
                if not self.segment_is_free(position, dock):
                    continue
            score = cost + 2.0 * abs(standoff - self.config.approach_preferred)
            candidates.append((score, cell, position, standoff))
        for _, _, position, standoff in sorted(candidates):
            try:
                route = self.plan(start, position)
            except NoPathError:
                continue
            yaw = math.atan2(target[1] - position[1], target[0] - position[0])
            return ApproachPlan(item["track_id"], position, yaw, standoff, route)
        raise NoPathError("no reachable litter approach pose within the standoff range")

    def approach_bin(self, start, bin_landmark, *, margin=0.05, band=0.25):
        """Pose facing a circular bin from just outside the padded obstacle.

        bin_landmark is a map landmark record with a Circle shape. The line of
        sight to the bin rim must be raw-free and on the same surface height.
        """
        shape = bin_landmark["shape"]
        centre, radius = (shape["x"], shape["y"]), shape["r"]
        source = self.grid.world_to_cell(*start)
        if self.blocked[source]:
            raise NoPathError("start is blocked after robot-clearance padding")
        nearest = radius + self.config.clearance_radius + margin
        distance, _ = self._search(source)
        candidates = []
        for cell, cost in distance.items():
            position = self.grid.cell_center(*cell)
            standoff = math.dist(position, centre)
            if not nearest <= standoff <= nearest + band:
                continue
            rim = dock_point(position, centre, radius + 2 * self.grid.resolution)
            if not self.segment_is_free(position, rim, raw=True):
                continue
            rim_cell = self.grid.world_to_cell(*rim)
            if abs(self.map.layers["height"][cell] - self.map.layers["height"][rim_cell]) > 0.025:
                continue
            candidates.append((cost + 2.0 * (standoff - nearest), position, standoff))
        for _, position, standoff in sorted(candidates):
            try:
                route = self.plan(start, position)
            except NoPathError:
                continue
            yaw = math.atan2(centre[1] - position[1], centre[0] - position[0])
            return ApproachPlan(0, position, yaw, standoff, route)
        raise NoPathError(f"no reachable approach pose for {bin_landmark['name']}")


def dock_point(position, target, offset):
    """Point `offset` metres short of target on the line from position."""
    dx, dy = target[0] - position[0], target[1] - position[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        raise ValueError("dock direction is undefined")
    return (target[0] - offset * dx / length, target[1] - offset * dy / length)
