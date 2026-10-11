"""Metre-based grid indexing: arrays are [row north, column east]."""

from dataclasses import dataclass
from decimal import Decimal
import math
from numbers import Integral

import numpy as np

from src.environment.layout import Circle, Rect


def _finite_coordinate(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float, np.integer, np.floating))
            and math.isfinite(value))


@dataclass(frozen=True)
class GridSpec:
    """Bounds are [origin, origin + size); outside points raise IndexError.

    Dimensions must be whole multiples of resolution. Cell centres are for
    layer queries only: object positions must retain their source coordinates.
    """

    width: float
    depth: float
    resolution: float = 0.10
    origin: tuple = (0.0, 0.0)

    def __post_init__(self):
        for name in ("width", "depth", "resolution"):
            value = getattr(self, name)
            if not _finite_coordinate(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not isinstance(self.origin, (tuple, list)) or len(self.origin) != 2 or not all(_finite_coordinate(v) for v in self.origin):
            raise ValueError("origin must have two finite coordinates")
        object.__setattr__(self, "origin", tuple(self.origin))
        for size in (self.width, self.depth):
            cells = size / self.resolution
            if round(cells) < 1 or not math.isclose(cells, round(cells), abs_tol=1e-9, rel_tol=0):
                raise ValueError("dimensions must be whole multiples of resolution")

    @property
    def columns(self):
        return round(self.width / self.resolution)

    @property
    def rows(self):
        return round(self.depth / self.resolution)

    @property
    def shape(self):
        return (self.rows, self.columns)

    @property
    def extent(self):
        x, y = self.origin
        return (x, x + self.width, y, y + self.depth)

    def world_to_cell(self, x, y):
        if not all(_finite_coordinate(v) for v in (x, y)):
            raise ValueError("world coordinates must be finite")
        ox, oy = self.origin
        # Decimal strings avoid 0.3 / 0.1 becoming 2.9999999999999996;
        # unlike an epsilon, this still distinguishes nextafter(edge, -inf).
        dx = Decimal(str(x)) - Decimal(str(ox))
        dy = Decimal(str(y)) - Decimal(str(oy))
        if not (0 <= dx < Decimal(str(self.width)) and 0 <= dy < Decimal(str(self.depth))):
            raise IndexError("point is outside the half-open map bounds")
        step = Decimal(str(self.resolution))
        column = math.floor(dx / step)
        row = math.floor(dy / step)
        self.validate_cell(row, column)
        return (row, column)

    def validate_cell(self, row, column):
        if any(isinstance(v, bool) or not isinstance(v, Integral) for v in (row, column)):
            raise ValueError("cell indices must be integers")
        if not (0 <= row < self.rows and 0 <= column < self.columns):
            raise IndexError("cell is outside the map")

    def cell_center(self, row, column):
        self.validate_cell(row, column)
        return (self.origin[0] + (column + 0.5) * self.resolution,
                self.origin[1] + (row + 0.5) * self.resolution)

    def cell_bounds(self):
        """Broadcastable arrays for each cell's x0, y0, x1, y1."""
        x = self.origin[0] + np.arange(self.columns)[None, :] * self.resolution
        y = self.origin[1] + np.arange(self.rows)[:, None] * self.resolution
        return x, y, x + self.resolution, y + self.resolution


def footprint_mask(grid, shape):
    """Intersect footprints with cell areas, including circles missing centres.

    Rectangles require positive area overlap (shared edges do not close the
    ramp opening). Circles include tangencies. A 1e-12 m tolerance suppresses
    floating-point slivers at aligned rectangle edges.
    """
    x0, y0, x1, y1 = grid.cell_bounds()
    if isinstance(shape, Rect):
        return ((np.minimum(x1, shape.x1) - np.maximum(x0, shape.x0) > 1e-12)
                & (np.minimum(y1, shape.y1) - np.maximum(y0, shape.y0) > 1e-12))
    if isinstance(shape, Circle):
        dx = np.maximum(np.maximum(x0 - shape.x, shape.x - x1), 0)
        dy = np.maximum(np.maximum(y0 - shape.y, shape.y - y1), 0)
        return dx * dx + dy * dy <= shape.r * shape.r
    raise TypeError(f"unsupported footprint: {type(shape).__name__}")


def clearance_mask(blocked, resolution, radius):
    """Conservatively expand blocked cell squares by radius metres.

    Padding is a separate planning layer; it never changes raw occupancy.
    Adjacent cells touching an expanded square are included for positive radii.
    This is grid geometry, not a certification of the quadruped's footprint.
    """
    if isinstance(radius, bool) or not math.isfinite(radius) or radius < 0:
        raise ValueError("clearance radius must be nonnegative and finite")
    padded = blocked.copy()
    if radius == 0:
        return padded
    rows, columns = blocked.shape
    span = math.ceil(radius / resolution) + 1
    for dr in range(-min(span, rows - 1), min(span, rows - 1) + 1):
        for dc in range(-min(span, columns - 1), min(span, columns - 1) + 1):
            gap_y = max(abs(dr) - 1, 0) * resolution
            gap_x = max(abs(dc) - 1, 0) * resolution
            if math.hypot(gap_x, gap_y) > radius + 1e-12:
                continue
            r0, r1 = max(0, dr), min(rows, rows + dr)
            c0, c1 = max(0, dc), min(columns, columns + dc)
            padded[r0:r1, c0:c1] |= blocked[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
    return padded
