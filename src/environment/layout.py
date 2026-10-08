"""
CleanQuad campus layout: WHAT is in the world.

Pure data plus geometric queries. No MuJoCo here, so the
same description drives the simulator (terrain.py,
objects.py), the litter placement (litter.py), the map
plots, and later the planners' ground truth.

World frame: x east, y north, z up. Units: metres, degrees.

    y=6 +-----------------------------+-------------+
        |  GRASS                      |  ENTRANCE   |
        |   tree              bench   |  == stairs  |
        |        [PLANTER]            |             |
        |                             |  PLAZA  bin2|
    3.2 +-----------------------------+-------------+
        |  PAVEMENT (raised)       pole        bin1 |
    2.0 +======== curb =======| RAMP |=== curb ====+
        |  ROAD                                     |
        |  [BASE]                                   |
    y=0 +-------------------------------------------+
       x=0                                         x=8

Terrain design (from experiments/18_terrain_traversal_test.py,
V2_FAST gait at 0.3 m/s)
------------------------------------------------------------
- ramps up to 15 deg: crossed both ways      -> ramp is walkable
- curbs >= 4 cm: the robot falls going up    -> curb is a no-go line
So pavement, grass and plaza are raised by CURB_HEIGHT and the
only way up from the road is the ramp (curb cut).
"""

from dataclasses import dataclass, field
import math


# =========================================================
# PARAMETERS
# =========================================================

MAP_WIDTH = 8.0
MAP_DEPTH = 6.0

# Surface friction (see terrain.py): the A1 feet always use
# their own friction, so this only changes the lower-calf
# contacts. 1.0 is the MuJoCo default = the Stage-1 floor.
# Grass is mainly a label and a planning cost.
FRICTION_PAVED = 1.0
FRICTION_GRASS = 0.6

CURB_HEIGHT = 0.06
RAMP_SLOPE_DEG = 8.0
RAMP_X = (3.4, 4.6)

PLANTER_HEIGHT = 0.06       # above the pavement level

STAIR_RISE = 0.06
STAIR_DEPTH = 0.20
STAIR_COUNT = 3

WALL_HEIGHT = 0.40
WALL_THICKNESS = 0.10


# =========================================================
# SHAPES
# =========================================================

@dataclass(frozen=True)
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    def contains(self, x, y):
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    @property
    def center(self):
        return (0.5 * (self.x0 + self.x1), 0.5 * (self.y0 + self.y1))

    @property
    def half_size(self):
        return (0.5 * (self.x1 - self.x0), 0.5 * (self.y1 - self.y0))


@dataclass(frozen=True)
class Circle:
    x: float
    y: float
    r: float

    def contains(self, x, y):
        return (x - self.x) ** 2 + (y - self.y) ** 2 <= self.r ** 2

    @property
    def center(self):
        return (self.x, self.y)


# =========================================================
# WORLD ELEMENTS
# =========================================================

@dataclass(frozen=True)
class Surface:
    """A flat region with its top at `height`."""

    kind: str           # zone label, also geom name "terrain_<kind>"
    rect: Rect
    height: float
    friction: float


@dataclass(frozen=True)
class Ramp:
    """Slope rising along +y from z_bottom (at y_bottom) to z_top."""

    kind: str
    x0: float
    x1: float
    y_bottom: float
    z_bottom: float
    z_top: float
    slope_deg: float
    friction: float

    @property
    def length(self):
        return (self.z_top - self.z_bottom) / math.tan(math.radians(self.slope_deg))

    @property
    def y_top(self):
        return self.y_bottom + self.length

    @property
    def rect(self):
        return Rect(self.x0, self.y_bottom, self.x1, self.y_top)

    def height_at(self, y):
        t = min(max((y - self.y_bottom) / self.length, 0.0), 1.0)
        return self.z_bottom + t * (self.z_top - self.z_bottom)


@dataclass(frozen=True)
class Stairs:
    """Steps rising along +y, starting at y0 on a surface at base_height."""

    kind: str
    x0: float
    x1: float
    y0: float
    base_height: float
    count: int
    rise: float
    depth: float
    friction: float

    @property
    def rect(self):
        return Rect(self.x0, self.y0, self.x1, self.y0 + self.count * self.depth)

    @property
    def top_height(self):
        """Height of the landing above the last step."""
        return self.base_height + (self.count + 1) * self.rise

    def steps(self):
        """Yield (Rect, top_height) for every step."""
        for k in range(self.count):
            y = self.y0 + k * self.depth
            yield (
                Rect(self.x0, y, self.x1, y + self.depth),
                self.base_height + (k + 1) * self.rise,
            )


@dataclass(frozen=True)
class Obstacle:
    """Solid object standing on the ground. Bins are obstacles too."""

    name: str           # geom name, e.g. "obstacle_tree", "bin_1"
    kind: str           # "tree" | "bench" | "pole" | "bin"
    shape: object       # Rect or Circle footprint
    height: float


# =========================================================
# LAYOUT
# =========================================================

# Zone label -> can the Stage-1 robot walk there?
DEFAULT_TRAVERSABLE = {
    "road": True,
    "pavement": True,
    "plaza": True,
    "grass": True,
    "ramp": True,
    "curb": False,
    "planter": False,
    "stairs": False,
    "entrance": False,
    "obstacle": False,
    "outside": False,
}


@dataclass
class CampusLayout:
    """
    surfaces are listed in query priority order: the first
    one containing a point defines its zone.
    """

    width: float
    depth: float
    surfaces: tuple
    ramp: Ramp
    stairs: Stairs
    obstacles: tuple
    base_pose: tuple            # (x, y, yaw)
    curb_y: float               # road / pavement boundary
    curb_height: float
    curb_band: float = 0.05     # half-width of the "curb" query strip
    wall_height: float = WALL_HEIGHT
    wall_thickness: float = WALL_THICKNESS
    traversable: dict = field(default_factory=lambda: dict(DEFAULT_TRAVERSABLE))

    @property
    def bins(self):
        return [o for o in self.obstacles if o.kind == "bin"]

    def surface(self, kind):
        for s in self.surfaces:
            if s.kind == kind:
                return s
        raise KeyError(kind)

    # -----------------------------------------------------
    # QUERIES
    # -----------------------------------------------------

    def zone_at(self, x, y):
        """
        Zone label at (x, y), highest priority first:
        outside, obstacle, stairs, ramp, curb, surfaces.
        """

        if not (0.0 <= x <= self.width and 0.0 <= y <= self.depth):
            return "outside"

        for o in self.obstacles:
            if o.shape.contains(x, y):
                return "obstacle"

        if self.stairs.rect.contains(x, y):
            return "stairs"

        if self.ramp.rect.contains(x, y):
            return "ramp"

        on_ramp_span = self.ramp.x0 <= x <= self.ramp.x1

        if abs(y - self.curb_y) <= self.curb_band and not on_ramp_span:
            return "curb"

        for s in self.surfaces:
            if s.rect.contains(x, y):
                return s.kind

        return "outside"

    def ground_height_at(self, x, y):
        """Top-surface height at (x, y), ignoring obstacles."""

        if self.ramp.rect.contains(x, y):
            return self.ramp.height_at(y)

        for rect, top in self.stairs.steps():
            if rect.contains(x, y):
                return top

        for s in self.surfaces:
            if s.rect.contains(x, y):
                return s.height

        return 0.0

    def is_traversable(self, x, y):
        return self.traversable[self.zone_at(x, y)]


# =========================================================
# THE CAMPUS
# =========================================================

def default_layout():

    h = CURB_HEIGHT
    curb_y = 2.0

    # Curb cut: the ramp sits on the road and ends at the curb.
    ramp_length = h / math.tan(math.radians(RAMP_SLOPE_DEG))

    ramp = Ramp(
        kind="ramp",
        x0=RAMP_X[0],
        x1=RAMP_X[1],
        y_bottom=curb_y - ramp_length,
        z_bottom=0.0,
        z_top=h,
        slope_deg=RAMP_SLOPE_DEG,
        friction=FRICTION_PAVED,
    )

    stairs = Stairs(
        kind="stairs",
        x0=5.8,
        x1=7.4,
        y0=4.6,
        base_height=h,
        count=STAIR_COUNT,
        rise=STAIR_RISE,
        depth=STAIR_DEPTH,
        friction=FRICTION_PAVED,
    )

    # Raised features first: they override the zone below.
    surfaces = (
        Surface("entrance", Rect(stairs.x0, stairs.rect.y1, stairs.x1, MAP_DEPTH),
                stairs.top_height, FRICTION_PAVED),
        Surface("planter", Rect(2.0, 3.8, 3.4, 4.2), h + PLANTER_HEIGHT, FRICTION_GRASS),
        Surface("road", Rect(0.0, 0.0, MAP_WIDTH, curb_y), 0.0, FRICTION_PAVED),
        Surface("pavement", Rect(0.0, curb_y, MAP_WIDTH, 3.2), h, FRICTION_PAVED),
        Surface("grass", Rect(0.0, 3.2, 5.0, MAP_DEPTH), h, FRICTION_GRASS),
        Surface("plaza", Rect(5.0, 3.2, MAP_WIDTH, MAP_DEPTH), h, FRICTION_PAVED),
    )

    obstacles = (
        Obstacle("obstacle_tree", "tree", Circle(1.0, 5.0, 0.15), 1.2),
        Obstacle("obstacle_bench", "bench", Rect(3.4, 5.1, 4.4, 5.45), 0.45),
        Obstacle("obstacle_pole", "pole", Circle(5.4, 2.7, 0.06), 2.0),
        Obstacle("bin_1", "bin", Circle(7.4, 2.6, 0.18), 0.5),
        Obstacle("bin_2", "bin", Circle(7.4, 3.9, 0.18), 0.5),
    )

    return CampusLayout(
        width=MAP_WIDTH,
        depth=MAP_DEPTH,
        surfaces=surfaces,
        ramp=ramp,
        stairs=stairs,
        obstacles=obstacles,
        base_pose=(0.8, 1.0, 0.0),
        curb_y=curb_y,
        curb_height=h,
    )
