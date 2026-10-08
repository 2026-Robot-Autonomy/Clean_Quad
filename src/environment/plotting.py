"""
Top-down campus map with matplotlib (for reports and later
for overlaying maps, paths and plans).
"""

from matplotlib.patches import Circle as CirclePatch
from matplotlib.patches import Rectangle

from src.environment.layout import Circle


ZONE_COLOURS = {
    "road": "#45464a",
    "pavement": "#a3a39e",
    "plaza": "#c4b9a3",
    "grass": "#5c9a4a",
    "planter": "#735938",
    "ramp": "#d1b45c",
    "stairs": "#99806a",
    "entrance": "#8c6b52",
}

OBSTACLE_COLOURS = {
    "tree": "#6b4423",
    "bench": "#80542e",
    "pole": "#333333",
    "bin": "#1a66a6",
}

LITTER_MARKERS = {
    "bottle": ("o", "#7fd98f"),
    "can": ("s", "#d92626"),
    "cup": ("^", "#ffffff"),
    "wrapper": ("D", "#f2cc19"),
}


def add_rect(ax, rect, colour, **kwargs):
    ax.add_patch(Rectangle(
        (rect.x0, rect.y0),
        rect.x1 - rect.x0,
        rect.y1 - rect.y0,
        facecolor=colour,
        **kwargs,
    ))


def plot_campus(ax, layout, litter=None, path=None):

    # Reverse so raised features (listed first) are drawn on top.
    # Hatched = no-go.
    for s in reversed(layout.surfaces):
        no_go = not layout.traversable[s.kind]
        add_rect(
            ax, s.rect, ZONE_COLOURS[s.kind],
            edgecolor="black" if no_go else "none", lw=0.8,
            hatch="//" if no_go else None,
        )

    for rect, _ in layout.stairs.steps():
        add_rect(ax, rect, ZONE_COLOURS["stairs"], edgecolor="black", lw=0.5, hatch="//")

    add_rect(ax, layout.ramp.rect, ZONE_COLOURS["ramp"], edgecolor="black", lw=0.5, label="ramp")

    curb = dict(color="#f0d000", lw=2.5)
    ax.plot([0, layout.ramp.x0], [layout.curb_y] * 2, **curb,
            label=f"curb {layout.curb_height * 100:.0f} cm (no-go)")
    ax.plot([layout.ramp.x1, layout.width], [layout.curb_y] * 2, **curb)

    labelled = set()

    for o in layout.obstacles:
        colour = OBSTACLE_COLOURS[o.kind]
        label = None if o.kind in labelled else o.kind
        labelled.add(o.kind)
        if isinstance(o.shape, Circle):
            ax.add_patch(CirclePatch(o.shape.center, o.shape.r, color=colour, label=label))
        else:
            add_rect(ax, o.shape, colour, label=label)

    bx, by, _ = layout.base_pose
    ax.plot(bx, by, marker="*", ms=16, color="#f28c1a", mec="black", ls="none", label="base")

    if litter:
        for kind, (marker, colour) in LITTER_MARKERS.items():
            pts = [it.position for it in litter if it.kind == kind and not it.collected]
            if pts:
                ax.plot(
                    [p[0] for p in pts], [p[1] for p in pts],
                    marker=marker, color=colour, mec="black", ls="none", ms=7, label=kind,
                )

    if path is not None:
        ax.plot(path[:, 0], path[:, 1], color="#ff3b3b", lw=1.5, label="robot path")

    for s in layout.surfaces:
        ax.text(
            s.rect.x0 + 0.08, s.rect.y0 + 0.08, s.kind.upper(),
            ha="left", va="bottom", color="white", alpha=0.8, fontsize=8,
        )

    ax.set_xlim(-0.2, layout.width + 0.2)
    ax.set_ylim(-0.2, layout.depth + 0.2)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
