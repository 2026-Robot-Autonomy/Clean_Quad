"""Headless semantic-map rendering, alongside the existing campus plot."""

from pathlib import Path

import numpy as np


def save_figure(semantic_map, layout, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from src.environment.plotting import LITTER_MARKERS, ZONE_COLOURS, plot_campus

    colours = {**ZONE_COLOURS, "curb": "#f0d000", "obstacle": "#393939", "outside": "#eeeeee"}
    labels = sorted(np.unique(semantic_map.layers["terrain"]).tolist())
    codes = np.zeros(semantic_map.grid.shape, dtype=int)
    for code, label in enumerate(labels):
        codes[semantic_map.layers["terrain"] == label] = code
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.7), layout="constrained")
    ax = axes[0]
    ax.imshow(codes, origin="lower", extent=semantic_map.grid.extent,
              cmap=ListedColormap([colours.get(label, "#dddddd") for label in labels]),
              vmin=-0.5, vmax=len(labels) - 0.5, interpolation="nearest")
    blocked = ~semantic_map.layers["traversable"]
    overlay = np.zeros((*blocked.shape, 4))
    overlay[blocked] = (0.85, 0.08, 0.10, 0.40)
    ax.imshow(overlay, origin="lower", extent=semantic_map.grid.extent, interpolation="nearest")
    padded_only = semantic_map.layers["clearance_blocked"] & ~blocked
    overlay = np.zeros((*blocked.shape, 4))
    overlay[padded_only] = (0.30, 0.16, 0.80, 0.38)
    ax.imshow(overlay, origin="lower", extent=semantic_map.grid.extent, interpolation="nearest")
    x, y, _ = semantic_map.landmarks["base_pose"]
    ax.plot(x, y, "*", ms=15, color="#f28c1a", mec="black", label="base")
    for number, bin_data in enumerate(semantic_map.landmarks["bins"], 1):
        shape = bin_data["shape"]
        ax.plot(shape["x"], shape["y"], "s", color="#1a66a6", ms=8,
                label="bins" if number == 1 else None)
    ax.set(xlabel="x east [m]", ylabel="y north [m]", title=f"Semantic grid: {semantic_map.grid.resolution * 100:g} cm cells")
    ax.set_aspect("equal")
    terrain_handles = [Patch(facecolor=colours.get(label, "#dddddd"), label=label) for label in labels]
    terrain_handles.append(Patch(facecolor="#dc6b6d", label="blocked for traversal"))
    if np.any(padded_only):
        terrain_handles.append(Patch(facecolor="#9b80d1", label=f"clearance +{semantic_map.clearance_radius:g} m"))
    plot_campus(axes[1], layout)
    axes[1].get_legend().remove()
    axes[1].set_title("Known campus layout + estimated litter")
    for panel in axes:
        for kind, (marker, colour) in LITTER_MARKERS.items():
            items = [item for item in semantic_map.active_items() if item["class"] == kind]
            if items:
                panel.scatter([item["position"][0] for item in items],
                              [item["position"][1] for item in items], marker=marker,
                              c=colour, edgecolors="black", s=65, zorder=5, label=kind)
        for item in semantic_map.items.values():
            x, y = item["position"][:2]
            panel.annotate(f"{item['track_id']} {item['class']}" + (" (collected)" if item["collected"] else ""),
                           (x, y), xytext=(5, 7), textcoords="offset points", fontsize=7,
                           color="#111111", bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1),
                           zorder=6)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=terrain_handles + handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7)
    fig.suptitle(f"Stage 5 | scene seed {semantic_map.seed} | {len(semantic_map.active_items())} uncollected items")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)

