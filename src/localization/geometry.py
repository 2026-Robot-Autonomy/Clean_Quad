"""Camera geometry and localization using RGB boxes and depth only.

MuJoCo camera coordinates are x right, y up, z backward. Depth is
positive distance along -z, rather than Euclidean distance from the lens.
Segmentation, object identities, and simulator object positions are never
inputs to either estimator.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class LocalizationConfig:
    min_score: float = 0.38
    min_depth: float = 0.05
    max_range: float = 8.0
    min_depth_pixels: int = 9
    depth_crop: float = 0.6
    max_extent: float = 0.35
    ground_z: float = 0.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (
            self.min_score, self.min_depth, self.max_range,
            self.depth_crop, self.max_extent, self.ground_z,
        )):
            raise ValueError("localization parameters must be finite")
        if not 0 <= self.min_score <= 1:
            raise ValueError("min_score must be in [0, 1]")
        if not 0 < self.min_depth < self.max_range:
            raise ValueError("depth limits must be positive and increasing")
        if not 0 < self.depth_crop <= 1 or not isinstance(self.min_depth_pixels, int) or self.min_depth_pixels < 1:
            raise ValueError("depth_crop must be in (0, 1], and min_depth_pixels >= 1")
        if self.max_extent <= 0:
            raise ValueError("max_extent must be positive")


@dataclass(frozen=True)
class LocalizedDetection:
    source_index: int
    class_id: int
    score: float
    position: np.ndarray
    time: float
    distance: float
    method: str
    depth_pixels: int = 0
    depth_mad: float = 0.0


def pixel_to_world(frame, u, v, depth):
    """Back-project pixel coordinates and axial depths; accepts arrays."""

    u, v, depth = np.broadcast_arrays(u, v, depth)
    k = frame.intrinsics
    camera_points = np.stack([
        (u - k.cx) * depth / k.fx,
        -(v - k.cy) * depth / k.fy,
        -depth,
    ], axis=-1)
    return camera_points @ frame.cam_mat.T + frame.cam_pos


def pixel_ray(frame, u, v):
    """Unnormalised world ray, with a camera-frame axial depth of one."""
    return pixel_to_world(frame, u, v, 1.0) - frame.cam_pos


def _box(frame, detection):
    values = np.asarray(detection.box, dtype=float)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        return None
    x0, y0, x1, y1 = values
    if x1 <= x0 or y1 <= y0:
        return None
    k = frame.intrinsics
    x0, x1 = np.clip([x0, x1], 0, k.width)
    y0, y1 = np.clip([y0, y1], 0, k.height)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return float(x0), float(y0), float(x1), float(y1)


def localize(frame, detection, config=None, method="depth", source_index=0):
    """Return (observation, rejection reason); an accepted reason is None.

    Depth uses the median of a central crop to reduce background pixels in
    detector boxes. The estimate is a visible surface point near the box
    centre, so error against an object's centre includes surface bias.

    The ablation intersects the bottom-centre ray with one fixed horizontal
    ground plane. It does not read depth or consult the campus terrain map.
    """

    config = config or LocalizationConfig()
    if method not in ("depth", "ground_plane"):
        raise ValueError(f"unknown localization method: {method}")
    if not math.isfinite(detection.score) or detection.score < config.min_score:
        return None, "score"
    if not 0 <= detection.class_id < 4:
        return None, "class"
    box = _box(frame, detection)
    if box is None:
        return None, "box"
    x0, y0, x1, y1 = box
    u, v = (x0 + x1) / 2, (y0 + y1) / 2
    k = frame.intrinsics
    count, mad = 0, 0.0

    if method == "depth":
        if frame.depth is None:
            return None, "missing_depth"
        if frame.depth.shape != (k.height, k.width):
            raise ValueError("depth image shape does not match camera intrinsics")
        half_w = (x1 - x0) * config.depth_crop / 2
        half_h = (y1 - y0) * config.depth_crop / 2
        left, right = math.floor(u - half_w), math.ceil(u + half_w)
        top, bottom = math.floor(v - half_h), math.ceil(v + half_h)
        values = frame.depth[top:bottom, left:right]
        values = values[np.isfinite(values) & (values >= config.min_depth) & (values <= config.max_range)]
        count = int(values.size)
        if count < config.min_depth_pixels:
            return None, "depth_pixels"
        axial_depth = float(np.median(values))
        mad = float(np.median(np.abs(values - axial_depth)))
        position = pixel_to_world(frame, u, v, axial_depth)
    else:
        ray = pixel_ray(frame, u, min(y1, k.height) - 0.5)
        if ray[2] >= -1e-8:
            return None, "ground_ray"
        axial_depth = (config.ground_z - frame.cam_pos[2]) / ray[2]
        if axial_depth <= 0:
            return None, "ground_ray"
        position = frame.cam_pos + axial_depth * ray

    distance = float(np.linalg.norm(position - frame.cam_pos))
    if not np.all(np.isfinite(position)) or not config.min_depth <= distance <= config.max_range:
        return None, "range"
    # Approximate transverse size at the estimated depth. This removes large
    # scenery (notably bin-as-can detections) without using simulator labels.
    extent = max((x1 - x0) / k.fx, (y1 - y0) / k.fy) * axial_depth
    if extent > config.max_extent:
        return None, "extent"

    return LocalizedDetection(
        source_index, int(detection.class_id), float(detection.score),
        np.asarray(position, dtype=float).copy(), float(frame.time), distance,
        method, count, mad,
    ), None
