"""
Robot head camera: RGB, depth and segmentation images, plus
the camera model (intrinsics + pose) that Stage 4 needs to turn
pixels into world positions.

Rendering needs an OpenGL context. On this laptop the default
GLFW backend works (inside a desktop session); EGL does not.
"""

from dataclasses import dataclass
import math

import numpy as np
import mujoco

from src.simulation.robot_setup import HEAD_CAMERA


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def matrix(self):
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ])


@dataclass
class Frame:
    rgb: np.ndarray             # (H, W, 3) uint8
    depth: np.ndarray           # (H, W) float32, metres along the view axis
    geom_ids: np.ndarray        # (H, W) int32 geom id per pixel, -1 = none
    cam_pos: np.ndarray         # (3,) camera position, world frame
    cam_mat: np.ndarray         # (3, 3) columns = camera x (right), y (up), z (backward)
    intrinsics: CameraIntrinsics
    time: float

    def project(self, point):
        """
        World point -> (u, v, depth). u, v in pixels (u right,
        v down); depth > 0 means in front of the camera.
        """

        p = self.cam_mat.T @ (np.asarray(point) - self.cam_pos)
        depth = -p[2]
        k = self.intrinsics

        return (
            k.cx + k.fx * p[0] / depth,
            k.cy - k.fy * p[1] / depth,
            depth,
        )


class RobotCamera:
    """
    with RobotCamera(model, data) as camera:
        frame = camera.capture()
    """

    def __init__(self, model, data, width=640, height=480, camera=HEAD_CAMERA):

        self.model = model
        self.data = data
        self.camera = camera
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)

        if self.cam_id < 0:
            raise ValueError(f"camera '{camera}' not in model")

        self.renderer = mujoco.Renderer(model, height, width)

        fy = 0.5 * height / math.tan(math.radians(model.cam_fovy[self.cam_id]) / 2.0)
        self.intrinsics = CameraIntrinsics(width, height, fy, fy, 0.5 * width, 0.5 * height)

    def capture(self, depth=True, segmentation=True):

        r = self.renderer
        r.update_scene(self.data, camera=self.camera)
        rgb = r.render().copy()

        depth_image = None
        if depth:
            r.enable_depth_rendering()
            r.update_scene(self.data, camera=self.camera)
            depth_image = r.render().astype(np.float32)
            r.disable_depth_rendering()

        geom_ids = None
        if segmentation:
            r.enable_segmentation_rendering()
            r.update_scene(self.data, camera=self.camera)
            seg = r.render()
            r.disable_segmentation_rendering()
            is_geom = seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM
            geom_ids = np.where(is_geom, seg[..., 0], -1).astype(np.int32)

        return Frame(
            rgb=rgb,
            depth=depth_image,
            geom_ids=geom_ids,
            cam_pos=self.data.cam_xpos[self.cam_id].copy(),
            cam_mat=self.data.cam_xmat[self.cam_id].reshape(3, 3).copy(),
            intrinsics=self.intrinsics,
            time=self.data.time,
        )

    def close(self):
        self.renderer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
