"""
build_campus(): the single entry point for the Stage-2 world.
"""

import math

import numpy as np
import mujoco

from src.environment.layout import default_layout
from src.environment.terrain import add_terrain
from src.environment.objects import add_objects
from src.environment.litter import (
    HIDDEN_POSITION,
    add_litter,
    place_litter,
)
from src.environment.distractors import add_distractors, place_distractors
from src.simulation.robot_setup import add_head_camera, load_a1_spec, reset_robot
from src.utils.orientation import quaternion_to_euler


class Campus:
    """
    The compiled world plus its ground-truth description.

    model, data   MuJoCo (robot has a "head_camera")
    layout        CampusLayout (zones, heights, obstacles, bins)
    litter        list of GroundObject, the targets
    distractors   list of GroundObject that are not litter
    lighting      (sun_dir, brightness)
    """

    def __init__(self, spec, model, layout, litter, distractors, lighting, seed):

        self.spec = spec
        self.model = model
        self.data = mujoco.MjData(model)
        self.layout = layout
        self.litter = litter
        self.distractors = distractors
        self.lighting = lighting
        self.seed = seed

        self._mocap_id = {
            item.name: model.body_mocapid[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, item.name)
            ]
            for item in litter
        }

        self.reset()

    # -----------------------------------------------------

    def reset(self):
        """Robot standing at base, all litter back in place."""

        x, y, yaw = self.layout.base_pose

        reset_robot(
            self.model,
            self.data,
            x=x,
            y=y,
            yaw=yaw,
            ground_z=self.layout.ground_height_at(x, y),
        )

        for item in self.litter:
            item.collected = False
            mid = self._mocap_id[item.name]
            self.data.mocap_pos[mid] = item.position
            self.data.mocap_quat[mid] = item.quat

        mujoco.mj_forward(self.model, self.data)

    def collect(self, name):
        """Remove one litter item from the world (Stage 7 pickup)."""

        item = self.litter_item(name)
        item.collected = True
        self.data.mocap_pos[self._mocap_id[name]] = HIDDEN_POSITION

    def litter_item(self, name):
        for item in self.litter:
            if item.name == name:
                return item
        raise KeyError(name)

    @property
    def remaining_litter(self):
        return [item for item in self.litter if not item.collected]

    def robot_pose(self):
        """(x, y, yaw) of the trunk."""

        _, _, yaw = quaternion_to_euler(self.data.qpos[3:7])
        return float(self.data.qpos[0]), float(self.data.qpos[1]), float(yaw)


def random_lighting(rng):
    """Sun from a random azimuth, 35-80 deg above the horizon."""

    azimuth = rng.uniform(0.0, 2.0 * math.pi)
    elevation = math.radians(rng.uniform(35.0, 80.0))

    sun_dir = (
        -math.cos(elevation) * math.cos(azimuth),
        -math.cos(elevation) * math.sin(azimuth),
        -math.sin(elevation),
    )

    return sun_dir, rng.uniform(0.55, 1.0)


def build_campus(
    seed=0,
    layout=None,
    litter_count=8,
    distractor_count=10,
    vary_lighting=True,
    cleaning_tools=False,
):
    """
    Everything random (litter, distractors, lighting) comes
    from `seed`, so a seed always gives the same scene.

    cleaning_tools adds the visual-only Stage 7 nozzle and hopper.
    """

    layout = layout or default_layout()
    rng = np.random.default_rng(seed)

    litter = place_litter(layout, rng, count=litter_count)
    distractors = place_distractors(layout, rng, count=distractor_count, existing=litter)
    lighting = random_lighting(rng) if vary_lighting else ((-0.3, -0.2, -1.0), 0.8)

    spec = load_a1_spec()
    spec.modelname = "cleanquad_campus"

    add_head_camera(spec)
    if cleaning_tools:
        from src.manipulation.tools import add_cleaning_tools
        add_cleaning_tools(spec)
    add_terrain(spec, layout, *lighting)
    add_objects(spec, layout)
    add_litter(spec, litter)
    add_distractors(spec, distractors)

    return Campus(spec, spec.compile(), layout, litter, distractors, lighting, seed)
