"""Viewer-only display of carried litter: on the hopper, then into the bin.

Collected items normally vanish (Campus.collect parks them below the ground).
For the viewer they are instead stacked on the back hopper and thrown into the
bin during a dump. Their geoms are moved to the tool render group, which
mujoco.Renderer hides, so the head camera never sees carried litter and
perception is unchanged. Positions are visual only: the simulator's collection
state and every evaluation use Campus.litter, not these mocap positions.
"""

import math

import numpy as np
import mujoco

from src.environment.litter import HIDDEN_POSITION
from src.manipulation.tools import HOPPER_POS, HOPPER_SIZE, TOOL_GROUP

THROW_START = 0.6       # seconds into the dump posture
THROW_SECONDS = 1.2


class CarryDisplay:
    def __init__(self, campus):
        self.campus = campus
        model = campus.model
        self.trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.mocap = {item.name: model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, item.name)]
                      for item in campus.litter}
        self.geom = {item.name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, item.geom_name)
                     for item in campus.litter}
        self.carried = []
        self.throw = None       # (start time, bin top, names)

    def pick(self, names):
        for name in names:
            if name not in self.carried:
                self.carried.append(name)
                self.campus.model.geom_group[self.geom[name]] = TOOL_GROUP

    def slot(self, index):
        """World position of the index-th item stacked on the hopper."""
        data = self.campus.data
        matrix = data.xmat[self.trunk].reshape(3, 3)
        column, layer = index % 3, index // 3
        local = np.array([HOPPER_POS[0] + (column - 1) * 0.045, HOPPER_POS[1],
                          HOPPER_POS[2] + HOPPER_SIZE[2] + 0.03 + 0.05 * layer])
        return data.xpos[self.trunk] + matrix @ local

    def start_dump(self, bin_landmark):
        shape = bin_landmark["shape"]
        top = np.array([shape["x"], shape["y"], bin_landmark["height"] + 0.05])
        self.throw = (float(self.campus.data.time), top, list(self.carried))
        self.carried = []

    def update(self, _campus=None):
        data = self.campus.data
        for index, name in enumerate(self.carried):
            data.mocap_pos[self.mocap[name]] = self.slot(index)
        if self.throw is None:
            return
        start, top, names = self.throw
        progress = (float(data.time) - start - THROW_START) / THROW_SECONDS
        for index, name in enumerate(names):
            if progress <= 0:
                data.mocap_pos[self.mocap[name]] = self.slot(index)
            elif progress < 1:
                # A ballistic-looking arc from the hopper over the head into the bin.
                origin = self.slot(index)
                point = (1 - progress) * origin + progress * top
                point[2] += 0.45 * math.sin(math.pi * progress)
                data.mocap_pos[self.mocap[name]] = point
            else:
                data.mocap_pos[self.mocap[name]] = HIDDEN_POSITION
        if progress >= 1:
            self.throw = None
