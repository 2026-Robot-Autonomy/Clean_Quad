import numpy as np

# should match the actuator order from XML
STAND_POSITION = np.array([
    0.0, 0.9, -1.8,   # Front Right
    0.0, 0.9, -1.8,   # Front Left
    0.0, 0.9, -1.8,   # Rear Right
    0.0, 0.9, -1.8,   # Rear Left
])


def get_stand_command():
    return STAND_POSITION.copy()