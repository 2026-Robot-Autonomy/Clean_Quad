from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


A1_MODEL_DIR = (
    PROJECT_ROOT
    / "assets"
    / "mujoco_menagerie"
    / "unitree_a1"
)


# Robot only (no floor). Used by MjSpec-built worlds.
A1_XML_PATH = (
    A1_MODEL_DIR
    / "a1.xml"
)


# Menagerie flat-ground scene used for Stage-1 certification.
A1_SCENE_PATH = (
    A1_MODEL_DIR
    / "scene.xml"
)
