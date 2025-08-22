from __future__ import annotations

import os
import numpy as np


def _get_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


def _get_bool_env(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on")


# Global constants
FPS = 30
MIN_SEC = 10


# Enhancement parameters (disabled for neural inference alignment)
SCI_STRENGTH = _get_float_env("SCI_STRENGTH", 0.0)
SCI_GAMMA = _get_float_env("SCI_GAMMA", 1.0)

# Face crop toggle (detect once and lock ROI)
FACE_CROP_ENABLED = _get_bool_env("FACE_CROP", True)

# Forehead-only ROI within face crop for higher SNR
FOREHEAD_ONLY = _get_bool_env("FOREHEAD_ONLY", False)

# Optional green-channel emphasis (1.0 = off)
GREEN_EMPHASIS = _get_float_env("GREEN_EMPHASIS", 1.08)

# Optional green-trace blend weight into neural output (0.0–0.5)
GREEN_BLEND = _get_float_env("GREEN_BLEND", 0.12)

# Face box scale (1.0 = detected box, >1 zooms out, <1 zooms in)
FACE_BOX_SCALE = _get_float_env("FACE_BOX_SCALE", 1.50)

# Neural rPPG toggles
USE_NEURAL_RPPG = _get_bool_env("USE_NEURAL_RPPG", True)
RPPG_INFER_CONFIG = os.getenv(
    "RPPG_INFER_CONFIG",
    os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "configs",
        "infer_configs",
        "PURE_UBFC-rPPG_PHYSNET_BASIC.yaml",
    ),
)


# HR constraints and filtering parameters
HR_MIN_BPM = int(_get_float_env("HR_MIN_BPM", 40.0))
HR_MAX_BPM = int(_get_float_env("HR_MAX_BPM", 200.0))
ANCHOR_BW_HZ = _get_float_env("ANCHOR_BW_HZ", 0.30)  # half-width around anchor

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
APP_DIR = os.path.dirname(__file__)
ENH_OUT_DIR = os.path.join(APP_DIR, "model_outputs", "enhanced_previews")
os.makedirs(ENH_OUT_DIR, exist_ok=True)

# Haar cascade path (falls back to original location if needed)
HAAR_CASCADE_PATH = os.path.join(APP_DIR, "dataset", "haarcascade_frontalface_default.xml")
FALLBACK_HAAR_PATH = os.path.join(ROOT_DIR, "dataset", "haarcascade_frontalface_default.xml")


# Torch availability
try:
    import torch  # noqa: F401
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


# POS availability (import lazily in wrapper but expose flag here)
try:
    from unsupervised_methods.methods.POS_WANG import POS_WANG  # noqa: F401
    POS_AVAILABLE = True
except Exception:
    POS_AVAILABLE = False


# NumPy 2.0 shim for np.mat
try:
    if not hasattr(np, "mat") and hasattr(np, "asmatrix"):
        np.mat = np.asmatrix  # type: ignore[attr-defined]
except Exception:
    pass


