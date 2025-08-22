from __future__ import annotations

import numpy as np

# NumPy 2.0 shim (safety if not already applied)
try:
    if not hasattr(np, "mat") and hasattr(np, "asmatrix"):
        np.mat = np.asmatrix  # type: ignore[attr-defined]
except Exception:
    pass

try:
    from unsupervised_methods.methods.POS_WANG import POS_WANG as _POS
    POS_AVAILABLE = True
except Exception:
    POS_AVAILABLE = False
    def _POS(frames, fps):  # type: ignore
        return np.random.randn(len(frames))


def run_pos(frames, fps):
    return _POS(frames, fps)


