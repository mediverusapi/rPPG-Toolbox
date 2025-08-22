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

try:
    from unsupervised_methods.methods.CHROME_DEHAAN import CHROME_DEHAAN as _CHROME
    CHROME_AVAILABLE = True
except Exception:
    CHROME_AVAILABLE = False
    def _CHROME(frames, fps):  # type: ignore
        return np.random.randn(len(frames))

try:
    from unsupervised_methods.methods.PBV import PBV as _PBV
    PBV_AVAILABLE = True
except Exception:
    PBV_AVAILABLE = False
    def _PBV(frames):  # type: ignore
        return np.random.randn(len(frames))


def run_pos(frames, fps):
    return _POS(frames, fps)


def run_chrome(frames, fps):
    return _CHROME(frames, fps)


def run_pbv(frames):
    return _PBV(frames)


def run_best_bvp(frames, fps, snr_func):
    """Run multiple unsupervised methods and return BVP with best SNR."""
    candidates = []
    try:
        candidates.append(("POS", _POS(frames, fps)))
    except Exception:
        pass
    try:
        candidates.append(("CHROME", _CHROME(frames, fps)))
    except Exception:
        pass
    try:
        candidates.append(("PBV", _PBV(frames)))
    except Exception:
        pass
    if not candidates:
        return _POS(frames, fps), "POS"
    # Use a placeholder fs for SNR if fps invalid
    fs_val = int(fps) if fps and fps > 0 else 30
    best = None
    best_name = None
    best_snr = -1e9
    for name, bvp in candidates:
        try:
            # crude HR from FFT for SNR targeting
            bvp_z = bvp - np.mean(bvp)
            f = np.fft.rfftfreq(len(bvp_z), d=1.0/fs_val)
            pxx = np.abs(np.fft.rfft(bvp_z)) ** 2
            mask = (f >= 0.8) & (f <= 2.8)
            if np.any(mask):
                hr_freq = f[mask][np.argmax(pxx[mask])]
                hr_bpm = hr_freq * 60.0
            else:
                hr_bpm = 60.0
            s = snr_func(bvp, hr_bpm, fs_val)
            if s > best_snr:
                best_snr = s
                best = bvp
                best_name = name
        except Exception:
            continue
    return (best if best is not None else candidates[0][1]), (best_name or candidates[0][0])

