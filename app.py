"""
app.py  🚀  –  FastAPI wrapper around rPPG-Toolbox’s POS_WANG algorithm.

• POST /predict  (multipart/form-data: video=<mp4>)
      ↳ returns {"bpm": float, "quality": float}

Test locally:
    uvicorn app:app --reload --port 8000
    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import os
import tempfile
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from scipy import signal

# --- POS algorithm from rPPG-Toolbox ---------------------------------
from unsupervised_methods.methods.POS_WANG import POS_WANG  # noqa: E402

# ---------------------------------------------------------------------
app = FastAPI(title="rPPG POS Demo")

FPS = 30        # expected camera frame-rate
MIN_SEC = 10    # minimum clip length for a reliable HR estimate


# ---------------------------------------------------------------------
def _video_to_frames(file_bytes: bytes) -> List[np.ndarray]:
    """
    Save uploaded bytes to a temporary .mp4 file, read frames with OpenCV,
    return a list of BGR images.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    frames: List[np.ndarray] = []
    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        os.remove(tmp_path)
        raise ValueError("Could not decode video")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)

    cap.release()
    os.remove(tmp_path)
    return frames


def _bpm_from_bvp(bvp: np.ndarray, fs: int) -> float:
    """Peak-pick the dominant heart-rate frequency inside 45–180 bpm band."""
    bvp = bvp - np.mean(bvp)
    f, pxx = signal.periodogram(bvp, fs)
    band = (f >= 0.75) & (f <= 3.0)        # 0.75–3 Hz → 45–180 bpm
    peak_freq = f[band][np.argmax(pxx[band])]
    return float(peak_freq * 60)


def _quality_snr(bvp: np.ndarray) -> float:
    """Quick & dirty SNR-style quality metric (0-1)."""
    signal_power = np.var(signal.detrend(bvp))
    noise_power = np.var(bvp - signal.detrend(bvp))
    return float(max(0.0, min(1.0, signal_power / (signal_power + noise_power))))


# ---------------------------------------------------------------------
@app.post("/predict")
async def predict_vitals(video: UploadFile):
    """
    Accept an MP4 clip, run POS_WANG, return BPM + a simple quality score.
    """
    raw = await video.read()

    try:
        frames = _video_to_frames(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if len(frames) < MIN_SEC * FPS:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {MIN_SEC} s of video at ~{FPS} fps",
        )

    bvp = POS_WANG(frames, FPS)
    bpm = _bpm_from_bvp(bvp, FPS)
    quality = _quality_snr(bvp)

    return {"bpm": round(bpm, 1), "quality": round(quality, 2)}
