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
from fastapi.middleware.cors import CORSMiddleware
from scipy import signal

# --- POS algorithm from rPPG-Toolbox ---------------------------------
from unsupervised_methods.methods.POS_WANG import POS_WANG  # noqa: E402

# ---------------------------------------------------------------------
app = FastAPI(title="rPPG POS Demo")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

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


def _peak_bpm_from_bvp(bvp: np.ndarray, fs: int) -> float:
    """Calculate heart rate using peak detection."""
    ppg_peaks, _ = signal.find_peaks(bvp)
    if len(ppg_peaks) < 2:
        return 0.0  # Not enough peaks
    hr_peak = 60 / (np.mean(np.diff(ppg_peaks)) / fs)
    return float(hr_peak)


def _respiratory_rate_from_bvp(bvp: np.ndarray, fs: int) -> float:
    """Extract respiratory rate from BVP signal using frequency analysis."""
    # Apply low-pass filter to isolate respiratory component
    b, a = signal.butter(4, 0.5 / (fs / 2), btype='low')  # 0.5 Hz cutoff
    resp_signal = signal.filtfilt(b, a, bvp)
    
    # FFT-based respiratory rate calculation (8-30 breaths per minute)
    f, pxx = signal.periodogram(resp_signal, fs)
    resp_band = (f >= 0.13) & (f <= 0.5)  # 0.13-0.5 Hz → 8-30 breaths/min
    if np.any(resp_band):
        peak_freq = f[resp_band][np.argmax(pxx[resp_band])]
        return float(peak_freq * 60)
    return 0.0


def _calculate_hrv_metrics(bvp: np.ndarray, fs: int) -> dict:
    """Calculate basic HRV metrics from BVP signal."""
    ppg_peaks, _ = signal.find_peaks(bvp, distance=fs//3)  # Minimum distance between peaks
    
    if len(ppg_peaks) < 3:
        return {"rmssd": 0.0, "sdnn": 0.0, "mean_rr": 0.0}
    
    # Calculate RR intervals (in milliseconds)
    rr_intervals = np.diff(ppg_peaks) / fs * 1000
    
    # RMSSD: Root mean square of successive differences
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
    
    # SDNN: Standard deviation of NN intervals
    sdnn = np.std(rr_intervals)
    
    # Mean RR interval
    mean_rr = np.mean(rr_intervals)
    
    return {
        "rmssd": float(rmssd),
        "sdnn": float(sdnn), 
        "mean_rr": float(mean_rr)
    }


def _calculate_snr(bvp: np.ndarray, hr_bpm: float, fs: int) -> float:
    """Calculate Signal-to-Noise Ratio of the BVP signal."""
    # Convert HR to Hz
    hr_freq = hr_bpm / 60
    deviation = 6 / 60  # 6 beats/min converted to Hz
    
    # Calculate FFT
    f, pxx = signal.periodogram(bvp, fs)
    
    # Find signal power around HR frequency and its harmonic
    hr_band = (f >= (hr_freq - deviation)) & (f <= (hr_freq + deviation))
    harmonic_band = (f >= (2 * hr_freq - deviation)) & (f <= (2 * hr_freq + deviation))
    
    signal_power = np.sum(pxx[hr_band]) + np.sum(pxx[harmonic_band])
    
    # Find noise power (rest of the spectrum in physiological range)
    noise_band = (f >= 0.6) & (f <= 3.3) & ~hr_band & ~harmonic_band
    noise_power = np.sum(pxx[noise_band])
    
    if noise_power == 0:
        return 0.0
    
    snr_db = 10 * np.log10(signal_power / noise_power)
    return float(snr_db)


def _quality_snr(bvp: np.ndarray) -> float:
    """Quick & dirty SNR-style quality metric (0-1)."""
    signal_power = np.var(signal.detrend(bvp))
    noise_power = np.var(bvp - signal.detrend(bvp))
    return float(max(0.0, min(1.0, signal_power / (signal_power + noise_power))))


# ---------------------------------------------------------------------
@app.post("/predict")
async def predict_vitals(video: UploadFile):
    """
    Accept an MP4 clip, run POS_WANG, return comprehensive vital signs.
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
    
    # Calculate multiple vital signs
    hr_fft = _bpm_from_bvp(bvp, FPS)
    hr_peak = _peak_bpm_from_bvp(bvp, FPS)
    respiratory_rate = _respiratory_rate_from_bvp(bvp, FPS)
    hrv_metrics = _calculate_hrv_metrics(bvp, FPS)
    snr = _calculate_snr(bvp, hr_fft, FPS)
    quality = _quality_snr(bvp)

    return {
        "heart_rate": {
            "fft_bpm": round(hr_fft, 1),
            "peak_bpm": round(hr_peak, 1),
            "method": "Both FFT and peak detection"
        },
        "respiratory_rate": {
            "breaths_per_minute": round(respiratory_rate, 1)
        },
        "heart_rate_variability": {
            "rmssd_ms": round(hrv_metrics["rmssd"], 2),
            "sdnn_ms": round(hrv_metrics["sdnn"], 2),
            "mean_rr_ms": round(hrv_metrics["mean_rr"], 2)
        },
        "signal_quality": {
            "snr_db": round(snr, 2),
            "quality_score": round(quality, 3)
        },
        "raw_data": {
            "bvp_waveform": bvp.tolist(),
            "sample_rate": FPS,
            "duration_seconds": len(bvp) / FPS
        }
    }
