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
import importlib.util
from typing import List, Union, Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from scipy import signal

# --- POS algorithm from rPPG-Toolbox ---------------------------------
from unsupervised_methods.methods.POS_WANG import POS_WANG  # noqa: E402

# ---------------------------------------------------------------------
app = FastAPI(title="rPPG POS Demo")

# ---------------------------------------------------------------------
# Basic health-check route for container orchestration (AWS App Runner etc.)
# ---------------------------------------------------------------------


@app.get("/")
async def health():
    """Return 200 OK with simple JSON body so health checks pass."""
    return {"status": "ok"}


@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    """Handle preflight OPTIONS requests for CORS."""
    return {"status": "ok"}


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://circadify.com",
        "http://localhost:3000",  # For local development
        "http://localhost:8000",  # For local testing
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

FPS = 30        # expected camera frame-rate
MIN_SEC = 10    # minimum clip length for a reliable HR estimate

# ---------------------------------------------------------------------
# Optional Blood-Pressure (BP) predictor setup
# ---------------------------------------------------------------------

_BP_PREDICTOR = None  # Will stay None if any import / load step fails

try:
    # Import the model directly
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "ppg_bp"))
    from model import M5_fusion_transformer
    
    BP_MODEL_PATH = os.getenv(
        "BP_MODEL_PATH",
        os.path.join(os.path.dirname(__file__), "ppg_bp", "output", "ppg2bp_custom.pth"),
    )
    
    print(f"DEBUG: Looking for BP model at: {BP_MODEL_PATH}")
    if os.path.exists(BP_MODEL_PATH):
        print("DEBUG: Model file found, initializing predictor...")
        
        # Load model directly
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _BP_MODEL = M5_fusion_transformer(n_input=1, n_output=2)
        _BP_MODEL.load_state_dict(torch.load(BP_MODEL_PATH, map_location=device))
        _BP_MODEL.eval()
        
        # Load normalization parameters
        norm_path = os.path.join(os.path.dirname(BP_MODEL_PATH), "bp_norm_params.npy")
        if os.path.exists(norm_path):
            _BP_NORM = np.load(norm_path, allow_pickle=True).item()
        else:
            _BP_NORM = {'sbp_min': 80, 'sbp_max': 200, 'dbp_min': 40, 'dbp_max': 120}
        
        _BP_PREDICTOR = True  # Flag that BP prediction is available
        print("DEBUG: ✓ BP predictor loaded successfully!")
    else:
        print(f"DEBUG: ✗ Model file not found at {BP_MODEL_PATH}")
        
except Exception as _bp_exc:  # noqa: BLE001
    print(f"DEBUG: ✗ BP predictor failed to load: {_bp_exc}")
    _BP_PREDICTOR = None


def _bp_from_ppg(ppg: np.ndarray, age: Optional[float] = None, bmi: Optional[float] = None) -> Optional[dict]:  # noqa: D401
    """Return systolic/diastolic BP estimate from PPG signal if predictor loaded."""
    if _BP_PREDICTOR is None:
        return None

    age_val = age if age is not None else 65.0  # default values if not provided
    bmi_val = bmi if bmi is not None else 25.0

    try:
        # Ensure we have at least 512 samples
        if len(ppg) < 512:
            ppg = np.pad(ppg, (0, 512 - len(ppg)), mode='constant')
        elif len(ppg) > 512:
            ppg = ppg[:512]
        
        # Normalize PPG signal (min-max normalization)
        ppg_min, ppg_max = ppg.min(), ppg.max()
        if ppg_max > ppg_min:
            ppg_norm = (ppg - ppg_min) / (ppg_max - ppg_min)
        else:
            ppg_norm = ppg * 0
        
        with torch.no_grad():
            # Convert to tensors
            ppg_tensor = torch.tensor(ppg_norm).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 512)
            
            # Age and BMI need to be broadcast to temporal dimension (128 after pooling)
            age_tensor = torch.tensor([[age_val]]).float().unsqueeze(2).repeat(1, 1, 128)  # (1, 1, 128)
            bmi_tensor = torch.tensor([[bmi_val]]).float().unsqueeze(2).repeat(1, 1, 128)  # (1, 1, 128)
            
            # Predict
            output = _BP_MODEL(ppg_tensor, age_tensor, bmi_tensor).squeeze()
            
            # Denormalize predictions
            sbp_norm, dbp_norm = output[0].item(), output[1].item()
            
            systolic = sbp_norm * (_BP_NORM['sbp_max'] - _BP_NORM['sbp_min']) + _BP_NORM['sbp_min']
            diastolic = dbp_norm * (_BP_NORM['dbp_max'] - _BP_NORM['dbp_min']) + _BP_NORM['dbp_min']
            
            return {
                "systolic_mmHg": round(max(60, min(250, systolic)), 1),  # Clamp to reasonable range
                "diastolic_mmHg": round(max(30, min(150, diastolic)), 1),
            }
    except Exception:  # noqa: BLE001
        return None


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
async def predict_vitals(
    video: UploadFile,
    age: Optional[float] = Form(None),
    bmi: Optional[float] = Form(None),
):
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

    # Optional BP estimation
    bp_res = _bp_from_ppg(bvp, age, bmi)

    response = {
        "heart_rate": {
            "fft_bpm": round(hr_fft, 1),
            "peak_bpm": round(hr_peak, 1),
            "method": "Both FFT and peak detection",
        },
        "respiratory_rate": {
            "breaths_per_minute": round(respiratory_rate, 1),
        },
        "heart_rate_variability": {
            "rmssd_ms": round(hrv_metrics["rmssd"], 2),
            "sdnn_ms": round(hrv_metrics["sdnn"], 2),
            "mean_rr_ms": round(hrv_metrics["mean_rr"], 2),
        },
        "signal_quality": {
            "snr_db": round(snr, 2),
            "quality_score": round(quality, 3),
        },
    }

    if bp_res is not None:
        response["blood_pressure"] = bp_res
    else:
        # Add debug info when BP prediction fails
        response["blood_pressure_status"] = "BP predictor not available" if _BP_PREDICTOR is None else "BP prediction failed"

    return response
