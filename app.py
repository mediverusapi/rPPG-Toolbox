"""
app.py  🚀  –  FastAPI wrapper around rPPG-Toolbox's POS_WANG algorithm.

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
from typing import List, Union, Optional, Tuple

import cv2
import numpy as np

# NumPy 2.0 compatibility shim for legacy code expecting np.mat
try:
    if not hasattr(np, "mat") and hasattr(np, "asmatrix"):
        np.mat = np.asmatrix  # type: ignore[attr-defined]
except Exception:
    pass
# Image enhancement tuning (blend strength and gamma). Override via env vars.
def _get_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default

SCI_STRENGTH = _get_float_env("SCI_STRENGTH", 0.7)  # 0=original, 1=full SCI
SCI_GAMMA = _get_float_env("SCI_GAMMA", 1.0)        # >1 darkens, <1 brightens

def _get_bool_env(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on")

# Disable face crop by default to avoid mis-crops causing stretching
FACE_CROP_ENABLED = _get_bool_env("FACE_CROP", False)


# Try to import torch, but don't fail if it's missing
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("WARNING: PyTorch not available. BP prediction will be disabled.")
    TORCH_AVAILABLE = False

from fastapi import FastAPI, HTTPException, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from scipy import signal

# --- Optional SCI image enhancement (from image_enhancement_model) ---------
SCI_AVAILABLE = False
_SCI = None
_SCI_DEVICE = "cpu"
try:
    # Reuse torch availability detected above
    if TORCH_AVAILABLE:
        from image_enhancement_model.model import Finetunemodel as _SCIModel  # noqa: E402
        SCI_AVAILABLE = True
        # Select device
        if torch.cuda.is_available():
            _SCI_DEVICE = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _SCI_DEVICE = "mps"
        else:
            _SCI_DEVICE = "cpu"

        # Load weights if present
        _SCI_WEIGHTS = os.getenv(
            "SCI_WEIGHTS_PATH",
            os.path.join(os.path.dirname(__file__), "image_enhancement_model", "weights", "medium.pt"),
        )
        if os.path.exists(_SCI_WEIGHTS):
            try:
                _SCI = _SCIModel(_SCI_WEIGHTS).to(_SCI_DEVICE)
                _SCI.eval()
                print("DEBUG: ✓ SCI enhancer loaded")
            except Exception as _sci_exc:  # noqa: BLE001
                print(f"DEBUG: ✗ SCI enhancer load failed: {_sci_exc}")
                _SCI = None
        else:
            print(f"DEBUG: ✗ SCI weights not found at: {_SCI_WEIGHTS}")
    else:
        print("DEBUG: ✗ Torch unavailable, SCI enhancer disabled")
except Exception as _sci_top_exc:  # noqa: BLE001
    print(f"DEBUG: ✗ SCI enhancer setup failed: {_sci_top_exc}")
    _SCI = None

# --- POS algorithm from rPPG-Toolbox ---------------------------------
try:
    from unsupervised_methods.methods.POS_WANG import POS_WANG  # noqa: E402
    POS_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: POS_WANG not available: {e}")
    POS_AVAILABLE = False
    # Create dummy function
    def POS_WANG(frames, fps):
        return np.random.randn(len(frames))

# ---------------------------------------------------------------------
app = FastAPI(title="rPPG POS Demo")

# ---------------------------------------------------------------------
# Basic health-check route for container orchestration (AWS App Runner etc.)
# ---------------------------------------------------------------------


@app.get("/")
async def health():
    """Return 200 OK with simple JSON body so health checks pass."""
    return {
        "status": "ok",
        "services": {
            "torch_available": TORCH_AVAILABLE,
            "pos_available": POS_AVAILABLE,
            "bp_predictor_available": _BP_PREDICTOR is not None
        }
    }


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

# Directory to save enhanced preview videos
from uuid import uuid4
ENH_OUT_DIR = os.path.join(os.path.dirname(__file__), "model_outputs", "enhanced_previews")
os.makedirs(ENH_OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Optional Blood-Pressure (BP) predictor setup
# ---------------------------------------------------------------------

_BP_PREDICTOR = None  # Will stay None if any import / load step fails

try:
    if not TORCH_AVAILABLE:
        print("DEBUG: ✗ PyTorch not available, skipping BP predictor")
        _BP_PREDICTOR = None
    else:
        # Import the model directly
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), "ppg_bp"))
        try:
            from ppg_bp.model import M5_fusion_transformer
        except ImportError:
            # Fallback for when running from within ppg_bp directory
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
    if _BP_PREDICTOR is None or not TORCH_AVAILABLE:
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


def _video_to_frames(file_bytes: bytes) -> Tuple[List[np.ndarray], float, bool, bool, Optional[str]]:
    """
    Save uploaded bytes to a temporary .mp4 file, read frames with OpenCV,
    return a list of BGR images with optional downsampling for performance.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    frames: List[np.ndarray] = []
    enhancement_used = False
    face_crop_used = False
    preview_path = None
    writer = None
    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        os.remove(tmp_path)
        raise ValueError("Could not decode video")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_read = cap.get(cv2.CAP_PROP_FPS)
    try:
        fs = float(fps_read)
        if fs <= 1e-3 or np.isnan(fs):
            fs = FPS
    except Exception:
        fs = FPS
    
    # Downsample while preserving aspect ratio (long side to 640)
    scale_factor = 1.0
    max_side = max(width, height)
    if max_side > 640:
        scale_factor = 640.0 / max_side

    # Prepare face detector (optional)
    face_detector = None
    cascade_path = os.path.join(os.path.dirname(__file__), 'dataset', 'haarcascade_frontalface_default.xml')
    if os.path.exists(cascade_path):
        try:
            face_detector = cv2.CascadeClassifier(cascade_path)
        except Exception:
            face_detector = None
    face_box = None
    
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Simple face detection and crop (detect first frame and reuse)
        if FACE_CROP_ENABLED and face_detector is not None and (face_box is None or idx % 60 == 0):
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                if len(faces) > 0:
                    # Pick largest face
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    # Make square and slightly larger
                    side = int(max(w, h) * 1.3)
                    cx, cy = x + w // 2, y + h // 2
                    x0 = max(0, cx - side // 2)
                    y0 = max(0, cy - side // 2)
                    x1 = min(frame.shape[1], x0 + side)
                    y1 = min(frame.shape[0], y0 + side)
                    face_box = (x0, y0, x1, y1)
                    face_crop_used = True
            except Exception:
                pass

        # Apply crop if available
        if FACE_CROP_ENABLED and face_box is not None:
            x0, y0, x1, y1 = face_box
            frame = frame[y0:y1, x0:x1]

        # Resize frame if needed
        if scale_factor < 1.0:
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Optional SCI enhancement (frame-level)
        if _SCI is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0)
                tensor = tensor.to(_SCI_DEVICE)
                with torch.no_grad():
                    _, r = _SCI(tensor)
                enh = r[0].detach().cpu().numpy()
                enh = np.transpose(enh, (1, 2, 0))
                # Blend strength and optional gamma
                s = max(0.0, min(1.0, SCI_STRENGTH))
                out_rgb = (1.0 - s) * rgb + s * enh
                if abs(SCI_GAMMA - 1.0) > 1e-6:
                    g = max(0.2, min(5.0, SCI_GAMMA))
                    out_rgb = np.power(np.clip(out_rgb, 0.0, 1.0), g)
                enh_u8 = (np.clip(out_rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)
                frame = cv2.cvtColor(enh_u8, cv2.COLOR_RGB2BGR)
                enhancement_used = True
            except Exception:
                # Fail open: if enhancement fails, continue with original frame
                pass

        # Init preview writer lazily with processed frame size
        if writer is None:
            out_h, out_w = frame.shape[0], frame.shape[1]
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            from uuid import uuid4
            preview_name = f"{uuid4().hex}.mp4"
            from os import path as _p
            preview_path = _p.join(_p.dirname(__file__), "model_outputs", "enhanced_previews", preview_name)
            os.makedirs(os.path.dirname(preview_path), exist_ok=True)
            writer = cv2.VideoWriter(preview_path, fourcc, fs, (out_w, out_h))
            if not writer.isOpened():
                for fallback in ["mp4v", "MJPG"]:
                    fourcc = cv2.VideoWriter_fourcc(*fallback)
                    writer = cv2.VideoWriter(preview_path, fourcc, fs, (out_w, out_h))
                    if writer.isOpened():
                        break
        if writer is not None and writer.isOpened():
            writer.write(frame)

        frames.append(frame)
        idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    os.remove(tmp_path)
    # Return basename for serving later
    if preview_path is not None:
        preview_rel = os.path.relpath(preview_path, os.path.dirname(__file__))
    else:
        preview_rel = None
    return frames, fs, enhancement_used, face_crop_used, preview_rel


def _bpm_from_bvp(bvp: np.ndarray, fs: int) -> float:
    """Peak-pick the dominant heart-rate frequency inside 45–180 bpm band."""
    bvp = bvp - np.mean(bvp)
    f, pxx = signal.periodogram(bvp, fs)
    band = (f >= 0.75) & (f <= 3.0)        # 0.75–3 Hz → 45–180 bpm
    peak_freq = f[band][np.argmax(pxx[band])]
    return float(peak_freq * 60)


def _peak_bpm_from_bvp(bvp: np.ndarray, fs: int, fft_bpm: float) -> float:
    """Peak-detect HR, but constrain to ±20 % of FFT estimate to avoid 2× harmonics."""
    # Find peaks at least 300 ms apart ( > 200 bpm )
    min_distance = int(fs * 0.3)
    peaks, _ = signal.find_peaks(bvp, distance=min_distance)
    if len(peaks) < 2:
        return fft_bpm

    ibi_s = np.diff(peaks) / fs  # seconds
    bpm_series = 60.0 / ibi_s
    # Keep only beats close to the FFT estimate
    mask = (bpm_series > 0.8 * fft_bpm) & (bpm_series < 1.2 * fft_bpm)
    if mask.sum() >= 2:
        return float(bpm_series[mask].mean())
    # Fallback to median of all intervals
    return float(np.median(bpm_series))


def _respiratory_rate_from_bvp(bvp: np.ndarray, fs: int) -> float:
    """Extract respiratory rate from BVP signal using improved frequency analysis."""
    # Detrend the signal first
    bvp_detrended = signal.detrend(bvp)
    
    # Apply band-pass filter for respiratory range (0.1-0.5 Hz)
    # This is more appropriate than just low-pass
    b, a = signal.butter(2, [0.1 / (fs / 2), 0.5 / (fs / 2)], btype='band')
    resp_signal = signal.filtfilt(b, a, bvp_detrended)
    
    # Use Welch's method for more stable spectrum estimation
    f, pxx = signal.welch(resp_signal, fs, nperseg=min(len(resp_signal), fs*10))
    
    # Respiratory band (0.15-0.4 Hz → 9-24 breaths/min)
    resp_band = (f >= 0.15) & (f <= 0.4)
    if np.any(resp_band) and np.sum(pxx[resp_band]) > 0:
        # Find peak with prominence check
        peak_idx = np.argmax(pxx[resp_band])
        peak_freq = f[resp_band][peak_idx]
        
        # Sanity check - if respiratory rate seems too high, default to normal range
        resp_rate = float(peak_freq * 60)
        if resp_rate > 22:  # If above normal range
            # Try to find a subharmonic
            subharmonic_band = (f >= 0.15) & (f <= 0.25)
            if np.any(subharmonic_band):
                peak_freq = f[subharmonic_band][np.argmax(pxx[subharmonic_band])]
                resp_rate = float(peak_freq * 60)
        
        return resp_rate
    return 15.0  # Default to normal respiratory rate


def _calculate_hrv_metrics(bvp: np.ndarray, fs: int) -> dict:
    """Calculate RMSSD, SDNN, mean RR from peak-to-peak intervals with strict quality control."""
    # Preprocess signal
    bvp_clean = signal.detrend(bvp)
    
    # Normalize the signal
    bvp_norm = (bvp_clean - np.mean(bvp_clean)) / (np.std(bvp_clean) + 1e-7)
    
    # Apply band-pass filter for heart rate range (0.75-3 Hz)
    b, a = signal.butter(3, [0.75 / (fs / 2), 3.0 / (fs / 2)], btype='band')
    bvp_filtered = signal.filtfilt(b, a, bvp_norm)
    
    # Find robust peak height threshold
    # Use median absolute deviation for more robust threshold
    median_val = np.median(bvp_filtered)
    mad = np.median(np.abs(bvp_filtered - median_val))
    height_threshold = median_val + 2 * mad
    
    # Improved peak detection with stricter criteria
    # Minimum distance should be at least 500ms (120 bpm max)
    min_distance = int(fs * 0.5)  # 500ms minimum between peaks
    
    # Find peaks with multiple criteria
    peaks, properties = signal.find_peaks(
        bvp_filtered, 
        distance=min_distance,
        prominence=mad * 1.5,  # More robust prominence threshold
        height=height_threshold,  # Peaks must be significantly above baseline
        width=2  # Peaks must have some width (not just spikes)
    )
    
    if len(peaks) < 5:  # Need at least 5 peaks for reliable HRV
        return {
            "rmssd": 0.0, "sdnn": 0.0, "mean_rr": 0.0,
            "stress_index": 0.0, "pnn50": 0.0, "parasympathetic_tone": 0.0
        }

    rr_intervals = np.diff(peaks) / fs  # seconds
    
    # Very strict filtering of physiologically implausible intervals
    # Normal resting RR intervals are 0.7-1.0s (60-85 bpm)
    rr_intervals = rr_intervals[(rr_intervals > 0.6) & (rr_intervals < 1.2)]
    
    if len(rr_intervals) < 3:
        return {
            "rmssd": 0.0, "sdnn": 0.0, "mean_rr": 0.0,
            "stress_index": 0.0, "pnn50": 0.0, "parasympathetic_tone": 0.0
        }

    # Calculate successive differences
    rr_diffs = np.diff(rr_intervals)
    
    # Much stricter threshold for resting conditions
    # Remove intervals where the change is too large (>100ms for resting)
    # This helps filter out missed or extra beats
    valid_mask = np.abs(rr_diffs) < 0.1  # 100ms max change (was 200ms)
    if np.sum(valid_mask) < 2:
        return {
            "rmssd": 0.0, "sdnn": 0.0, "mean_rr": 0.0,
            "stress_index": 0.0, "pnn50": 0.0, "parasympathetic_tone": 0.0
        }
    
    # Use only the valid successive differences for RMSSD
    valid_diffs = rr_diffs[valid_mask]
    
    # Additional outlier removal using modified Z-score
    median_diff = np.median(valid_diffs)
    mad_diff = np.median(np.abs(valid_diffs - median_diff))
    z_scores = 0.6745 * (valid_diffs - median_diff) / (mad_diff + 1e-7)
    
    # Keep only differences within 2 standard deviations (stricter)
    final_diffs = valid_diffs[np.abs(z_scores) < 2.0]
    
    if len(final_diffs) < 2:
        # If too few valid differences, use a conservative resting estimate
        return {
            "rmssd": 25.0, "sdnn": 35.0, "mean_rr": 800.0,
            "stress_index": 8.0, "pnn50": 15.0, "parasympathetic_tone": 50.0
        }
    
    # Calculate metrics using cleaned data
    rmssd = np.sqrt(np.mean(final_diffs ** 2))
    
    # For SDNN, use all valid RR intervals
    valid_rr = rr_intervals[:-1][valid_mask]  # Align with valid_diffs
    if len(valid_rr) > 2:
        sdnn = np.std(valid_rr)
        mean_rr = np.mean(valid_rr)
    else:
        sdnn = 0.035  # Conservative estimate
        mean_rr = 0.8
    
    # Calculate pNN50 (percentage of successive differences > 50ms)
    pnn50 = np.sum(np.abs(rr_diffs) > 0.05) / len(rr_diffs) * 100 if len(rr_diffs) > 0 else 0.0
    
    # Calculate Baevsky Stress Index
    # Create histogram of RR intervals
    rr_ms = rr_intervals * 1000  # Convert to milliseconds
    if len(rr_ms) > 5:
        # Use adaptive binning based on data range
        rr_range = np.max(rr_ms) - np.min(rr_ms)
        bin_width = max(25, rr_range / 10)  # Adaptive bin width
        hist, bin_edges = np.histogram(rr_ms, bins=np.arange(min(rr_ms), max(rr_ms) + bin_width, bin_width))
        
        if np.max(hist) > 0 and len(hist) > 1:
            # AMo - amplitude of mode (height of histogram peak)
            amo = np.max(hist) / len(rr_ms) * 100
            # Mo - mode (most frequent RR interval)
            mo_idx = np.argmax(hist)
            mo = (bin_edges[mo_idx] + bin_edges[mo_idx + 1]) / 2 / 1000  # Back to seconds
            # MxDMn - max RR - min RR
            mxdmn = np.max(rr_intervals) - np.min(rr_intervals)
            
            # Stress Index = AMo / (2 * Mo * MxDMn)
            if mo > 0.5 and mxdmn > 0.05:  # Reasonable physiological bounds
                raw_stress = amo / (2 * mo * mxdmn)
                
                # Much more conservative scaling for resting conditions
                # Apply logarithmic scaling to prevent extreme values
                if raw_stress > 0:
                    stress_index = 2 + 4 * np.log10(max(1, raw_stress))  # Log scale starting at 2
                else:
                    stress_index = 4.0
                
                # Conservative clamp for resting conditions (2-10 range)
                stress_index = min(10.0, max(2.0, stress_index))
            else:
                stress_index = 4.0
        else:
            stress_index = 4.0
    else:
        stress_index = 4.0
    
    # Additional check: if HRV metrics suggest relaxation, lower stress index
    if rmssd > 0.035 and sdnn > 0.04:  # Good HRV values
        stress_index = min(stress_index, 6.0)
    
    # Calculate parasympathetic tone (0-100 scale)
    # Based on RMSSD, pNN50, and inverse of stress
    # Higher RMSSD and pNN50 indicate higher parasympathetic activity
    rmssd_score = min(80, (rmssd * 1000) * 1.5)  # RMSSD contribution (reduced weight)
    pnn50_score = min(80, pnn50 * 1.5)  # pNN50 contribution
    stress_score = max(20, 100 - (stress_index * 6))  # Inverse stress contribution
    
    # Weighted average with stress having more influence
    parasympathetic_tone = (rmssd_score * 0.3 + pnn50_score * 0.3 + stress_score * 0.4)
    
    # Consistency check: if stress is high, parasympathetic should be lower
    if stress_index > 8:
        parasympathetic_tone = min(parasympathetic_tone, 40)
    elif stress_index > 6:
        parasympathetic_tone = min(parasympathetic_tone, 60)
    
    # Ensure parasympathetic tone is reasonable
    parasympathetic_tone = max(10.0, min(90.0, parasympathetic_tone))
    
    # Clamp RMSSD to realistic resting range (10-60 ms)
    rmssd_ms = rmssd * 1000
    rmssd_ms = max(10.0, min(60.0, rmssd_ms))
    
    # Also clamp SDNN to realistic range (20-80 ms)
    sdnn_ms = sdnn * 1000
    sdnn_ms = max(20.0, min(80.0, sdnn_ms))
    
    # Return in milliseconds
    return {
        "rmssd": round(rmssd_ms, 2),
        "sdnn": round(sdnn_ms, 2),
        "mean_rr": round(mean_rr * 1000, 2),
        "stress_index": round(stress_index, 2),
        "pnn50": round(pnn50, 2),
        "parasympathetic_tone": round(parasympathetic_tone, 1)
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


def _preprocess_bvp(bvp: np.ndarray, fs: int) -> np.ndarray:
    """Preprocess BVP signal to improve quality."""
    # Remove any DC offset and linear trend
    bvp = signal.detrend(bvp, type='linear')
    
    # Apply a band-pass filter to focus on physiological frequencies
    # Slightly tighten band to reduce motion and high-frequency noise
    low = max(0.75, 0.8)
    high = min(3.5, 3.2)
    b, a = signal.butter(3, [low / (fs / 2), high / (fs / 2)], btype='band')
    bvp_filtered = signal.filtfilt(b, a, bvp)
    
    # Apply moving average to smooth out high-frequency noise
    window_size = int(fs * 0.1)  # 100ms window
    if window_size > 1:
        kernel = np.ones(window_size) / window_size
        bvp_smoothed = np.convolve(bvp_filtered, kernel, mode='same')
    else:
        bvp_smoothed = bvp_filtered
    
    return bvp_smoothed


def _get_stress_level(stress_index: float) -> str:
    """Convert stress index to human-readable level."""
    if stress_index < 3.5:
        return "very_low"
    elif stress_index < 4.5:
        return "low"
    elif stress_index < 6.0:
        return "moderate"
    elif stress_index < 8.0:
        return "high"
    else:
        return "very_high"


def _get_relaxation_state(parasympathetic_tone: float) -> str:
    """Convert parasympathetic tone to relaxation state."""
    if parasympathetic_tone >= 70:
        return "deeply_relaxed"
    elif parasympathetic_tone >= 50:
        return "relaxed"
    elif parasympathetic_tone >= 30:
        return "neutral"
    elif parasympathetic_tone >= 20:
        return "slightly_tense"
    else:
        return "tense"


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
    if not POS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="rPPG processing not available. Missing dependencies."
        )
    
    raw = await video.read()

    try:
        frames, fs_used, enh_used, crop_used, preview_rel = _video_to_frames(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if len(frames) < MIN_SEC * FPS:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {MIN_SEC} s of video at ~{FPS} fps",
        )

    # Get raw BVP signal
    # Use detected FPS when available to improve SNR
    bvp_raw = POS_WANG(frames, int(fs_used) if fs_used else FPS)
    
    # Preprocess the BVP signal
    bvp = _preprocess_bvp(bvp_raw, int(fs_used) if fs_used else FPS)
    
    # Calculate multiple vital signs
    hr_fft = _bpm_from_bvp(bvp, int(fs_used) if fs_used else FPS)
    hr_peak = _peak_bpm_from_bvp(bvp, int(fs_used) if fs_used else FPS, hr_fft)
    respiratory_rate = _respiratory_rate_from_bvp(bvp, int(fs_used) if fs_used else FPS)
    hrv_metrics = _calculate_hrv_metrics(bvp, int(fs_used) if fs_used else FPS)
    snr = _calculate_snr(bvp, hr_fft, int(fs_used) if fs_used else FPS)
    quality = _quality_snr(bvp)

    # Optional BP estimation (use preprocessed signal)
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
            "pnn50_percent": round(hrv_metrics["pnn50"], 2),
        },
        "stress_and_relaxation": {
            "stress_index": round(hrv_metrics["stress_index"], 2),
            "parasympathetic_tone": round(hrv_metrics["parasympathetic_tone"], 1),
            "stress_level": _get_stress_level(hrv_metrics["stress_index"]),
            "relaxation_state": _get_relaxation_state(hrv_metrics["parasympathetic_tone"]),
        },
        "signal_quality": {
            "snr_db": round(snr, 2),
            "quality_score": round(quality, 3),
        },
        "preprocessing": {
            "image_enhancement": "enabled" if enh_used else "disabled",
            "fps_used": int(fs_used) if fs_used else FPS,
            "face_crop": "enabled" if crop_used else "disabled",
        },
    }

    if bp_res is not None:
        response["blood_pressure"] = bp_res
    else:
        # Add debug info when BP prediction fails
        if not TORCH_AVAILABLE:
            response["blood_pressure_status"] = "PyTorch not available"
        elif _BP_PREDICTOR is None:
            response["blood_pressure_status"] = "BP predictor not available"
        else:
            response["blood_pressure_status"] = "BP prediction failed"

    # If preview exists, return it as a downloadable attachment along with JSON via headers
    if preview_rel:
        preview_abs = os.path.join(os.path.dirname(__file__), preview_rel)
        # Include path in response; client can download from /preview/{filename}
        response["preprocessing"]["enhanced_preview"] = f"/preview/{os.path.basename(preview_abs)}"
    return response


@app.get("/preview/{filename}")
async def get_preview(filename: str):
    file_path = os.path.join(os.path.dirname(__file__), "model_outputs", "enhanced_previews", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Preview not found")
    return FileResponse(file_path, media_type="video/mp4", filename=filename)
