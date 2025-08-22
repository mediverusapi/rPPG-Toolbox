from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, Form

from ..config import FPS, MIN_SEC
from ..services.video_io import video_to_frames
from ..services.rppg_wrapper import run_pos
from ..services.signal_proc import (
    preprocess_bvp,
    bpm_fft,
    bpm_peaks,
    respiratory_rate,
    hrv_metrics,
    snr_db,
    quality_score,
    emphasize_green_mean_rgb,
)
from ..services.bp_predictor import bp_from_ppg


router = APIRouter()


@router.post("/predict")
async def predict_vitals(
    video: UploadFile,
    age: Optional[float] = Form(None),
    bmi: Optional[float] = Form(None),
):
    raw = await video.read()
    try:
        frames, fs_used, enh_used, crop_used, preview_name = video_to_frames(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if len(frames) < MIN_SEC * FPS:
        raise HTTPException(status_code=400, detail=f"Need at least {MIN_SEC} s of video at ~{FPS} fps")

    # Compose POS with optional green-emphasis fallback: if SNR low, mix traces
    bvp_raw = run_pos(frames, int(fs_used) if fs_used else FPS)
    # Optional: blend a small fraction of green-emphasis trace to stabilize
    green_trace = emphasize_green_mean_rgb(frames)
    if green_trace.size == bvp_raw.size:
        # Increase blend to 20%
        bvp_raw = 0.8 * bvp_raw + 0.2 * (green_trace - green_trace.mean())
    bvp = preprocess_bvp(bvp_raw, int(fs_used) if fs_used else FPS)
    hr_fft = bpm_fft(bvp, int(fs_used) if fs_used else FPS)
    hr_peak = bpm_peaks(bvp, int(fs_used) if fs_used else FPS, hr_fft)
    rr = respiratory_rate(bvp, int(fs_used) if fs_used else FPS)
    hrv = hrv_metrics(bvp, int(fs_used) if fs_used else FPS)
    # Use FFT HR for SNR targeting (undo previous change)
    snr = snr_db(bvp, hr_fft, int(fs_used) if fs_used else FPS)
    quality = quality_score(bvp)

    bp_res = bp_from_ppg(bvp, age, bmi)

    response = {
        "heart_rate": {
            "fft_bpm": round(hr_fft, 1),
            "peak_bpm": round(hr_peak, 1),
            "method": "Both FFT and peak detection",
        },
        "respiratory_rate": {"breaths_per_minute": round(rr, 1)},
        "heart_rate_variability": {
            "rmssd_ms": round(hrv["rmssd"], 2),
            "sdnn_ms": round(hrv["sdnn"], 2),
            "mean_rr_ms": round(hrv["mean_rr"], 2),
            "pnn50_percent": round(hrv["pnn50"], 2),
        },
        "stress_and_relaxation": {
            "stress_index": round(hrv["stress_index"], 2),
            "parasympathetic_tone": round(hrv["parasympathetic_tone"], 1),
            "stress_level": _get_stress_level(hrv["stress_index"]),
            "relaxation_state": _get_relaxation_state(hrv["parasympathetic_tone"]),
        },
        "signal_quality": {"snr_db": round(snr, 2), "quality_score": round(quality, 3)},
        "preprocessing": {
            "image_enhancement": "enabled" if enh_used else "disabled",
            "fps_used": int(fs_used) if fs_used else FPS,
            "face_crop": "enabled" if crop_used else "disabled",
            "enhanced_preview": f"/preview/{preview_name}" if preview_name else None,
        },
    }

    if bp_res is not None:
        response["blood_pressure"] = bp_res
    return response


def _get_stress_level(stress_index: float) -> str:
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


