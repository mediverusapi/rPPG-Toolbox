from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, Form
import os
import numpy as np

from ..config import FPS, MIN_SEC, USE_NEURAL_RPPG, GREEN_BLEND
from ..services.video_io import video_to_frames
from ..services.rppg_wrapper import run_best_bvp, run_pos, run_chrome, run_pbv
from ..services.neural_rppg import neural_bvp
from ..services.signal_proc import (
    preprocess_bvp,
    preprocess_bvp_anchored,
    bpm_fft,
    bpm_peaks,
    bpm_autocorr,
    refine_hr_harmonic,
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

    # Choose BVP source
    method_used = "auto_best"
    if USE_NEURAL_RPPG:
        try:
            bvp_raw, mname = neural_bvp(frames)
            method_used = f"neural:{mname}"
            # Optional: blend in a mild green-channel trace to stabilize color noise
            green_trace = emphasize_green_mean_rgb(frames)
            if green_trace.size and green_trace.size != bvp_raw.size:
                # simple resample to match length
                import numpy as _np
                green_trace = _np.interp(
                    _np.linspace(0, len(green_trace) - 1, num=bvp_raw.size),
                    _np.arange(len(green_trace)),
                    green_trace,
                )
            if green_trace.size == bvp_raw.size and GREEN_BLEND > 0.0:
                w = max(0.0, min(0.5, float(GREEN_BLEND)))
                bvp_raw = (1.0 - w) * bvp_raw + w * (green_trace - green_trace.mean())
                method_used += "+green"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Neural rPPG failed: {type(e).__name__}: {e}")
    # Build per-method results across neural + classical and pick best by SNR
    fs_eval = int(fs_used) if fs_used else FPS
    per_method = {}
    candidates: dict[str, np.ndarray] = {}
    if 'neural' in method_used:
        candidates[method_used] = bvp_raw
    # Classical methods (best-effort; ignore failures)
    try:
        candidates['POS'] = run_pos(frames, fs_eval)
    except Exception:
        pass
    try:
        candidates['CHROME'] = run_chrome(frames, fs_eval)
    except Exception:
        pass
    try:
        candidates['PBV'] = run_pbv(frames)
    except Exception:
        pass

    best_name = None
    best_snr = -1e12
    best_bvp = None
    for name, raw in candidates.items():
        try:
            tmp = preprocess_bvp(raw, fs_eval)
            hr0 = bpm_fft(tmp, fs_eval)
            s = snr_db(tmp, hr0, fs_eval)
            per_method[name] = {"snr_db": float(round(s, 2)), "fft_bpm": float(round(hr0, 1))}
            if s > best_snr:
                best_snr = s
                best_name = name
                best_bvp = raw
        except Exception:
            continue
    if best_bvp is not None and best_name is not None:
        bvp_raw = best_bvp
        method_used = best_name
    # Two-stage: broad band-pass → get HR → HR-anchored narrow band-pass
    fs_eval = int(fs_used) if fs_used else FPS
    bvp_broad = preprocess_bvp(bvp_raw, fs_eval)
    hr_fft0 = bpm_fft(bvp_broad, fs_eval)
    hr_ac = bpm_autocorr(bvp_broad, fs_eval)
    hr_fft = refine_hr_harmonic(bvp_broad, fs_eval, 0.6 * hr_fft0 + 0.4 * hr_ac)
    bvp = preprocess_bvp_anchored(bvp_raw, fs_eval, hr_fft)
    hr_peak = bpm_peaks(bvp, fs_eval, hr_fft)
    hrv = hrv_metrics(bvp, fs_eval)
    # Use FFT HR for SNR targeting
    snr = snr_db(bvp, hr_fft, fs_eval)
    quality = quality_score(bvp)

    bp_res = bp_from_ppg(bvp, age, bmi)

    # Render waveform PNG and save alongside preview
    waveform_name = None
    try:
        import io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 3))
        t = np.arange(len(bvp)) / (int(fs_used) if fs_used else FPS)
        ax.plot(t, bvp - np.mean(bvp), linewidth=1.2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("BVP (anchored)")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        from ..config import ENH_OUT_DIR
        waveform_name = f"{os.urandom(8).hex()}.png"
        with open(os.path.join(ENH_OUT_DIR, waveform_name), "wb") as f:
            f.write(buf.getvalue())
    except Exception:
        waveform_name = None

    # Render spectrum PNG (Welch) around HR
    spectrum_name = None
    try:
        import io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy import signal as _sig

        fs_eval = int(fs_used) if fs_used else FPS
        b = bvp - np.mean(bvp)
        f, pxx = _sig.welch(b, fs_eval, nperseg=min(len(b), max(256, 8*fs_eval)))
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(f * 60.0, pxx, linewidth=1.2)
        ax.axvline(hr_fft, color='r', linestyle='--', alpha=0.7, label='FFT HR')
        ax.axvline(hr_peak, color='g', linestyle=':', alpha=0.7, label='Peak HR')
        ax.set_xlim(40, 180)
        ax.set_xlabel("Frequency (BPM)")
        ax.set_ylabel("Power")
        ax.set_title("Welch Spectrum")
        ax.legend(loc='upper right')
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        from ..config import ENH_OUT_DIR
        spectrum_name = f"{os.urandom(8).hex()}.png"
        with open(os.path.join(ENH_OUT_DIR, spectrum_name), "wb") as f:
            f.write(buf.getvalue())
    except Exception:
        spectrum_name = None

    # Render peaks overlay PNG
    peaks_name = None
    try:
        import io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy import signal as _sig

        fs_eval = int(fs_used) if fs_used else FPS
        min_distance = int(fs_eval * 0.3)
        pks, _ = _sig.find_peaks(bvp, distance=min_distance)
        fig, ax = plt.subplots(figsize=(10, 3))
        t = np.arange(len(bvp)) / fs_eval
        ax.plot(t, bvp - np.mean(bvp), linewidth=1.1)
        if len(pks) > 0:
            ax.scatter(t[pks], (bvp - np.mean(bvp))[pks], s=10, color='r', alpha=0.8, label='peaks')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("BVP with detected peaks")
        if len(pks) > 0:
            ax.legend(loc='upper right')
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        from ..config import ENH_OUT_DIR
        peaks_name = f"{os.urandom(8).hex()}.png"
        with open(os.path.join(ENH_OUT_DIR, peaks_name), "wb") as f:
            f.write(buf.getvalue())
    except Exception:
        peaks_name = None

    response = {
        "heart_rate": {
            "fft_bpm": round(hr_fft, 1),
            "peak_bpm": round(hr_peak, 1),
            "method": "Both FFT and peak detection",
        },
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
        "signal_quality": {"snr_db": round(snr, 2), "quality_score": round(quality, 3), "method": method_used},
        "per_method": per_method,
        "preprocessing": {
            "image_enhancement": "enabled" if enh_used else "disabled",
            "fps_used": int(fs_used) if fs_used else FPS,
            "face_crop": "enabled" if crop_used else "disabled",
            "enhanced_preview": f"/preview/{preview_name}" if preview_name else None,
            "waveform_preview": f"/waveform/{waveform_name}" if waveform_name else None,
            "spectrum_preview": f"/waveform/{spectrum_name}" if spectrum_name else None,
            "peaks_preview": f"/waveform/{peaks_name}" if peaks_name else None,
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


