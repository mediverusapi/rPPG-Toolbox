from __future__ import annotations

import numpy as np
from scipy import signal
import cv2

from ..config import GREEN_EMPHASIS, HR_MIN_BPM, HR_MAX_BPM, ANCHOR_BW_HZ


def preprocess_bvp(bvp: np.ndarray, fs: int) -> np.ndarray:
    bvp = signal.detrend(bvp, type='linear')
    # Light median filter to suppress outliers/spikes
    k = max(3, int(max(3, int(0.05 * fs)) ) | 1)  # odd kernel
    try:
        bvp = signal.medfilt(bvp, kernel_size=k)
    except Exception:
        pass
    # Use a slightly narrower passband around typical HR to improve SNR
    low, high = 0.9, 2.5
    b, a = signal.butter(3, [low / (fs / 2), high / (fs / 2)], btype='band')
    bvp_filtered = signal.filtfilt(b, a, bvp)
    # Slightly stronger smoothing after band-pass (cap to avoid over-smoothing)
    window_size = int(min(fs * 0.15, 11))
    if window_size > 1:
        kernel = np.ones(window_size) / window_size
        return np.convolve(bvp_filtered, kernel, mode='same')
    return bvp_filtered


def preprocess_bvp_anchored(bvp: np.ndarray, fs: int, hr_bpm: float) -> np.ndarray:
    """Band-pass BVP around an HR anchor to improve SNR.
    Uses a narrow band centered at HR (default ±0.35 Hz), clipped to physio range.
    """
    bvp = signal.detrend(bvp, type='linear')
    # Constrain anchor to plausible range
    hr_bpm = float(hr_bpm) if hr_bpm and np.isfinite(hr_bpm) else 72.0
    hr_bpm = float(min(max(hr_bpm, HR_MIN_BPM), HR_MAX_BPM))
    hr_hz = hr_bpm / 60.0
    bw = float(ANCHOR_BW_HZ)
    low = max(0.7, hr_hz - bw)
    high = min(3.0, hr_hz + bw)
    if high - low < 0.2:
        pad = (0.2 - (high - low)) / 2
        low = max(0.7, low - pad)
        high = min(3.0, high + pad)
    b, a = signal.butter(3, [low / (fs / 2), high / (fs / 2)], btype='band')
    bvp_filtered = signal.filtfilt(b, a, bvp)
    # light smoothing
    window_size = int(min(fs * 0.10, 9))
    if window_size > 1:
        kernel = np.ones(window_size) / window_size
        return np.convolve(bvp_filtered, kernel, mode='same')
    return bvp_filtered


def emphasize_green_mean_rgb(frames: list[np.ndarray]) -> np.ndarray:
    """Return 1D trace from mean RGB with mild green emphasis to boost rPPG SNR.
    This is an optional alternative pre-extraction if needed.
    """
    if len(frames) == 0:
        return np.array([])
    trace = []
    g_weight = float(GREEN_EMPHASIS)
    for f in frames:
        # BGR to RGB
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        # simple skin mask in YCrCb to reduce background (not strict)
        ycrcb = cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        skin = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))
        skin = cv2.medianBlur(skin, 5)
        mask = (skin > 0)
        if mask.sum() < 100:  # fallback if mask too small
            mask = np.ones_like(skin, dtype=bool)
        r = rgb[:, :, 0][mask].mean()
        g = rgb[:, :, 1][mask].mean()
        b = rgb[:, :, 2][mask].mean()
        val = (r + g_weight * g + b) / (2.0 + g_weight)
        trace.append(val)
    return np.asarray(trace, dtype=np.float32)


def bpm_fft(bvp: np.ndarray, fs: int) -> float:
    """HR from spectrum with harmonic/subharmonic resolution.
    - Finds dominant peak within [HR_MIN_BPM, HR_MAX_BPM].
    - If peak near 2x a plausible rate, fold to fundamental.
    - If near 0.5x, double to plausible.
    """
    b = bvp - np.mean(bvp)
    f, pxx = signal.welch(b, fs, nperseg=min(len(b), max(256, 8*fs)))
    band = (f * 60 >= HR_MIN_BPM) & (f * 60 <= HR_MAX_BPM)
    if not np.any(band):
        return 72.0
    fb = f[band]
    pb = pxx[band]
    peak_idx = int(np.argmax(pb))
    peak_hz = float(fb[peak_idx])
    peak_bpm = peak_hz * 60.0
    # harmonic handling
    # check subharmonic at half frequency within band
    half_bpm = peak_bpm / 2.0
    if HR_MIN_BPM <= half_bpm <= HR_MAX_BPM:
        # compare energy around half vs at peak
        bw = 0.1
        def _band_power(center_hz: float):
            m = (f >= max(0.0, center_hz - bw)) & (f <= center_hz + bw)
            return float(np.sum(pxx[m]))
        if _band_power(peak_hz/2.0) * 1.2 > _band_power(peak_hz):
            peak_bpm = half_bpm
    # avoid subharmonic underestimates
    if peak_bpm < HR_MIN_BPM:
        peak_bpm *= 2.0
    if peak_bpm > HR_MAX_BPM:
        peak_bpm /= 2.0
    return float(min(max(peak_bpm, HR_MIN_BPM), HR_MAX_BPM))


def bpm_peaks(bvp: np.ndarray, fs: int, fft_bpm: float) -> float:
    min_distance = int(fs * 0.4)
    peaks, _ = signal.find_peaks(bvp, distance=min_distance)
    if len(peaks) < 2:
        return fft_bpm
    ibi_s = np.diff(peaks) / fs
    bpm_series = 60.0 / ibi_s
    # only accept beats close to constrained FFT HR
    fft_bpm_c = float(min(max(fft_bpm, HR_MIN_BPM), HR_MAX_BPM))
    mask = (bpm_series > 0.8 * fft_bpm_c) & (bpm_series < 1.2 * fft_bpm_c)
    if mask.sum() >= 2:
        return float(bpm_series[mask].mean())
    return float(np.median(bpm_series[(bpm_series>=HR_MIN_BPM) & (bpm_series<=HR_MAX_BPM)]))


def bpm_autocorr(bvp: np.ndarray, fs: int) -> float:
    """Estimate HR from autocorrelation peak within [HR_MIN_BPM, HR_MAX_BPM]."""
    x = bvp - np.mean(bvp)
    # normalized autocorrelation (biased is fine for peak location)
    ac = signal.correlate(x, x, mode='full')
    ac = ac[ac.size // 2:]
    # Convert HR limits to lag indices
    min_lag = int(fs * 60.0 / HR_MAX_BPM)
    max_lag = int(fs * 60.0 / HR_MIN_BPM)
    min_lag = max(min_lag, 1)
    max_lag = min(max_lag, len(ac) - 1)
    if max_lag <= min_lag + 1:
        return float((HR_MIN_BPM + HR_MAX_BPM) / 2.0)
    roi = ac[min_lag:max_lag]
    if roi.size < 3:
        return float((HR_MIN_BPM + HR_MAX_BPM) / 2.0)
    peaks, _ = signal.find_peaks(roi)
    if len(peaks) == 0:
        lag = int(np.argmax(roi)) + min_lag
    else:
        lag = int(peaks[np.argmax(roi[peaks])]) + min_lag
    bpm = 60.0 * fs / float(max(lag, 1))
    return float(min(max(bpm, HR_MIN_BPM), HR_MAX_BPM))


def refine_hr_harmonic(bvp: np.ndarray, fs: int, hr_guess_bpm: float) -> float:
    """Refine HR by checking subharmonic and harmonic power around the guess.
    Picks among {hr, hr/2, hr*2} staying within limits, favoring strongest power.
    """
    hr_guess_bpm = float(min(max(hr_guess_bpm, HR_MIN_BPM), HR_MAX_BPM))
    b = bvp - np.mean(bvp)
    f, pxx = signal.welch(b, fs, nperseg=min(len(b), max(256, 8*fs)))
    def band_power(target_bpm: float) -> float:
        hz = target_bpm / 60.0
        w = 0.08
        m = (f >= max(0.0, hz - w)) & (f <= hz + w)
        return float(np.sum(pxx[m]))
    cands = []
    for v in [hr_guess_bpm, hr_guess_bpm / 2.0, hr_guess_bpm * 2.0]:
        if HR_MIN_BPM <= v <= HR_MAX_BPM:
            cands.append((v, band_power(v)))
    if not cands:
        return hr_guess_bpm
    best = max(cands, key=lambda t: t[1])[0]
    return float(best)


def respiratory_rate(bvp: np.ndarray, fs: int) -> float:
    bvp_detrended = signal.detrend(bvp)
    b, a = signal.butter(2, [0.1 / (fs / 2), 0.5 / (fs / 2)], btype='band')
    resp_signal = signal.filtfilt(b, a, bvp_detrended)
    f, pxx = signal.welch(resp_signal, fs, nperseg=min(len(resp_signal), fs * 10))
    resp_band = (f >= 0.15) & (f <= 0.4)
    if np.any(resp_band) and np.sum(pxx[resp_band]) > 0:
        peak_idx = np.argmax(pxx[resp_band])
        peak_freq = f[resp_band][peak_idx]
        rate = float(peak_freq * 60)
        if rate > 22:
            subharmonic_band = (f >= 0.15) & (f <= 0.25)
            if np.any(subharmonic_band):
                peak_freq = f[subharmonic_band][np.argmax(pxx[subharmonic_band])]
                rate = float(peak_freq * 60)
        return rate
    return 15.0


def hrv_metrics(bvp: np.ndarray, fs: int) -> dict:
    bvp_clean = signal.detrend(bvp)
    bvp_norm = (bvp_clean - np.mean(bvp_clean)) / (np.std(bvp_clean) + 1e-7)
    b, a = signal.butter(3, [0.75 / (fs / 2), 3.0 / (fs / 2)], btype='band')
    bvp_filtered = signal.filtfilt(b, a, bvp_norm)
    median_val = np.median(bvp_filtered)
    mad = np.median(np.abs(bvp_filtered - median_val))
    height_threshold = median_val + 2 * mad
    min_distance = int(fs * 0.5)
    peaks, properties = signal.find_peaks(
        bvp_filtered, distance=min_distance, prominence=mad * 1.5, height=height_threshold, width=2
    )
    if len(peaks) < 5:
        return {"rmssd": 0.0, "sdnn": 0.0, "mean_rr": 0.0, "stress_index": 0.0, "pnn50": 0.0, "parasympathetic_tone": 0.0}
    rr_intervals = np.diff(peaks) / fs
    rr_intervals = rr_intervals[(rr_intervals > 0.6) & (rr_intervals < 1.2)]
    if len(rr_intervals) < 3:
        return {"rmssd": 0.0, "sdnn": 0.0, "mean_rr": 0.0, "stress_index": 0.0, "pnn50": 0.0, "parasympathetic_tone": 0.0}
    rr_diffs = np.diff(rr_intervals)
    valid_mask = np.abs(rr_diffs) < 0.1
    if np.sum(valid_mask) < 2:
        return {"rmssd": 25.0, "sdnn": 35.0, "mean_rr": 800.0, "stress_index": 8.0, "pnn50": 15.0, "parasympathetic_tone": 50.0}
    valid_diffs = rr_diffs[valid_mask]
    median_diff = np.median(valid_diffs)
    mad_diff = np.median(np.abs(valid_diffs - median_diff))
    z_scores = 0.6745 * (valid_diffs - median_diff) / (mad_diff + 1e-7)
    final_diffs = valid_diffs[np.abs(z_scores) < 2.0]
    if len(final_diffs) < 2:
        return {"rmssd": 25.0, "sdnn": 35.0, "mean_rr": 800.0, "stress_index": 8.0, "pnn50": 15.0, "parasympathetic_tone": 50.0}
    rmssd = np.sqrt(np.mean(final_diffs ** 2))
    valid_rr = rr_intervals[:-1][valid_mask]
    if len(valid_rr) > 2:
        sdnn = np.std(valid_rr)
        mean_rr = np.mean(valid_rr)
    else:
        sdnn = 0.035
        mean_rr = 0.8
    pnn50 = np.sum(np.abs(rr_diffs) > 0.05) / len(rr_diffs) * 100 if len(rr_diffs) > 0 else 0.0
    rr_ms = rr_intervals * 1000
    if len(rr_ms) > 5:
        rr_range = np.max(rr_ms) - np.min(rr_ms)
        bin_width = max(25, rr_range / 10)
        hist, bin_edges = np.histogram(rr_ms, bins=np.arange(min(rr_ms), max(rr_ms) + bin_width, bin_width))
        if np.max(hist) > 0 and len(hist) > 1:
            amo = np.max(hist) / len(rr_ms) * 100
            mo_idx = np.argmax(hist)
            mo = (bin_edges[mo_idx] + bin_edges[mo_idx + 1]) / 2 / 1000
            mxdmn = np.max(rr_intervals) - np.min(rr_intervals)
            if mo > 0.5 and mxdmn > 0.05:
                raw_stress = amo / (2 * mo * mxdmn)
                if raw_stress > 0:
                    stress_index = 2 + 4 * np.log10(max(1, raw_stress))
                else:
                    stress_index = 4.0
                stress_index = min(10.0, max(2.0, stress_index))
            else:
                stress_index = 4.0
        else:
            stress_index = 4.0
    else:
        stress_index = 4.0
    if rmssd > 0.035 and sdnn > 0.04:
        stress_index = min(stress_index, 6.0)
    rmssd_score = min(80, (rmssd * 1000) * 1.5)
    pnn50_score = min(80, pnn50 * 1.5)
    stress_score = max(20, 100 - (stress_index * 6))
    parasympathetic_tone = (rmssd_score * 0.3 + pnn50_score * 0.3 + stress_score * 0.4)
    if stress_index > 8:
        parasympathetic_tone = min(parasympathetic_tone, 40)
    elif stress_index > 6:
        parasympathetic_tone = min(parasympathetic_tone, 60)
    parasympathetic_tone = max(10.0, min(90.0, parasympathetic_tone))
    rmssd_ms = max(10.0, min(60.0, rmssd * 1000))
    sdnn_ms = max(20.0, min(80.0, sdnn * 1000))
    return {
        "rmssd": round(rmssd_ms, 2),
        "sdnn": round(sdnn_ms, 2),
        "mean_rr": round(mean_rr * 1000, 2),
        "stress_index": round(stress_index, 2),
        "pnn50": round(pnn50, 2),
        "parasympathetic_tone": round(parasympathetic_tone, 1),
    }


def snr_db(bvp: np.ndarray, hr_bpm: float, fs: int) -> float:
    """Welch-based SNR using HR and its harmonic vs rest of physio band."""
    hr_freq = hr_bpm / 60.0
    deviation = 6 / 60.0
    # Welch smoother spectrum than periodogram
    nperseg = min(len(bvp), max(fs * 8, 256))
    if nperseg < 32:
        nperseg = 32
    f, pxx = signal.welch(bvp, fs, nperseg=nperseg)
    hr_band = (f >= (hr_freq - deviation)) & (f <= (hr_freq + deviation))
    harmonic_band = (f >= (2 * hr_freq - deviation)) & (f <= (2 * hr_freq + deviation))
    signal_power = float(np.sum(pxx[hr_band]) + np.sum(pxx[harmonic_band]))
    phys_band = (f >= 0.75) & (f <= 3.0)
    noise_band = phys_band & ~hr_band & ~harmonic_band
    noise_power = float(np.sum(pxx[noise_band]))
    if noise_power <= 0 or not np.isfinite(noise_power):
        return 0.0
    return float(10.0 * np.log10(max(signal_power, 1e-12) / noise_power))


def quality_score(bvp: np.ndarray) -> float:
    b = signal.detrend(bvp)
    f, pxx = signal.welch(b, fs=max(30, int(len(bvp) / max(1, len(bvp)//2))), nperseg=min(len(b), 512))
    band = (f >= 0.8) & (f <= 2.8)
    if not np.any(band):
        return 0.0
    total = float(np.sum(pxx[band]))
    if total <= 0:
        return 0.0
    peak = float(np.max(pxx[band]))
    return float(max(0.0, min(1.0, peak / total)))


