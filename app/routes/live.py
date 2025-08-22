from __future__ import annotations

import base64
import io
import time
from collections import deque
from typing import Deque, List

import numpy as np
import cv2
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..config import (
    FPS,
    MIN_SEC,
    HAAR_CASCADE_PATH,
    FALLBACK_HAAR_PATH,
    FACE_CROP_ENABLED,
    FACE_BOX_SCALE,
    FOREHEAD_ONLY,
)
from ..services.signal_proc import (
    emphasize_green_mean_rgb,
    preprocess_bvp,
    preprocess_bvp_anchored,
    bpm_fft,
    bpm_peaks,
    hrv_metrics,
    snr_db,
)
from ..services.rppg_wrapper import run_pos, run_chrome, run_pbv


router = APIRouter()


@router.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await websocket.accept()
    # Keep ~20s of frames for robust estimation
    max_seconds = max(20, MIN_SEC)
    frames: Deque[np.ndarray] = deque()
    stamps: Deque[float] = deque()
    # Lazy-load Haar cascade
    haar = None
    if FACE_CROP_ENABLED:
        try:
            path = HAAR_CASCADE_PATH
            if not cv2.haveImageReader(path):
                path = FALLBACK_HAAR_PATH
            haar = cv2.CascadeClassifier(path)
        except Exception:
            haar = None
    roi = None  # (x, y, w, h)
    detect_every = 10
    frame_idx = 0
    try:
        while True:
            msg = await websocket.receive_json()
            if not isinstance(msg, dict):
                continue
            kind = msg.get("type")
            if kind == "close":
                await websocket.close()
                break
            if kind != "frame":
                continue
            data_url = msg.get("data", "")
            ts_ms = float(msg.get("ts", time.time() * 1000.0))
            # data URL: data:image/jpeg;base64,....
            if "," in data_url:
                b64_part = data_url.split(",", 1)[1]
            else:
                b64_part = data_url
            try:
                raw = base64.b64decode(b64_part)
                buf = np.frombuffer(raw, dtype=np.uint8)
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            except Exception:
                continue
            if frame is None:
                continue
            # Downscale for speed (long side <= 480)
            h, w = frame.shape[:2]
            scale = 480.0 / max(h, w)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            # Optional face/forehead ROI detection (stabilized)
            frame_idx += 1
            if FACE_CROP_ENABLED and haar is not None and (roi is None or frame_idx % detect_every == 0):
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
                    if len(faces) > 0:
                        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                        # scale box
                        cx, cy = fx + fw / 2, fy + fh / 2
                        side = max(fw, fh) * float(FACE_BOX_SCALE)
                        sx, sy = int(cx - side / 2), int(cy - side / 2)
                        ex, ey = sx + int(side), sy + int(side)
                        # clamp
                        sx = max(0, sx); sy = max(0, sy)
                        ex = min(frame.shape[1], ex); ey = min(frame.shape[0], ey)
                        nx, ny, nw, nh = sx, sy, ex - sx, ey - sy
                        # Forehead-only region
                        if FOREHEAD_ONLY:
                            ny = sy + int(0.15 * nh)
                            nh = int(0.35 * nh)
                        new_roi = (nx, ny, nw, nh)
                        if roi is None:
                            roi = new_roi
                        else:
                            ax = 0.8
                            rx, ry, rw, rh = roi
                            nx, ny, nw, nh = new_roi
                            roi = (int(ax * rx + (1-ax) * nx), int(ax * ry + (1-ax) * ny), int(ax * rw + (1-ax) * nw), int(ax * rh + (1-ax) * nh))
                except Exception:
                    pass

            frames.append(frame if roi is None else frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]])
            stamps.append(ts_ms)
            # Trim by time window
            while len(stamps) > 1 and (stamps[-1] - stamps[0]) > max_seconds * 1000.0:
                frames.popleft()
                stamps.popleft()

            # Compute metrics when we have enough history
            if len(frames) < max(FPS * 3, MIN_SEC * FPS) or (stamps[-1] - stamps[0]) <= 2000:
                await websocket.send_json({"type": "status", "state": "warming_up", "frames": len(frames)})
                continue

            # Estimate effective frame rate from timestamps
            duration_s = max(1e-3, (stamps[-1] - stamps[0]) / 1000.0)
            fs_eff = int(round((len(frames) - 1) / duration_s)) if len(frames) > 1 else FPS
            fs_eff = max(10, min(60, fs_eff))

            # Build green-emphasized trace with skin mask inside ROI
            fr_list: List[np.ndarray] = list(frames)
            trace = emphasize_green_mean_rgb(fr_list)
            if trace.size < fs_eff * 2:
                await websocket.send_json({"type": "status", "state": "warming_up", "frames": len(frames)})
                continue
            # Build candidates: Green, POS, CHROME, PBV → choose highest SNR
            candidates = {}
            candidates['GREEN'] = trace
            try:
                candidates['POS'] = run_pos(fr_list, fs_eff)
            except Exception:
                pass
            try:
                candidates['CHROME'] = run_chrome(fr_list, fs_eff)
            except Exception:
                pass
            try:
                candidates['PBV'] = run_pbv(fr_list)
            except Exception:
                pass

            best_name = None
            best_snr = -1e12
            best_raw = None
            per_method = {}
            for name, raw in candidates.items():
                try:
                    tmp = preprocess_bvp(raw, fs_eff)
                    hr0 = bpm_fft(tmp, fs_eff)
                    s = snr_db(tmp, hr0, fs_eff)
                    per_method[name] = float(s)
                    if s > best_snr:
                        best_snr = s
                        best_name = name
                        best_raw = raw
                except Exception:
                    continue
            if best_raw is None:
                best_raw = trace
                best_name = 'GREEN'

            # Two-stage: broad → anchored for cleaner waveform
            bvp_broad = preprocess_bvp(best_raw, fs_eff)
            hr_fft = bpm_fft(bvp_broad, fs_eff)
            bvp = preprocess_bvp_anchored(best_raw, fs_eff, hr_fft)
            hr_peak = bpm_peaks(bvp, fs_eff, hr_fft)
            snr = snr_db(bvp, hr_fft, fs_eff)
            hrv = hrv_metrics(bvp, fs_eff)

            # Waveform for UI: last ~20s, downsample to 500 points
            want_s = 20
            n_tail = min(len(bvp_broad), fs_eff * want_s)
            tail = bvp[-n_tail:]
            if tail.size > 500:
                xs = np.linspace(0, tail.size - 1, 500)
                waveform = np.interp(xs, np.arange(tail.size), tail)
            else:
                waveform = tail

            await websocket.send_json(
                {
                    "type": "metrics",
                    "heart_rate": {
                        "fft_bpm": round(float(hr_fft), 1),
                        "peak_bpm": round(float(hr_peak), 1),
                    },
                    "heart_rate_variability": {
                        "rmssd_ms": round(float(hrv.get("rmssd", 0.0)), 2),
                        "sdnn_ms": round(float(hrv.get("sdnn", 0.0)), 2),
                        "mean_rr_ms": round(float(hrv.get("mean_rr", 0.0)), 2),
                        "pnn50_percent": round(float(hrv.get("pnn50", 0.0)), 2),
                    },
                    "signal_quality": {"snr_db": round(float(snr), 2), "method": best_name},
                    "waveform": [float(x) for x in waveform.tolist()],
                    "fs": fs_eff,
                }
            )
    except WebSocketDisconnect:
        return
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass

