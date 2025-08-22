from __future__ import annotations

import base64
import io
import time
from collections import deque
from typing import Deque, List

import numpy as np
import cv2
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..config import FPS, MIN_SEC
from ..services.signal_proc import (
    emphasize_green_mean_rgb,
    preprocess_bvp,
    bpm_fft,
    bpm_peaks,
    hrv_metrics,
    snr_db,
)


router = APIRouter()


@router.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await websocket.accept()
    # Keep ~20s of frames for robust estimation
    max_seconds = max(20, MIN_SEC)
    frames: Deque[np.ndarray] = deque()
    stamps: Deque[float] = deque()
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

            frames.append(frame)
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

            # Build green-emphasized trace and process
            fr_list: List[np.ndarray] = list(frames)
            trace = emphasize_green_mean_rgb(fr_list)
            if trace.size < fs_eff * 2:
                await websocket.send_json({"type": "status", "state": "warming_up", "frames": len(frames)})
                continue
            bvp_broad = preprocess_bvp(trace, fs_eff)
            hr_fft = bpm_fft(bvp_broad, fs_eff)
            hr_peak = bpm_peaks(bvp_broad, fs_eff, hr_fft)
            snr = snr_db(bvp_broad, hr_fft, fs_eff)
            hrv = hrv_metrics(bvp_broad, fs_eff)

            # Waveform for UI: last ~8s, downsample to 300 points
            want_s = 8
            n_tail = min(len(bvp_broad), fs_eff * want_s)
            tail = bvp_broad[-n_tail:]
            if tail.size > 300:
                xs = np.linspace(0, tail.size - 1, 300)
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
                    "signal_quality": {"snr_db": round(float(snr), 2)},
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

