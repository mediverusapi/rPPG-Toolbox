from __future__ import annotations

import os
import tempfile
from typing import List, Tuple, Optional

import cv2
import numpy as np

from ..config import FPS, ENH_OUT_DIR, FACE_CROP_ENABLED, HAAR_CASCADE_PATH, FALLBACK_HAAR_PATH, FOREHEAD_ONLY
from .enhancement import maybe_enhance_frame


def _load_face_detector():
    path = HAAR_CASCADE_PATH if os.path.exists(HAAR_CASCADE_PATH) else FALLBACK_HAAR_PATH
    if os.path.exists(path):
        try:
            return cv2.CascadeClassifier(path)
        except Exception:
            return None
    return None


def video_to_frames(file_bytes: bytes) -> Tuple[List[np.ndarray], float, bool, bool, Optional[str]]:
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

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_read = cap.get(cv2.CAP_PROP_FPS)
    try:
        fs = float(fps_read)
        if fs <= 1e-3 or np.isnan(fs):
            fs = FPS
    except Exception:
        fs = FPS

    # Aspect-ratio preserving scale: cap long side to 640
    scale_factor = 1.0
    max_side = max(width, height)
    if max_side > 640:
        scale_factor = 640.0 / max_side

    # Optional face detector
    face_detector = _load_face_detector()
    face_box = None

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Detect once, then lock ROI to avoid crop jitter
        if FACE_CROP_ENABLED and face_detector is not None and face_box is None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    side = int(max(w, h) * 1.5)
                    cx, cy = x + w // 2, y + h // 2
                    x0 = max(0, cx - side // 2)
                    y0 = max(0, cy - side // 2)
                    x1 = min(frame.shape[1], x0 + side)
                    y1 = min(frame.shape[0], y0 + side)
                    face_box = (x0, y0, x1, y1)
                    face_crop_used = True
            except Exception:
                pass

        if FACE_CROP_ENABLED and face_box is not None:
            x0, y0, x1, y1 = face_box
            roi = frame[y0:y1, x0:x1]
            if FOREHEAD_ONLY:
                h = roi.shape[0]
                # take upper 35% as forehead region
                fh = max(10, int(0.35 * h))
                roi = roi[0:fh, :]
            frame = roi

        if scale_factor < 1.0:
            new_w = int(round(frame.shape[1] * scale_factor))
            new_h = int(round(frame.shape[0] * scale_factor))
            # Ensure even dims for codecs
            if new_w % 2 == 1:
                new_w += 1
            if new_h % 2 == 1:
                new_h += 1
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        frame, enhanced = maybe_enhance_frame(frame)
        enhancement_used = enhancement_used or enhanced

        if writer is None:
            out_h, out_w = frame.shape[0], frame.shape[1]
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            preview_name = f"{os.urandom(8).hex()}.mp4"
            preview_path = os.path.join(ENH_OUT_DIR, preview_name)
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

    return frames, fs, enhancement_used, face_crop_used, (
        os.path.basename(preview_path) if preview_path else None
    )


