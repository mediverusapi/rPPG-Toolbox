from __future__ import annotations

import os
import numpy as np
import cv2

from ..config import TORCH_AVAILABLE, SCI_STRENGTH, SCI_GAMMA


_SCI = None
_SCI_DEVICE = "cpu"
_SCI_READY = False


def _lazy_init():
    global _SCI, _SCI_DEVICE, _SCI_READY
    if _SCI_READY:
        return
    try:
        if not TORCH_AVAILABLE:
            return
        import torch
        from image_enhancement_model.model import Finetunemodel as _SCIModel

        if torch.cuda.is_available():
            _SCI_DEVICE = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _SCI_DEVICE = "mps"
        else:
            _SCI_DEVICE = "cpu"

        weights = os.getenv(
            "SCI_WEIGHTS_PATH",
            os.path.join(os.path.dirname(__file__), "..", "..", "image_enhancement_model", "weights", "medium.pt"),
        )
        if os.path.exists(weights):
            _SCI = _SCIModel(weights).to(_SCI_DEVICE)
            _SCI.eval()
            _SCI_READY = True
    except Exception:
        _SCI = None
        _SCI_READY = False


def maybe_enhance_frame(frame_bgr):
    _lazy_init()
    if not _SCI_READY or _SCI is None:
        return frame_bgr, False

    try:
        import torch

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0).to(_SCI_DEVICE)
        with torch.no_grad():
            _, r = _SCI(tensor)
        enh = r[0].detach().cpu().numpy()
        enh = np.transpose(enh, (1, 2, 0))

        s = max(0.0, min(1.0, SCI_STRENGTH))
        out_rgb = (1.0 - s) * rgb + s * enh
        if abs(SCI_GAMMA - 1.0) > 1e-6:
            g = max(0.2, min(5.0, SCI_GAMMA))
            out_rgb = np.power(np.clip(out_rgb, 0.0, 1.0), g)
        enh_u8 = (np.clip(out_rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        out_bgr = cv2.cvtColor(enh_u8, cv2.COLOR_RGB2BGR)
        return out_bgr, True
    except Exception:
        return frame_bgr, False


