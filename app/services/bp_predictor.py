from __future__ import annotations

import os
import numpy as np

from ..config import TORCH_AVAILABLE


BP_PREDICTOR_AVAILABLE = False
_MODEL = None
_NORM = None
_DEVICE = "cpu"


def _lazy_init():
    global BP_PREDICTOR_AVAILABLE, _MODEL, _NORM, _DEVICE
    if _MODEL is not None or not TORCH_AVAILABLE:
        BP_PREDICTOR_AVAILABLE = _MODEL is not None
        return
    try:
        import torch
        import sys
        from os import path as _p

        sys.path.append(_p.join(_p.dirname(_p.dirname(__file__)), "ppg_bp"))
        try:
            from ppg_bp.model import M5_fusion_transformer
        except Exception:
            from model import M5_fusion_transformer  # type: ignore

        model_path = os.getenv(
            "BP_MODEL_PATH",
            _p.join(_p.dirname(_p.dirname(__file__)), "ppg_bp", "output", "ppg2bp_custom.pth"),
        )
        if not _p.exists(model_path):
            BP_PREDICTOR_AVAILABLE = False
            return

        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL = M5_fusion_transformer(n_input=1, n_output=2)
        _MODEL.load_state_dict(torch.load(model_path, map_location=_DEVICE))
        _MODEL.eval()

        norm_path = _p.join(_p.dirname(model_path), "bp_norm_params.npy")
        if os.path.exists(norm_path):
            _NORM = np.load(norm_path, allow_pickle=True).item()
        else:
            _NORM = {'sbp_min': 80, 'sbp_max': 200, 'dbp_min': 40, 'dbp_max': 120}
        BP_PREDICTOR_AVAILABLE = True
    except Exception:
        _MODEL = None
        BP_PREDICTOR_AVAILABLE = False


def bp_from_ppg(ppg: np.ndarray, age: float | None, bmi: float | None):
    _lazy_init()
    if _MODEL is None or not BP_PREDICTOR_AVAILABLE or not TORCH_AVAILABLE:
        return None
    try:
        import torch

        if len(ppg) < 512:
            ppg = np.pad(ppg, (0, 512 - len(ppg)), mode='constant')
        elif len(ppg) > 512:
            ppg = ppg[:512]
        ppg_min, ppg_max = ppg.min(), ppg.max()
        ppg_norm = (ppg - ppg_min) / (ppg_max - ppg_min) if ppg_max > ppg_min else ppg * 0

        age_val = age if age is not None else 65.0
        bmi_val = bmi if bmi is not None else 25.0

        with torch.no_grad():
            ppg_tensor = torch.tensor(ppg_norm).float().unsqueeze(0).unsqueeze(0)
            age_tensor = torch.tensor([[age_val]]).float().unsqueeze(2).repeat(1, 1, 128)
            bmi_tensor = torch.tensor([[bmi_val]]).float().unsqueeze(2).repeat(1, 1, 128)
            output = _MODEL(ppg_tensor, age_tensor, bmi_tensor).squeeze()
            sbp_norm, dbp_norm = output[0].item(), output[1].item()
            systolic = sbp_norm * (_NORM['sbp_max'] - _NORM['sbp_min']) + _NORM['sbp_min']
            diastolic = dbp_norm * (_NORM['dbp_max'] - _NORM['dbp_min']) + _NORM['dbp_min']
            return {
                "systolic_mmHg": round(max(60, min(250, systolic)), 1),
                "diastolic_mmHg": round(max(30, min(150, diastolic)), 1),
            }
    except Exception:
        return None


