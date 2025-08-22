from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np

from ..config import TORCH_AVAILABLE, RPPG_INFER_CONFIG, ROOT_DIR

_MODEL = None
_DEVICE = "cpu"
_CFG = None


def _load_yaml_config(path: str) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _lazy_init():
    global _MODEL, _DEVICE, _CFG
    if _MODEL is not None:
        return
    if not TORCH_AVAILABLE:
        return
    import torch
    # Ensure MPS missing ops fall back to CPU; prefer explicit CPU to avoid runtime errors
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    _CFG = _load_yaml_config(RPPG_INFER_CONFIG)
    model_name = str(_CFG.get("MODEL", {}).get("NAME", "Physnet")).lower()
    fs = int(_CFG.get("TEST", {}).get("DATA", {}).get("FS", 30))
    infer = _CFG.get("INFERENCE", {})
    model_path = os.path.expanduser(infer.get("MODEL_PATH", ""))

    # Resolve weights path: prefer absolute, otherwise relative to repo ROOT_DIR
    if not os.path.isabs(model_path):
        cand1 = os.path.join(ROOT_DIR, model_path.lstrip("./"))
        cand2 = os.path.join(ROOT_DIR, model_path)
        model_path = cand1 if os.path.exists(cand1) else cand2

    if torch.cuda.is_available():
        _DEVICE = "cuda"
    else:
        _DEVICE = "cpu"

    if model_name in ("physnet", "physnet_padding_encoder_decoder_max", "physnet_padding_encoder_decoder_max(nn.module)"):
        from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX as PhysNet
        frame_num = int(_CFG.get("MODEL", {}).get("PHYSNET", {}).get("FRAME_NUM", 128))
        _MODEL = PhysNet(frames=frame_num)
    elif model_name in ("tscan", "ts_can", "ts-can"):
        from neural_methods.model.TS_CAN import TSCAN
        frame_depth = int(_CFG.get("MODEL", {}).get("TSCAN", {}).get("FRAME_DEPTH", 10))
        img_size = int(_CFG.get("TEST", {}).get("DATA", {}).get("RESIZE", {}).get("W", 72))
        _MODEL = TSCAN(frame_depth=frame_depth, img_size=img_size)
    else:
        raise RuntimeError(f"Unsupported RPPG model in config: {model_name}")

    if not os.path.exists(model_path):
        raise RuntimeError(f"RPPG weights not found: {model_path}")

    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
        # optional: strip prefixes if saved with DataParallel
        state = {k.replace("module.", ""): v for k, v in state.items()}
    try:
        _MODEL.load_state_dict(state, strict=False)
    except Exception:
        _MODEL.load_state_dict(state)
    _MODEL.eval()
    _MODEL.to(_DEVICE)


def _preprocess_frames_to_clip_batch(frames: List[np.ndarray], cfg: dict) -> Tuple[np.ndarray, int]:
    # Follow BaseLoader: crop assumed done; resize elsewhere; build DiffNormalized/Standardized
    import cv2
    data_cfg = cfg.get("TEST", {}).get("DATA", {})
    resize = data_cfg.get("RESIZE", {})
    W = int(resize.get("W", 72))
    H = int(resize.get("H", 72))
    types = cfg.get("TEST", {}).get("DATA", {}).get("PREPROCESS", {}).get("DATA_TYPE", ["DiffNormalized"]) or ["DiffNormalized"]

    # resize preserving aspect, then center-crop to W,H
    resized = []
    for f in frames:
        if f.shape[0] != H or f.shape[1] != W:
            f2 = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        else:
            f2 = f
        resized.append(f2)
    arr = np.asarray(resized).astype(np.float32)

    def diff_normalize(data):
        n, h, w, c = data.shape
        dlen = n - 1
        d = np.zeros((dlen, h, w, c), dtype=np.float32)
        pad = np.zeros((1, h, w, c), dtype=np.float32)
        for j in range(dlen):
            d[j] = (data[j + 1] - data[j]) / (data[j + 1] + data[j] + 1e-7)
        d = d / (np.std(d) + 1e-8)
        d = np.concatenate([d, pad], axis=0)
        d[np.isnan(d)] = 0
        return d

    def standardized(data):
        z = data - np.mean(data)
        z = z / (np.std(z) + 1e-8)
        z[np.isnan(z)] = 0
        return z

    channels = []
    for t in types:
        if t.lower() == "diffnormalized":
            channels.append(diff_normalize(arr))
        elif t.lower() == "standardized":
            channels.append(standardized(arr))
        elif t.lower() == "raw":
            channels.append(arr.copy())
        else:
            raise RuntimeError(f"Unsupported DATA_TYPE: {t}")
    data = np.concatenate(channels, axis=-1)

    # chunk to windows per config
    infer = cfg.get("INFERENCE", {})
    win_sec = int(infer.get("EVALUATION_WINDOW", {}).get("WINDOW_SIZE", 10))
    fs = int(cfg.get("TEST", {}).get("DATA", {}).get("FS", 30))
    chunk_len = int(cfg.get("TEST", {}).get("DATA", {}).get("PREPROCESS", {}).get("CHUNK_LENGTH", 128))
    if chunk_len <= 0:
        chunk_len = max(1, int(win_sec * fs))
    clip_num = data.shape[0] // chunk_len
    clips = [data[i * chunk_len:(i + 1) * chunk_len] for i in range(clip_num)]
    if len(clips) == 0:
        clips = [data]
    batch = np.stack(clips, axis=0)  # [N, T, H, W, C]
    # transpose to NCHWT or NTHWC as model expects
    # Returns batch shaped [N, T, H, W, C] and chunk length
    return batch, chunk_len


def neural_bvp(frames: List[np.ndarray]) -> Tuple[np.ndarray, str]:
    """Return BVP from neural model using infer config.
    Returns: (bvp_1d, model_name)
    """
    _lazy_init()
    if _MODEL is None:
        raise RuntimeError("Neural rPPG model not available/loaded")

    import torch

    batch, chunk_len = _preprocess_frames_to_clip_batch(frames, _CFG)
    model_name = str(_CFG.get("MODEL", {}).get("NAME", "physnet")).lower()
    fs = int(_CFG.get("TEST", {}).get("DATA", {}).get("FS", 30))

    with torch.no_grad():
        if model_name.startswith("physnet"):
            # PhysNet expects [B,C,T,H,W] with C=3
            x = batch[..., :3]  # first 3 channels
            x = np.transpose(x, (0, 4, 1, 2, 3))
            x_t = torch.from_numpy(x).float().contiguous().to(_DEVICE)
            rppg_list = []
            for clip in x_t:  # [C,T,H,W]
                clip_b = clip.contiguous().unsqueeze(0)
                y, *_ = _MODEL(clip_b)
                rppg = y.contiguous().squeeze(0).detach().cpu().numpy()
                rppg_list.append(rppg)
            bvp = np.concatenate(rppg_list, axis=0)
        else:
            # TSCAN expects stacked frames: [nt, c, h, w], with c=6 (DiffNormalized+Standardized)
            # Ensure time length is divisible by frame_depth (n_segment)
            frame_depth = int(_CFG.get("MODEL", {}).get("TSCAN", {}).get("FRAME_DEPTH", 10))
            out_list = []
            for clip in batch:  # [T,H,W,C]
                T = clip.shape[0]
                C = clip.shape[-1]
                # transpose to [T,C,H,W]
                x_tc = np.transpose(clip, (0, 3, 1, 2))
                # if channels not 6, try to pad/repeat to 6
                if C < 6:
                    reps = (6 + C - 1) // C
                    x_tc = np.concatenate([x_tc] * reps, axis=1)[:, :6]
                elif C > 6:
                    x_tc = x_tc[:, :6]
                # trim to multiple of frame_depth
                T_adj = (T // frame_depth) * frame_depth
                if T_adj == 0:
                    continue
                x_tc = x_tc[:T_adj]
                x_t = torch.from_numpy(x_tc).float().contiguous().to(_DEVICE)  # [nt,c,h,w]
                out = _MODEL(x_t)
                out_np = out.detach().cpu().contiguous().numpy().reshape(-1)
                out_list.append(out_np)
            if not out_list:
                raise RuntimeError("No valid TSCAN clips to process")
            bvp = np.concatenate(out_list, axis=0)

    return bvp.astype(np.float32), model_name


