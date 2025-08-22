import os
import argparse
import cv2
import numpy as np
import torch
import math

from model import Finetunemodel


def parse_args():
    parser = argparse.ArgumentParser("SCI video enhancement")
    parser.add_argument("--input", required=True, type=str, help="path to input video (e.g., .mp4/.mov)")
    parser.add_argument("--output", type=str, default=None, help="path to output enhanced video (.mp4)")
    parser.add_argument("--model", type=str, default="./weights/medium.pt", help="path to SCI weights")
    parser.add_argument("--max_size", type=int, default=640, help="max side length for processing (0=keep)")
    parser.add_argument("--codec", type=str, default="mp4v", help="fourcc codec (e.g., mp4v, avc1, MJPG)")
    parser.add_argument("--fps", type=float, default=None, help="override output FPS; default uses input FPS")
    parser.add_argument("--strength", type=float, default=1.0, help="blend factor: 0=original, 1=full enhancement")
    parser.add_argument("--gamma", type=float, default=1.0, help="gamma correction on output (>1 darkens, <1 brightens)")
    return parser.parse_args()


def compute_output_size(width: int, height: int, max_size: int) -> tuple:
    if max_size is None or max_size <= 0:
        return width, height
    scale = min(max_size / max(width, height), 1.0)
    out_w = int(round(width * scale))
    out_h = int(round(height * scale))
    # Ensure even dimensions for some codecs
    if out_w % 2 == 1:
        out_w += 1
    if out_h % 2 == 1:
        out_h += 1
    return out_w, out_h


def enhance_video(input_path: str, output_path: str, model_path: str, max_size: int, codec: str, fps_override: float = None, strength: float = 1.0, gamma: float = 1.0):
    # Select device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Load model (weights load on CPU in Finetunemodel; then move to device)
    model = Finetunemodel(model_path)
    model = model.to(device)
    model.eval()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    in_fps = cap.get(cv2.CAP_PROP_FPS)
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_w, out_h = compute_output_size(in_w, in_h, max_size)
    # Determine FPS robustly
    if fps_override is not None and fps_override > 0:
        fps = float(fps_override)
    else:
        try:
            fps_candidate = float(in_fps)
        except Exception:
            fps_candidate = 0.0
        if math.isnan(fps_candidate) or fps_candidate <= 1e-3:
            fps = 30.0
        else:
            fps = fps_candidate

    # Prepare writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    # Fallback to widely compatible codecs if needed
    if not writer.isOpened():
        # Try H.264 if available
        for fallback_codec in ["avc1", "mp4v", "MJPG"]:
            fourcc = cv2.VideoWriter_fourcc(*fallback_codec)
            writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
            if writer.isOpened():
                break
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter for: {output_path} with codec {codec} or fallbacks")

    frame_idx = 0
    with torch.no_grad():
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if (in_w, in_h) != (out_w, out_h):
                frame_bgr = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)

            # BGR -> RGB, normalize to [0,1]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb_f = (frame_rgb.astype(np.float32) / 255.0)

            # HWC -> CHW tensor
            tensor = torch.from_numpy(np.transpose(frame_rgb_f, (2, 0, 1))).unsqueeze(0)
            tensor = tensor.to(device)

            i, r = model(tensor)
            # r: (1, 3, H, W) in [0,1]
            r_np = r[0].detach().cpu().numpy()
            r_np = np.transpose(r_np, (1, 2, 0))  # HWC, RGB float [0,1]

            # Blend with original to control enhancement strength
            s = max(0.0, min(1.0, float(strength)))
            out_rgb = (1.0 - s) * frame_rgb_f + s * r_np
            out_rgb = np.clip(out_rgb, 0.0, 1.0)

            # Optional gamma correction (gamma > 1 darkens)
            if abs(gamma - 1.0) > 1e-6:
                g = max(0.2, min(5.0, float(gamma)))
                out_rgb = np.power(out_rgb, g)

            out_u8 = (out_rgb * 255.0).round().astype(np.uint8)
            out_bgr = cv2.cvtColor(out_u8, cv2.COLOR_RGB2BGR)

            writer.write(out_bgr)
            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx} frames...")

    cap.release()
    writer.release()
    print(f"Saved enhanced video to: {output_path}")


def main():
    args = parse_args()
    input_path = args.input
    if args.output is None:
        base, _ = os.path.splitext(input_path)
        output_path = base + "_enhanced.mp4"
    else:
        output_path = args.output

    enhance_video(
        input_path=input_path,
        output_path=output_path,
        model_path=args.model,
        max_size=args.max_size,
        codec=args.codec,
        fps_override=args.fps,
        strength=args.strength,
        gamma=args.gamma,
    )


if __name__ == "__main__":
    main()


