import cv2
import torch
import numpy as np
import os
import sys
import requests
import time
from PIL import Image

# --- CONFIGURATION ---
TOKENIZER_URL = "https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz"

WEBCAM_INDEX = 0
PROMPT_DEFAULT = "person"

# Speed knobs
MAX_SIDE = 768              # downscale frames for SAM3 (None to disable)
USE_AMP = True              # autocast on CUDA
CONF_THRESH = 0.20          # processor confidence threshold
MASK_ALPHA = 0.50           # overlay strength
SHOW_FPS = True

WINDOW = "SAM 3 Live (t=change prompt, q=quit)"

# 1. SETUP PATHS
current_file_path = os.path.abspath(__file__)
sam3_repo_root = os.path.abspath(os.path.join(os.path.dirname(current_file_path), "..", ".."))
if sam3_repo_root not in sys.path:
    sys.path.append(sam3_repo_root)

# Check Tokenizer
assets_dir = os.path.join(sam3_repo_root, "assets")
bpe_path = os.path.join(assets_dir, "bpe_simple_vocab_16e6.txt.gz")
if not os.path.exists(bpe_path):
    print(f"[AUTO-FIX] Downloading tokenizer to {bpe_path}...")
    try:
        os.makedirs(assets_dir, exist_ok=True)
        r = requests.get(TOKENIZER_URL, timeout=30)
        r.raise_for_status()
        with open(bpe_path, "wb") as f:
            f.write(r.content)
    except Exception as e:
        print(f"Error downloading tokenizer: {e}")
        sys.exit(1)

# 2. IMPORTS (your repo version)
try:
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)


def resize_keep_aspect(frame_bgr: np.ndarray, max_side: int):
    h, w = frame_bgr.shape[:2]
    if max_side is None or max(h, w) <= max_side:
        return frame_bgr, 1.0
    scale = max_side / float(max(h, w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def extract_masks_scores(result: dict):
    """
    Handles common key variants across repo versions.
    Expected masks shape: (N, H, W) or (N, 1, H, W) or (H, W)
    """
    masks = None
    scores = None

    for k in ["masks", "binary_masks", "out_binary_masks"]:
        if k in result:
            masks = result[k]
            break

    for k in ["scores", "out_scores", "mask_scores"]:
        if k in result:
            scores = result[k]
            break

    return masks, scores


def overlay_green(frame_bgr: np.ndarray, mask_bool: np.ndarray, alpha: float):
    vis = frame_bgr.copy()
    if mask_bool is None or not np.any(mask_bool):
        return vis

    green = np.zeros_like(vis, dtype=np.uint8)
    green[:, :, 1] = 255

    roi = vis[mask_bool]
    blended = cv2.addWeighted(roi, 1 - alpha, green[mask_bool], alpha, 0)
    vis[mask_bool] = blended

    # optional contour
    m255 = (mask_bool.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)

    return vis


def main():
    print("--- SAM 3 Live (Official) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # small speed helpers
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # 3. LOAD MODEL
    try:
        model = build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=None
        ).to(device)
        model.eval()

        processor = Sam3Processor(model, confidence_threshold=CONF_THRESH)
        print("Model Loaded!")
    except Exception as e:
        print(f"Load Error: {e}")
        return

    text_prompt = input("\nObject to segment (e.g. person, cup): ").strip() or PROMPT_DEFAULT
    print(f"Segmenting: '{text_prompt}'")

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    # Webcam settings (often helps FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    fps = 0.0
    t_prev = time.time()

    amp_ctx = torch.cuda.amp.autocast if (device == "cuda" and USE_AMP) else None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Downscale for model speed
        frame_small, scale = resize_keep_aspect(frame, MAX_SIDE)

        frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        mask_bool_full = None

        try:
            with torch.inference_mode():
                if amp_ctx:
                    with amp_ctx(dtype=torch.float16):
                        state = processor.set_image(pil_image)
                        processor.reset_all_prompts(state)
                        result = processor.set_text_prompt(state=state, prompt=text_prompt)
                else:
                    state = processor.set_image(pil_image)
                    processor.reset_all_prompts(state)
                    result = processor.set_text_prompt(state=state, prompt=text_prompt)

                masks, scores = extract_masks_scores(result)

                if masks is not None and scores is not None and len(scores) > 0:
                    # pick best
                    best_idx = torch.argmax(scores).item()
                    m = masks[best_idx]

                    # tensor -> numpy
                    if isinstance(m, torch.Tensor):
                        m = m.detach().float().cpu().numpy()

                    # squeeze (1,H,W) if needed
                    if m.ndim == 3:
                        m = m[0]

                    # threshold (logits or probs)
                    mask_small = (m > 0)

                    # scale back to full frame
                    if scale != 1.0:
                        mask_small_u8 = (mask_small.astype(np.uint8) * 255)
                        mask_full_u8 = cv2.resize(
                            mask_small_u8,
                            (frame.shape[1], frame.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        )
                        mask_bool_full = (mask_full_u8 > 0)
                    else:
                        mask_bool_full = mask_small

        except Exception:
            # Keep UI alive even if a frame fails
            mask_bool_full = None

        vis = overlay_green(frame, mask_bool_full, MASK_ALPHA)

        # FPS
        t_now = time.time()
        dt = t_now - t_prev
        t_prev = t_now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        if SHOW_FPS:
            cv2.putText(
                vis,
                f"Prompt: {text_prompt} | FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(WINDOW, vis)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("t"):
            # pause + change prompt
            cv2.waitKey(1)
            new_prompt = input("\nNew prompt: ").strip()
            if new_prompt:
                text_prompt = new_prompt
                print(f"Updated prompt -> '{text_prompt}'")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
