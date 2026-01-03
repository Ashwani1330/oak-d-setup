import cv2
import torch
import numpy as np
import os
import sys
import requests
from PIL import Image

# --- CONFIGURATION ---
TOKENIZER_URL = "https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz"

# OPTIMIZATION 1: Drastically lower resolution for speed
# 320x240 allows near real-time performance on RTX 4080
WEBCAM_RES = (320, 240) 

# OPTIMIZATION 2: Run AI every single frame for instant tracking
# Set to 2 if it's still slightly laggy, but 1 should be fine at this resolution.
INFERENCE_INTERVAL = 1

CONFIDENCE_THRESH = 0.2

# OPTIMIZATION 3: Display Scale
# We make the window 3x bigger so you can see the 320x240 image clearly
DISPLAY_SCALE = 1.0

# 1. SETUP PATHS
current_file_path = os.path.abspath(__file__)
sam3_repo_root = os.path.abspath(os.path.join(os.path.dirname(current_file_path), "..", ".."))
if sam3_repo_root not in sys.path:
    sys.path.append(sam3_repo_root)

# Check Tokenizer
assets_dir = os.path.join(sam3_repo_root, "assets")
bpe_path = os.path.join(assets_dir, "bpe_simple_vocab_16e6.txt.gz")
if not os.path.exists(bpe_path):
    pass 

# 2. IMPORTS
try:
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def main():
    print("--- SAM 3 'Turbo' Live Feed ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} (with Mixed Precision)")

    # 3. LOAD MODEL
    try:
        model = build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=None 
        )
        model = model.to(device)
        model.eval()
        processor = Sam3Processor(model, confidence_threshold=CONFIDENCE_THRESH)
        print("Model Loaded!")
    except Exception as e:
        print(f"Load Error: {e}")
        return

    text_prompt = input("\nObject to segment (e.g. person, cup): ").strip() or "person"
    print(f"Tracking: '{text_prompt}'... Press 'q' to quit.")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_RES[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_RES[1])
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_count = 0
    last_mask = None # Memory of the last valid mask

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Always copy the fresh frame for display (keeps video smooth)
        vis_frame = frame.copy()

        # Only run AI on specific intervals
        if frame_count % INFERENCE_INTERVAL == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            try:
                # ENABLE FP16 ACCELERATION
                with torch.amp.autocast("cuda"): 
                    with torch.no_grad():
                        inference_state = processor.set_image(pil_image)
                        processor.reset_all_prompts(inference_state)
                        result = processor.set_text_prompt(
                            state=inference_state, 
                            prompt=text_prompt
                        )

                        # Extract Mask
                        if 'scores' in result and len(result['scores']) > 0:
                            best_idx = torch.argmax(result['scores'])
                            mask_tensor = result['masks'][best_idx]
                            
                            # Move to CPU immediately to free GPU
                            mask = mask_tensor.cpu().numpy().astype(np.uint8)
                            if mask.ndim > 2: mask = mask[0]
                            
                            # Save to memory
                            last_mask = mask
                        else:
                            last_mask = None # Object lost

            except Exception:
                pass # Skip frame on error

        # VISUALIZATION (Runs every frame using 'last_mask')
        if last_mask is not None:
            # Resize mask if frame size changed (robustness)
            if last_mask.shape != vis_frame.shape[:2]:
                last_mask = cv2.resize(last_mask, (vis_frame.shape[1], vis_frame.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Create Green Overlay
            mask_bool = last_mask > 0
            
            # Standard Green Fill
            green_layer = np.zeros_like(vis_frame)
            green_layer[mask_bool] = [0, 255, 0]
            vis_frame = cv2.addWeighted(vis_frame, 1.0, green_layer, 0.5, 0)

        # Scale up for display so it's not tiny
        display_frame = cv2.resize(vis_frame, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_NEAREST)
        
        cv2.imshow('SAM 3 Turbo (Press q)', display_frame)
        frame_count += 1

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
