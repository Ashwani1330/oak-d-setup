import os
import sys

# --- CRITICAL: SET CACHE TO D: DRIVE ---
os.environ["HF_HOME"] = r"D:\sam3\huggingface_cache"

import cv2
import torch
import numpy as np
import requests
from PIL import Image

# --- CONFIGURATION ---
TOKENIZER_URL = "https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz"
WEBCAM_RES = (320, 240) 
INFERENCE_INTERVAL = 1  
CONFIDENCE_THRESH = 0.20 # Adjust this! Lower = more objects, Higher = fewer mistakes.
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
    print("--- SAM 3 'Multi-Object' Live Feed ---")
    print(f"Cache Location: {os.environ['HF_HOME']}")
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

    text_prompt = input("\nObject to segment (e.g. person, bottle): ").strip() or "person"
    print(f"Tracking ALL '{text_prompt}'s... Press 'q' to quit.")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_RES[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_RES[1])
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_count = 0
    last_mask = None 

    while True:
        ret, frame = cap.read()
        if not ret: break

        vis_frame = frame.copy()

        if frame_count % INFERENCE_INTERVAL == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            try:
                with torch.amp.autocast("cuda"): 
                    with torch.no_grad():
                        inference_state = processor.set_image(pil_image)
                        processor.reset_all_prompts(inference_state)
                        result = processor.set_text_prompt(
                            state=inference_state, 
                            prompt=text_prompt
                        )

                        # --- MULTI-OBJECT LOGIC ---
                        if 'scores' in result and len(result['scores']) > 0:
                            scores = result['scores']
                            masks = result['masks']
                            
                            # Filter: Find indices of all scores > threshold
                            # Squeeze ensures shape [N] instead of [N, 1] for indexing
                            valid_indices = torch.where(scores.flatten() > CONFIDENCE_THRESH)[0]
                            
                            if len(valid_indices) > 0:
                                # Select all valid masks
                                valid_masks = masks[valid_indices]
                                
                                # Merge them: Logical OR across all detected masks
                                # valid_masks shape: [N, 1, H, W] -> sum dim 0 -> [1, H, W]
                                merged_mask = torch.sum(valid_masks, dim=0) > 0 
                                
                                # Move to CPU
                                mask = merged_mask.cpu().numpy().astype(np.uint8)
                                if mask.ndim > 2: mask = mask[0]
                                last_mask = mask
                            else:
                                last_mask = None
                        else:
                            last_mask = None

            except Exception as e:
                # print(e) # Uncomment to debug
                pass 

        if last_mask is not None:
            if last_mask.shape != vis_frame.shape[:2]:
                last_mask = cv2.resize(last_mask, (vis_frame.shape[1], vis_frame.shape[0]), interpolation=cv2.INTER_NEAREST)

            mask_bool = last_mask > 0
            green_layer = np.zeros_like(vis_frame)
            green_layer[mask_bool] = [0, 255, 0]
            vis_frame = cv2.addWeighted(vis_frame, 1.0, green_layer, 0.5, 0)

        display_frame = cv2.resize(vis_frame, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_NEAREST)
        
        cv2.imshow('SAM 3 Multi (Press q)', display_frame)
        frame_count += 1

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

