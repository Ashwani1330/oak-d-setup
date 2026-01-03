import cv2
import torch
import numpy as np
import os
import sys
import requests
from PIL import Image

# --- CONFIGURATION ---
TOKENIZER_URL = "https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz"

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
        response = requests.get(TOKENIZER_URL)
        with open(bpe_path, 'wb') as f: f.write(response.content)
    except Exception as e:
        print(f"Error downloading tokenizer: {e}")
        sys.exit(1)

# 2. IMPORTS
try:
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def main():
    print("--- SAM 3 Live (Official) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 3. LOAD MODEL
    try:
        # Load model (letting it infer defaults)
        model = build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=None 
        )
        model = model.to(device)
        model.eval()
        
        # Lower threshold slightly to ensure we see things
        processor = Sam3Processor(model, confidence_threshold=0.2)
        print("Model Loaded!")
    except Exception as e:
        print(f"Load Error: {e}")
        return

    text_prompt = input("\nObject to segment (e.g. person, cup): ").strip() or "person"
    print(f"Tracking: '{text_prompt}'")

    cap = cv2.VideoCapture(0)
    # Optimized for speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        vis_frame = frame.copy()

        try:
            with torch.no_grad():
                # A. Encode Image
                inference_state = processor.set_image(pil_image)
                
                # B. Prompt with Text
                # We must reset prompts per frame for a fresh detection
                processor.reset_all_prompts(inference_state)
                result = processor.set_text_prompt(
                    state=inference_state, 
                    prompt=text_prompt
                )

                # C. Extract Best Mask
                # The result dict contains 'masks' (tensor) and 'scores' (tensor)
                if 'masks' in result and 'scores' in result:
                    masks = result['masks']
                    scores = result['scores']
                    
                    # Pick the mask with the highest confidence score
                    if len(scores) > 0:
                        best_idx = torch.argmax(scores)
                        mask_tensor = masks[best_idx] # Get best mask
                        
                        # Convert to Numpy
                        mask = mask_tensor.cpu().numpy()
                        
                        # Handle dimensions: (1, H, W) -> (H, W)
                        if mask.ndim > 2: mask = mask[0]
                        
                        # Resize if needed (sometimes model output differs from input)
                        if mask.shape != frame.shape[:2]:
                            mask = cv2.resize(mask.astype(np.float32), (frame.shape[1], frame.shape[0]))

                        # D. Draw Green Overlay
                        # Create green layer
                        color_mask = np.zeros_like(frame)
                        color_mask[:, :] = [0, 255, 0] # Green
                        
                        # Binary mask (threshold 0 for logits/masks)
                        mask_bool = mask > 0
                        
                        # Blend only where mask is True
                        # This avoids darkening the whole image
                        roi = vis_frame[mask_bool]
                        blended = cv2.addWeighted(roi, 0.5, color_mask[mask_bool], 0.5, 0)
                        vis_frame[mask_bool] = blended

        except Exception as e:
            # print(e) # Uncomment for debug
            pass

        cv2.imshow('SAM 3 Live (Press q to quit)', vis_frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
