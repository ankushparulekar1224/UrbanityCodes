import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm

# --- CONFIG ---
BASE_DIR = r"C:/Users/softlabs_group/Documents/swab"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
LABEL_DIR = os.path.join(BASE_DIR, "labels")
MODEL_ID = "microsoft/Florence-2-large"

# Create labels folder
os.makedirs(LABEL_DIR, exist_ok=True)

def main():
    print("Initializing Florence-2 with Python 3.14 Patch...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- PATCH START ---
    # We manually load the config and set the missing attribute to bypass the error
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    if hasattr(config, 'text_config'):
        # Force the attribute into the sub-config where it's failing
        config.text_config.forced_bos_token_id = None
    # --- PATCH END ---

    print(f"Loading model on {device.upper()}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        config=config, # Pass our patched config
        trust_remote_code=True
    ).to(device).eval()
    
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(valid_extensions)]
    
    print(f"Found {len(image_files)} images. Processing...")

    for img_name in tqdm(image_files):
        try:
            img_path = os.path.join(IMAGE_DIR, img_name)
            image = Image.open(img_path).convert("RGB")
            
            # Use specific gauze prompt
            prompt = "<CAPTION_TO_PHRASE_GROUNDING>blood-stained surgical gauze. swab."
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3
                )
            
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(generated_text, task="<CAPTION_TO_PHRASE_GROUNDING>", image_size=image.size)
            
            # Save results in YOLO format
            res = parsed.get("<CAPTION_TO_PHRASE_GROUNDING>", {})
            with open(os.path.join(LABEL_DIR, os.path.splitext(img_name)[0] + ".txt"), "w") as f:
                for bbox in res.get('bboxes', []):
                    x1, y1, x2, y2 = bbox
                    w_img, h_img = image.size
                    cx = ((x1 + x2) / 2) / w_img
                    cy = ((y1 + y2) / 2) / h_img
                    w = abs(x2 - x1) / w_img
                    h = abs(y2 - y1) / h_img
                    f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    
        except Exception as e:
            print(f"Error on {img_name}: {e}")

    print("Success! Labels are in c:\\newData\\labels")

if __name__ == "__main__":
    main()