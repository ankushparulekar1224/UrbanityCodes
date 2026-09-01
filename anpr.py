import cv2
import re
import logging
from paddleocr import PaddleOCR

# Suppress Logging for Cleaner Output
logging.getLogger("ppocr").setLevel(logging.ERROR)

# Initialize stable PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def test_ocr_on_image(image_path):
    """Perform OCR on a given image file and return detected text."""
    
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Error: Could not read image at '{image_path}'. Check the file path.")
        return None
        
    # The original stable API that the GitHub repo used
    results = ocr.ocr(frame, det=True, rec=True, cls=False)
    
    detected_texts = []
    if results and isinstance(results, list) and len(results) > 0 and results[0]:
        for res in results[0]:
            text = res[1][0]  
            confidence = res[1][1]  
            
            pattern = re.compile(r'[\W]')
            text = pattern.sub('', text)
            text = text.replace("???", "").strip()
            text = text.replace("O", "0").strip() 
            text = text.replace("粤", "").strip()
            
            if confidence > 0.7 and text not in {".", "?"}:
                detected_texts.append(text)

    current_text = " ".join(detected_texts)
    return current_text   

if __name__ == "__main__":
    IMAGE_FILE_PATH = r"c:\Users\thero\Downloads\Screenshot 2026-07-20 101723.png"
    
    print(f"Running OCR on: {IMAGE_FILE_PATH}")
    extracted_plate = test_ocr_on_image(IMAGE_FILE_PATH)
    
    print("-" * 30)
    print(f"Detected License Plate: {extracted_plate}")
    print("-" * 30)