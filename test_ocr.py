import cv2
import numpy as np
from OCR_Scanner import OCRScanner

def main():
    print("OCR Test Script")
    print("===============")
    
    # Create a simple image with text
    img = np.zeros((300, 600), dtype=np.uint8)
    img.fill(255)  # White background
    
    # Add text to the image
    text = "This is a test for OCR functionality"
    cv2.putText(img, text, (50, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    
    # Save the test image
    test_img_path = "ocr_test_image.jpg"
    cv2.imwrite(test_img_path, img)
    print(f"Created test image: {test_img_path}")
    
    # Initialize OCR scanner
    scanner = OCRScanner()
    print(f"OCR Scanner initialized. Tesseract available: {scanner.tesseract_available}")
    
    if scanner.tesseract_available:
        # Test OCR on the image
        print("\nPerforming OCR on test image...")
        extracted_text = scanner.extract_text(test_img_path)
        
        print("\nExtracted text:")
        print("--------------")
        print(extracted_text)
        
        # Show success/failure
        if "test" in extracted_text.lower():
            print("\n✅ OCR TEST SUCCESSFUL!")
        else:
            print("\n❌ OCR TEST FAILED - Text not detected properly")
    else:
        print("\n❌ OCR TEST FAILED - Tesseract not available")

if __name__ == "__main__":
    main() 