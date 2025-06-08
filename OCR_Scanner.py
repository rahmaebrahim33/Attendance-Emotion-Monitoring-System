import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
from tkinter import Tk, filedialog, simpledialog, messagebox

class OCRScanner:
    """
    Optical Character Recognition (OCR) module for the Vision-Based Attendance System.
    Extracts text from documents, attendance sheets, and IDs using Tesseract OCR.
    """
    
    def __init__(self, tesseract_path=r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
        """
        Initialize the OCR Scanner.
        
        Parameters:
        - tesseract_path: Path to tesseract executable (optional)
        """
        # Set Tesseract path
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Verify Tesseract installation
        try:
            pytesseract.get_tesseract_version()
            print("Tesseract OCR initialized successfully.")
            self.tesseract_available = True
        except Exception as e:
            print(f"Warning: Tesseract not properly configured - {str(e)}")
            print("Please install Tesseract OCR and ensure it's in your PATH")
            print("or provide the path using the tesseract_path parameter.")
            print("OCR features will be disabled. Get Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
            self.tesseract_available = False
        
        # Default preprocessing parameters
        self.preprocessing_mode = "adaptive"  # Options: "simple", "adaptive", "otsu"
        self.resize_factor = 1.5  # Scale factor for resizing before OCR
        self.blur_kernel = 3     # Blur kernel size for noise reduction
        
        # Create directory for OCR results
        self.results_dir = "ocr_results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    def _preprocess_image(self, image, mode=None):
        """
        Preprocess image for better OCR results.
        
        Parameters:
        - image: Input image
        - mode: Preprocessing mode (overrides instance setting if provided)
        
        Returns:
        - processed_image: Processed image ready for OCR
        """
        if mode is None:
            mode = self.preprocessing_mode
        
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize image to improve OCR (optional)
        gray = cv2.resize(gray, None, fx=self.resize_factor, fy=self.resize_factor, 
                         interpolation=cv2.INTER_CUBIC)
        
        # Apply slight blur to reduce noise
        gray = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        
        # Apply different binarization methods based on mode
        if mode == "simple":
            # Simple thresholding
            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        
        elif mode == "adaptive":
            # Adaptive thresholding (good for varying lighting)
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
        
        elif mode == "otsu":
            # Otsu's thresholding (good for bimodal images)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        else:
            # Default to adaptive
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
        
        # Perform morphological operations to clean up image
        kernel = np.ones((1, 1), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
    
    def extract_text(self, image, lang='eng', preprocess=True):
        """
        Extract text from an image using OCR.
        
        Parameters:
        - image: Input image or image path
        - lang: Language for OCR (default: 'eng')
        - preprocess: Whether to preprocess the image
        
        Returns:
        - text: Extracted text
        """
        if not self.tesseract_available:
            print("Error: Tesseract OCR is not properly configured.")
            return "ERROR: Tesseract OCR not available"
        
        # Load image if path provided
        if isinstance(image, str):
            print(f"Loading image from {image}")
            image = cv2.imread(image)
            if image is None:
                print(f"Error: Could not read image from {image}")
                return "ERROR: Could not read image"
        
        # Preprocess image if requested
        if preprocess:
            print("Preprocessing image to improve OCR results...")
            processed_image = self._preprocess_image(image)
            # Show the processed image
            cv2.namedWindow("OCR Processing", cv2.WINDOW_NORMAL)
            cv2.imshow("OCR Processing", processed_image)
            cv2.waitKey(1000)  # Show for 1 second
            cv2.destroyAllWindows()
        else:
            processed_image = image
        
        # Extract text using Tesseract
        try:
            print(f"Running OCR with Tesseract (language: {lang})...")
            start_time = time.time()
            text = pytesseract.image_to_string(processed_image, lang=lang, config='--psm 1')
            elapsed_time = time.time() - start_time
            print(f"OCR completed in {elapsed_time:.2f} seconds")
            
            if not text.strip():
                print("Warning: No text extracted. Try adjusting preprocessing settings.")
                
            return text.strip()
        except Exception as e:
            print(f"Error during OCR: {str(e)}")
            return f"ERROR: {str(e)}"
    
    def extract_text_with_layout(self, image, lang='eng'):
        """
        Extract text with layout information.
        
        Parameters:
        - image: Input image or image path
        - lang: Language for OCR
        
        Returns:
        - data: Dictionary with text and layout information
        """
        if not self.tesseract_available:
            print("Error: Tesseract OCR is not properly configured.")
            return None
        
        # Load image if path provided
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                print(f"Error: Could not read image from {image}")
                return None
        
        # Preprocess image
        processed_image = self._preprocess_image(image)
        
        # Extract data
        try:
            # Get detailed OCR information
            data = pytesseract.image_to_data(processed_image, lang=lang, output_type=pytesseract.Output.DICT)
            
            # Display results
            n_boxes = len(data['text'])
            overlay = image.copy()
            
            for i in range(n_boxes):
                # Filter out empty text and low-confidence results
                if int(float(data['conf'][i])) > 60 and data['text'][i].strip() != '':
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    
                    # Adjust coordinates for resize factor
                    x = int(x / self.resize_factor)
                    y = int(y / self.resize_factor)
                    w = int(w / self.resize_factor)
                    h = int(h / self.resize_factor)
                    
                    # Draw rectangle for word/line
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Add text
                    cv2.putText(overlay, data['text'][i], (x, y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Combine with original using transparency
            alpha = 0.4
            result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            
            return {
                'text': data['text'],
                'confidence': data['conf'],
                'boxes': list(zip(data['left'], data['top'], data['width'], data['height'])),
                'visualization': result
            }
            
        except Exception as e:
            print(f"Error during structured OCR: {str(e)}")
            return None
    
    def scan_document(self, image=None, real_time_mode=True):
        """
        Scan a document and extract text with visualization.
        
        Parameters:
        - image: Input image or None to capture from webcam
        - real_time_mode: Whether to show real-time OCR results during capture
        
        Returns:
        - extracted_text: The text extracted from the document
        """
        if image is None:
            # Capture from webcam
            image = self._capture_from_webcam()
            if image is None:
                return None
        
        # Check if Tesseract is available
        if not self.tesseract_available:
            return self._process_document_without_ocr(image)
        
        # Process the document with Tesseract
        data = self.extract_text_with_layout(image)
        if data is None:
            return None
        
        # Combine all text
        extracted_text = ' '.join([word for word in data['text'] if word.strip() != ''])
        
        # Display the visualization
        cv2.namedWindow("OCR Results", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("OCR Results", 800, 600)
        
        # Create a display image with text overlay
        result_image = data['visualization'].copy()
        h, w = result_image.shape[:2]
        
        # Add text overlay at the bottom
        text_lines = []
        words = extracted_text.split()
        current_line = ""
        max_width = w - 40
        
        for word in words[:100]:  # Limit to first 100 words for display
            test_line = current_line + " " + word if current_line else word
            # Check if adding this word would make the line too long
            text_size = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            if text_size[0] > max_width:
                text_lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        
        if current_line:  # Add the last line
            text_lines.append(current_line)
        
        # Create text panel at bottom
        text_panel_height = min(len(text_lines) * 30 + 40, 300)
        result_with_text = np.zeros((h + text_panel_height, w, 3), dtype=np.uint8)
        result_with_text[:h, :] = result_image
        result_with_text[h:, :] = (30, 30, 30)  # Dark background for text
        
        # Add header
        cv2.putText(result_with_text, "Extracted Text:", (20, h + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Add text lines
        for i, line in enumerate(text_lines):
            if i >= 8:  # Limit displayed lines
                cv2.putText(result_with_text, "... (more text available in saved file)", 
                           (20, h + 60 + 8 * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
                break
            cv2.putText(result_with_text, line, (20, h + 60 + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show the result
        cv2.imshow("OCR Results", result_with_text)
        
        # Wait for key press
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save results
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(self.results_dir, f"ocr_result_{timestamp}.txt")
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        
        # Save the visualization image
        img_filename = os.path.join(self.results_dir, f"ocr_visual_{timestamp}.jpg")
        cv2.imwrite(img_filename, result_with_text)
        
        print(f"OCR results saved to {filename}")
        print(f"OCR visualization saved to {img_filename}")
        
        return extracted_text
        
    def _process_document_without_ocr(self, image):
        """
        Process document image without OCR capabilities.
        Shows enhanced image and provides instructions to install Tesseract.
        
        Parameters:
        - image: Document image
        
        Returns:
        - message: Information about Tesseract installation
        """
        # Apply basic image processing to enhance text visibility
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Display enhanced image
        cv2.namedWindow("Enhanced Document", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Enhanced Document", 800, 600)
        cv2.imshow("Enhanced Document", binary)
        cv2.waitKey(3000)  # Show for 3 seconds
        cv2.destroyAllWindows()
        
        # Save the enhanced image
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(self.results_dir, f"enhanced_doc_{timestamp}.jpg")
        cv2.imwrite(filename, binary)
        
        message = (
            "OCR processing is unavailable because Tesseract is not installed.\n"
            f"Enhanced document image saved to: {filename}\n\n"
            "To enable full OCR capabilities, please install Tesseract OCR:\n"
            "1. Download from: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "2. Install with default options\n"
            "3. Restart this application\n"
        )
        
        print(message)
        
        # Display the message in a dialog if tkinter is available
        try:
            from tkinter import messagebox, Tk
            root = Tk()
            root.withdraw()
            messagebox.showinfo("OCR Not Available", message)
            root.destroy()
        except:
            pass
            
        return message
    
    def _capture_from_webcam(self):
        """
        Capture document image from webcam with live preview.
        
        Returns:
        - image: Captured image
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return None
        
        # Set moderate resolution for the camera capture (not too large)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Create window with reasonable size
        cv2.namedWindow("Document Scanner", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Document Scanner", 800, 600)
        
        print("Position document in front of camera and press SPACE to capture.")
        print("Press ESC to cancel.")
        
        countdown = 0
        countdown_start = 0
        capturing = False
        last_processed = time.time()
        processing_interval = 0.5  # Process frame for document detection every 0.5 seconds
        last_ocr_time = time.time() - 2.0  # Initialize to trigger OCR on first detection
        ocr_interval = 2.0  # Process OCR every 2 seconds when document is detected
        
        # Initialize variables for document detection
        last_corners = None
        document_detected = False
        current_text = ""
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            display_frame = frame.copy()
            current_time = time.time()
            
            # Process frame for document detection at intervals to improve performance
            if current_time - last_processed > processing_interval and not capturing:
                last_processed = current_time
                document_detected, doc_corners = self._detect_document(frame)
                if document_detected:
                    last_corners = doc_corners
                    
                    # Perform OCR on detected document at intervals
                    if current_time - last_ocr_time > ocr_interval:
                        last_ocr_time = current_time
                        # Extract document and perform quick OCR
                        doc_image = self._get_warped_document(frame, last_corners)
                        if doc_image is not None:
                            try:
                                # Process with quick OCR settings
                                gray = cv2.cvtColor(doc_image, cv2.COLOR_BGR2GRAY)
                                binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                            cv2.THRESH_BINARY, 11, 2)
                                
                                # Use simpler OCR config for speed
                                if self.tesseract_available:
                                    text = pytesseract.image_to_string(binary, config='--psm 1')
                                    if text.strip():
                                        current_text = text.strip()
                            except Exception as e:
                                print(f"Live OCR error: {e}")
            
            # Draw guide overlay
            h, w = frame.shape[:2]
            
            if document_detected and last_corners is not None:
                # Draw detected document outline
                cv2.polylines(display_frame, [last_corners], True, (0, 255, 0), 2)
                cv2.putText(display_frame, "Document Detected!", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display detected text
                if current_text:
                    # Create text background
                    text_y = 60
                    max_width = w - 20
                    
                    # Split text into lines to fit on screen
                    text_lines = []
                    words = current_text.split()
                    current_line = ""
                    
                    for word in words[:50]:  # Limit to first 50 words for display
                        test_line = current_line + " " + word if current_line else word
                        # Check if adding this word would make the line too long
                        text_size = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        if text_size[0] > max_width:
                            text_lines.append(current_line)
                            current_line = word
                        else:
                            current_line = test_line
                    
                    if current_line:  # Add the last line
                        text_lines.append(current_line)
                    
                    # Draw text overlay background
                    overlay_height = min(len(text_lines) * 25 + 10, 200)  # Limit height
                    overlay = display_frame.copy()
                    cv2.rectangle(overlay, (5, text_y - 20), (w - 5, text_y + overlay_height), 
                                (0, 0, 0), -1)
                    alpha = 0.7
                    cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)
                    
                    # Draw text header
                    cv2.putText(display_frame, "Detected Text:", (10, text_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Draw text lines
                    for i, line in enumerate(text_lines[:8]):  # Limit to 8 lines
                        y_pos = text_y + 25 + i * 20
                        cv2.putText(display_frame, line, (15, y_pos),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    if len(text_lines) > 8:
                        cv2.putText(display_frame, "...", (15, text_y + 25 + 8 * 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                # Draw guide rectangle if no document detected
                guide_margin = 50
                guide_color = (0, 255, 0)
                cv2.rectangle(display_frame, 
                             (guide_margin, guide_margin), 
                             (w - guide_margin, h - guide_margin), 
                             guide_color, 2)
            
            # Add instructions
            cv2.putText(display_frame, "Position document inside rectangle", 
                       (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(display_frame, "SPACE to capture, ESC to cancel", 
                       (10, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Handle countdown if active
            if capturing:
                elapsed = time.time() - countdown_start
                remaining = 3 - int(elapsed)
                
                if remaining > 0:
                    # Show countdown
                    cv2.putText(display_frame, f"Capturing in {remaining}...", 
                              (w//2 - 150, h//2), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                else:
                    # Capture the document
                    cv2.putText(display_frame, "CAPTURING!", 
                              (w//2 - 120, h//2), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    
                    # Show final frame
                    cv2.imshow("Document Scanner", display_frame)
                    cv2.waitKey(500)  # Brief pause to show capture message
                    
                    # Extract document area
                    if document_detected and last_corners is not None:
                        # Get perspective transform if we have corners
                        captured_image = self._get_warped_document(frame, last_corners)
                    else:
                        # Extract just the document area using guide rectangle
                        guide_margin = 50
                        captured_image = frame[guide_margin:h-guide_margin, 
                                            guide_margin:w-guide_margin]
                    break
            
            # Show the live preview frame
            cv2.imshow("Document Scanner", display_frame)
            
            # Process key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                captured_image = None
                break
            elif key == 32 and not capturing:  # SPACE
                # Start countdown
                capturing = True
                countdown_start = time.time()
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        if capturing and 'captured_image' in locals() and captured_image is not None:
            # Apply preprocessing to improve OCR
            # Enhance contrast
            gray = cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY)
            enhanced = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
            
            # Show the enhanced version briefly
            cv2.namedWindow("Enhanced Document", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Enhanced Document", 800, 600)
            cv2.imshow("Enhanced Document", enhanced)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
            
            # Save the captured image for reference
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(self.results_dir, f"captured_doc_{timestamp}.jpg")
            cv2.imwrite(filename, captured_image)
            print(f"Document image saved to: {filename}")
            
            return captured_image
        
        return None
    
    def _detect_document(self, image):
        """
        Detect document in image.
        
        Parameters:
        - image: Input image
        
        Returns:
        - detected: Boolean indicating if document was detected
        - corners: Corners of detected document
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply edge detection
        edges = cv2.Canny(blurred, 75, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        document_contour = None
        
        # Find the contour with 4 corners (document)
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            # If contour has 4 corners, it's likely a document
            if len(approx) == 4:
                document_contour = approx
                break
        
        # If document contour found, return corners
        if document_contour is not None:
            return True, document_contour
        
        return False, None
    
    def _get_warped_document(self, image, corners):
        """
        Apply perspective transform to get straightened document.
        
        Parameters:
        - image: Input image
        - corners: Four corners of document
        
        Returns:
        - warped: Perspective-corrected document image
        """
        # Reshape corners to required format
        corners = corners.reshape(4, 2)
        
        # Order points in top-left, top-right, bottom-right, bottom-left order
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Sum of coordinates
        s = corners.sum(axis=1)
        rect[0] = corners[np.argmin(s)]  # Top-left has smallest sum
        rect[2] = corners[np.argmax(s)]  # Bottom-right has largest sum
        
        # Difference of coordinates
        diff = np.diff(corners, axis=1)
        rect[1] = corners[np.argmin(diff)]  # Top-right has smallest difference
        rect[3] = corners[np.argmax(diff)]  # Bottom-left has largest difference
        
        # Calculate width and height of new image
        width_a = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
        width_b = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
        max_width = max(int(width_a), int(width_b))
        
        height_a = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
        height_b = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        
        # Destination points for transform
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)
        
        # Calculate perspective transform matrix
        M = cv2.getPerspectiveTransform(rect, dst)
        
        # Apply transform
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
        
        return warped
    
    def scan_from_file(self):
        """
        Scan a document from a file selected by the user.
        
        Returns:
        - extracted_text: The text extracted from the document
        """
        # Open file dialog
        root = Tk()
        root.withdraw()
        
        file_path = filedialog.askopenfilename(
            title="Select Document Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        root.destroy()
        
        if not file_path:
            print("No file selected")
            return None
        
        # Process the selected file
        return self.scan_document(file_path)
    
    def show_processing_options(self):
        """Show a dialog to adjust processing options."""
        root = Tk()
        root.title("OCR Processing Options")
        root.geometry("400x300")
        
        # Mode selection
        from tkinter import ttk
        
        ttk.Label(root, text="Preprocessing Mode:").pack(pady=10)
        
        mode_var = ttk.StringVar(value=self.preprocessing_mode)
        mode_combo = ttk.Combobox(root, textvariable=mode_var)
        mode_combo['values'] = ("simple", "adaptive", "otsu")
        mode_combo.pack(pady=5)
        
        # Resize factor
        ttk.Label(root, text="Resize Factor:").pack(pady=10)
        
        resize_var = ttk.DoubleVar(value=self.resize_factor)
        resize_scale = ttk.Scale(root, from_=1.0, to=3.0, variable=resize_var, 
                               orient="horizontal", length=200)
        resize_scale.pack(pady=5)
        
        # Blur kernel
        ttk.Label(root, text="Blur Kernel Size:").pack(pady=10)
        
        blur_var = ttk.IntVar(value=self.blur_kernel)
        blur_combo = ttk.Combobox(root, textvariable=blur_var)
        blur_combo['values'] = (1, 3, 5, 7)
        blur_combo.pack(pady=5)
        
        # Save button
        def save_settings():
            self.preprocessing_mode = mode_var.get()
            self.resize_factor = resize_var.get()
            self.blur_kernel = blur_var.get()
            if self.blur_kernel % 2 == 0:  # Ensure odd kernel size
                self.blur_kernel += 1
            messagebox.showinfo("Settings Saved", "OCR processing options updated successfully.")
            root.destroy()
        
        ttk.Button(root, text="Save Settings", command=save_settings).pack(pady=20)
        
        root.mainloop()

    def continuous_ocr_scan(self):
        """
        Run OCR in continuous mode, showing detected text in real-time.
        This mode runs until ESC is pressed, constantly updating text recognition.
        """
        if not self.tesseract_available:
            print("Error: Tesseract OCR is not properly configured.")
            return self._process_document_without_ocr(None)
            
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return None
        
        # Set moderate resolution for the camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Create window with reasonable size
        cv2.namedWindow("Live OCR Scanner", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Live OCR Scanner", 800, 600)
        
        print("Continuous OCR mode active.")
        print("Move document in front of camera to see real-time OCR results.")
        print("Press ESC to exit, SPACE to save current results.")
        
        last_processed = time.time()
        processing_interval = 0.1  # Process document detection more frequently
        ocr_interval = 0.5  # Process OCR more frequently for faster updates
        last_ocr_time = time.time() - ocr_interval
        
        # Initialize variables
        last_corners = None
        document_detected = False
        current_text = "Waiting for text..."
        saved_count = 0
        
        # Always perform OCR even without document detection
        always_ocr = True
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            display_frame = frame.copy()
            current_time = time.time()
            
            # Process frame for document detection at intervals
            if current_time - last_processed > processing_interval:
                last_processed = current_time
                document_detected, doc_corners = self._detect_document(frame)
                
                # If document detected, use it for OCR
                if document_detected:
                    last_corners = doc_corners
                
                # Perform OCR at intervals
                if (current_time - last_ocr_time > ocr_interval and 
                    (document_detected or always_ocr)):
                    last_ocr_time = current_time
                    
                    # Extract document if detected, otherwise use the whole frame
                    if document_detected and last_corners is not None:
                        doc_image = self._get_warped_document(frame, last_corners)
                    else:
                        # Use central portion of the frame
                        h, w = frame.shape[:2]
                        margin = int(w * 0.15)  # 15% margin
                        doc_image = frame[margin:h-margin, margin:w-margin]
                    
                    if doc_image is not None:
                        try:
                            # Process with OCR
                            gray = cv2.cvtColor(doc_image, cv2.COLOR_BGR2GRAY)
                            
                            # Try different preprocessing methods for better results
                            preprocessed = []
                            
                            # Method 1: Adaptive thresholding
                            binary1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                        cv2.THRESH_BINARY, 11, 2)
                            preprocessed.append(binary1)
                            
                            # Method 2: Otsu's thresholding
                            _, binary2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                            preprocessed.append(binary2)
                            
                            # Method 3: Simple thresholding
                            _, binary3 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                            preprocessed.append(binary3)
                            
                            # Try each preprocessing method until we get text
                            detected_text = ""
                            
                            for i, binary in enumerate(preprocessed):
                                if self.tesseract_available:
                                    # Use faster PSM modes for real-time
                                    # PSM 3: Fully automatic page segmentation, but no OSD
                                    # PSM 6: Assume a single uniform block of text
                                    for psm in [3, 6]:
                                        text = pytesseract.image_to_string(binary, config=f'--psm {psm}')
                                        if text.strip():
                                            detected_text = text.strip()
                                            break
                                
                                if detected_text:
                                    break
                            
                            if detected_text:
                                current_text = detected_text
                                # Show which binary and PSM worked
                                print(f"Text detected: {len(current_text)} characters")
                            
                            # Show the processed image in a corner
                            h_small = 150
                            w_small = int(doc_image.shape[1] * h_small / doc_image.shape[0])
                            
                            # Use the binary that worked best
                            best_binary = preprocessed[0]  # Default to first one
                            small_binary = cv2.resize(best_binary, (w_small, h_small))
                            
                            # Place in top-right corner
                            try:
                                display_frame[10:10+h_small, display_frame.shape[1]-w_small-10:display_frame.shape[1]-10] = \
                                    cv2.cvtColor(small_binary, cv2.COLOR_GRAY2BGR)
                            except:
                                pass  # Skip if dimensions don't match
                                
                        except Exception as e:
                            print(f"Live OCR error: {e}")
            
            # Draw guide overlay
            h, w = frame.shape[:2]
            
            if document_detected and last_corners is not None:
                # Draw detected document outline
                cv2.polylines(display_frame, [last_corners], True, (0, 255, 0), 2)
                cv2.putText(display_frame, "Document Detected", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # Draw guide rectangle
                guide_margin = 50
                guide_color = (0, 255, 0)
                cv2.rectangle(display_frame, 
                             (guide_margin, guide_margin), 
                             (w - guide_margin, h - guide_margin), 
                             guide_color, 2)
                cv2.putText(display_frame, "Position document in view", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Always display text area (even when empty)
            # Create text background
            text_y = 60
            text_panel_height = 300  # Larger height for text panel
            
            # Create semi-transparent overlay for text
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (5, text_y - 20), (w - 5, text_y + text_panel_height), 
                        (0, 0, 0), -1)
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)
            
            # Draw text header
            cv2.putText(display_frame, "Detected Text:", (10, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Always show text (even if it's "Waiting for text...")
            # Split text into lines to fit on screen
            text_lines = []
            words = current_text.split()
            current_line = ""
            max_width = w - 20
            
            for word in words[:150]:  # Increased limit to 150 words
                test_line = current_line + " " + word if current_line else word
                # Check if adding this word would make the line too long
                text_size = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                if text_size[0] > max_width:
                    text_lines.append(current_line)
                    current_line = word
                else:
                    current_line = test_line
            
            if current_line:  # Add the last line
                text_lines.append(current_line)
            
            # Draw text lines
            for i, line in enumerate(text_lines[:12]):  # Increased to 12 lines
                y_pos = text_y + 30 + i * 22
                cv2.putText(display_frame, line, (15, y_pos),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            if len(text_lines) > 12:
                cv2.putText(display_frame, "...", (15, text_y + 30 + 12 * 22),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Add instructions at bottom
            cv2.putText(display_frame, "ESC: Exit | SPACE: Save current text", 
                       (10, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Show the frame
            cv2.imshow("Live OCR Scanner", display_frame)
            
            # Process key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32 and current_text and current_text != "Waiting for text...":  # SPACE - save current results
                # Save the text
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = os.path.join(self.results_dir, f"live_ocr_{timestamp}.txt")
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(current_text)
                
                # Also save the current image
                if document_detected and last_corners is not None:
                    doc_image = self._get_warped_document(frame, last_corners)
                    if doc_image is not None:
                        img_filename = os.path.join(self.results_dir, f"live_ocr_img_{timestamp}.jpg")
                        cv2.imwrite(img_filename, doc_image)
                
                saved_count += 1
                print(f"Saved OCR text to {filename} (#{saved_count})")
                
                # Show brief confirmation
                confirm_overlay = display_frame.copy()
                cv2.rectangle(confirm_overlay, (w//4, h//2-30), (3*w//4, h//2+30), (0, 0, 0), -1)
                cv2.putText(confirm_overlay, f"Text saved! (#{saved_count})", 
                           (w//4 + 20, h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Live OCR Scanner", confirm_overlay)
                cv2.waitKey(500)  # Show confirmation for 500ms
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        return current_text

if __name__ == "__main__":
    # Demo usage
    scanner = OCRScanner()
    
    # Print welcome message
    print("OCR Document Scanner Demo")
    print("=========================")
    print("1. Scan document from webcam")
    print("2. Scan document from file")
    print("3. Adjust processing options")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ")
    
    if choice == '1':
        text = scanner.scan_document()
        if text:
            print("\nExtracted Text:")
            print("--------------")
            print(text)
    elif choice == '2':
        text = scanner.scan_from_file()
        if text:
            print("\nExtracted Text:")
            print("--------------")
            print(text)
    elif choice == '3':
        scanner.show_processing_options()
    elif choice == '4':
        print("Exiting...")
    else:
        print("Invalid choice. Exiting...") 