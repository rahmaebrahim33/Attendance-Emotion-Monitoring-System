import cv2
import pytesseract
import os
import tkinter as tk
from tkinter import messagebox, filedialog, Button, Label, Frame, TOP, BOTTOM, LEFT, RIGHT, X, Y, BOTH
import numpy as np
import time
from PIL import Image, ImageTk
import threading

class StandaloneOCR:
    def __init__(self):
        # Set Tesseract path
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Create directories
        self.results_dir = "ocr_results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        
        # Check if Tesseract is available
        try:
            pytesseract.get_tesseract_version()
            print("Tesseract OCR initialized successfully")
            self.tesseract_available = True
        except Exception as e:
            print(f"Error initializing Tesseract: {e}")
            self.tesseract_available = False
            
        # Live mode variables
        self.live_mode_active = False
        self.live_thread = None
        self.last_ocr_text = ""
        self.ocr_cooldown = 1.0  # seconds between OCR attempts in live mode
        self.last_ocr_time = 0
    
    def capture_from_webcam(self):
        """Capture document from webcam"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return None
        
        # Set high resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Create window
        cv2.namedWindow("Document Scanner", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Document Scanner", 1280, 720)
        
        print("Position document in front of camera and press SPACE to capture")
        print("Press ESC to cancel")
        
        # Add countdown timer variables
        countdown_active = False
        countdown_start = 0
        
        # Add edge detection variables for better guidance
        edges_visible = True
        
        # Initialize captured_frame as None
        captured_frame = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Create a copy for display
            display_frame = frame.copy()
            
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Edge detection to help position document
            if edges_visible:
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blurred, 75, 200)
                
                # Overlay edges on display frame in red
                display_frame[edges > 0] = (0, 0, 255)  # Simpler approach to overlay edges
            
            # Draw guide rectangle for document placement
            h, w = frame.shape[:2]
            margin = 100  # Increased margin for better document visibility
            cv2.rectangle(display_frame, (margin, margin), (w-margin, h-margin), (0, 255, 0), 2)
            
            # Add guidelines for better alignment
            # Horizontal center line
            cv2.line(display_frame, (margin, h//2), (w-margin, h//2), (0, 255, 0), 1, cv2.LINE_AA)
            # Vertical center line
            cv2.line(display_frame, (w//2, margin), (w//2, h-margin), (0, 255, 0), 1, cv2.LINE_AA)
            
            # Add instructions
            cv2.putText(display_frame, "Position document inside rectangle", 
                       (margin, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display_frame, "Align with center guidelines", 
                       (margin, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display_frame, "SPACE to capture, ESC to cancel, E to toggle edge detection", 
                       (margin, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Handle countdown if active
            if countdown_active:
                elapsed = time.time() - countdown_start
                remaining = 3 - int(elapsed)  # 3 second countdown
                
                if remaining > 0:
                    # Display countdown
                    cv2.putText(display_frame, f"Capturing in {remaining}...", 
                              (w//2 - 150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)
                else:
                    # Display "CAPTURING" message
                    cv2.putText(display_frame, "CAPTURING!", 
                              (w//2 - 150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)
                    
                    # Show the final frame with "CAPTURING" message
                    cv2.imshow("Document Scanner", display_frame)
                    cv2.waitKey(500)  # Brief pause to show capturing message
                    
                    # Take the best frame
                    captured_frame = frame[margin:h-margin, margin:w-margin]
                    break
            
            # Show frame
            cv2.imshow("Document Scanner", display_frame)
            
            # Process key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                captured_frame = None
                break
            elif key == 32 and not countdown_active:  # SPACE
                # Start countdown
                countdown_active = True
                countdown_start = time.time()
            elif key == ord('e'):  # Toggle edge detection
                edges_visible = not edges_visible
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        return captured_frame
    
    def process_image(self, image):
        """Process image for better OCR results"""
        if image is None:
            print("Error: Cannot process None image")
            return None
            
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Increase contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Apply noise reduction
            denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
            
            # Try different thresholding methods
            # 1. Adaptive thresholding
            thresh1 = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
            
            # 2. Otsu's thresholding
            _, thresh2 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Save the processed images for reference
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            cv2.imwrite(os.path.join(self.results_dir, f"original_{timestamp}.jpg"), image)
            cv2.imwrite(os.path.join(self.results_dir, f"processed_adaptive_{timestamp}.jpg"), thresh1)
            cv2.imwrite(os.path.join(self.results_dir, f"processed_otsu_{timestamp}.jpg"), thresh2)
            
            try:
                # Display the processing steps
                # Ensure all images are valid and properly formatted
                if image is not None and gray is not None and thresh1 is not None and thresh2 is not None:
                    # Create a stack of images
                    h, w = image.shape[:2]
                    h_new = h // 2
                    w_new = w // 2
                    
                    # Ensure minimum size
                    h_new = max(h_new, 200)
                    w_new = max(w_new, 200)
                    
                    # Resize images for display
                    img_resized = cv2.resize(image, (w_new, h_new))
                    gray_resized = cv2.resize(gray, (w_new, h_new))
                    thresh1_resized = cv2.resize(thresh1, (w_new, h_new))
                    thresh2_resized = cv2.resize(thresh2, (w_new, h_new))
                    
                    # Convert grayscale to BGR for stacking
                    gray_bgr = cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2BGR)
                    thresh1_bgr = cv2.cvtColor(thresh1_resized, cv2.COLOR_GRAY2BGR)
                    thresh2_bgr = cv2.cvtColor(thresh2_resized, cv2.COLOR_GRAY2BGR)
                    
                    # Create display layout (2x2 grid)
                    top_row = np.hstack((img_resized, gray_bgr))
                    bottom_row = np.hstack((thresh1_bgr, thresh2_bgr))
                    
                    # Check that the arrays can be stacked
                    if top_row.shape == bottom_row.shape:
                        display = np.vstack((top_row, bottom_row))
                        
                        # Add labels
                        cv2.putText(display, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(display, "Grayscale", (w_new + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(display, "Adaptive Threshold", (10, h_new + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(display, "Otsu Threshold", (w_new + 10, h_new + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        # Create window
                        cv2.namedWindow("Processing Steps", cv2.WINDOW_NORMAL)
                        cv2.resizeWindow("Processing Steps", 1280, 720)
                        
                        # Show the display
                        cv2.imshow("Processing Steps", display)
                        cv2.waitKey(2000)  # Show for 2 seconds
                        cv2.destroyAllWindows()
                    else:
                        print("Warning: Could not create processing display due to mismatched shapes")
            except Exception as e:
                print(f"Warning: Could not display processing steps: {e}")
                # Continue with processing even if display fails
            
            # For better OCR results, use the adaptive thresholding
            return thresh1
            
        except Exception as e:
            print(f"Error during image processing: {e}")
            return None
    
    def perform_ocr(self, image, lang='eng'):
        """Extract text from image using OCR"""
        if not self.tesseract_available:
            print("Tesseract OCR is not available")
            return None
            
        if image is None:
            print("Error: Cannot perform OCR on None image")
            return None
        
        # Process the image
        processed_image = self.process_image(image)
        
        if processed_image is None:
            print("Error: Image processing failed")
            return None
        
        # Ensure image is in the right format for pytesseract
        # Convert to PIL Image format which is compatible with pytesseract
        pil_image = Image.fromarray(processed_image)
        
        # Define custom Tesseract configuration
        custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
        # OEM modes:
        # 0: Legacy engine only
        # 1: Neural nets LSTM engine only
        # 2: Legacy + LSTM engines
        # 3: Default, based on what is available
        
        # PSM modes:
        # 0: Orientation and script detection (OSD) only
        # 1: Automatic page segmentation with OSD
        # 3: Fully automatic page segmentation, but no OSD (default)
        # 4: Assume a single column of text of variable sizes
        # 6: Assume a single uniform block of text
        # 7: Treat the image as a single text line
        # 10: Treat the image as a single character
        
        # First try with the default page segmentation mode (6 - uniform block of text)
        print("Running OCR with mode 6...")
        start_time = time.time()
        text = pytesseract.image_to_string(pil_image, lang=lang, config=custom_config)
        elapsed_time = time.time() - start_time
        print(f"OCR completed in {elapsed_time:.2f} seconds")
        
        # If no text found, try another page segmentation mode (3 - fully automatic)
        if not text.strip():
            print("No text found. Trying with mode 3...")
            start_time = time.time()
            custom_config = r'--oem 3 --psm 3'
            text = pytesseract.image_to_string(pil_image, lang=lang, config=custom_config)
            elapsed_time = time.time() - start_time
            print(f"OCR (mode 3) completed in {elapsed_time:.2f} seconds")
        
        # If still no text, try with whitelist of common characters
        if not text.strip():
            print("Still no text. Trying with character whitelist...")
            start_time = time.time()
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:\'\"()-+/ "'
            text = pytesseract.image_to_string(pil_image, lang=lang, config=custom_config)
            elapsed_time = time.time() - start_time
            print(f"OCR (with whitelist) completed in {elapsed_time:.2f} seconds")
        
        return text
    
    def perform_quick_ocr(self, image, lang='eng'):
        """Extract text from image using OCR with faster settings for live mode"""
        if not self.tesseract_available or image is None:
            return None
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Increase contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Apply adaptive thresholding (faster than multiple methods)
            thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(thresh)
            
            # Use faster OCR settings
            custom_config = r'--oem 3 --psm 6 -l eng'
            text = pytesseract.image_to_string(pil_image, lang=lang, config=custom_config)
            
            return text
        except Exception as e:
            print(f"Error in quick OCR: {e}")
            return None
    
    def save_results(self, text):
        """Save OCR results to file"""
        if text is None or text.strip() == "":
            print("Warning: No text to save")
            text = ""  # Ensure text is at least an empty string
            
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(self.results_dir, f"ocr_result_{timestamp}.txt")
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Results saved to {filename}")
        return filename
    
    def run_live_ocr(self):
        """Run OCR in live mode with continuous webcam feed and real-time results"""
        # Create the main window
        root = tk.Tk()
        root.title("Live OCR")
        root.geometry("1600x900")
        root.configure(bg="#2c3e50")
        
        # Create frames
        top_frame = Frame(root, bg="#2c3e50")
        top_frame.pack(side=TOP, fill=X, expand=False, padx=10, pady=10)
        
        # Create a horizontal layout for video and text
        content_frame = Frame(root, bg="#2c3e50")
        content_frame.pack(side=TOP, fill=BOTH, expand=True, padx=10, pady=10)
        
        # Video frame on the left
        video_frame = Frame(content_frame, bg="#34495e", width=800, height=600)
        video_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=5, pady=5)
        video_frame.pack_propagate(False)  # Prevent frame from shrinking
        
        # Text frame on the right
        text_frame = Frame(content_frame, bg="#34495e", width=600)
        text_frame.pack(side=RIGHT, fill=BOTH, expand=True, padx=5, pady=5)
        
        # Add controls to top frame
        title_label = Label(top_frame, text="Live OCR Scanner", font=("Arial", 18, "bold"), bg="#2c3e50", fg="white")
        title_label.pack(side=LEFT, padx=10)
        
        start_button = Button(top_frame, text="Start", font=("Arial", 12), bg="#27ae60", fg="white", 
                             command=lambda: self.toggle_live_mode(True, text_output, root))
        start_button.pack(side=LEFT, padx=5)
        
        stop_button = Button(top_frame, text="Stop", font=("Arial", 12), bg="#c0392b", fg="white",
                            command=lambda: self.toggle_live_mode(False, None, None))
        stop_button.pack(side=LEFT, padx=5)
        
        save_button = Button(top_frame, text="Save Text", font=("Arial", 12), bg="#2980b9", fg="white",
                            command=lambda: self.save_live_text(text_output.get("1.0", "end-1c")))
        save_button.pack(side=LEFT, padx=5)
        
        exit_button = Button(top_frame, text="Exit", font=("Arial", 12), bg="#7f8c8d", fg="white",
                            command=root.destroy)
        exit_button.pack(side=RIGHT, padx=10)
        
        # Add video label to video frame
        video_label = Label(video_frame, text="Camera Feed", font=("Arial", 14), bg="#34495e", fg="white")
        video_label.pack(anchor="nw", padx=10, pady=5)
        
        # Panel for video display
        self.panel = Label(video_frame)
        self.panel.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Add text output to text frame
        text_label = Label(text_frame, text="Live OCR Results:", font=("Arial", 14), bg="#34495e", fg="white")
        text_label.pack(anchor="nw", padx=10, pady=5)
        
        # Create a frame for the text output with a scrollbar
        text_container = Frame(text_frame, bg="#34495e")
        text_container.pack(fill=BOTH, expand=True, padx=10, pady=5)
        
        # Add scrollbar
        scrollbar = tk.Scrollbar(text_container)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        # Text widget with scrollbar
        text_output = tk.Text(text_container, height=20, width=50, font=("Courier", 12), 
                             bg="#ecf0f1", fg="#2c3e50", wrap=tk.WORD,
                             yscrollcommand=scrollbar.set)
        text_output.pack(fill=BOTH, expand=True)
        scrollbar.config(command=text_output.yview)
        
        # Status bar at the bottom
        status_frame = Frame(root, bg="#34495e", height=25)
        status_frame.pack(side=BOTTOM, fill=X, padx=10, pady=5)
        
        self.status_label = Label(status_frame, text="Ready. Press Start to begin OCR.", 
                                 bg="#34495e", fg="white", anchor="w")
        self.status_label.pack(fill=X, padx=10)
        
        # Store references to UI elements
        self.video_frame = video_frame
        self.text_output = text_output
        self.root = root
        
        # Set OCR cooldown to be more responsive
        self.ocr_cooldown = 0.5  # Process every 0.5 seconds
        
        # Start the main loop
        root.protocol("WM_DELETE_WINDOW", lambda: self.on_closing(root))
        root.mainloop()
    
    def update_video(self):
        """Update the video feed for live OCR mode"""
        if self.live_mode_active and hasattr(self, 'cap') and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Create a copy for display
                display_frame = frame.copy()
                
                # Draw guide rectangle for document placement
                h, w = frame.shape[:2]
                margin = 100
                cv2.rectangle(display_frame, (margin, margin), (w-margin, h-margin), (0, 255, 0), 2)
                
                # Add center guidelines
                cv2.line(display_frame, (margin, h//2), (w-margin, h//2), (0, 255, 0), 1)
                cv2.line(display_frame, (w//2, margin), (w//2, h-margin), (0, 255, 0), 1)
                
                # Add instructions
                cv2.putText(display_frame, "Position document inside rectangle", 
                          (margin, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Convert to RGB for tkinter
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
                # Get video frame dimensions
                vf_width = self.video_frame.winfo_width() - 20  # Account for padding
                vf_height = self.video_frame.winfo_height() - 60  # Account for label and padding
                
                # Ensure minimum dimensions
                vf_width = max(vf_width, 640)
                vf_height = max(vf_height, 480)
                
                # Resize to fit the window
                rgb_frame = cv2.resize(rgb_frame, (vf_width, vf_height))
                
                # Convert to PIL Image
                img = Image.fromarray(rgb_frame)
                
                # Update the panel
                self.photo = ImageTk.PhotoImage(image=img)
                self.panel.configure(image=self.photo)
                self.panel.image = self.photo
                
                # Perform OCR on a schedule
                current_time = time.time()
                if current_time - self.last_ocr_time > self.ocr_cooldown:
                    # Extract document region
                    doc_region = frame[margin:h-margin, margin:w-margin]
                    
                    # Run OCR in a separate thread to avoid freezing UI
                    if not hasattr(self, 'ocr_thread') or not self.ocr_thread.is_alive():
                        self.ocr_thread = threading.Thread(
                            target=self.process_frame_for_ocr, 
                            args=(doc_region, self.text_output)
                        )
                        self.ocr_thread.daemon = True
                        self.ocr_thread.start()
                        self.last_ocr_time = current_time
                        
                        # Update status
                        self.status_label.config(text=f"Processing OCR... Last update: {time.strftime('%H:%M:%S')}")
            
            # Schedule the next update
            if hasattr(self, 'root') and self.live_mode_active:
                self.root.after(30, self.update_video)
        else:
            # Camera not available
            if hasattr(self, 'status_label'):
                self.status_label.config(text="Camera not available. Please restart the application.")
    
    def toggle_live_mode(self, start, text_output, root):
        """Toggle live OCR mode on/off"""
        if start and not self.live_mode_active:
            # Start live mode
            self.live_mode_active = True
            self.cap = cv2.VideoCapture(0)
            
            # Set high resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam")
                self.live_mode_active = False
                return
            
            # Clear text output
            if text_output:
                text_output.delete("1.0", "end")
            
            # Reset OCR text
            self.last_ocr_text = ""
            
            # Update status
            if hasattr(self, 'status_label'):
                self.status_label.config(text="Live OCR active. Position document in the green rectangle.")
            
            # Start video update
            if root:
                root.after(30, self.update_video)
        
        elif not start and self.live_mode_active:
            # Stop live mode
            self.live_mode_active = False
            if hasattr(self, 'cap'):
                self.cap.release()
                
            # Update status
            if hasattr(self, 'status_label'):
                self.status_label.config(text="OCR stopped. Press Start to begin again.")
    
    def process_frame_for_ocr(self, frame, text_output):
        """Process a frame for OCR and update the text output"""
        if frame is None or frame.size == 0:
            return
        
        # Perform quick OCR
        text = self.perform_quick_ocr(frame)
        
        if text and text.strip():
            # Update the text output in the main thread
            text_output.master.after(0, lambda: self.update_text_output(text_output, text))
    
    def update_text_output(self, text_output, text):
        """Update the text output widget with new OCR results"""
        if text and text.strip():
            # Only update if text has changed
            if text.strip() != self.last_ocr_text:
                # Clear previous text
                text_output.delete("1.0", "end")
                
                # Insert new text
                text_output.insert("1.0", text)
                
                # Store the last OCR text
                self.last_ocr_text = text.strip()
                
                # Update status
                if hasattr(self, 'status_label'):
                    self.status_label.config(text=f"OCR updated at {time.strftime('%H:%M:%S')}")
    
    def save_live_text(self, text):
        """Save the current text from live OCR"""
        if text and text.strip():
            filename = self.save_results(text)
            messagebox.showinfo("Text Saved", f"Text saved to {filename}")
            
            # Update status
            if hasattr(self, 'status_label'):
                self.status_label.config(text=f"Text saved to {os.path.basename(filename)}")
        else:
            messagebox.showwarning("No Text", "No text to save")
            
            # Update status
            if hasattr(self, 'status_label'):
                self.status_label.config(text="No text to save. Position a document in the frame.")
    
    def on_closing(self, root):
        """Handle window closing"""
        self.live_mode_active = False
        if hasattr(self, 'cap'):
            self.cap.release()
        root.destroy()
    
    def run(self):
        """Run the OCR application"""
        # Ask user if they want to use standard mode or live mode
        root = tk.Tk()
        root.withdraw()
        choice = messagebox.askyesno("OCR Mode", "Do you want to run in live mode?\n\nYes: Live OCR mode\nNo: Standard OCR mode")
        root.destroy()
        
        if choice:  # Live mode
            self.run_live_ocr()
            return
        
        # Standard mode
        root = tk.Tk()
        root.withdraw()
        choice = messagebox.askyesno("OCR Method", "Do you want to capture from webcam?\n\nYes: Use webcam\nNo: Open image file")
        root.destroy()
        
        if choice:  # Webcam
            # Capture image from webcam
            image = self.capture_from_webcam()
            
            if image is None:
                print("No image captured")
                root = tk.Tk()
                root.withdraw()
                messagebox.showerror("OCR Error", "No image was captured. Please try again.")
                root.destroy()
                return
                
            # Save the captured image
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            image_filename = os.path.join(self.results_dir, f"captured_image_{timestamp}.jpg")
            cv2.imwrite(image_filename, image)
            print(f"Image saved to {image_filename}")
        else:  # File
            # Open file dialog
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(
                title="Select an image file",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                    ("All files", "*.*")
                ]
            )
            root.destroy()
            
            if not file_path:
                print("No file selected")
                return
                
            # Load the image
            print(f"Loading image from: {file_path}")
            image = cv2.imread(file_path)
            
            if image is None:
                print(f"Error: Could not load image from {file_path}")
                root = tk.Tk()
                root.withdraw()
                messagebox.showerror("OCR Error", f"Could not load image from {file_path}. The file may be corrupted or in an unsupported format.")
                root.destroy()
                return
            
            image_filename = file_path
        
        # Perform OCR
        text = self.perform_ocr(image)
        
        if text:
            # Save results
            result_file = self.save_results(text)
            
            # Show results
            print("\nExtracted Text:")
            print("---------------")
            print(text)
            
            # Show in a message box
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo("OCR Results", f"Text extracted:\n\n{text}\n\nSaved to: {result_file}")
            root.destroy()
        else:
            print("No text extracted")
            
            root = tk.Tk()
            root.withdraw()
            messagebox.showwarning("OCR Results", "No text could be extracted from the image.")
            root.destroy()

if __name__ == "__main__":
    print("Standalone OCR Application")
    print("=========================")
    
    ocr = StandaloneOCR()
    ocr.run() 