import cv2
import numpy as np
import os
import pygame
import time
import datetime
import pandas as pd
from deepface import DeepFace
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, Frame, Label, Button, Scale, HORIZONTAL
# Import custom modules
from Attendance_Emotion import AttendanceEmotionSystem
from Camera_Calibration import CameraCalibration
from Face_Emotion_Detection import FaceEmotionDetection
from Live_UI_Feedback import LiveUIFeedback
from Hand_Gesture import HandGestureDetector
from OCR_Scanner import OCRScanner

class VisionBasedAttendanceSystem:
    def __init__(self):
        """Initialize the integrated vision-based attendance and emotion monitoring system."""
        # Create necessary directories
        self.dirs = {
            "known_faces": "known_faces",
            "calibration": "camera_calibration",
            "attendance": "attendance_data",
            "snapshots": "snapshots",
            "sounds": "sounds"
        }
        
        for dir_path in self.dirs.values():
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"Created directory: {dir_path}")
        
        # Initialize component modules
        self.attendance_system = AttendanceEmotionSystem(
            known_faces_dir=self.dirs["known_faces"],
            attendance_file=os.path.join(self.dirs["attendance"], "attendance_log.csv")
        )
        
        self.camera_calibration = CameraCalibration(
            calibration_dir=self.dirs["calibration"]
        )
        
        self.face_detection = FaceEmotionDetection()
        
        self.ui_feedback = LiveUIFeedback(
            sound_dir=self.dirs["sounds"]
        )
        
        self.hand_gesture = HandGestureDetector()
        
        self.ocr_scanner = OCRScanner()
        
        # System state
        self.is_running = False
        self.recognized_users = set()
        self.emotions_data = {}
        
        # Initialize controller for gesture commands
        self.command_controller = {
            "volume_up": self._handle_volume_up,
            "volume_down": self._handle_volume_down,
            "next_page": self._handle_next_page,
            "play_pause": self._handle_play_pause,
            "mark_attendance": self._handle_mark_attendance
        }
        
        # System settings
        self.enable_face_recognition = True
        self.enable_emotion_detection = True
        self.enable_hand_gestures = True
        self.enable_audio_feedback = True
        
        print("Vision-Based Attendance and Emotion Monitoring System initialized.")
        
        # Print status of optional components
        if not self.ocr_scanner.tesseract_available:
            print("\nNOTE: OCR functionality is disabled. Install Tesseract for document scanning capabilities.")
            print("This is optional and the system will work without it.")
            print("Download from: https://github.com/UB-Mannheim/tesseract/wiki\n")
    
    # Command handlers for gestures
    def _handle_volume_up(self):
        print("Volume Up command received")
        # Implementation would go here
        
    def _handle_volume_down(self):
        print("Volume Down command received")
        # Implementation would go here
        
    def _handle_next_page(self):
        print("Next Page command received")
        # Implementation would go here
        
    def _handle_play_pause(self):
        print("Play/Pause command received")
        # Implementation would go here
        
    def _handle_mark_attendance(self):
        print("Mark Attendance command received")
        # Implementation would capture current attendance snapshot
        
    def run_system(self, duration=None):
        """
        Run the complete attendance and emotion monitoring system.
        
        Parameters:
        - duration: Duration in seconds (None for continuous operation until 'q' is pressed)
        """
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set higher resolution for the camera capture
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Get initial frame to determine dimensions
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read from webcam")
            cap.release()
            return
        
        # Apply camera calibration if available
        if self.camera_calibration.is_calibrated:
            print("Using calibrated camera parameters")
        
        # Create window for display
        cv2.namedWindow("Attendance & Emotion Monitoring System", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Attendance & Emotion Monitoring System", 1280, 960)
        
        start_time = time.time()
        frame_count = 0
        processing_times = []
        
        print("Starting attendance and emotion monitoring...")
        print("Press 'q' to quit, 's' to save snapshot, 'g' to toggle gesture recognition")
        
        self.is_running = True
        last_attendance_check = time.time()
        attendance_check_interval = 5  # Check for new faces every 5 seconds
        
        # For UI purposes
        show_gestures = True
        show_debug_info = True
        
        while self.is_running:
            # Check if duration limit reached
            if duration is not None and time.time() - start_time > duration:
                break
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Start timing frame processing
            frame_process_start = time.time()
            
            # Apply camera calibration if available
            if self.camera_calibration.is_calibrated:
                frame = self.camera_calibration.undistort_image(frame)
            
            # Apply image optimization
            frame = self.camera_calibration.apply_optimizations(frame)
            
            # Create a copy for display
            display_frame = frame.copy()
            
            # Detect faces and emotions if enabled
            faces_data = []
            if self.enable_face_recognition:
                processed_frame, face_results = self.face_detection.detect_faces_emotions(frame)
                display_frame = processed_frame
                
                # Prepare data for attendance system and UI
                for face_result in face_results:
                    x, y, w, h = face_result['location']
                    emotion = face_result['emotion']
                    
                    # Attempt to recognize the face
                    face_img = frame[y:y+h, x:x+w]
                    name, _, _ = self.attendance_system.recognize_face(frame)
                    
                    if name is None:
                        name = "Unknown"
                    
                    face_data = {
                        'location': (x, y, w, h),
                        'name': name,
                        'emotion': emotion,
                        'emotion_scores': face_result['emotion_scores']
                    }
                    
                    faces_data.append(face_data)
                    
                    # Update emotions data for statistics
                    if name != "Unknown":
                        if name not in self.emotions_data:
                            self.emotions_data[name] = {}
                        
                        if emotion in self.emotions_data[name]:
                            self.emotions_data[name][emotion] += 1
                        else:
                            self.emotions_data[name][emotion] = 1
                    
                    # Mark attendance if not already marked and not unknown
                    current_time = time.time()
                    if (current_time - last_attendance_check >= attendance_check_interval and 
                        name != "Unknown" and name not in self.recognized_users):
                        success, message = self.attendance_system.mark_attendance(name, emotion)
                        if success:
                            self.recognized_users.add(name)
                            self.ui_feedback.play_sound("attendance_marked")
                            print(f"Marked attendance for {name}")
            
            # Update attendance check time if needed
            current_time = time.time()
            if current_time - last_attendance_check >= attendance_check_interval:
                last_attendance_check = current_time
            
            # Process hand gestures if enabled
            gesture_command = None
            if self.enable_hand_gestures and show_gestures:
                gesture_frame, gesture, command = self.hand_gesture.process_frame(display_frame)
                display_frame = gesture_frame
                
                # Execute command if received
                if command:
                    gesture_command = command
                    if command in self.command_controller:
                        self.command_controller[command]()
            
            # Apply UI feedback
            display_frame = self.ui_feedback.process_frame(display_frame, faces_data)
            
            # Calculate FPS and processing time
            frame_process_time = time.time() - frame_process_start
            processing_times.append(frame_process_time)
            
            # Limit to last 30 frames for average calculation
            if len(processing_times) > 30:
                processing_times.pop(0)
            
            avg_process_time = sum(processing_times) / len(processing_times)
            fps = 1.0 / avg_process_time if avg_process_time > 0 else 0
            
            # Add FPS display
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add system mode indicators
            mode_text = []
            if self.enable_face_recognition:
                mode_text.append("Face Recognition: ON")
            if self.enable_emotion_detection:
                mode_text.append("Emotion Detection: ON")
            if self.enable_hand_gestures and show_gestures:
                mode_text.append("Gesture Control: ON")
            
            # Display mode indicators
            if show_debug_info:
                for i, text in enumerate(mode_text):
                    cv2.putText(display_frame, text, (10, 60 + i * 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow("Attendance & Emotion Monitoring System", display_frame)
            
            # Process key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save snapshot
                self.ui_feedback.save_snapshot(display_frame, faces_data, self.dirs["snapshots"])
            elif key == ord('g'):
                # Toggle gesture recognition
                show_gestures = not show_gestures
                print(f"Gesture recognition: {'ON' if show_gestures else 'OFF'}")
            elif key == ord('d'):
                # Toggle debug info
                show_debug_info = not show_debug_info
            elif key == ord('o'):
                # Launch continuous OCR by default (more useful)
                self.run_continuous_ocr()
            elif key == ord('m'):
                # Manual OCR scanner (original mode)
                self.run_ocr_scanner()
                
            # Display keyboard shortcuts on frame if debug info is enabled
            if show_debug_info:
                shortcut_text = [
                    "Press 'q' to quit",
                    "Press 's' to save snapshot",
                    "Press 'g' to toggle gestures",
                    "Press 'd' to toggle debug info",
                    "Press 'o' for Live OCR (recommended)",
                    "Press 'm' for manual OCR scan"
                ]
                
                # Calculate base y position (bottom of frame with margin)
                base_y = display_frame.shape[0] - 150
                
                for i, text in enumerate(shortcut_text):
                    y_pos = base_y + i * 25
                    cv2.putText(display_frame, text, (10, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(display_frame, text, (10, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.is_running = False
        
        # Show summary
        attendance_summary = f"Session Summary:\n"
        attendance_summary += f"Total recognized users: {len(self.recognized_users)}\n"
        attendance_summary += f"Users: {', '.join(self.recognized_users) if self.recognized_users else 'None'}"
        
        print("\n" + attendance_summary)
        
        # Generate and show emotion summary
        if self.emotions_data:
            self.generate_emotion_summary()
        
        # Show in message box
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Attendance Summary", attendance_summary)
        root.destroy()
    
    def run_ocr_scanner(self):
        """Run the OCR scanner in a separate process."""
        print("Starting OCR scanner...")
        # Pause the main system
        self.is_running = False
        
        # Show message to user
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("OCR Scanner", "Starting document scanner. Position document in front of camera and press SPACE to capture, or ESC to cancel.")
        root.destroy()
        
        # Run OCR scanner
        extracted_text = self.ocr_scanner.scan_document()
        
        if extracted_text:
            print("OCR Result:")
            print(extracted_text)
            
            # Display results to user
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo("OCR Results", f"Text extracted:\n\n{extracted_text[:500]}...")
            root.destroy()
            
            # Save results to file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join("ocr_results", f"ocr_result_{timestamp}.txt")
            
            # Create directory if it doesn't exist
            os.makedirs("ocr_results", exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            
            print(f"OCR results saved to {filename}")
        else:
            print("OCR scanning cancelled or no text detected")
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo("OCR Scanner", "OCR scanning cancelled or no text detected.")
            root.destroy()
        
        # Resume the main system
        self.is_running = True
    
    def run_continuous_ocr(self):
        """Run the OCR scanner in continuous mode that updates in real-time."""
        print("Starting continuous OCR scanner...")
        # Pause the main system
        self.is_running = False
        
        # Show message to user
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Continuous OCR", "Starting live OCR mode. Text will be recognized automatically as you show documents to the camera.\n\nPress ESC to exit, SPACE to save current results.")
        root.destroy()
        
        # Run continuous OCR scanner
        self.ocr_scanner.continuous_ocr_scan()
        
        # Resume the main system
        self.is_running = True
    
    def generate_emotion_summary(self):
        """Generate and display a summary of emotions."""
        if not self.emotions_data:
            print("No emotion data available")
            return
        
        print("\nEmotion Summary:")
        print("="*50)
        
        # Overall emotion counts
        all_emotions = {}
        for user, emotions in self.emotions_data.items():
            for emotion, count in emotions.items():
                if emotion in all_emotions:
                    all_emotions[emotion] += count
                else:
                    all_emotions[emotion] = count
        
        # Print overall stats
        print("Overall emotion distribution:")
        total = sum(all_emotions.values())
        for emotion, count in sorted(all_emotions.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total) * 100
            print(f"  - {emotion}: {count} ({percentage:.1f}%)")
        
        print("\nPer-user emotion distribution:")
        for user, emotions in self.emotions_data.items():
            print(f"{user}:")
            user_total = sum(emotions.values())
            for emotion, count in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / user_total) * 100
                print(f"  - {emotion}: {count} ({percentage:.1f}%)")
        
        # We could add visualization here using matplotlib
    
    def display_menu(self):
        """Display the main menu interface using Tkinter."""
        root = tk.Tk()
        root.title("Vision-Based Attendance and Emotion Monitoring System")
        root.geometry("1024x800")
        root.configure(bg="#f0f0f0")
        
        # Header
        header_frame = Frame(root, bg="#2c3e50", padx=20, pady=20)
        header_frame.pack(fill="x")
        
        header_label = Label(header_frame, 
                           text="Vision-Based Attendance and Emotion Monitoring System",
                           font=("Arial", 16, "bold"), fg="white", bg="#2c3e50")
        header_label.pack()
        
        # Main content
        content_frame = Frame(root, bg="#f0f0f0", padx=20, pady=20)
        content_frame.pack(fill="both", expand=True)
        
        # Attendance System
        attendance_frame = Frame(content_frame, bg="#ecf0f1", padx=15, pady=15, 
                               highlightbackground="#bdc3c7", highlightthickness=1)
        attendance_frame.pack(fill="x", pady=10)
        
        Label(attendance_frame, text="Attendance System", font=("Arial", 14, "bold"), 
             bg="#ecf0f1").pack(anchor="w")
        
        Button(attendance_frame, text="Add New Face (Webcam)", padx=10, pady=5,
              command=self.attendance_system.add_new_face_from_webcam).pack(fill="x", pady=5)
        
        Button(attendance_frame, text="Add New Face (File)", padx=10, pady=5,
              command=self.attendance_system.add_new_face_from_file).pack(fill="x", pady=5)
        
        Button(attendance_frame, text="Run Attendance System", padx=10, pady=5,
              command=self.run_system).pack(fill="x", pady=5)
        
        Button(attendance_frame, text="Generate Attendance Reports", padx=10, pady=5,
              command=self.attendance_system.generate_reports).pack(fill="x", pady=5)
        
        # Camera Calibration
        calibration_frame = Frame(content_frame, bg="#ecf0f1", padx=15, pady=15,
                                highlightbackground="#bdc3c7", highlightthickness=1)
        calibration_frame.pack(fill="x", pady=10)
        
        Label(calibration_frame, text="Camera Calibration", font=("Arial", 14, "bold"),
             bg="#ecf0f1").pack(anchor="w")
        
        Button(calibration_frame, text="Calibrate Camera", padx=10, pady=5,
              command=lambda: self.camera_calibration.calibrate_with_chessboard()).pack(fill="x", pady=5)
        
        Button(calibration_frame, text="Test Calibration", padx=10, pady=5,
              command=self.camera_calibration.calibration_test).pack(fill="x", pady=5)
        
        # Hand Gesture and OCR
        advanced_frame = Frame(content_frame, bg="#ecf0f1", padx=15, pady=15,
                             highlightbackground="#bdc3c7", highlightthickness=1)
        advanced_frame.pack(fill="x", pady=10)
        
        Label(advanced_frame, text="Advanced Features", font=("Arial", 14, "bold"),
             bg="#ecf0f1").pack(anchor="w")
        
        Button(advanced_frame, text="Test Hand Gesture Recognition", padx=10, pady=5,
              command=self.hand_gesture.run_demo).pack(fill="x", pady=5)
        
        # Make continuous OCR more prominent with a different style
        continuous_ocr_button = Button(advanced_frame, text="Live OCR Text Recognition", padx=10, pady=8,
              command=self.run_continuous_ocr, bg="#2980b9", fg="white", font=("Arial", 10, "bold"))
        continuous_ocr_button.pack(fill="x", pady=8)
        
        Button(advanced_frame, text="Manual Document Scan (OCR)", padx=10, pady=5,
              command=self.ocr_scanner.scan_document).pack(fill="x", pady=5)
        
        Button(advanced_frame, text="OCR from File", padx=10, pady=5,
              command=self.ocr_scanner.scan_from_file).pack(fill="x", pady=5)
        
        # Settings
        settings_frame = Frame(content_frame, bg="#ecf0f1", padx=15, pady=15,
                             highlightbackground="#bdc3c7", highlightthickness=1)
        settings_frame.pack(fill="x", pady=10)
        
        Label(settings_frame, text="Settings", font=("Arial", 14, "bold"),
             bg="#ecf0f1").pack(anchor="w")
        
        Button(settings_frame, text="UI Settings", padx=10, pady=5,
              command=self.ui_feedback.show_settings_dialog).pack(fill="x", pady=5)
        
        Button(settings_frame, text="OCR Settings", padx=10, pady=5,
              command=self.ocr_scanner.show_processing_options).pack(fill="x", pady=5)
        
        Button(settings_frame, text="Exit", padx=10, pady=5,
              command=root.destroy).pack(fill="x", pady=5)
        
        # Footer
        footer_frame = Frame(root, bg="#2c3e50", padx=20, pady=10)
        footer_frame.pack(fill="x", side="bottom")
        
        footer_label = Label(footer_frame, text="© 2023 Vision-Based Attendance System",
                           font=("Arial", 10), fg="white", bg="#2c3e50")
        footer_label.pack()
        
        root.mainloop()

def main():
    """Main function to start the application."""
    system = VisionBasedAttendanceSystem()
    system.display_menu()

if __name__ == "__main__":
    main() 