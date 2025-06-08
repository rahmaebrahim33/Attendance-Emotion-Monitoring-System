import cv2
import time
import threading
import numpy as np
import os
import pygame
from tkinter import Tk, Label, Frame, Button, StringVar, OptionMenu, Scale, HORIZONTAL, messagebox

class LiveUIFeedback:
    def __init__(self, sound_dir="sounds"):
        # Initialize default settings and statistics
        self.sound_dir = sound_dir
        self.emotion_colors = {
            "happy": (0, 255, 255),
            "sad": (255, 0, 0),
            "angry": (0, 0, 255),
            "surprise": (255, 255, 0),
            "fear": (0, 255, 0),
            "neutral": (200, 200, 200),
            "disgust": (0, 140, 255)
        }
        self.stats = {"emotions_recognized": {}}
        self.show_stats = True
        self.themes = {"light": {}, "dark": {}}
        self.current_theme = "light"
        self.sound_enabled = True
        
        # Create sounds directory if it doesn't exist
        if not os.path.exists(sound_dir):
            os.makedirs(sound_dir)
            print(f"Created directory for sounds: {sound_dir}")
        
        # Initialize sound engine
        self._init_sound_engine()
        
        # Motion detection for UI attention
        self.prev_frame = None
        self.motion_threshold = 25
        self.motion_detected = False
        self.fade_overlay = False
        self.overlay_opacity = 1.0
    
    def _init_sound_engine(self):
        """Initialize the sound engine for notifications."""
        try:
            pygame.mixer.init()
            self.sound_loaded = True
            
            # Generate default notification sounds if they don't exist
            self._ensure_default_sounds()
            
            # Load sounds
            self.sounds = {}
            sound_files = ["attendance_marked.wav", "face_detected.wav", 
                          "unknown_person.wav", "calibration_complete.wav"]
            
            for sound_file in sound_files:
                path = os.path.join(self.sound_dir, sound_file)
                if os.path.exists(path):
                    self.sounds[os.path.splitext(sound_file)[0]] = pygame.mixer.Sound(path)
            
            # Set volume
            for sound in self.sounds.values():
                sound.set_volume(0.5)
                
            print("Sound engine initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize sound engine: {str(e)}")
            self.sound_loaded = False
    
    def _ensure_default_sounds(self):
        """Create default notification sounds if they don't exist."""
        # Check if sounds exist, if not, create placeholders
        sound_files = ["attendance_marked.wav", "face_detected.wav", 
                      "unknown_person.wav", "calibration_complete.wav"]
        
        for sound_file in sound_files:
            path = os.path.join(self.sound_dir, sound_file)
            if not os.path.exists(path):
                print(f"Sound file {sound_file} not found. Please add sound files to the {self.sound_dir} directory.")
    
    def play_sound(self, sound_type):
        """Play a notification sound."""
        if not self.sound_enabled or not hasattr(self, 'sound_loaded') or not self.sound_loaded:
            return
            
        if sound_type in self.sounds:
            # Play sound in a separate thread to avoid blocking
            threading.Thread(target=self.sounds[sound_type].play).start()
    
    def update_stats(self, faces_data, elapsed_time):
        """Update statistics based on processing results."""
        # Update face count
        self.stats["faces_detected"] = len(faces_data)
        
        # Update emotions counts
        emotions_in_frame = {}
        for face in faces_data:
            if 'emotion' in face:
                emotion = face['emotion']
                if emotion in emotions_in_frame:
                    emotions_in_frame[emotion] += 1
                else:
                    emotions_in_frame[emotion] = 1
        
        self.stats["emotions_recognized"] = emotions_in_frame
        
        # Count unknown faces
        unknown_count = sum(1 for face in faces_data if face.get('name', '') == 'Unknown')
        self.stats["unknown_faces"] = unknown_count
        
        # Count marked attendance
        self.stats["attendance_marked"] = sum(1 for face in faces_data if face.get('name', '') != 'Unknown')
    
    def detect_motion(self, frame):
        """Detect motion in the frame to determine when to show/hide UI elements."""
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return 0
            
        # Convert current frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference between current and previous frame
        frame_diff = cv2.absdiff(self.prev_frame, gray)
        
        # Apply threshold to detect significant changes
        _, thresh = cv2.threshold(frame_diff, 15, 255, cv2.THRESH_BINARY)
        
        # Calculate percentage of pixels that changed
        motion_level = (np.sum(thresh) / (thresh.shape[0] * thresh.shape[1] * 255)) * 100
        
        # Update previous frame
        self.prev_frame = gray
        
        # Update motion detection state
        self.motion_detected = motion_level > self.motion_threshold
        
        # Update overlay opacity based on motion
        if self.fade_overlay:
            if self.motion_detected:
                self.overlay_opacity = min(1.0, self.overlay_opacity + 0.1)
            else:
                self.overlay_opacity = max(0.3, self.overlay_opacity - 0.05)
        
        return motion_level
    
    def apply_visual_feedback(self, frame, faces_data):
        """Apply visual feedback overlays to the frame."""
        display_frame = frame.copy()
        
        # Apply face detection and recognition overlays
        for face in faces_data:
            if 'location' in face:
                x, y, w, h = face['location']
                
                # Get name and emotion
                name = face.get('name', 'Unknown')
                emotion = face.get('emotion', 'unknown')
                
                # Choose box color based on emotion
                box_color = self.emotion_colors.get(emotion, (0, 255, 0))
                
                # Draw rectangle around face
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), box_color, 2)
                
                # Create label with name and emotion
                label = f"{name}"
                if emotion != 'unknown':
                    label += f" - {emotion}"
                
                # Add text background for better readability
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(display_frame, (x, y-text_size[1]-10), (x+text_size[0], y), (0, 0, 0, 180), -1)
                
                # Display text
                cv2.putText(display_frame, label, (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show attendance status if available
                if name != 'Unknown':
                    cv2.putText(display_frame, "✓ Attendance", (x, y+h+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add basic stats overlay
        if self.show_stats:
            stats_text = [
                f"Detected: {self.stats.get('faces_detected', 0)}",
                f"Recognized: {self.stats.get('attendance_marked', 0)}",
                f"Unknown: {self.stats.get('unknown_faces', 0)}"
            ]
            
            h, w = display_frame.shape[:2]
            margin = 10
            
            for i, text in enumerate(stats_text):
                y_pos = 60 + i * 30
                cv2.putText(display_frame, text, (margin, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return display_frame
    
    def draw_emotion_distribution(self, frame, emotions_data):
        """Draw emotion distribution chart."""
        if not emotions_data:
            return frame
            
        # Define chart dimensions and position
        chart_width = 200
        chart_height = 150
        margin = 10
        h, w = frame.shape[:2]
        chart_x = margin
        chart_y = margin
        
        # Draw chart background
        overlay = frame.copy()
        cv2.rectangle(overlay, (chart_x, chart_y), 
                     (chart_x + chart_width, chart_y + chart_height), 
                     (0, 0, 0), -1)
        
        # Apply transparency
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw title
        cv2.putText(frame, "Emotion Distribution", 
                   (chart_x + 10, chart_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Find max value for scaling
        max_count = max(emotions_data.values()) if emotions_data else 1
        
        # Draw bars
        bar_height = 15
        bar_spacing = 5
        bar_start_y = chart_y + 40
        
        for i, (emotion, count) in enumerate(emotions_data.items()):
            # Calculate bar dimensions
            y_pos = bar_start_y + i * (bar_height + bar_spacing)
            bar_length = int((count / max_count) * (chart_width - 90))
            
            # Draw emotion label
            cv2.putText(frame, emotion, 
                       (chart_x + 10, y_pos + bar_height - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                       
            # Draw bar background
            cv2.rectangle(frame, 
                         (chart_x + 70, y_pos), 
                         (chart_x + chart_width - 10, y_pos + bar_height), 
                         (100, 100, 100), -1)
                         
            # Draw emotion-specific colored bar
            bar_color = self.emotion_colors.get(emotion, (200, 200, 200))
            cv2.rectangle(frame, 
                         (chart_x + 70, y_pos), 
                         (chart_x + 70 + bar_length, y_pos + bar_height), 
                         bar_color, -1)
                         
            # Draw count text
            cv2.putText(frame, str(count), 
                       (chart_x + 75 + bar_length, y_pos + bar_height - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def process_frame(self, frame, faces_data=None):
        """Process a video frame with all visual feedback effects."""
        if faces_data is None:
            faces_data = []
        
        # Detect motion in frame
        motion_level = self.detect_motion(frame)
        
        # Apply face detection overlays
        processed_frame = self.apply_visual_feedback(frame, faces_data)
        
        # Draw emotion distribution chart if we have multiple emotions
        emotion_counts = self.stats.get("emotions_recognized", {})
        if len(emotion_counts) > 1 and self.show_stats:
            processed_frame = self.draw_emotion_distribution(processed_frame, emotion_counts)
        
        # Update statistics
        self.update_stats(faces_data, 0)  # Elapsed time not used here
        
        return processed_frame
    
    def snapshot_with_overlay(self, frame, faces_data):
        """Create a snapshot with overlay information."""
        snapshot = frame.copy()
        snapshot = self.apply_visual_feedback(snapshot, faces_data)
        h, w = snapshot.shape[:2]
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        footer_height = 40
        
        # Create footer
        footer = np.zeros((footer_height, w, 3), dtype=np.uint8)
        footer[:] = (0, 0, 0)  # Black background
        
        # Add timestamp and system name
        cv2.putText(footer, f"Captured: {timestamp}", (10, footer_height-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(footer, "Vision-Based Attendance System", (w-250, footer_height-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Combine frame and footer
        result = np.vstack([snapshot, footer])
        
        return result
    
    def save_snapshot(self, frame, faces_data, directory="snapshots"):
        """Save a snapshot with overlays to file."""
        if not os.path.exists(directory):
            os.makedirs(directory)
        try:
            snapshot = self.snapshot_with_overlay(frame, faces_data)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{directory}/snapshot_{timestamp}.jpg"
            cv2.imwrite(filename, snapshot)
            self.play_sound("attendance_marked")
            self.display_notification(f"Snapshot saved: {filename}", "success")
            return True, filename
        except Exception as e:
            print(f"Error saving snapshot: {str(e)}")
            self.display_notification(f"Failed to save snapshot: {str(e)}", "error")
            return False, None
    
    def display_notification(self, message, notification_type="info", duration=3000):
        """Display a temporary notification message."""
        print(f"[{notification_type.upper()}] {message}")
        
        # Here we'd normally create a popup notification, but we'll just print to console
        # In a full application, this could use a Tkinter popup or overlay on the video

    def show_settings_dialog(self):
        """Show a settings dialog for UI customization."""
        root = Tk()
        root.title("UI Settings")
        root.geometry("400x400")
        
        # Theme selection
        theme_frame = Frame(root, padx=10, pady=10)
        theme_frame.pack(fill="x")
        
        Label(theme_frame, text="Theme:").pack(side="left")
        theme_var = StringVar(root)
        theme_var.set(self.current_theme)
        theme_menu = OptionMenu(theme_frame, theme_var, "standard", "dark", "light", "minimal")
        theme_menu.pack(side="left", padx=10)
        
        # Show/hide options
        options_frame = Frame(root, padx=10, pady=10)
        options_frame.pack(fill="x")
        
        show_names_var = StringVar(root, value="1" if self.show_names else "0")
        show_emotions_var = StringVar(root, value="1" if self.show_emotions else "0")
        show_stats_var = StringVar(root, value="1" if self.show_stats else "0")
        sound_enabled_var = StringVar(root, value="1" if self.sound_enabled else "0")
        
        Label(options_frame, text="Display Options:").pack(anchor="w")
        Button(options_frame, text="Show Names", width=15, 
              command=lambda: show_names_var.set("1" if show_names_var.get() == "0" else "0")).pack(anchor="w")
        Button(options_frame, text="Show Emotions", width=15,
              command=lambda: show_emotions_var.set("1" if show_emotions_var.get() == "0" else "0")).pack(anchor="w")
        Button(options_frame, text="Show Stats", width=15,
              command=lambda: show_stats_var.set("1" if show_stats_var.get() == "0" else "0")).pack(anchor="w")
        Button(options_frame, text="Enable Sound", width=15,
              command=lambda: sound_enabled_var.set("1" if sound_enabled_var.get() == "0" else "0")).pack(anchor="w")
        
        # Controls
        controls_frame = Frame(root, padx=10, pady=10)
        controls_frame.pack(fill="x")
        
        Label(controls_frame, text="Test Sound:").pack(anchor="w")
        Button(controls_frame, text="Test Notification", command=lambda: self.play_sound("attendance_marked")).pack(anchor="w")
        
        # Save button
        Button(root, text="Save Settings", command=lambda: self._save_settings(
            theme_var.get(),
            show_names_var.get() == "1",
            show_emotions_var.get() == "1",
            show_stats_var.get() == "1",
            sound_enabled_var.get() == "1",
            root
        )).pack(pady=20)
        
        root.mainloop()
    
    def _save_settings(self, theme, show_names, show_emotions, show_stats, sound_enabled, root):
        """Save UI settings and close dialog."""
        self.current_theme = theme
        self.show_names = show_names
        self.show_emotions = show_emotions
        self.show_stats = show_stats
        self.sound_enabled = sound_enabled
        
        print("Settings saved")
        root.destroy()

if __name__ == "__main__":
    ui_feedback = LiveUIFeedback()
    # Run a simple demo
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        # Set higher resolution for the camera capture
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        cv2.namedWindow("Live UI Demo", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Live UI Demo", 1280, 720)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Demo face data
            faces_data = [
                {"location": (100, 100, 120, 120), "name": "John Doe", "emotion": "happy"},
                {"location": (300, 150, 110, 110), "name": "Jane Smith", "emotion": "neutral"}
            ]
            
            processed = ui_feedback.process_frame(frame, faces_data)
            cv2.imshow("Live UI Demo", processed)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Could not open webcam")