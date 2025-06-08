import cv2
import numpy as np
import time
import pygame
import threading
from tkinter import Tk, Label, Frame, Button, StringVar, OptionMenu, Scale, HORIZONTAL, messagebox
import os

class LiveUIFeedback:
    """
    Live UI Feedback Module for Vision-Based Attendance and Emotion Monitoring System.
    
    This module provides real-time feedback through:
    - Visual feedback overlays on detected faces
    - Sound notifications for successful detections
    - Real-time statistics display
    - User-friendly controls for system adjustment
    """
    
    def __init__(self, sound_dir="sounds"):
        """
        Initialize the Live UI Feedback module.
        
        Parameters:
        - sound_dir: Directory containing notification sounds
        """
        self.sound_dir = sound_dir
        self.stats = {
            "faces_detected": 0,
            "emotions_recognized": {},
            "attendance_marked": 0,
            "unknown_faces": 0,
            "processing_fps": 0
        }
        
        # Create sounds directory if it doesn't exist
        if not os.path.exists(sound_dir):
            os.makedirs(sound_dir)
            print(f"Created directory for sounds: {sound_dir}")
        
        # UI configuration
        self.show_names = True
        self.show_emotions = True
        self.show_confidence = True
        self.show_stats = True
        self.sound_enabled = True
        self.overlay_style = "standard"  # Options: "standard", "minimal", "detailed"
        
        # Animation properties
        self.animation_frames = 15
        self.current_frame = 0
        self.pulse_opacity = 0.5
        self.pulse_direction = 1
        
        # Performance tracking
        self.frame_times = []
        self.max_frame_times = 30  # Track FPS over last 30 frames
        self.last_frame_time = time.time()
        
        # Initialize sound engine
        self._init_sound_engine()
        
        # Motion detection for UI attention
        self.prev_frame = None
        self.motion_threshold = 25
        self.motion_detected = False
        self.fade_overlay = False
        self.overlay_opacity = 1.0
        
        # UI themes
        self.themes = {
            "standard": {
                "box_color": (0, 255, 0),
                "text_color": (255, 255, 255),
                "background": (0, 0, 0, 120),
                "highlight": (0, 255, 0)
            },
            "dark": {
                "box_color": (0, 140, 255),
                "text_color": (230, 230, 230),
                "background": (40, 40, 40, 160),
                "highlight": (0, 140, 255)
            },
            "light": {
                "box_color": (255, 100, 0),
                "text_color": (50, 50, 50),
                "background": (240, 240, 240, 180),
                "highlight": (255, 100, 0)
            },
            "minimal": {
                "box_color": (0, 200, 200),
                "text_color": (255, 255, 255),
                "background": (0, 0, 0, 80),
                "highlight": (0, 200, 200)
            }
        }
        self.current_theme = "standard"
        
        # Emotion colors (for visual feedback)
        self.emotion_colors = {
            "angry": (0, 0, 255),      # Red
            "disgust": (0, 140, 255),  # Orange
            "fear": (0, 69, 255),      # Pink
            "happy": (0, 255, 0),      # Green
            "sad": (255, 0, 0),        # Blue
            "surprise": (0, 255, 255), # Yellow
            "neutral": (200, 200, 200) # Gray
        }
        
    def _init_sound_engine(self):
        """Initialize the sound engine for notifications."""
        try:
            pygame.mixer.init()
            self.sound_loaded = True
            
            # Generate default notification sounds if they don't exist
            self._ensure_default_sounds()
            
            # Load sounds
            self.sounds = {
                "attendance_marked": pygame.mixer.Sound(os.path.join(self.sound_dir, "attendance_marked.wav")),
                "face_detected": pygame.mixer.Sound(os.path.join(self.sound_dir, "face_detected.wav")),
                "unknown_person": pygame.mixer.Sound(os.path.join(self.sound_dir, "unknown_person.wav")),
                "calibration_complete": pygame.mixer.Sound(os.path.join(self.sound_dir, "calibration_complete.wav"))
            }
            
            # Set volume
            for sound in self.sounds.values():
                sound.set_volume(0.5)
                
            print("Sound engine initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize sound engine: {str(e)}")
            self.sound_loaded = False
    
    def _ensure_default_sounds(self):
        """Create default notification sounds if they don't exist."""
        # This is a placeholder. In a real implementation, you'd include default sounds
        # or generate them programmatically using a library like pydub
        
        # Check if sounds exist, if not, create placeholders
        sound_files = ["attendance_marked.wav", "face_detected.wav", 
                      "unknown_person.wav", "calibration_complete.wav"]
        
        for sound_file in sound_files:
            path = os.path.join(self.sound_dir, sound_file)
            if not os.path.exists(path):
                print(f"Sound file {sound_file} not found. Please add sound files to the {self.sound_dir} directory.")
    
    def play_sound(self, sound_type):
        """
        Play a notification sound.
        
        Parameters:
        - sound_type: Type of sound to play (e.g., "attendance_marked")
        """
        if not self.sound_enabled or not self.sound_loaded:
            return
            
        if sound_type in self.sounds:
            # Play sound in a separate thread to avoid blocking
            threading.Thread(target=self.sounds[sound_type].play).start()
    
    def update_stats(self, faces_data, elapsed_time):
        """
        Update statistics based on processing results.
        
        Parameters:
        - faces_data: List of dictionaries containing face detection results
        - elapsed_time: Time taken to process the frame
        """
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
        
        # Update FPS calculation
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.max_frame_times:
            self.frame_times.pop(0)
            
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        self.stats["processing_fps"] = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
    
    def detect_motion(self, frame):
        """
        Detect motion in the frame to determine when to show/hide UI elements.
        
        Parameters:
        - frame: Current video frame
        
        Returns:
        - motion_level: Level of motion detected (0-100)
        """
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
    
    def _create_animation_mask(self, frame_shape, face_location, animation_type="pulse"):
        """
        Create animation mask for face highlight effects.
        
        Parameters:
        - frame_shape: Shape of the video frame
        - face_location: Location of the detected face (x, y, w, h)
        - animation_type: Type of animation effect
        
        Returns:
        - mask: Animation mask
        """
        mask = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.uint8)
        x, y, w, h = face_location
        
        if animation_type == "pulse":
            # Calculate expanded rectangle based on animation frame
            expansion = int(5 * (self.pulse_opacity))
            x1 = max(0, x - expansion)
            y1 = max(0, y - expansion)
            x2 = min(frame_shape[1], x + w + expansion)
            y2 = min(frame_shape[0], y + h + expansion)
            
            # Draw expanding rectangle
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, 2)
            
        elif animation_type == "corners":
            # Draw corners with animation
            corner_length = int(min(w, h) * 0.3)
            thickness = 2
            
            # Top-left corner
            cv2.line(mask, (x, y), (x + corner_length, y), 255, thickness)
            cv2.line(mask, (x, y), (x, y + corner_length), 255, thickness)
            
            # Top-right corner
            cv2.line(mask, (x + w, y), (x + w - corner_length, y), 255, thickness)
            cv2.line(mask, (x + w, y), (x + w, y + corner_length), 255, thickness)
            
            # Bottom-left corner
            cv2.line(mask, (x, y + h), (x + corner_length, y + h), 255, thickness)
            cv2.line(mask, (x, y + h), (x, y + h - corner_length), 255, thickness)
            
            # Bottom-right corner
            cv2.line(mask, (x + w, y + h), (x + w - corner_length, y + h), 255, thickness)
            cv2.line(mask, (x + w, y + h), (x + w, y + h - corner_length), 255, thickness)
        
        return mask
    
    def update_animation(self):
        """Update animation state for visual effects."""
        # Update pulse animation
        self.pulse_opacity += 0.05 * self.pulse_direction
        if self.pulse_opacity >= 1.0:
            self.pulse_opacity = 1.0
            self.pulse_direction = -1
        elif self.pulse_opacity <= 0.3:
            self.pulse_opacity = 0.3
            self.pulse_direction = 1
            
        # Update frame counter
        self.current_frame = (self.current_frame + 1) % self.animation_frames
    
    def apply_visual_feedback(self, frame, faces_data):
        """
        Apply visual feedback overlays on the video frame.
        
        Parameters:
        - frame: Video frame to modify
        - faces_data: List of dictionaries containing face detection results
        
        Returns:
        - processed_frame: Frame with visual feedback added
        """
        # Create a copy of the frame to work on
        display_frame = frame.copy()
        
        # Get current theme
        theme = self.themes[self.current_theme]
        
        # Update animation state
        self.update_animation()
        
        # Process each detected face
        for face in faces_data:
            if 'location' not in face:
                continue
                
            # Extract face information
            x, y, w, h = face['location']
            name = face.get('name', 'Unknown')
            emotion = face.get('emotion', 'neutral')
            emotion_scores = face.get('emotion_scores', {})
            
            # Determine border color based on emotion
            if emotion in self.emotion_colors:
                border_color = self.emotion_colors[emotion]
            else:
                border_color = theme['box_color']
            
            # Apply different visual styles based on settings
            if self.overlay_style == "standard":
                # Standard rectangle with name and emotion
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), border_color, 2)
                
                # Create text background
                if self.show_names or self.show_emotions:
                    label_parts = []
                    if self.show_names:
                        label_parts.append(name)
                    if self.show_emotions:
                        label_parts.append(emotion)
                        
                    label = " - ".join(label_parts)
                    
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(display_frame, (x, y-30), (x+text_size[0]+10, y), theme['background'][:3], -1)
                    cv2.putText(display_frame, label, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, theme['text_color'], 2)
                
                # Show confidence if enabled
                if self.show_confidence and emotion in emotion_scores:
                    confidence = emotion_scores[emotion]
                    conf_label = f"{confidence:.1f}%"
                    conf_pos = (x + w - 70, y - 10)
                    cv2.putText(display_frame, conf_label, conf_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, theme['text_color'], 1)
            
            elif self.overlay_style == "minimal":
                # Minimalist style - only corners
                corner_length = int(min(w, h) * 0.2)
                thickness = 2
                
                # Top-left corner
                cv2.line(display_frame, (x, y), (x + corner_length, y), border_color, thickness)
                cv2.line(display_frame, (x, y), (x, y + corner_length), border_color, thickness)
                
                # Top-right corner
                cv2.line(display_frame, (x + w, y), (x + w - corner_length, y), border_color, thickness)
                cv2.line(display_frame, (x + w, y), (x + w, y + corner_length), border_color, thickness)
                
                # Bottom-left corner
                cv2.line(display_frame, (x, y + h), (x + corner_length, y + h), border_color, thickness)
                cv2.line(display_frame, (x, y + h), (x, y + h - corner_length), border_color, thickness)
                
                # Bottom-right corner
                cv2.line(display_frame, (x + w, y + h), (x + w - corner_length, y + h), border_color, thickness)
                cv2.line(display_frame, (x + w, y + h), (x + w, y + h - corner_length), border_color, thickness)
                
                # Small label at the bottom
                if self.show_names or self.show_emotions:
                    mini_label = name if self.show_names else emotion
                    cv2.putText(display_frame, mini_label, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, border_color, 1)
            
            elif self.overlay_style == "detailed":
                # Detailed overlay with emotion distribution
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), border_color, 2)
                
                # Create background for text area
                info_height = 15 * (len(emotion_scores) + 2) if self.show_confidence else 40
                cv2.rectangle(display_frame, (x, y-info_height), (x+w, y), (0, 0, 0, 150), -1)
                
                # Show name and primary emotion
                header = f"{name} - {emotion}"
                cv2.putText(display_frame, header, (x+5, y-info_height+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, theme['text_color'], 1)
                
                # Show emotion scores if enabled
                if self.show_confidence:
                    line = 2
                    # Sort emotions by score
                    sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
                    for em, score in sorted_emotions[:3]:  # Show top 3 emotions
                        text = f"{em}: {score:.1f}%"
                        cv2.putText(display_frame, text, (x+5, y-info_height+(15*line)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, theme['text_color'], 1)
                        line += 1
            
            # Add animation effects based on recognition
            if name != "Unknown":
                # Create animation mask
                animation_mask = self._create_animation_mask(display_frame.shape, (x, y, w, h), "pulse")
                
                # Create color overlay
                color_overlay = np.zeros_like(display_frame)
                color_overlay[animation_mask > 0] = (*border_color, 0)  # Use emotion color for animation
                
                # Blend with original frame using opacity
                cv2.addWeighted(color_overlay, self.pulse_opacity * 0.3, display_frame, 1.0, 0, display_frame)
        
        # Add statistics overlay if enabled
        if self.show_stats:
            self._add_stats_overlay(display_frame)
        
        return display_frame
    
    def _add_stats_overlay(self, frame):
        """
        Add statistics overlay to the frame.
        
        Parameters:
        - frame: Video frame to modify
        """
        # Define overlay parameters
        overlay_margin = 10
        overlay_width = 220
        stats_height = 130
        
        # Create semi-transparent overlay
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Draw background rectangle
        theme = self.themes[self.current_theme]
        cv2.rectangle(overlay, 
                     (w - overlay_width - overlay_margin, overlay_margin), 
                     (w - overlay_margin, overlay_margin + stats_height), 
                     theme['background'][:3], -1)
        
        # Add stats text
        text_x = w - overlay_width - overlay_margin + 10
        text_y = overlay_margin + 25
        line_spacing = 20
        
        # Title
        cv2.putText(overlay, "SYSTEM STATISTICS", (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, theme['highlight'], 2)
        text_y += line_spacing + 5
        
        # FPS
        fps_text = f"FPS: {self.stats['processing_fps']:.1f}"
        cv2.putText(overlay, fps_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, theme['text_color'], 1)
        text_y += line_spacing
        
        # Faces detected
        faces_text = f"Faces: {self.stats['faces_detected']}"
        cv2.putText(overlay, faces_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, theme['text_color'], 1)
        text_y += line_spacing
        
        # Attendance count
        attendance_text = f"Attendance: {self.stats['attendance_marked']}"
        cv2.putText(overlay, attendance_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, theme['text_color'], 1)
        text_y += line_spacing
        
        # Emotions summary (most common emotion)
        if self.stats['emotions_recognized']:
            main_emotion = max(self.stats['emotions_recognized'].items(), key=lambda x: x[1])[0]
            emotion_text = f"Main emotion: {main_emotion}"
            cv2.putText(overlay, emotion_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, theme['text_color'], 1)
        
        # Apply the overlay with transparency
        overlay_opacity = self.overlay_opacity if self.fade_overlay else 0.8
        cv2.addWeighted(overlay, overlay_opacity, frame, 1 - overlay_opacity, 0, frame)
    
    def show_settings_dialog(self):
        """Display a settings dialog for adjusting UI feedback options."""
        # Create a Tkinter window for settings
        root = Tk()
        root.title("Live UI Feedback Settings")
        root.geometry("500x600")
        root.resizable(False, False)
        
        # Main frame
        main_frame = Frame(root, padx=20, pady=20)
        main_frame.pack(fill='both', expand=True)
        
        # Title
        Label(main_frame, text="UI Feedback Settings", font=("Arial", 16, "bold")).pack(pady=(0, 20))
        
        # Visual Settings Section
        Label(main_frame, text="Visual Settings", font=("Arial", 12, "bold")).pack(anchor='w', pady=(10, 5))
        
        # Theme selection
        theme_frame = Frame(main_frame)
        theme_frame.pack(fill='x', pady=5)
        Label(theme_frame, text="UI Theme:", width=15, anchor='w').pack(side='left')
        
        theme_var = StringVar(value=self.current_theme)
        theme_menu = OptionMenu(theme_frame, theme_var, *self.themes.keys())
        theme_menu.pack(side='left', fill='x', expand=True)
        
        # Overlay style selection
        style_frame = Frame(main_frame)
        style_frame.pack(fill='x', pady=5)
        Label(style_frame, text="Overlay Style:", width=15, anchor='w').pack(side='left')
        
        style_var = StringVar(value=self.overlay_style)
        style_menu = OptionMenu(style_frame, style_var, "standard", "minimal", "detailed")
        style_menu.pack(side='left', fill='x', expand=True)
        
        # Show checkboxes for display options
        options_frame = Frame(main_frame)
        options_frame.pack(fill='x', pady=10)
        
        from tkinter import IntVar, Checkbutton
        
        show_names_var = IntVar(value=int(self.show_names))
        show_emotions_var = IntVar(value=int(self.show_emotions))
        show_confidence_var = IntVar(value=int(self.show_confidence))
        show_stats_var = IntVar(value=int(self.show_stats))
        fade_overlay_var = IntVar(value=int(self.fade_overlay))
        
        options = [
            ("Show Names", show_names_var),
            ("Show Emotions", show_emotions_var),
            ("Show Confidence", show_confidence_var),
            ("Show Statistics", show_stats_var),
            ("Auto-hide Overlay", fade_overlay_var)
        ]
        
        for i, (text, var) in enumerate(options):
            Checkbutton(options_frame, text=text, variable=var).grid(row=i//2, column=i%2, sticky='w', padx=5, pady=3)
        
        # Sound Settings Section
        Label(main_frame, text="Sound Settings", font=("Arial", 12, "bold")).pack(anchor='w', pady=(20, 5))
        
        sound_frame = Frame(main_frame)
        sound_frame.pack(fill='x', pady=5)
        
        sound_enabled_var = IntVar(value=int(self.sound_enabled))
        Checkbutton(sound_frame, text="Enable Sound Notifications", variable=sound_enabled_var).pack(anchor='w')
        
        # Volume slider
        volume_frame = Frame(main_frame)
        volume_frame.pack(fill='x', pady=5)
        Label(volume_frame, text="Volume:", width=15, anchor='w').pack(side='left')
        
        volume_var = Scale(volume_frame, from_=0, to=100, orient=HORIZONTAL)
        volume_var.set(50)  # Default volume
        volume_var.pack(side='left', fill='x', expand=True)
        
        # Animation Settings Section
        Label(main_frame, text="Animation Settings", font=("Arial", 12, "bold")).pack(anchor='w', pady=(20, 5))
        
        # Motion detection sensitivity slider
        motion_frame = Frame(main_frame)
        motion_frame.pack(fill='x', pady=5)
        Label(motion_frame, text="Motion Sensitivity:", width=15, anchor='w').pack(side='left')
        
        motion_var = Scale(motion_frame, from_=5, to=50, orient=HORIZONTAL)
        motion_var.set(self.motion_threshold)
        motion_var.pack(side='left', fill='x', expand=True)
        
        # Animation speed slider
        animation_frame = Frame(main_frame)
        animation_frame.pack(fill='x', pady=5)
        Label(animation_frame, text="Animation Speed:", width=15, anchor='w').pack(side='left')
        
        animation_var = Scale(animation_frame, from_=1, to=10, orient=HORIZONTAL)
        animation_var.set(5)  # Default speed
        animation_var.pack(side='left', fill='x', expand=True)
        
        # Test sound button
        def test_sound():
            if self.sound_loaded:
                self.play_sound("face_detected")
                
        Button(main_frame, text="Test Sound", command=test_sound).pack(pady=10)
        
        # Save/Cancel buttons
        buttons_frame = Frame(main_frame)
        buttons_frame.pack(fill='x', pady=20)
        
        def save_settings():
            # Update settings
            self.current_theme = theme_var.get()
            self.overlay_style = style_var.get()
            self.show_names = bool(show_names_var.get())
            self.show_emotions = bool(show_emotions_var.get())
            self.show_confidence = bool(show_confidence_var.get())
            self.show_stats = bool(show_stats_var.get())
            self.fade_overlay = bool(fade_overlay_var.get())
            self.sound_enabled = bool(sound_enabled_var.get())
            self.motion_threshold = motion_var.get()
            
            # Set volume for all sounds
            if self.sound_loaded:
                volume = volume_var.get() / 100.0
                for sound in self.sounds.values():
                    sound.set_volume(volume)
            
            messagebox.showinfo("Settings", "Settings saved successfully!")
            root.destroy()
            
        def cancel():
            root.destroy()
            
        Button(buttons_frame, text="Save", command=save_settings, width=10).pack(side='right', padx=5)
        Button(buttons_frame, text="Cancel", command=cancel, width=10).pack(side='right', padx=5)
        
        # Run the window
        root.mainloop()
    
    def display_notification(self, message, notification_type="info", duration=3000):
        """
        Display a temporary notification message.
        
        Parameters:
        - message: Notification message
        - notification_type: Type of notification (info, success, warning, error)
        - duration: Display duration in milliseconds
        """
        # Create notification window
        root = Tk()
        root.overrideredirect(True)  # Remove window decorations
        root.attributes('-topmost', True)  # Keep on top
        
        # Position in bottom right corner
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        # Color based on notification type
        colors = {
            "info": "#3498db",
            "success": "#2ecc71",
            "warning": "#f39c12",
            "error": "#e74c3c"
        }
        bg_color = colors.get(notification_type, colors["info"])
        
        # Create notification frame
        notif_frame = Frame(root, bg=bg_color, padx=20, pady=15)
        notif_frame.pack(fill='both', expand=True)
        
        # Add message
        Label(notif_frame, text=message, bg=bg_color, fg="white", font=("Arial", 11)).pack()
        
        # Calculate width based on message length
        message_width = min(len(message) * 10, 400)
        width = max(200, message_width)
        height = 60
        
        # Position window
        x_pos = screen_width - width - 20
        y_pos = screen_height - height - 60  # Above taskbar
        root.geometry(f"{width}x{height}+{x_pos}+{y_pos}")
        
        # Auto-close after duration
        def close_notification():
            root.destroy()
            
        root.after(duration, close_notification)
        root.mainloop()

