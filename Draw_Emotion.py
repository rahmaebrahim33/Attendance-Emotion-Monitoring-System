def draw_emotion_distribution(self, frame, emotions_data):
    """
    Draw emotion distribution chart.
    
    Parameters:
    - frame: Video frame to draw on
    - emotions_data: Dictionary of emotions and their counts
    
    Returns:
    - modified_frame: Frame with chart added
    """
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
    cv2.rectangle(frame, (chart_x, chart_y), 
                 (chart_x + chart_width, chart_y + chart_height), 
                 (0, 0, 0, 150), -1)
    
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

def apply_emotion_filter(self, frame, emotion_filter=None):
    """
    Apply visual filter based on detected emotions.
    
    Parameters:
    - frame: Video frame
    - emotion_filter: Emotion to apply filter for (optional)
    
    Returns:
    - filtered_frame: Frame with visual filter applied
    """
    if emotion_filter is None:
        # If no specific emotion, use most common emotion in stats
        if not self.stats["emotions_recognized"]:
            return frame
        
        emotion_filter = max(self.stats["emotions_recognized"].items(), 
                            key=lambda x: x[1])[0]
    
    # Create a copy of the frame
    filtered_frame = frame.copy()
    
    # Apply different visual filters based on emotion
    if emotion_filter == "happy":
        # Bright, warm filter for happy
        filtered_frame = cv2.convertScaleAbs(filtered_frame, alpha=1.1, beta=10)
        
    elif emotion_filter == "sad":
        # Blue-tinted, darker filter for sad
        filtered_frame = cv2.convertScaleAbs(filtered_frame, alpha=0.9, beta=-10)
        b, g, r = cv2.split(filtered_frame)
        b = cv2.convertScaleAbs(b, alpha=1.1, beta=0)
        filtered_frame = cv2.merge([b, g, r])
        
    elif emotion_filter == "angry":
        # Red tint for angry
        b, g, r = cv2.split(filtered_frame)
        r = cv2.convertScaleAbs(r, alpha=1.2, beta=0)
        filtered_frame = cv2.merge([b, g, r])
        
    elif emotion_filter == "surprise":
        # High contrast for surprise
        filtered_frame = cv2.convertScaleAbs(filtered_frame, alpha=1.3, beta=0)
        
    elif emotion_filter == "fear":
        # Darker, green-blue tint for fear
        filtered_frame = cv2.convertScaleAbs(filtered_frame, alpha=0.85, beta=-5)
        b, g, r = cv2.split(filtered_frame)
        g = cv2.convertScaleAbs(g, alpha=1.05, beta=0)
        b = cv2.convertScaleAbs(b, alpha=1.05, beta=0)
        filtered_frame = cv2.merge([b, g, r])
    
    # Apply any filter with reduced opacity to maintain visibility
    return cv2.addWeighted(filtered_frame, 0.7, frame, 0.3, 0)

def generate_attendance_report(self, attendance_data):
    """
    Generate a formatted attendance report.
    
    Parameters:
    - attendance_data: Dictionary of names and attendance status
    
    Returns:
    - report_text: Formatted report text
    """
    if not attendance_data:
        return "No attendance data available."
    
    # Get current date and time
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Create report header
    report = [
        "==================================",
        "       ATTENDANCE REPORT          ",
        "==================================",
        f"Generated on: {now}",
        "==================================",
        ""
    ]
    
    # Count statistics
    present_count = sum(1 for status in attendance_data.values() if status)
    absent_count = len(attendance_data) - present_count
    
    # Add summary
    report.extend([
        f"Total students: {len(attendance_data)}",
        f"Present: {present_count}",
        f"Absent: {absent_count}",
        f"Attendance rate: {(present_count / len(attendance_data) * 100):.1f}%",
        ""
    ])
    
    # Add detailed list
    report.append("Detailed Attendance:")
    report.append("-" * 35)
    
    for i, (name, present) in enumerate(sorted(attendance_data.items()), 1):
        status = "✓ Present" if present else "✗ Absent"
        report.append(f"{i:2d}. {name:20s} - {status}")
    
    return "\n".join(report)

def export_attendance_report(self, attendance_data, filename=None):
    """
    Export attendance report to a file.
    
    Parameters:
    - attendance_data: Dictionary of names and attendance status
    - filename: Output filename (optional)
    
    Returns:
    - success: Boolean indicating success
    - file_path: Path to the saved file
    """
    if filename is None:
        # Generate default filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"attendance_report_{timestamp}.txt"
    
    try:
        # Generate report text
        report_text = self.generate_attendance_report(attendance_data)
        
        # Save to file
        with open(filename, 'w') as f:
            f.write(report_text)
        
        # Show notification
        self.display_notification(f"Report saved to {filename}", "success")
        return True, filename
    
    except Exception as e:
        print(f"Error exporting report: {str(e)}")
        self.display_notification(f"Failed to save report: {str(e)}", "error")
        return False, None

def process_frame(self, frame, faces_data=None):
    """
    Process a video frame with all visual feedback effects.
    
    Parameters:
    - frame: Video frame to process
    - faces_data: List of dictionaries containing face detection results (optional)
    
    Returns:
    - processed_frame: Frame with all visual effects applied
    """
    if faces_data is None:
        faces_data = []
    
    # Measure processing time for FPS calculation
    start_time = time.time()
    
    # Detect motion in frame
    motion_level = self.detect_motion(frame)
    
    # Apply face detection overlays
    processed_frame = self.apply_visual_feedback(frame, faces_data)
    
    # Draw emotion distribution chart if we have multiple faces with emotions
    emotion_counts = self.stats.get("emotions_recognized", {})
    if len(emotion_counts) > 1 and self.show_stats:
        processed_frame = self.draw_emotion_distribution(processed_frame, emotion_counts)
    
    # Update statistics
    elapsed_time = time.time() - start_time
    self.update_stats(faces_data, elapsed_time)
    
    return processed_frame

def run_demo(self, video_source=0):
    """
    Run a demonstration of the UI feedback system.
    
    Parameters:
    - video_source: Camera index or video file path
    """
    try:
        # Open video source
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source}")
            return
        
        # Display welcome message
        self.display_notification("Demo Mode - Live UI Feedback System", "info", 3000)
        
        # Demo data for faces
        demo_faces = [
            {"location": (100, 100, 120, 120), "name": "John Doe", "emotion": "happy", 
             "emotion_scores": {"happy": 95.2, "neutral": 3.5, "surprise": 1.3}},
            {"location": (300, 150, 110, 110), "name": "Jane Smith", "emotion": "neutral", 
             "emotion_scores": {"neutral": 85.7, "happy": 10.2, "sad": 4.1}}
        ]
        
        # Sometimes add an unknown face
        demo_unknown = {"location": (500, 120, 100, 100), "name": "Unknown", "emotion": "surprise", 
                        "emotion_scores": {"surprise": 75.8, "fear": 15.3, "neutral": 8.9}}
        
        frame_count = 0
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
                
            # Every 50 frames, modify demo data slightly to show animation
            if frame_count % 50 == 0:
                # Randomly move faces around a bit
                for face in demo_faces:
                    x, y, w, h = face["location"]
                    face["location"] = (
                        max(0, min(frame.shape[1] - w, x + np.random.randint(-20, 21))),
                        max(0, min(frame.shape[0] - h, y + np.random.randint(-20, 21))),
                        w, h
                    )
                
                # Randomly change emotions
                emotions = list(self.emotion_colors.keys())
                for face in demo_faces:
                    if np.random.random() < 0.3:  # 30% chance to change emotion
                        new_emotion = np.random.choice(emotions)
                        face["emotion"] = new_emotion
                        face["emotion_scores"] = {new_emotion: np.random.randint(70, 96)}
            
            # Every 100 frames, toggle the unknown face
            faces_to_process = demo_faces.copy()
            if frame_count % 100 < 50:  # Show unknown face for 50 frames every 100 frames
                faces_to_process.append(demo_unknown)
                if frame_count % 100 == 0:  # Play sound on first appearance
                    self.play_sound("unknown_person")
            
            # Process frame with UI feedback
            display_frame = self.process_frame(frame, faces_to_process)
            
            # Display the resulting frame
            cv2.imshow('UI Feedback Demo', display_frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == ord('s'):  # Press 's' to open settings
                # Process in another thread to avoid blocking
                threading.Thread(target=self.show_settings_dialog).start()
            elif key == ord('t'):  # Press 't' to change theme
                themes = list(self.themes.keys())
                current_index = themes.index(self.current_theme)
                next_index = (current_index + 1) % len(themes)
                self.current_theme = themes[next_index]
                self.display_notification(f"Theme changed to {self.current_theme}", "info")
            elif key == ord('m'):  # Press 'm' to toggle sound
                self.sound_enabled = not self.sound_enabled
                status = "enabled" if self.sound_enabled else "disabled"
                self.display_notification(f"Sound notifications {status}", "info")
            
            frame_count += 1
            
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error in demo: {str(e)}")
        self.display_notification(f"Demo error: {str(e)}", "error")

def snapshot_with_overlay(self, frame, faces_data):
    """
    Create a snapshot with enhanced visual overlays for saving.
    
    Parameters:
    - frame: Video frame
    - faces_data: Face detection data
    
    Returns:
    - snapshot: Enhanced frame for saving
    """
    # Create a copy of the frame
    snapshot = frame.copy()
    
    # Apply standard visual feedback
    snapshot = self.apply_visual_feedback(snapshot, faces_data)
    
    # Add timestamp and footer
    h, w = snapshot.shape[:2]
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Add semi-transparent footer
    footer_height = 40
    footer = snapshot[h-footer_height:h, :].copy()
    overlay = np.zeros_like(footer)
    cv2.addWeighted(overlay, 0.7, footer, 0.3, 0, footer)
    snapshot[h-footer_height:h, :] = footer
    
    # Add text to footer
    cv2.putText(snapshot, f"Captured: {timestamp}", 
               (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.putText(snapshot, "Vision-Based Attendance System", 
               (w-250, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return snapshot

def save_snapshot(self, frame, faces_data, directory="snapshots"):
    """
    Save a snapshot with overlays to disk.
    
    Parameters:
    - frame: Video frame
    - faces_data: Face detection data
    - directory: Directory to save snapshots
    
    Returns:
    - success: Boolean indicating success
    - file_path: Path to the saved file
    """
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    try:
        # Generate enhanced snapshot
        snapshot = self.snapshot_with_overlay(frame, faces_data)
        
        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{directory}/snapshot_{timestamp}.jpg"
        
        # Save image
        cv2.imwrite(filename, snapshot)
        
        # Play sound notification
        self.play_sound("attendance_marked")
        
        # Show notification
        self.display_notification(f"Snapshot saved: {filename}", "success")
        
        return True, filename
        
    except Exception as e:
        print(f"Error saving snapshot: {str(e)}")
        self.display_notification(f"Failed to save snapshot: {str(e)}", "error")
        return False, None

# Main entry point for standalone testing
if __name__ == "__main__":
    # Create and test the LiveUIFeedback module
    ui_feedback = LiveUIFeedback()
    ui_feedback.run_demo()