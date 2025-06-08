import cv2
import numpy as np
import mediapipe as mp
import time

class HandGestureDetector:
    """
    Hand gesture detection and recognition module using MediaPipe.
    Recognizes predefined gestures for controlling the system.
    """
    
    def __init__(self):
        """Initialize the hand gesture detector."""
        # MediaPipe hands components
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize MediaPipe Hands with appropriate settings
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Define gesture thresholds and parameters
        self.finger_thresh = 0.4  # Threshold for finger bent/straight detection
        
        # Store last detected gesture for debouncing
        self.last_gesture = None
        self.gesture_start_time = 0
        self.gesture_hold_time = 0.5  # Hold gesture for this long to activate
        
        # Gesture command mappings for system control
        self.gesture_commands = {
            "open_palm": "volume_up",
            "closed_fist": "volume_down",
            "pointing_up": "next_page",
            "victory": "play_pause",
            "thumbs_up": "mark_attendance"
        }
        
        print("Hand gesture detection module initialized")
    
    def detect_hands(self, frame):
        """
        Detect hands in the frame using MediaPipe.
        
        Parameters:
        - frame: Input video frame
        
        Returns:
        - results: MediaPipe hand detection results
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame for hand detection
        results = self.hands.process(rgb_frame)
        
        return results
    
    def draw_hand_landmarks(self, frame, results):
        """
        Draw hand landmarks and connections on the frame.
        
        Parameters:
        - frame: Input video frame
        - results: MediaPipe hand detection results
        
        Returns:
        - frame: Frame with landmarks drawn
        """
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks and connections
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Draw a circle at the wrist for better visibility
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                h, w, _ = frame.shape
                cx, cy = int(wrist.x * w), int(wrist.y * h)
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
        
        return frame
    
    def _calculate_finger_angles(self, landmarks):
        """
        Calculate angles for each finger to determine if it's extended or bent.
        
        Parameters:
        - landmarks: Hand landmarks from MediaPipe
        
        Returns:
        - finger_states: Dictionary with state of each finger (True=extended, False=bent)
        """
        # Get landmark coordinates
        points = []
        for landmark in landmarks.landmark:
            points.append((landmark.x, landmark.y, landmark.z))
        
        # Check thumb
        thumb_angle = self._angle_between_points(
            points[0],  # WRIST
            points[1],  # THUMB_CMC
            points[4]   # THUMB_TIP
        )
        thumb_extended = thumb_angle > self.finger_thresh
        
        # Check index finger
        index_angle = self._angle_between_points(
            points[0],  # WRIST
            points[5],  # INDEX_FINGER_MCP
            points[8]   # INDEX_FINGER_TIP
        )
        index_extended = index_angle > self.finger_thresh
        
        # Check middle finger
        middle_angle = self._angle_between_points(
            points[0],  # WRIST
            points[9],  # MIDDLE_FINGER_MCP
            points[12]  # MIDDLE_FINGER_TIP
        )
        middle_extended = middle_angle > self.finger_thresh
        
        # Check ring finger
        ring_angle = self._angle_between_points(
            points[0],  # WRIST
            points[13], # RING_FINGER_MCP
            points[16]  # RING_FINGER_TIP
        )
        ring_extended = ring_angle > self.finger_thresh
        
        # Check pinky finger
        pinky_angle = self._angle_between_points(
            points[0],  # WRIST
            points[17], # PINKY_FINGER_MCP
            points[20]  # PINKY_FINGER_TIP
        )
        pinky_extended = pinky_angle > self.finger_thresh
        
        # Return finger states
        return {
            "thumb": thumb_extended,
            "index": index_extended,
            "middle": middle_extended,
            "ring": ring_extended,
            "pinky": pinky_extended
        }
    
    def _angle_between_points(self, p1, p2, p3):
        """
        Calculate the angle between three points.
        
        Parameters:
        - p1, p2, p3: 3D points (x, y, z)
        
        Returns:
        - angle: Angle in normalized units
        """
        # Calculate vectors
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]])
        
        # Normalize vectors
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm == 0 or v2_norm == 0:
            return 0
        
        v1 = v1 / v1_norm
        v2 = v2 / v2_norm
        
        # Calculate dot product and angle
        dot_product = np.dot(v1, v2)
        dot_product = np.clip(dot_product, -1.0, 1.0)  # Ensure value is in range [-1, 1]
        angle = np.arccos(dot_product)
        
        # Normalize to [0, 1] range
        return angle / np.pi
    
    def recognize_gesture(self, landmarks):
        """
        Recognize hand gesture based on finger states.
        
        Parameters:
        - landmarks: Hand landmarks from MediaPipe
        
        Returns:
        - gesture: Recognized gesture name or None
        """
        # Calculate finger states
        finger_states = self._calculate_finger_angles(landmarks)
        
        # Recognize gestures based on finger states
        # Open palm: all fingers extended
        if (finger_states["thumb"] and finger_states["index"] and 
            finger_states["middle"] and finger_states["ring"] and 
            finger_states["pinky"]):
            return "open_palm"
        
        # Closed fist: all fingers bent
        elif (not finger_states["thumb"] and not finger_states["index"] and 
              not finger_states["middle"] and not finger_states["ring"] and 
              not finger_states["pinky"]):
            return "closed_fist"
        
        # Pointing up: only index finger extended
        elif (not finger_states["thumb"] and finger_states["index"] and 
              not finger_states["middle"] and not finger_states["ring"] and 
              not finger_states["pinky"]):
            return "pointing_up"
        
        # Victory: index and middle fingers extended
        elif (not finger_states["thumb"] and finger_states["index"] and 
              finger_states["middle"] and not finger_states["ring"] and 
              not finger_states["pinky"]):
            return "victory"
        
        # Thumbs up: only thumb extended
        elif (finger_states["thumb"] and not finger_states["index"] and 
              not finger_states["middle"] and not finger_states["ring"] and 
              not finger_states["pinky"]):
            return "thumbs_up"
        
        # No recognized gesture
        return None
    
    def process_frame(self, frame):
        """
        Process a frame to detect and recognize hand gestures.
        
        Parameters:
        - frame: Input video frame
        
        Returns:
        - frame: Processed frame with visualizations
        - gesture: Recognized gesture if stable, else None
        - gesture_command: Associated command if gesture recognized, else None
        """
        # Detect hands
        results = self.detect_hands(frame)
        
        # Get frame dimensions
        h, w, _ = frame.shape
        
        # Draw hand landmarks
        frame = self.draw_hand_landmarks(frame, results)
        
        # Process detected hands
        current_gesture = None
        gesture_command = None
        
        if results.multi_hand_landmarks:
            # We'll process the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Recognize gesture
            current_gesture = self.recognize_gesture(hand_landmarks)
            
            # Handle gesture stability and debouncing
            current_time = time.time()
            
            if current_gesture == self.last_gesture:
                # Same gesture held
                if self.last_gesture is not None:
                    hold_duration = current_time - self.gesture_start_time
                    
                    # Display hold progress
                    if hold_duration < self.gesture_hold_time:
                        progress = int((hold_duration / self.gesture_hold_time) * 100)
                        cv2.putText(frame, f"Gesture: {current_gesture} ({progress}%)", 
                                   (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        # Gesture held long enough, trigger command
                        gesture_command = self.gesture_commands.get(current_gesture)
                        cv2.putText(frame, f"Command: {gesture_command}", 
                                   (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # New gesture, reset timer
                self.last_gesture = current_gesture
                self.gesture_start_time = current_time
        else:
            # No hand detected, reset gesture state
            self.last_gesture = None
        
        return frame, current_gesture, gesture_command
    
    def run_demo(self):
        """Run a demonstration of the hand gesture detection."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set higher resolution for the camera capture
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        cv2.namedWindow("Hand Gesture Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Hand Gesture Detection", 1280, 720)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process the frame
            processed_frame, gesture, command = self.process_frame(frame)
            
            # Display information
            h, w, _ = processed_frame.shape
            cv2.putText(processed_frame, "Hand Gesture Detection Demo", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if gesture:
                cv2.putText(processed_frame, f"Detected: {gesture}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if command:
                cv2.putText(processed_frame, f"Command: {command}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show the frame
            cv2.imshow("Hand Gesture Detection", processed_frame)
            
            # Check for exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Run demo if script is executed directly
    gesture_detector = HandGestureDetector()
    gesture_detector.run_demo() 