import cv2
import numpy as np
import pandas as pd
import datetime
import os
import time
from deepface import DeepFace  # For face and emotion detection
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, simpledialog, messagebox

class AttendanceEmotionSystem:
    def __init__(self, known_faces_dir="known_faces", attendance_file="attendance_log.csv"):
        """
        Initialize the attendance and emotion monitoring system.
        Parameters:
        - known_faces_dir: Directory containing images of known individuals
        - attendance_file: CSV file to store attendance records
        """
        self.known_faces_dir = known_faces_dir
        self.attendance_file = attendance_file
        self.known_face_names = []
        self.emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

        # Ensure directories exist
        if not os.path.exists(known_faces_dir):
            os.makedirs(known_faces_dir)
            print(f"Created directory for known faces: {known_faces_dir}")

        # Initialize face detection and recognition
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Initialize attendance log file if it doesn't exist
        if not os.path.exists(attendance_file):
            self._create_attendance_file()

        # Load known faces
        self._load_known_faces()

    def _create_attendance_file(self):
        """Create a new attendance log file with appropriate headers."""
        df = pd.DataFrame(columns=["Name", "Date", "Time", "Emotion"])
        df.to_csv(self.attendance_file, index=False)
        print(f"Created attendance log file: {self.attendance_file}")

    def _load_known_faces(self):
        """Load all known faces from the directory."""
        print("Loading known faces...")
        if not os.path.exists(self.known_faces_dir):
            print(f"Warning: Known faces directory '{self.known_faces_dir}' does not exist.")
            return

        for filename in os.listdir(self.known_faces_dir):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                name = os.path.splitext(filename)[0]
                print(f"Loading face: {name}")
                self.known_face_names.append(name)
                print(f"Successfully loaded face for {name}")

        print(f"Loaded {len(self.known_face_names)} known faces")

    def browse_for_image(self, title="Select an image"):
        """Open a file dialog to browse for an image file."""
        root = Tk()
        root.withdraw()  # Hide the main window
        
        file_path = filedialog.askopenfilename(
            title=title,
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        root.destroy()
        return file_path if file_path else None

    def add_new_face_from_file(self):
        """Add a new face from an image file selected via dialog."""
        # Get name for the person
        root = Tk()
        root.withdraw()
        name = simpledialog.askstring("Input", "Enter name for the person:")
        root.destroy()
        
        if not name:
            print("Operation cancelled")
            return
            
        # Select image file
        image_path = self.browse_for_image(f"Select image for {name}")
        
        if not image_path:
            print("No image selected, operation cancelled")
            return
            
        try:
            # Read the image
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Failed to read image from {image_path}")
                return
                
            # Save the image to known faces directory
            img_path = os.path.join(self.known_faces_dir, f"{name}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"Face saved as {img_path}")
            
            # Add to known faces
            self.known_face_names.append(name)
            print(f"Added {name} to known faces")
            
            # Show confirmation
            root = Tk()
            root.withdraw()
            messagebox.showinfo("Success", f"Successfully added {name} to known faces.")
            root.destroy()
            
        except Exception as e:
            print(f"Error adding face: {str(e)}")
            root = Tk()
            root.withdraw()
            messagebox.showerror("Error", f"Failed to add face: {str(e)}")
            root.destroy()

    def add_new_face_from_webcam(self):
        """Capture and add a new face from webcam."""
        # Get name for the person
        root = Tk()
        root.withdraw()
        name = simpledialog.askstring("Input", "Enter name for the person:")
        root.destroy()
        
        if not name:
            print("Operation cancelled")
            return
            
        try:
            # Initialize webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open webcam")
                return
                
            print("Preparing to capture face. Please look at the camera...")
            
            # Create window for preview
            cv2.namedWindow("Capture Face", cv2.WINDOW_NORMAL)
            
            # Countdown display
            for i in range(3, 0, -1):
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                    
                # Display countdown
                cv2.putText(frame, f"Capturing in {i}...", (30, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.imshow("Capture Face", frame)
                cv2.waitKey(1000)  # Wait for 1 second
            
            # Capture the frame
            ret, frame = cap.read()
            if ret:
                # Display "Captured!" message
                cv2.putText(frame, "Captured!", (30, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.imshow("Capture Face", frame)
                cv2.waitKey(1000)  # Show the captured frame for 1 second
                
                # Save the image
                img_path = os.path.join(self.known_faces_dir, f"{name}.jpg")
                cv2.imwrite(img_path, frame)
                print(f"Face captured and saved as {img_path}")
                
                # Add to known faces
                self.known_face_names.append(name)
                print(f"Added {name} to known faces")
                
                # Show confirmation
                root = Tk()
                root.withdraw()
                messagebox.showinfo("Success", f"Successfully added {name} to known faces.")
                root.destroy()
                
            else:
                print("Failed to capture image")
                
            # Release resources
            cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error adding face: {str(e)}")
            root = Tk()
            root.withdraw()
            messagebox.showerror("Error", f"Failed to add face: {str(e)}")
            root.destroy()

    def mark_attendance(self, name, emotion):
        """
        Mark attendance for a recognized face.

        Parameters:
        - name: Name of the recognized person
        - emotion: Detected emotion
        """
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        # Check if attendance already marked for today
        df = pd.read_csv(self.attendance_file)
        today_records = df[(df["Name"] == name) & (df["Date"] == date_str)]

        if today_records.empty:
            new_record = pd.DataFrame({"Name": [name],
                                      "Date": [date_str],
                                      "Time": [time_str],
                                      "Emotion": [emotion]})

            df = pd.concat([df, new_record], ignore_index=True)
            df.to_csv(self.attendance_file, index=False)
            return True, f"Marked attendance for {name}"
        else:
            return False, f"{name} already marked for today"

    def recognize_face(self, frame):
        """
        Recognize a face in the given frame using DeepFace.

        Parameters:
        - frame: Image frame to analyze

        Returns:
        - name: Name of the recognized person or "Unknown"
        - emotion: Detected emotion
        - face_location: Location of the detected face
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]

            try:
                # Analyze face for emotion
                result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False, silent=True)

                emotion = result[0]['dominant_emotion']

                # Try to recognize the face
                name = "Unknown"
                if self.known_face_names:  # Only try recognition if we have known faces
                    try:
                        recognition = DeepFace.find(face_img, self.known_faces_dir, enforce_detection=False, silent=True)
                        if not recognition[0].empty:
                            # Extract the name from the path
                            identity_path = recognition[0].iloc[0]['identity']
                            name = os.path.splitext(os.path.basename(identity_path))[0]
                    except:
                        pass

                return name, emotion, (x, y, w, h)

            except:
                pass

        return None, None, None

    def process_image_file(self):
        """Process a single image file for attendance."""
        # Select image file
        image_path = self.browse_for_image("Select image for attendance")
        
        if not image_path:
            print("No image selected")
            return
            
        try:
            # Read the image
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Failed to read image from {image_path}")
                return
                
            # Create a copy for displaying results
            display_frame = frame.copy()
            
            # Detect and recognize faces
            marked_attendance = set()
            
            name, emotion, face_location = self.recognize_face(frame)
            
            if name and face_location:
                x, y, w, h = face_location
                
                # Draw rectangle around face
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Display name and emotion
                label = f"{name} - {emotion}"
                cv2.putText(display_frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Mark attendance if not already marked
                if name != "Unknown":
                    success, message = self.mark_attendance(name, emotion)
                    if success:
                        marked_attendance.add(name)
                    print(message)
                    
                    # Add status text to image
                    cv2.putText(display_frame, message, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display the processed image
            cv2.imshow("Attendance Processing Result", display_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Show summary
            if marked_attendance:
                message = f"Marked attendance for: {', '.join(marked_attendance)}"
            else:
                message = "No attendance marked"
                
            root = Tk()
            root.withdraw()
            messagebox.showinfo("Attendance Result", message)
            root.destroy()
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            root = Tk()
            root.withdraw()
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
            root.destroy()

    def run_attendance_system(self, duration=30):
        """
        Run the attendance system for a specified duration using webcam.

        Parameters:
        - duration: Number of seconds to run the system
        """
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
            
        start_time = time.time()
        marked_attendance = set()
        
        # Create window for display
        cv2.namedWindow("Attendance System", cv2.WINDOW_NORMAL)
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
                
            # Create a copy for displaying results
            display_frame = frame.copy()
            
            # Display time remaining
            time_remaining = int(duration - (time.time() - start_time))
            cv2.putText(display_frame, f"Time remaining: {time_remaining}s", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Detect and recognize faces
            name, emotion, face_location = self.recognize_face(frame)
            
            if name and face_location:
                x, y, w, h = face_location
                
                # Draw rectangle around face
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Display name and emotion
                label = f"{name} - {emotion}"
                cv2.putText(display_frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Mark attendance if not already marked
                if name != "Unknown" and name not in marked_attendance:
                    success, message = self.mark_attendance(name, emotion)
                    if success:
                        marked_attendance.add(name)
                    
                    # Display message on frame
                    cv2.putText(display_frame, message, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    print(message)
            
            # Display the frame
            cv2.imshow("Attendance System", display_frame)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        # Show summary
        print("\nAttendance Summary:")
        print(f"Total people recognized: {len(marked_attendance)}")
        print(f"People recognized: {', '.join(marked_attendance) if marked_attendance else 'None'}")
        
        if marked_attendance:
            message = f"Marked attendance for: {', '.join(marked_attendance)}"
        else:
            message = "No attendance marked"
            
        root = Tk()
        root.withdraw()
        messagebox.showinfo("Attendance Summary", message)
        root.destroy()

    def generate_reports(self):
        """Generate reports from the attendance data."""
        if not os.path.exists(self.attendance_file):
            print("No attendance data available")
            root = Tk()
            root.withdraw()
            messagebox.showinfo("Report", "No attendance data available")
            root.destroy()
            return

        df = pd.read_csv(self.attendance_file)

        if df.empty:
            print("No attendance records found")
            root = Tk()
            root.withdraw()
            messagebox.showinfo("Report", "No attendance records found")
            root.destroy()
            return

        # Create report text
        report_text = "===== ATTENDANCE REPORT =====\n\n"
        
        # Daily attendance report
        report_text += "=== Daily Attendance Report ===\n"
        daily_count = df.groupby('Date').size()
        report_text += daily_count.to_string() + "\n\n"

        # Person-wise report
        report_text += "=== Person-wise Attendance Report ===\n"
        person_count = df.groupby('Name').size()
        report_text += person_count.to_string() + "\n\n"

        # Emotion analysis
        report_text += "=== Emotion Analysis ===\n"
        emotion_count = df.groupby('Emotion').size()
        report_text += emotion_count.to_string() + "\n"

        # Print to console
        print(report_text)
        
        # Display in message box
        root = Tk()
        root.withdraw()
        messagebox.showinfo("Attendance Report", report_text)
        root.destroy()

        # Visualize emotion distribution
        plt.figure(figsize=(10, 6))
        emotion_count.plot(kind='bar', color='skyblue')
        plt.title('Emotion Distribution')
        plt.xlabel('Emotion')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()

def display_menu():
    """Display main menu options."""
    print("\nAttendance and Emotion Recognition System")
    print("=========================================")
    print("1. Add new face from webcam")
    print("2. Add new face from image file")
    print("3. Run attendance system (webcam)")
    print("4. Process image for attendance")
    print("5. Generate attendance reports")
    print("6. Exit")
    choice = input("\nEnter your choice (1-6): ")
    return choice

def main():
    """Main function to run the application."""
    system = AttendanceEmotionSystem()
    
    while True:
        choice = display_menu()
        
        if choice == '1':
            system.add_new_face_from_webcam()
        elif choice == '2':
            system.add_new_face_from_file()
        elif choice == '3':
            duration = input("Enter duration in seconds (default: 30): ")
            duration = int(duration) if duration.strip().isdigit() else 30
            system.run_attendance_system(duration=duration)
        elif choice == '4':
            system.process_image_file()
        elif choice == '5':
            system.generate_reports()
        elif choice == '6':
            print("Exiting application.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()