import cv2
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace
import time
import os
import sys
from tkinter import Tk, filedialog

class FaceEmotionDetection:
    def __init__(self):
        """Initialize the face and emotion detection system"""
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

        # Print initialization message
        print("Face & Emotion Detection System Initialized")
        print("Available emotions for detection:", ", ".join(self.emotions))

    def detect_faces_emotions(self, image):
        """
        Detect faces and emotions in an image

        Parameters:
        - image: Input image (numpy array)

        Returns:
        - processed_image: Image with detection results drawn
        - results: List of dictionaries containing face locations and emotions
        """
        # Create a copy of the image to draw on
        processed_image = image.copy()

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        # List to store results
        results = []

        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract the face region
            face_img = image[y:y+h, x:x+w]

            try:
                # Analyze face for emotion
                analysis = DeepFace.analyze(
                    face_img,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )

                # Get the dominant emotion
                emotion = analysis[0]['dominant_emotion']
                emotion_scores = analysis[0]['emotion']

                # Create result dictionary
                result = {
                    'location': (x, y, w, h),
                    'emotion': emotion,
                    'emotion_scores': emotion_scores
                }

                results.append(result)

                # Draw rectangle around face
                cv2.rectangle(processed_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Display emotion
                label = f"{emotion}"
                cv2.putText(processed_image, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            except Exception as e:
                print(f"Error processing face: {str(e)}")

        return processed_image, results

    def detect_from_image_path(self, image_path):
        """
        Detect faces and emotions from an image file

        Parameters:
        - image_path: Path to image file
        
        Returns:
        - results: List of dictionaries containing face locations and emotions
        """
        # Read the image
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return None

        # Process the image
        processed_image, results = self.detect_faces_emotions(image)

        # Display results
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        plt.title("Face & Emotion Detection Results")
        plt.axis('off')
        plt.show()

        # Print detailed results
        print(f"Detected {len(results)} faces:")
        for i, result in enumerate(results):
            print(f"Face {i+1}:")
            print(f"  - Emotion: {result['emotion']}")
            print("  - Emotion scores:")
            for emotion, score in result['emotion_scores'].items():
                print(f"    - {emotion}: {score:.2f}%")
            print()

        return results

    def detect_from_webcam(self, duration=30, display_type='window'):
        """
        Detect faces and emotions from webcam feed

        Parameters:
        - duration: Duration in seconds to run detection
        - display_type: 'window' for OpenCV window or 'matplotlib' for matplotlib display
        """
        # Initialize webcam
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        print(f"Running face and emotion detection for {duration} seconds...")
        start_time = time.time()

        if display_type == 'matplotlib':
            plt.figure(figsize=(12, 8))
            
        while time.time() - start_time < duration:
            # Read frame from webcam
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame from webcam")
                break

            # Process the frame
            processed_frame, results = self.detect_faces_emotions(frame)

            if display_type == 'window':
                # Display using OpenCV window (better for VS Code)
                cv2.imshow('Face & Emotion Detection', processed_frame)
            else:
                # Display using matplotlib
                plt.clf()
                plt.imshow(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.title(f"Face & Emotion Detection - Time remaining: {int(duration - (time.time() - start_time))}s")
                plt.pause(0.01)

            # Print detection results
            print(f"\rDetected {len(results)} faces. Dominant emotions: " +
                  ", ".join([f"{i+1}: {r['emotion']}" for i, r in enumerate(results)]), end="")

            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release webcam and close windows
        cap.release()
        cv2.destroyAllWindows()
        if display_type == 'matplotlib':
            plt.close()
        print("\nDetection complete")

    def browse_for_image(self):
        """
        Open a file dialog to browse for an image file
        
        Returns:
        - file_path: Path to the selected image file or None if canceled
        """
        # Hide the main tkinter window
        root = Tk()
        root.withdraw()
        
        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select an image file",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )
        
        # Destroy the tkinter instance
        root.destroy()
        
        return file_path if file_path else None

    def save_processed_image(self, processed_image, original_path=None):
        """
        Save the processed image with detection results
        
        Parameters:
        - processed_image: Image with detection results
        - original_path: Path of the original image to derive save location
        
        Returns:
        - saved_path: Path where the image was saved
        """
        if original_path:
            # Derive output filename from input
            filename, ext = os.path.splitext(original_path)
            output_path = f"{filename}_detected{ext}"
        else:
            # Use default location with timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = f"detected_faces_{timestamp}.jpg"
            
        # Save the image
        cv2.imwrite(output_path, processed_image)
        print(f"Processed image saved to: {output_path}")
        return output_path

# Check if running in Jupyter
def is_jupyter():
    try:
        get_ipython() # type: ignore
        return True
    except:
        return False

# Function to demonstrate usage in Jupyter
def run_jupyter_demo():
    detector = FaceEmotionDetection()
    
    print("Face Emotion Detection Tool - Jupyter Mode")
    print("\nChoose an option:")
    print("1. Run webcam detection")
    print("2. Select an image file")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == '1':
        duration = input("Enter duration in seconds (default: 30): ")
        duration = int(duration) if duration.isdigit() else 30
        
        # For Jupyter, use matplotlib display
        detector.detect_from_webcam(duration=duration, display_type='matplotlib')
    elif choice == '2':
        image_path = detector.browse_for_image()
        if image_path:
            print(f"Selected image: {image_path}")
            results = detector.detect_from_image_path(image_path)
            
            save_choice = input("Do you want to save the processed image? (y/n): ")
            if save_choice.lower() == 'y':
                # We need to reprocess the image to save it
                image = cv2.imread(image_path)
                processed_image, _ = detector.detect_faces_emotions(image)
                detector.save_processed_image(processed_image, image_path)
        else:
            print("No image selected")
    else:
        print("Invalid choice")

# Function to run command line interface
def run_command_line():
    import argparse
    
    parser = argparse.ArgumentParser(description='Face and Emotion Detection Tool')
    parser.add_argument('--image', help='Path to the image file')
    parser.add_argument('--webcam', action='store_true', help='Use webcam for detection')
    parser.add_argument('--duration', type=int, default=30, help='Duration for webcam detection in seconds')
    parser.add_argument('--browse', action='store_true', help='Open file browser to select an image')
    parser.add_argument('--save', action='store_true', help='Save the processed image with detection results')
    
    # Ignore unrecognized arguments (helps with Jupyter)
    args, unknown = parser.parse_known_args()
    
    # Initialize the detector
    detector = FaceEmotionDetection()
    
    if args.webcam:
        # Use webcam
        detector.detect_from_webcam(duration=args.duration, display_type='window')
    elif args.image:
        # Use specified image path
        image_path = args.image
        image = cv2.imread(image_path)
        if image is not None:
            processed_image, results = detector.detect_faces_emotions(image)
            
            # Display the image
            cv2.imshow('Face & Emotion Detection Results', processed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Save if requested
            if args.save:
                detector.save_processed_image(processed_image, image_path)
                
            # Print results
            print(f"Detected {len(results)} faces:")
            for i, result in enumerate(results):
                print(f"Face {i+1}:")
                print(f"  - Emotion: {result['emotion']}")
                print("  - Emotion scores:")
                for emotion, score in result['emotion_scores'].items():
                    print(f"    - {emotion}: {score:.2f}%")
                print()
        else:
            print(f"Error: Could not read image from {image_path}")
    elif args.browse:
        # Open file browser to select image
        image_path = detector.browse_for_image()
        if image_path:
            print(f"Selected image: {image_path}")
            image = cv2.imread(image_path)
            if image is not None:
                processed_image, results = detector.detect_faces_emotions(image)
                
                # Display the image
                cv2.imshow('Face & Emotion Detection Results', processed_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                # Save if requested
                if args.save:
                    detector.save_processed_image(processed_image, image_path)
                    
                # Print results
                print(f"Detected {len(results)} faces:")
                for i, result in enumerate(results):
                    print(f"Face {i+1}:")
                    print(f"  - Emotion: {result['emotion']}")
                    print("  - Emotion scores:")
                    for emotion, score in result['emotion_scores'].items():
                        print(f"    - {emotion}: {score:.2f}%")
                    print()
            else:
                print(f"Error: Could not read image from {image_path}")
        else:
            print("No image selected")
    else:
        # No arguments provided, show usage instructions
        print("Face & Emotion Detection Tool")
        print("\nUsage options:")
        print("1. Run with webcam: python face_emotion_detector.py --webcam")
        print("2. Analyze image file: python face_emotion_detector.py --image path/to/image.jpg")
        print("3. Browse for image: python face_emotion_detector.py --browse")
        print("4. Save detection results: Add --save to any command")
        print("\nFor more options, use: python face_emotion_detector.py --help")

# Main entry point
if __name__ == "__main__":
    if is_jupyter():
        # Running in Jupyter, use interactive mode
        run_jupyter_demo()
    else:
        # Running as a script, use command line mode
        run_command_line()