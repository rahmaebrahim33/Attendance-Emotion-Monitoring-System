import cv2
import numpy as np
import os
import glob
import pickle
from tkinter import Tk, filedialog, messagebox, simpledialog
import matplotlib.pyplot as plt

class CameraCalibration:
    """
    Camera calibration and optimization module for Vision-Based Attendance and Emotion Monitoring System.
    This class handles calibrating the camera to correct lens distortion and optimizing camera parameters
    for better face detection and emotion recognition.
    """
    
    def __init__(self, calibration_dir="camera_calibration"):
        """
        Initialize the camera calibration module.
        
        Parameters:
        - calibration_dir: Directory to store calibration data
        """
        self.calibration_dir = calibration_dir
        self.calibration_file = os.path.join(calibration_dir, "calibration_data.pkl")
        self.camera_matrix = None
        self.dist_coeffs = None
        self.resolution = None
        self.is_calibrated = False
        
        # Chessboard parameters for calibration
        self.chessboard_size = (9, 6)  # Number of internal corners on the chessboard
        self.square_size = 2.5         # Size of each square in cm (can be adjusted)
        
        # Auto brightness and contrast parameters
        self.alpha = 1.0  # Contrast control (1.0 means no change)
        self.beta = 0     # Brightness control (0 means no change)
        
        # Ensure calibration directory exists
        if not os.path.exists(calibration_dir):
            os.makedirs(calibration_dir)
            print(f"Created directory for calibration data: {calibration_dir}")
            
        # Try to load existing calibration data
        self._load_calibration()
    
    def _load_calibration(self):
        """Load calibration data from file if it exists."""
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, 'rb') as f:
                    calibration_data = pickle.load(f)
                    self.camera_matrix = calibration_data['camera_matrix']
                    self.dist_coeffs = calibration_data['dist_coeffs']
                    self.resolution = calibration_data['resolution']
                    self.is_calibrated = True
                    print(f"Loaded calibration data from {self.calibration_file}")
            except Exception as e:
                print(f"Error loading calibration data: {str(e)}")
                self.is_calibrated = False
    
    def _save_calibration(self):
        """Save calibration data to file."""
        try:
            calibration_data = {
                'camera_matrix': self.camera_matrix,
                'dist_coeffs': self.dist_coeffs,
                'resolution': self.resolution
            }
            
            with open(self.calibration_file, 'wb') as f:
                pickle.dump(calibration_data, f)
                
            print(f"Saved calibration data to {self.calibration_file}")
        except Exception as e:
            print(f"Error saving calibration data: {str(e)}")
    
    def calibrate_with_chessboard(self, num_images=10, preview_delay=500):
        """
        Calibrate camera using a chessboard pattern.
        
        Parameters:
        - num_images: Number of images to capture for calibration
        - preview_delay: Delay between captures in milliseconds
        
        Returns:
        - success: True if calibration was successful
        """
        print("\nCamera Calibration Process")
        print("==========================")
        print(f"This process requires you to show a chessboard pattern ({self.chessboard_size[0]}x{self.chessboard_size[1]} internal corners)")
        print("Slowly move and rotate the chessboard in front of the camera")
        print(f"The system will capture {num_images} images automatically")
        input("\nPress Enter to start the calibration process...")
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return False
        
        # Get camera resolution
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam")
            cap.release()
            return False
            
        self.resolution = (frame.shape[1], frame.shape[0])
        
        # Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2) * self.square_size
        
        # Arrays to store object points and image points
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        
        # Create window for preview
        cv2.namedWindow("Camera Calibration", cv2.WINDOW_NORMAL)
        
        images_captured = 0
        last_capture_time = cv2.getTickCount()
        
        print("\nStarting capture - position the chessboard in front of the camera...")
        
        while images_captured < num_images:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam")
                break
                
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            
            # Copy frame for display
            display_frame = frame.copy()
            
            # Display instructions
            cv2.putText(display_frame, f"Captures: {images_captured}/{num_images}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_frame, "Position chessboard in view and hold still", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # If found, add object points, image points
            if ret:
                # Draw and display the corners
                cv2.drawChessboardCorners(display_frame, self.chessboard_size, corners, ret)
                
                # Check if enough time has passed since last capture
                current_time = cv2.getTickCount()
                elapsed_time = (current_time - last_capture_time) / cv2.getTickFrequency() * 1000  # ms
                
                if elapsed_time > preview_delay:
                    objpoints.append(objp)
                    imgpoints.append(corners)
                    
                    images_captured += 1
                    print(f"Captured image {images_captured}/{num_images}")
                    
                    # Update last capture time
                    last_capture_time = current_time
            
            # Display the frame
            cv2.imshow("Camera Calibration", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        if images_captured < 3:
            print("Not enough images captured for calibration")
            return False
            
        print("\nProcessing calibration data...")
        
        try:
            # Calibrate camera
            ret, self.camera_matrix, self.dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, self.resolution, None, None)
                
            if ret:
                self.is_calibrated = True
                self._save_calibration()
                
                # Calculate reprojection error
                total_error = 0
                for i in range(len(objpoints)):
                    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], 
                                                     self.camera_matrix, self.dist_coeffs)
                    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                    total_error += error
                    
                avg_error = total_error / len(objpoints)
                print(f"Calibration complete! Average reprojection error: {avg_error}")
                
                # Show calibration summary in messagebox
                root = Tk()
                root.withdraw()
                messagebox.showinfo("Calibration Complete", 
                                   f"Camera calibration successful!\n\nAverage reprojection error: {avg_error:.4f}\n\n"
                                   f"This value should ideally be below 1.0. Lower values indicate better calibration.")
                root.destroy()
                
                return True
            else:
                print("Calibration failed")
                return False
                
        except Exception as e:
            print(f"Error during calibration: {str(e)}")
            return False
    
    def undistort_image(self, image):
        """
        Undistort an image using calibration data.
        
        Parameters:
        - image: Input image
        
        Returns:
        - undistorted_image: Corrected image
        """
        if not self.is_calibrated:
            return image
            
        # Check if image resolution matches calibration resolution
        h, w = image.shape[:2]
        if (w, h) != self.resolution:
            # Scale camera matrix for the current resolution
            fx = w / self.resolution[0]
            fy = h / self.resolution[1]
            scaled_camera_matrix = self.camera_matrix.copy()
            scaled_camera_matrix[0, 0] *= fx  # Scale fx
            scaled_camera_matrix[1, 1] *= fy  # Scale fy
            scaled_camera_matrix[0, 2] *= fx  # Scale cx
            scaled_camera_matrix[1, 2] *= fy  # Scale cy
            
            # Get optimal new camera matrix
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                scaled_camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))
            
            # Undistort
            undistorted = cv2.undistort(image, scaled_camera_matrix, self.dist_coeffs, None, new_camera_matrix)
            
            # Crop the image (optional)
            # x, y, w, h = roi
            # undistorted = undistorted[y:y+h, x:x+w]
            
            return undistorted
        else:
            # Get optimal new camera matrix
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, self.resolution, 1, self.resolution)
            
            # Undistort
            undistorted = cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)
            
            return undistorted
    
    def detect_chessboard(self, image):
        """
        Detect chessboard corners in an image.
        Useful for testing calibration.
        
        Parameters:
        - image: Input image
        
        Returns:
        - result_image: Image with detected corners
        - found: Whether corners were found
        """
        # Create a copy of the image
        result_image = image.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        found, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
        
        if found:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Draw the corners
            cv2.drawChessboardCorners(result_image, self.chessboard_size, corners2, found)
        
        return result_image, found
    
    def optimize_brightness_contrast(self, image, clip_hist_percent=1):
        """
        Automatically adjust image brightness and contrast using histogram equalization.
        
        Parameters:
        - image: Input image
        - clip_hist_percent: Percentage of histogram to clip
        
        Returns:
        - optimized_image: Image with adjusted brightness and contrast
        """
        # Convert to grayscale if image is BGR
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Calculate grayscale histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_size = len(hist)
        
        # Calculate cumulative distribution
        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index-1] + float(hist[index]))
            
        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_percent *= (maximum/100.0)
        clip_hist_percent /= 2.0
        
        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1
            
        # Locate right cut
        maximum_gray = hist_size - 1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1
            
        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha
        
        # Store the parameters
        self.alpha = alpha
        self.beta = beta
        
        # Adjust the image
        if len(image.shape) == 3:
            # For color images
            adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        else:
            # For grayscale images
            adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
            
        return adjusted
    
    def apply_optimizations(self, image):
        """
        Apply all optimizations to an image (undistortion, brightness/contrast adjustment).
        
        Parameters:
        - image: Input image
        
        Returns:
        - optimized_image: Image with all optimizations applied
        """
        # First undistort
        if self.is_calibrated:
            undistorted = self.undistort_image(image)
        else:
            undistorted = image
            
        # Then optimize brightness and contrast
        optimized = self.optimize_brightness_contrast(undistorted)
        
        return optimized
    
    def calibration_test(self):
        """
        Test the camera calibration by capturing images and showing before/after.
        """
        if not self.is_calibrated:
            print("Camera not calibrated. Please run calibration first.")
            return
            
        print("\nCalibration Test")
        print("===============")
        print("This will capture an image and show the before/after undistortion")
        input("Press Enter to capture test image...")
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
            
        # Create window for preview
        cv2.namedWindow("Calibration Test", cv2.WINDOW_NORMAL)
        
        print("Position yourself or an object in front of the camera...")
        print("Press 'c' to capture a test image or 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam")
                break
                
            # Display the frame
            cv2.imshow("Calibration Test", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                # Process the captured frame
                undistorted = self.undistort_image(frame)
                optimized = self.optimize_brightness_contrast(undistorted)
                
                # Display the results
                plt.figure(figsize=(15, 10))
                
                plt.subplot(221)
                plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                plt.title('Original Image')
                plt.axis('off')
                
                plt.subplot(222)
                plt.imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
                plt.title('Undistorted Image')
                plt.axis('off')
                
                plt.subplot(223)
                plt.imshow(cv2.cvtColor(self.optimize_brightness_contrast(frame), cv2.COLOR_BGR2RGB))
                plt.title('Brightness/Contrast Optimized')
                plt.axis('off')
                
                plt.subplot(224)
                plt.imshow(cv2.cvtColor(optimized, cv2.COLOR_BGR2RGB))
                plt.title('All Optimizations Applied')
                plt.axis('off')
                
                plt.tight_layout()
                plt.show()
                
                print("Test complete! Close the matplotlib window to continue.")
                
            elif key == ord('q'):
                break
                
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
    
    def adjust_camera_parameters(self):
        """
        Interactive adjustment of camera parameters (exposure, focus, etc.)
        """
        print("\nCamera Parameter Adjustment")
        print("==========================")
        print("This will help you adjust camera parameters for optimal face detection.")
        input("Press Enter to start...")
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
            
        # Create window for preview
        cv2.namedWindow("Camera Adjustment", cv2.WINDOW_NORMAL)
        
        # Get available camera properties
        properties = [
            ('Brightness', cv2.CAP_PROP_BRIGHTNESS),
            ('Contrast', cv2.CAP_PROP_CONTRAST),
            ('Saturation', cv2.CAP_PROP_SATURATION),
            ('Exposure', cv2.CAP_PROP_EXPOSURE),
            ('Gain', cv2.CAP_PROP_GAIN),
            ('Auto Exposure', cv2.CAP_PROP_AUTO_EXPOSURE)
        ]
        
        # Store original values
        original_values = {}
        current_values = {}
        for name, prop in properties:
            value = cap.get(prop)
            original_values[prop] = value
            current_values[prop] = value
            print(f"{name}: {value}")
        
        current_property_index = 0
        
        print("\nUse the following keys:")
        print("  'n' - Next property")
        print("  'p' - Previous property")
        print("  '+' - Increase value")
        print("  '-' - Decrease value")
        print("  'r' - Reset to original values")
        print("  's' - Save current values")
        print("  'q' - Quit without saving")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam")
                break
                
            # Apply optimizations if calibrated
            if self.is_calibrated:
                processed = self.apply_optimizations(frame)
            else:
                processed = frame.copy()
                
            # Display current property info
            name, prop = properties[current_property_index]
            value = current_values[prop]
            
            cv2.putText(processed, f"Adjusting: {name} = {value}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(processed, "Press 'n'/'p' to change property, '+'/' to adjust", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display the processed frame
            cv2.imshow("Camera Adjustment", processed)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('n'):
                # Next property
                current_property_index = (current_property_index + 1) % len(properties)
                
            elif key == ord('p'):
                # Previous property
                current_property_index = (current_property_index - 1) % len(properties)
                
            elif key == ord('+'):
                # Increase value
                name, prop = properties[current_property_index]
                value = current_values[prop]
                
                # Increment by an appropriate amount based on the property
                increment = 0.1
                if prop == cv2.CAP_PROP_AUTO_EXPOSURE:
                    increment = 1  # Auto exposure is usually 0 or 1
                    
                new_value = value + increment
                if cap.set(prop, new_value):
                    current_values[prop] = new_value
                    print(f"Increased {name} to {new_value}")
                    
            elif key == ord('-'):
                # Decrease value
                name, prop = properties[current_property_index]
                value = current_values[prop]
                
                # Decrement by an appropriate amount
                decrement = 0.1
                if prop == cv2.CAP_PROP_AUTO_EXPOSURE:
                    decrement = 1
                    
                new_value = max(0, value - decrement)  # Prevent negative values
                if cap.set(prop, new_value):
                    current_values[prop] = new_value
                    print(f"Decreased {name} to {new_value}")
                    
            elif key == ord('r'):
                # Reset values
                for name, prop in properties:
                    orig_value = original_values[prop]
                    cap.set(prop, orig_value)
                    current_values[prop] = orig_value
                    
                print("Reset all properties to original values")
                
            elif key == ord('s'):
                # Save values
                print("\nSaved camera parameters:")
                for name, prop in properties:
                    print(f"{name}: {current_values[prop]}")
                    
                # Optional: save to a config file
                try:
                    with open(os.path.join(self.calibration_dir, "camera_params.pkl"), 'wb') as f:
                        pickle.dump(current_values, f)
                    print("Parameters saved to file")
                except Exception as e:
                    print(f"Error saving parameters: {str(e)}")
                    
                break
                
            elif key == ord('q'):
                # Quit without saving
                print("Exiting without saving parameters")
                
                # Reset to original values
                for name, prop in properties:
                    cap.set(prop, original_values[prop])
                    
                break
                
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

def display_menu():
    """Display calibration menu options."""
    print("\nCamera Calibration & Optimization")
    print("================================")
    print("1. Run camera calibration")
    print("2. Test calibration (before/after)")
    print("3. Adjust camera parameters")
    print("4. Back to main menu")
    choice = input("\nEnter your choice (1-4): ")
    return choice

def main():
    """Main function to run the calibration module."""
    calibrator = CameraCalibration()
    
    while True:
        choice = display_menu()
        
        if choice == '1':
            # Run calibration
            num_images = input("Enter number of images to capture (default: 10): ")
            num_images = int(num_images) if num_images.strip().isdigit() else 10
            calibrator.calibrate_with_chessboard(num_images=num_images)
            
        elif choice == '2':
            # Test calibration
            calibrator.calibration_test()
            
        elif choice == '3':
            # Adjust camera parameters
            calibrator.adjust_camera_parameters()
            
        elif choice == '4':
            print("Returning to main menu.")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()