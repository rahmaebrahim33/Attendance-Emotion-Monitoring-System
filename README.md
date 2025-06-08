# Vision-Based Attendance and Emotion Monitoring System

## Overview
This system uses computer vision to automatically track student attendance through face recognition and monitor emotional states during a session. Additionally, it can recognize hand gestures for control, extract information from physical documents using OCR, and provide visual and audio feedback.

## Core Features
1. **User Authentication**:
   - Face recognition using DeepFace
   - Only authorized students are logged in the attendance sheet

2. **Emotion Detection**:
   - Detects emotions using DeepFace emotion module
   - Classifies emotions: happy, sad, neutral, angry, surprise, fear, disgust

3. **Attendance Logging**:
   - Automatically logs recognized students with timestamp
   - Prevents duplicate logging during the same session
   - Generates attendance reports

4. **Live Feedback Display**:
   - Shows detected face name, attendance status, and real-time emotion
   - Provides visual indicators for newly marked attendance
   - Displays emotion distribution statistics

5. **Camera Calibration**:
   - Uses OpenCV to calibrate the webcam using a chessboard pattern
   - Applies calibration to improve detection accuracy

6. **Hand Gesture Recognition**:
   - Recognizes predefined gestures (open palm, closed fist, pointing, etc.)
   - Maps gestures to system commands

7. **Document OCR**:
   - Extracts text from physical documents using Tesseract OCR
   - Processes and enhances document images for better OCR results

## Installation

### Prerequisites
- Python 3.7 or higher
- Webcam
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on your system (for document scanning)

### Setup
1. Clone this repository:
   ```
   git clone <repository-url>
   cd vision-based-attendance
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. If using OCR features, ensure Tesseract is installed and available in your PATH, or specify its location in the OCR_Scanner class.

## Usage

### Running the System
1. Launch the system:
   ```
   python main.py
   ```

2. The main menu will appear with the following options:
   - **Attendance System**: Add new faces, run the attendance system, generate reports
   - **Camera Calibration**: Calibrate your camera for optimal performance
   - **Advanced Features**: Test hand gesture recognition, scan documents with OCR
   - **Settings**: Configure UI and OCR settings

### Key Controls During Operation
- **q**: Quit the system
- **s**: Save a snapshot
- **g**: Toggle gesture recognition
- **d**: Toggle debug information
- **o**: Launch OCR scanner

### Adding New Faces
1. Select "Add New Face (Webcam)" or "Add New Face (File)" from the menu
2. Enter the person's name
3. For webcam capture, position face and wait for countdown
4. For file upload, select a clear photo of the person's face

### Camera Calibration
1. Select "Calibrate Camera" from the menu
2. Hold a chessboard pattern (9x6 internal corners) in front of the camera
3. Move the chessboard to different positions as instructed
4. The system will capture multiple images and calculate calibration parameters

### Using Hand Gestures
The system recognizes the following gestures:
- **Open Palm**: Volume up
- **Closed Fist**: Volume down
- **Pointing Up** (index finger): Next page
- **Victory Sign** (index and middle fingers): Play/pause
- **Thumbs Up**: Mark attendance

### Document Scanning (OCR)
1. Select "Scan Document (OCR)" or press 'o' during operation
2. Position the document in the frame and press SPACE
3. The system will extract text and display results
4. OCR results are saved in the "ocr_results" directory

## Directory Structure
- **known_faces/**: Images of registered users
- **camera_calibration/**: Camera calibration data
- **attendance_data/**: Attendance logs and reports
- **snapshots/**: Captured images during operation
- **sounds/**: Audio feedback files
- **ocr_results/**: Results from document scanning

## Troubleshooting

### Face Recognition Issues
- Ensure adequate lighting
- Position face directly toward the camera
- Try recalibrating the camera

### Hand Gesture Problems
- Keep hand clearly visible against a contrasting background
- Make gestures slowly and deliberately
- Maintain adequate distance from the camera

### OCR Difficulties
- Use well-lit, high-contrast documents
- Position document flat and fully visible in the frame
- Adjust OCR settings for different document types

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- DeepFace for face recognition and emotion detection
- MediaPipe for hand gesture recognition
- Tesseract OCR for document text extraction
- OpenCV for computer vision capabilities 