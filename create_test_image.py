import cv2
import numpy as np
import os

# Create a blank white image
img = np.ones((800, 1000, 3), dtype=np.uint8) * 255

# Add a title
cv2.putText(img, "OCR Test Document", (250, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)

# Add multiple lines of text
text_lines = [
    "This is a test document for Tesseract OCR.",
    "The quick brown fox jumps over the lazy dog.",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "abcdefghijklmnopqrstuvwxyz",
    "0123456789",
    "Special characters: !@#$%^&*()_+-=[]{}|;':\",./<>?",
    "",
    "Tesseract is an optical character recognition engine",
    "for various operating systems. It is free software,",
    "released under the Apache License, Version 2.0, and",
    "development has been sponsored by Google since 2006.",
    "",
    "This document tests if the OCR functionality is",
    "working properly with our Vision-Based Attendance",
    "and Emotion Monitoring System."
]

y_pos = 150
for line in text_lines:
    cv2.putText(img, line, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    y_pos += 40

# Save the image
if not os.path.exists("test_images"):
    os.makedirs("test_images")

filename = "test_images/ocr_test_document.jpg"
cv2.imwrite(filename, img)

print(f"Test image created: {filename}")

# Display the image
cv2.namedWindow("Test Image", cv2.WINDOW_NORMAL)
cv2.imshow("Test Image", img)
cv2.waitKey(2000)  # Show for 2 seconds
cv2.destroyAllWindows() 