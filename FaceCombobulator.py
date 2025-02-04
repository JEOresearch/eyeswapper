import cv2
import numpy as np

# Load the pre-trained Haar Cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Open a connection to the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break  # Exit if there was an issue grabbing the frame

    # Convert the frame to grayscale (face and eye detection works better on grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Define regions of interest (ROI) for color and grayscale images
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

        # If two eyes are detected, proceed to swap them
        if len(eyes) >= 2:
            # Select the two eyes with the highest confidence (assuming they are the actual eyes)
            eyes = sorted(eyes, key=lambda ex: ex[0])  # Sort eyes based on x-coordinate

            # Get coordinates for the two eyes
            (ex1, ey1, ew1, eh1) = eyes[0]
            (ex2, ey2, ew2, eh2) = eyes[1]

            # Ensure that ex1 is the left eye and ex2 is the right eye
            if ex1 > ex2:
                (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = (ex2, ey2, ew2, eh2), (ex1, ey1, ew1, eh1)

            # Extract the eye regions from the color frame
            eye1 = roi_color[ey1:ey1 + eh1, ex1:ex1 + ew1].copy()
            eye2 = roi_color[ey2:ey2 + eh2, ex2:ex2 + ew2].copy()

            # Resize eyes to match each other's size
            eye1_resized = cv2.resize(eye1, (ew2, eh2))
            eye2_resized = cv2.resize(eye2, (ew1, eh1))

            # Swap the eyes in the ROI
            roi_color[ey1:ey1 + eh1, ex1:ex1 + ew1] = eye2_resized
            roi_color[ey2:ey2 + eh2, ex2:ex2 + ew2] = eye1_resized

    # Display the resulting frame with swapped eyes
    cv2.imshow('Face with Swapped Eyes', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
