import cv2
import numpy as np

def cyclops_effect():
    # Load the Haar Cascade classifiers for face and eyes
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Define regions of interest (ROI) for face in both grayscale and color images
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # Detect eyes within the face ROI
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) == 0:
                continue

            # Sort eyes based on the x-coordinate (from left to right)
            eyes = sorted(eyes, key=lambda ex: ex[0])

            # Select the rightmost eye (assuming it's the right eye of the person)
            ex, ey, ew, eh = eyes[-1]

            # Extract the eye region
            eye_region = roi_color[ey:ey+eh, ex:ex+ew]

            # Create a circular mask with a blurred border
            mask = np.zeros((eh, ew), dtype=np.uint8)
            center = (ew // 2, eh // 2)
            radius = min(ew, eh) // 2 - 5  # Leave space for blending
            cv2.circle(mask, center, radius, 255, -1)
            mask_blur = cv2.GaussianBlur(mask, (11, 11), 5)

            # Convert mask to 3 channels
            mask_blur_3d = cv2.merge([mask_blur, mask_blur, mask_blur])

            # Calculate the position halfway between the forehead and original eye level
            eye_level_y = ey + y
            forehead_y = y - eh // 2
            mid_y = (eye_level_y + forehead_y) // 2  # Halfway position

            # Ensure the position is within image boundaries
            if mid_y < 0:
                mid_y = 0

            # Calculate x-coordinate for center placement
            forehead_x = x + w // 2 - ew // 2

            # Overlay eye with blending onto the face
            roi_forehead = frame[mid_y:mid_y+eh, forehead_x:forehead_x+ew]

            # Blend the eye into the forehead region using the blurred mask
            roi_forehead = roi_forehead.astype(float)
            eye_region = eye_region.astype(float)
            mask_blur_3d = mask_blur_3d.astype(float) / 255  # Normalize mask to range [0,1]

            blended = (mask_blur_3d * eye_region) + ((1 - mask_blur_3d) * roi_forehead)
            blended = blended.astype(np.uint8)

            # Place the blended eye on the frame
            frame[mid_y:mid_y+eh, forehead_x:forehead_x+ew] = blended

        # Display the resulting frame
        cv2.imshow('Cyclops Effect', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Run the cyclops effect function
cyclops_effect()
