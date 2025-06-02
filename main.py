import cv2
import mediapipe as mp
import numpy as np
from calculate_angle import calculate_angle_3d  # Import your 3D angle calculator

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize BlazePose GHUM pose detector
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty frame.")
        continue

    # Convert BGR to RGB for processing
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)

    # Convert back to BGR for OpenCV display
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        # Draw the pose skeleton on the image
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract 3D landmarks as numpy array [33, 3]
        landmarks = results.pose_landmarks.landmark
        landmarks_3d = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

        # Get relevant joints for right elbow angle calculation
        shoulder = landmarks_3d[12]  # RIGHT_SHOULDER
        elbow = landmarks_3d[14]     # RIGHT_ELBOW
        wrist = landmarks_3d[16]     # RIGHT_WRIST

        # Calculate the elbow angle in degrees
        elbow_angle = calculate_angle_3d(shoulder, elbow, wrist)

        # Get pixel coords for placing the angle text
        elbow_pixel = results.pose_landmarks.landmark[14]
        x = int(elbow_pixel.x * image.shape[1])
        y = int(elbow_pixel.y * image.shape[0])

        # Display the angle on the image near the right elbow
        cv2.putText(image, f'{int(elbow_angle)}°', (x, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow('BlazePose GHUM', image)

    # Exit when ESC is pressed
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
