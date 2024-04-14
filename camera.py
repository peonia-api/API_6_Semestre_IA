import cv2
import mediapipe as mp
import numpy as np
import requests

# Initialize MediaPipe Pose detection
mp_pose = mp.solutions.pose

# Open webcam
cap = cv2.VideoCapture(0)
line_left = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2 - 200)
line_right = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2 + 200)

# Define variables to control entry and exit
entered = False
exited = False
occurrence = None  # Initialize occurrence variable

# Initialize MediaPipe Pose detector
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_detector:
    while cap.isOpened():
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        results = pose_detector.process(rgb_frame)
        height, width, _ = frame.shape

        # Check if any poses are detected
        if results.pose_landmarks:
            # Initialize empty list to store landmark positions
            landmark_positions = []

            for pose_landmark in results.pose_landmarks.landmark:
                # Get the coordinates (x, y) of the landmark
                cx, cy = int(pose_landmark.x * width), int(pose_landmark.y * height)
                landmark_positions.append(cx)

            # Calculate the mean horizontal position of all landmarks
            cx_mean = np.mean(landmark_positions)

            # Check if the person crossed the entry line
            if cx_mean < line_left and not entered:
                entered = True
                exited = False
                occurrence = 1  # Register entry
                print("Entrou")

            # Check if the person crossed the exit line
            elif cx_mean > line_right and not exited:
                entered = False
                exited = True
                occurrence = 0  # Register exit
                print("Saiu")

            # Render detected landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the frame
        cv2.line(frame, (line_left, 0), (line_left, height), (0, 255, 0), 2)  # green (entrada)
        cv2.line(frame, (line_right, 0), (line_right, height), (0, 0, 255), 2)  # red (saida)
        cv2.imshow('Pose Detection', frame)

        # Check if occurrence is not None and send HTTP request
        if occurrence is not None:
            try:
                resp = requests.post("http://localhost:8080/record", json={"occurrence": occurrence, "room": "Laboratorio"})
                print(resp)
            except Exception as e:
                print(f"Erro ao enviar a solicitação HTTP: {e}")
            occurrence = None  # Reset occurrence after sending HTTP request

        # Check for exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
